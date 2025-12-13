// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#include "src/turbomind/models/llama/EagleModule.h"
#include "src/turbomind/models/llama/EagleBuffers.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/eagle_debug.h"

#include <fstream>
#include <new>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace turbomind {

namespace {

void logEagleError(const std::string& msg)
{
    TM_LOG_ERROR("[EAGLE][EagleModule::load] %s", msg.c_str());
}

template<typename T>
bool loadTensorFromFile(Tensor& tensor, const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        logEagleError("Failed to open " + path);
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    const auto expected = static_cast<std::streamsize>(tensor.size() * sizeof(T));
    if (size != expected) {
        logEagleError("Size mismatch for " + path + ": expected "
                      + std::to_string(expected) + ", got " + std::to_string(size));
        return false;
    }

    std::vector<T> host_data(tensor.size());
    if (!file.read(reinterpret_cast<char*>(host_data.data()), expected)) {
        logEagleError("Failed to read " + path);
        return false;
    }

    check_cuda_error(cudaMemcpy(tensor.data<T>(), host_data.data(), expected, cudaMemcpyHostToDevice));
    return true;
}

}  // namespace

// Coarse-grained progress logger for Eagle3 pipeline bring-up. This is
// defined at translation-unit scope so other EAGLE call sites (e.g.
// LlamaV2's dynamicDecodeWithSpecMulti) can reuse the same milestones.
// It is gated on the global EAGLE debug flag so normal inference runs
// are not flooded with logs.
void logEagleProgress(int percent, const char* stage)
{
    if (!isEagleDebugEnabled()) {
        return;
    }
    TM_LOG_WARNING("[EAGLE][Progress] %3d%% %s", percent, stage);
}

EagleModule::EagleModule(SizeType max_draft_path_len,
                         SizeType max_decoding_draft_tokens,
                         SizeType max_decoding_tokens,
                         SizeType max_non_leaf_nodes):
    max_draft_path_len_(max_draft_path_len),
    max_decoding_draft_tokens_(max_decoding_draft_tokens),
    max_decoding_tokens_(max_decoding_tokens),
    max_non_leaf_nodes_(max_non_leaf_nodes)
{
    initializeDefaultChoices();
}

void EagleModule::initializeDefaultChoices()
{
    // Default EAGLE tree structure: a simple chain. This is kept as a
    // safe fallback when no eagle_tree.yaml is present.
    default_eagle_choices_.clear();
    default_eagle_choices_.resize(max_decoding_tokens_);
    for (int i = 0; i + 1 < max_decoding_tokens_; ++i) {
        default_eagle_choices_[i].push_back(i + 1);
    }
}

void EagleModule::load(const std::string& model_dir, int /*device_id*/, cudaStream_t /*stream*/)
{
    enabled_ = false;
    bool success = true;

    logEagleProgress(5, "Begin EagleModule::load (parsing config.yaml)");

    const std::string config_path = model_dir + "/config.yaml";
    YAML::Node        config;
    try {
        config = YAML::LoadFile(config_path);
    }
    catch (const std::exception& e) {
        logEagleError("Failed to load config.yaml from " + model_dir + ": " + e.what());
        return;
    }

    if (!config["model_config"]) {
        logEagleError("config.yaml missing 'model_config' node");
        return;
    }

    YAML::Node model_config = config["model_config"];

    if (!model_config["hidden_units"] || !model_config["vocab_size"] || !model_config["head_num"]
        || !model_config["size_per_head"] || !model_config["inter_size"]) {
        logEagleError("config.yaml missing required fields in model_config");
        return;
    }

    int hidden_units = model_config["hidden_units"].as<int>();
    int vocab_size   = model_config["vocab_size"].as<int>();
    int head_num     = model_config["head_num"].as<int>();
    int head_dim     = model_config["size_per_head"].as<int>();
    const int attn_hidden_units = head_num * head_dim; // 4096 for GPT-OSS-120B

    if (hidden_units <= 0 || vocab_size <= 0 || head_num <= 0 || head_dim <= 0) {
        logEagleError("Invalid model_config values in config.yaml (non-positive)");
        return;
    }

    int        intermediate_size = 0;
    YAML::Node inter_node        = model_config["inter_size"];
    if (inter_node.IsSequence()) {
        intermediate_size = inter_node[0].as<int>();
    }
    else {
        intermediate_size = inter_node.as<int>();
    }
    if (intermediate_size <= 0) {
        logEagleError("Invalid inter_size in config.yaml");
        return;
    }

    // Parse optional Eagle3 geometry hints. When absent, we fall back to
    // the legacy EagleNet layout (single-hidden QKV / FC). For backward
    // compatibility with older config.yaml files that only contain the
    // geometry ints (eagle_qkv_in_dim, etc.) but no explicit
    // `eagle_mode` string, we treat the presence of those ints as an
    // implicit Eagle3 indicator.
    std::string eagle_mode_str;
    if (model_config["eagle_mode"]) {
        try {
            eagle_mode_str = model_config["eagle_mode"].as<std::string>();
        }
        catch (const std::exception&) {
            eagle_mode_str.clear();
        }
    }

    eagle_q_size_        = model_config["eagle_q_size"] ? model_config["eagle_q_size"].as<int>() : 0;
    eagle_kv_size_       = model_config["eagle_kv_size"] ? model_config["eagle_kv_size"].as<int>() : 0;
    eagle_qkv_in_dim_    = model_config["eagle_qkv_in_dim"] ? model_config["eagle_qkv_in_dim"].as<int>() : 0;
    eagle_qkv_in_factor_ = model_config["eagle_qkv_in_factor"] ? model_config["eagle_qkv_in_factor"].as<int>() : 0;
    eagle_fc_in_dim_     = model_config["eagle_fc_in_dim"] ? model_config["eagle_fc_in_dim"].as<int>() : 0;
    eagle_fc_in_factor_  = model_config["eagle_fc_in_factor"] ? model_config["eagle_fc_in_factor"].as<int>() : 0;
    if (model_config["eagle_capture_layers"] && model_config["eagle_capture_layers"].IsSequence()) {
        try {
            for (auto const& v : model_config["eagle_capture_layers"]) {
                eagle_capture_layers_cfg_.push_back(v.as<int>());
            }
        }
        catch (const std::exception&) {
            eagle_capture_layers_cfg_.clear();
        }
    }

    // Track the model's working hidden width (e.g. 2880 for GPT-OSS-120B)
    // separately from the Eagle3 attention width (e.g. eagle_q_size_ = 4096).
    // The former is used for FC / capture geometry and LM head, while the
    // latter is captured via eagle_q_size_ and attn_hidden_units.
    const int config_base_hidden = model_config["eagle_base_hidden"] ? model_config["eagle_base_hidden"].as<int>() : attn_hidden_units;

    base_hidden_units_  = hidden_units;
    draft_hidden_units_ = model_config["eagle_draft_hidden"]
        ? model_config["eagle_draft_hidden"].as<int>()
        : (eagle_qkv_in_dim_ > 0 ? eagle_qkv_in_dim_ / 2 : hidden_units);

    const bool has_eagle_geometry =
        eagle_qkv_in_dim_ > 0 || eagle_q_size_ > 0 || eagle_kv_size_ > 0 || eagle_fc_in_dim_ > 0;

    if (eagle_mode_str == "eagle3" || (eagle_mode_str.empty() && has_eagle_geometry)) {
        eagle_mode_ = EagleMode::kEagle3;
    }
    else {
        eagle_mode_ = EagleMode::kEagleNet;
    }

    // Keep hidden_units_ aligned with the model_config hidden width so it
    // matches the runtime hidden-state dimension coming from LlamaV2 /
    // UnifiedDecoder. Draft-specific tensors explicitly use
    // draft_hidden_units_ below, and attention uses base_hidden_units_ when
    // Eagle3 geometry is present.
    hidden_units_ = hidden_units;
    hidden_units  = hidden_units_;
    vocab_size_   = vocab_size;

    // Warn when base hidden does not match head geometry.
    if (base_hidden_units_ != head_num * head_dim) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule::load] base_hidden (%d) != head_num (%d) * size_per_head (%d); "
            "continuing with base_hidden=%d based on config.yaml",
            base_hidden_units_,
            head_num,
            head_dim,
            base_hidden_units_);
    }

    // Decide draft weight dtype from config.yaml. The converter records
    // the on-disk precision in `eagle_weight_dtype`:
    //   - "bf16"  -> kBfloat16
    //   - "fp16"  -> kFloat16 (default)
    DataType   dtype      = kFloat16;
    const auto dtype_node = model_config["eagle_weight_dtype"];
    if (dtype_node && dtype_node.IsScalar()) {
        try {
            const auto dtype_str = dtype_node.as<std::string>();
            if (dtype_str == "bf16" || dtype_str == "bfloat16") {
                dtype = kBfloat16;
            }
            else if (dtype_str == "fp16" || dtype_str == "float16" || dtype_str == "half") {
                dtype = kFloat16;
            }
        }
        catch (const std::exception&) {
            // Leave dtype at the default if parsing fails.
        }
    }
    weight_dtype_ = dtype;

    const bool eagle3 = eagle_mode_ == EagleMode::kEagle3;

    logEagleProgress(10, eagle3 ? "Parsed Eagle3 geometry from config.yaml"
                                : "Parsed legacy Eagle geometry from config.yaml");

    if (eagle3) {
        // For Eagle3 we require full geometry to be present and
        // consistent with the draft hidden size exported by the
        // converter. Geometry checks must line up with the HF
        // Eagle3 midlayer (fc + Q/K/V/O) instead of assuming legacy
        // base-hidden layouts.
        if (eagle_q_size_ <= 0 || eagle_kv_size_ <= 0 || eagle_qkv_in_dim_ <= 0 || eagle_fc_in_dim_ <= 0) {
            logEagleError(
                "Eagle3 config.yaml is missing required geometry "
                "(eagle_q_size/eagle_kv_size/eagle_qkv_in_dim/eagle_fc_in_dim)");
            return;
        }
        if (eagle_qkv_in_factor_ > 0 && eagle_qkv_in_dim_ != eagle_qkv_in_factor_ * draft_hidden_units_) {
            TM_LOG_ERROR(
                "[EAGLE][EagleModule::load] Eagle3 qkv_in_dim (%d) != factor (%d) * draft_hidden (%d); "
                "disabling Eagle3 draft due to geometry inconsistency.",
                eagle_qkv_in_dim_,
                eagle_qkv_in_factor_,
                draft_hidden_units_);
            success = false;
        }
        if (eagle_fc_in_factor_ > 0 && eagle_fc_in_dim_ != eagle_fc_in_factor_ * draft_hidden_units_) {
            TM_LOG_ERROR(
                "[EAGLE][EAGLE3][EagleModule::load] fc_in_dim (%d) != factor (%d) * draft_hidden (%d); "
                "disabling Eagle3 draft due to geometry inconsistency.",
                eagle_fc_in_dim_,
                eagle_fc_in_factor_,
                draft_hidden_units_);
            success = false;
        }
        // For Eagle3 we no longer require fc_in_dim to equal
        // capture_layers * base_hidden_units_. The converter encodes
        // fc_in_dim relative to draft_hidden (e.g. 3 * 2880 for
        // GPT‑OSS‑120B‑Eagle3), and UnifiedDecoder is responsible for
        // providing a captured_hidden_states buffer that matches this
        // dimension when Eagle3 is active. Keep capture_layers only as
        // a sanity check that some capture configuration exists.
        if (eagle_capture_layers_cfg_.empty()) {
            logEagleError(
                "[EAGLE][EAGLE3][EagleModule::load] eagle_capture_layers missing in config.yaml; "
                "disabling Eagle3 draft to avoid geometry drift.");
            success = false;
        }
    }

    // Token embeddings live in the same hidden space as the base model
    // hidden states (hidden_units_).
    weights_.embed_tokens = Tensor{Layout({vocab_size, hidden_units_}), dtype, kDEVICE};

    // Legacy shallow FC reduces captured features to the draft hidden width.
    // Legacy FC path stays in draft-hidden space.
    weights_.fc = Tensor{Layout({draft_hidden_units_ * 2, draft_hidden_units_}), dtype, kDEVICE};

    // Eagle3-specific FC weight that consumes the full concatenated
    // multi-layer hidden (e.g. 3 * draft_hidden_units_) before attention.
    if (eagle3 && eagle_fc_in_dim_ > 0) {
        weights_.eagle_fc = Tensor{Layout({eagle_fc_in_dim_, draft_hidden_units_}), dtype, kDEVICE};
    }

    // Layer weights (single EagleNet / Eagle3 shallow layer). Norms operate
    // in the same hidden space as the model hidden states (hidden_units_ /
    // draft_hidden_units_).
    weights_.input_norm  = Tensor{Layout({draft_hidden_units_}), dtype, kDEVICE};
    weights_.hidden_norm = Tensor{Layout({draft_hidden_units_}), dtype, kDEVICE};

    // Attention weight shapes differ slightly between EagleNet and Eagle3:
    //  - EagleNet: [hidden, 3 * hidden] and Wo [hidden, hidden]
    //  - Eagle3:   [2 * draft_hidden, q_size + 2 * kv_size] and Wo [q_size, base_hidden]
    int qkv_in_dim = hidden_units_;
    int q_size     = hidden_units_;
    int kv_size    = hidden_units_;
    if (eagle3) {
        // For Eagle3 we want attention over draft-hidden inputs:
        // QKV: [2 * draft_hidden_units_, eagle_q_size_ + 2 * eagle_kv_size_]
        // Wo:  [eagle_q_size_, base_hidden_units_]
        qkv_in_dim = eagle_qkv_in_dim_ > 0 ? eagle_qkv_in_dim_ : 2 * draft_hidden_units_;
        q_size     = eagle_q_size_ > 0 ? eagle_q_size_ : attn_hidden_units;
        kv_size    = eagle_kv_size_ > 0 ? eagle_kv_size_ : attn_hidden_units;
    }

    const int qkv_out_dim = q_size + 2 * kv_size;

    weights_.attn_qkv = Tensor{Layout({qkv_in_dim, qkv_out_dim}), dtype, kDEVICE};
    // Wo projects back to base_hidden to align with target model width.
    weights_.attn_o    = Tensor{Layout({q_size, base_hidden_units_}), dtype, kDEVICE};
    // This norm is used in the Eagle3 draft layer and therefore stays in
    // draft-hidden space.
    weights_.attn_norm = Tensor{Layout({draft_hidden_units_}), dtype, kDEVICE};

    // Draft FFN stays in draft_hidden space.
    weights_.mlp_gate_up = Tensor{Layout({intermediate_size * 2, draft_hidden_units_}), dtype, kDEVICE};
    weights_.mlp_down    = Tensor{Layout({intermediate_size, draft_hidden_units_}), dtype, kDEVICE};

    // Output norm and LM head operate in the same hidden space as the base
    // model hidden states (hidden_units_ / draft_hidden_units_).
    weights_.output_norm = Tensor{Layout({draft_hidden_units_}), dtype, kDEVICE};
    weights_.lm_head     = Tensor{Layout({draft_hidden_units_, vocab_size}), dtype, kDEVICE};

    // Optional native Eagle3 attention projections (non‑LLaMA geometry).
    // Only allocate these when we have full Eagle3 geometry; otherwise
    // they remain empty and the dedicated Eagle3Attention backend will
    // be considered unavailable.
    if (eagle3 && eagle_q_size_ > 0 && eagle_kv_size_ > 0 && eagle_qkv_in_dim_ > 0) {
        // midlayer.self_attn.q_proj.weight: [q_out, q_in]   = [eagle_q_size_, eagle_qkv_in_dim_]
        // midlayer.self_attn.k_proj.weight: [kv_out, q_in]  = [eagle_kv_size_, eagle_qkv_in_dim_]
        // midlayer.self_attn.v_proj.weight: [kv_out, q_in]  = [eagle_kv_size_, eagle_qkv_in_dim_]
        // midlayer.self_attn.o_proj.weight: [hidden, q_out] = [draft_hidden_units_, eagle_q_size_]
        weights_.eagle_q_proj = Tensor{Layout({eagle_q_size_, eagle_qkv_in_dim_}), dtype, kDEVICE};
        weights_.eagle_k_proj = Tensor{Layout({eagle_kv_size_, eagle_qkv_in_dim_}), dtype, kDEVICE};
        weights_.eagle_v_proj = Tensor{Layout({eagle_kv_size_, eagle_qkv_in_dim_}), dtype, kDEVICE};
        weights_.eagle_o_proj = Tensor{Layout({draft_hidden_units_, eagle_q_size_}), dtype, kDEVICE};
    }

    weights_.is_initialized = true;

    const std::string base = model_dir + "/";

    // Helper that tries both canonical and TP-split names (e.g.
    // `tok_embeddings.weight` and `tok_embeddings.0.weight`) when
    // loading weights from a TurboMind export directory.
    auto load_with_variants = [&](Tensor& tensor, const std::string& canonical) -> bool {
        const std::string primary = base + canonical;
        // Use the same element type as the allocated tensor.
        if (tensor.dtype() == kBfloat16) {
            if (loadTensorFromFile<bfloat16_t>(tensor, primary)) {
                return true;
            }
        }
        else {
            if (loadTensorFromFile<half>(tensor, primary)) {
                return true;
            }
        }

        // Fallback: TP-split layout (".0" before the last extension).
        auto pos = canonical.rfind('.');
        if (pos != std::string::npos) {
            std::string split_name = canonical.substr(0, pos) + ".0" + canonical.substr(pos);
            const std::string alt  = base + split_name;
            if (tensor.dtype() == kBfloat16) {
                if (loadTensorFromFile<bfloat16_t>(tensor, alt)) {
                    return true;
                }
            }
            else {
                if (loadTensorFromFile<half>(tensor, alt)) {
                    return true;
                }
            }
        }

        return false;
    };

    // Model-level weights
    // Note: embed_tokens and fc are optional for the current EagleModule
    // forward path (which uses target hidden states + draft LM head).
    // When missing, we log a warning but keep the module enabled.
    if (!load_with_variants(weights_.embed_tokens, "tok_embeddings.weight")) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule::load] tok_embeddings.weight missing; draft module will skip token embeddings");
    }

    bool fc_ok = load_with_variants(weights_.fc, "fc.weight");
    if (!fc_ok) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule::load] fc.weight missing; shallow Eagle block disabled for this engine");
    }

    bool eagle_fc_ok = true;
    if (weights_.eagle_fc) {
        eagle_fc_ok = load_with_variants(weights_.eagle_fc, "eagle_fc.weight");
        if (!eagle_fc_ok) {
            TM_LOG_WARNING(
                "[EAGLE][EagleModule::load] eagle_fc.weight missing; Eagle3 pre-FC will be disabled");
        }
    }

    // Layer 0 norms and attention (all optional – when absent we fall
    // back to a minimal RMSNorm + LM head path).
    if (!load_with_variants(weights_.input_norm, "layers.0.attention_norm.weight")) {
        TM_LOG_WARNING("[EAGLE][EagleModule::load] layers.0.attention_norm.weight missing");
    }
    if (!load_with_variants(weights_.hidden_norm, "layers.0.hidden_norm.weight")) {
        TM_LOG_WARNING("[EAGLE][EagleModule::load] layers.0.hidden_norm.weight missing");
    }
    bool attn_qkv_ok = load_with_variants(weights_.attn_qkv, "layers.0.attention.w_qkv.weight");
    if (!attn_qkv_ok) {
        TM_LOG_WARNING("[EAGLE][EagleModule::load] layers.0.attention.w_qkv.weight missing");
    }
    bool attn_o_ok = load_with_variants(weights_.attn_o, "layers.0.attention.wo.weight");
    if (!attn_o_ok) {
        TM_LOG_WARNING("[EAGLE][EagleModule::load] layers.0.attention.wo.weight missing");
    }
    // Native Eagle3 Q/K/V/O projections (optional). These are only
    // expected for true Eagle3 midlayer checkpoints; when absent we
    // simply keep the dedicated Eagle3 attention backend disabled.
    bool eagle_q_ok = false;
    bool eagle_k_ok = false;
    bool eagle_v_ok = false;
    bool eagle_o_ok = false;
    if (weights_.eagle_q_proj) {
        eagle_q_ok = load_with_variants(weights_.eagle_q_proj, "eagle3.q_proj.weight");
        if (!eagle_q_ok) {
            TM_LOG_WARNING(
                "[EAGLE][EagleModule::load] eagle3.q_proj.weight missing; Eagle3 native attention disabled for this engine");
        }
    }
    if (weights_.eagle_k_proj) {
        eagle_k_ok = load_with_variants(weights_.eagle_k_proj, "eagle3.k_proj.weight");
        if (!eagle_k_ok) {
            TM_LOG_WARNING(
                "[EAGLE][EagleModule::load] eagle3.k_proj.weight missing; Eagle3 native attention disabled for this engine");
        }
    }
    if (weights_.eagle_v_proj) {
        eagle_v_ok = load_with_variants(weights_.eagle_v_proj, "eagle3.v_proj.weight");
        if (!eagle_v_ok) {
            TM_LOG_WARNING(
                "[EAGLE][EagleModule::load] eagle3.v_proj.weight missing; Eagle3 native attention disabled for this engine");
        }
    }
    if (weights_.eagle_o_proj) {
        eagle_o_ok = load_with_variants(weights_.eagle_o_proj, "eagle3.o_proj.weight");
        if (!eagle_o_ok) {
            TM_LOG_WARNING(
                "[EAGLE][EagleModule::load] eagle3.o_proj.weight missing; Eagle3 native attention disabled for this engine");
        }
    }
    if (!load_with_variants(weights_.attn_norm, "layers.0.ffn_norm.weight")) {
        TM_LOG_WARNING("[EAGLE][EagleModule::load] layers.0.ffn_norm.weight missing");
    }

    // MLP gate/up: merge w1 and w3 into mlp_gate_up.
    Tensor w1{{intermediate_size, hidden_units}, dtype, kDEVICE};
    Tensor w3{{intermediate_size, hidden_units}, dtype, kDEVICE};
    bool   w1_ok = load_with_variants(w1, "layers.0.feed_forward.w1.weight");
    bool   w3_ok = load_with_variants(w3, "layers.0.feed_forward.w3.weight");
    if (w1_ok && w3_ok) {
        const size_t elem_count = static_cast<size_t>(w1.size());
        const size_t elem_bytes = static_cast<size_t>(byte_size(dtype, 1));
        const size_t w_bytes    = elem_count * elem_bytes;
        void*        dst_base   = weights_.mlp_gate_up.raw_data();
        void*        src_w1     = w1.raw_data();
        void*        src_w3     = w3.raw_data();

        // Copy w1 into the first half of mlp_gate_up, and w3 into the second
        // half, treating the buffers as contiguous ranges of `dtype` elements.
        check_cuda_error(
            cudaMemcpy(dst_base, src_w1, w_bytes, cudaMemcpyDeviceToDevice));
        check_cuda_error(cudaMemcpy(static_cast<char*>(dst_base) + w_bytes,
                                    src_w3,
                                    w_bytes,
                                    cudaMemcpyDeviceToDevice));
    }
    else {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule::load] feed_forward w1/w3 weights missing; skipping draft MLP gate/up block");
    }

    bool mlp_down_ok = load_with_variants(weights_.mlp_down, "layers.0.feed_forward.w2.weight");
    if (!mlp_down_ok) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule::load] layers.0.feed_forward.w2.weight missing; draft MLP down block disabled");
    }

    // Output norm and LM head are required for EagleModule to be usable.
    bool output_norm_ok = load_with_variants(weights_.output_norm, "norm.weight");
    bool lm_head_ok     = load_with_variants(weights_.lm_head, "output.weight");
    success &= output_norm_ok && lm_head_ok;

    // Prepare LM head wrapper for LlamaLinear. The input dim for the
    // draft LM head is taken from the first dimension of the loaded
    // weight tensor to avoid geometry mismatches when the draft head
    // lives in draft_hidden space (e.g. 2880) instead of the decoder
    // hidden width.
    if (weights_.lm_head) {
        int lm_in  = this->hidden_units_;
        int lm_out = this->vocab_size_;
        if (weights_.lm_head.ndim() == 2) {
            lm_in  = weights_.lm_head.shape(0);
            lm_out = weights_.lm_head.shape(1);
        }
        lm_head_input_dim_ = lm_in;
        vocab_size_        = lm_out;

        lm_head_weight_.emplace(lm_in, lm_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
        lm_head_weight_.weight      = weights_.lm_head.borrow();
        lm_head_weight_.bias        = {};
        lm_head_weight_.data_type   = weight_dtype_;
        lm_head_weight_.weight_type = weight_dtype_;
        lm_head_weight_.input_type  = weight_dtype_;
        lm_head_weight_.prepare(/*fused_moe=*/false);
        lm_head_prepared_ = true;
    }

    // Prepare shallow EagleNet / Eagle3 attention + FC block wrappers for LlamaLinear.
    if (!draft_block_prepared_) {
        const bool have_attn      = weights_.attn_qkv && weights_.attn_o;
        const bool have_legacy_fc = weights_.fc && fc_ok;
        const bool have_eagle_fc  = weights_.eagle_fc && eagle_fc_ok;

        if (have_attn && (have_legacy_fc || have_eagle_fc)) {
            if (eagle_mode_ == EagleMode::kEagle3) {
                const int expected_qkv_in  = eagle_qkv_in_dim_;
                const int expected_qkv_out = eagle_q_size_ + 2 * eagle_kv_size_;
                const int expected_attn_o0 = eagle_q_size_;
                const int expected_attn_o1 = base_hidden_units_;
                if (weights_.attn_qkv.shape(0) != expected_qkv_in || weights_.attn_qkv.shape(1) != expected_qkv_out
                    || weights_.attn_o.shape(0) != expected_attn_o0 || weights_.attn_o.shape(1) != expected_attn_o1) {
                    logEagleError(
                        "[EAGLE][EagleModule::load] Eagle3 attention weights have unexpected shapes; disabling EAGLE for this engine");
                    success = false;
                }
                const int expected_fc_in  = draft_hidden_units_ * 2;
                const int expected_fc_out = draft_hidden_units_;
                if (have_legacy_fc
                    && (weights_.fc.shape(0) != expected_fc_in || weights_.fc.shape(1) != expected_fc_out)) {
                    logEagleError(
                        "[EAGLE][EagleModule::load] Eagle3 legacy fc.weight shape mismatch; disabling EAGLE for this engine");
                    success = false;
                }
                if (have_eagle_fc
                    && (weights_.eagle_fc.shape(0) != eagle_fc_in_dim_
                        || weights_.eagle_fc.shape(1) != draft_hidden_units_)) {
                    logEagleError(
                        "[EAGLE][EagleModule::load] Eagle3 eagle_fc.weight shape mismatch; disabling EAGLE for this engine");
                    success = false;
                }
                if (!success) {
                    enabled_ = false;
                    return;
                }
            }

            const int qkv_in  = weights_.attn_qkv.shape(0);
            const int qkv_out = weights_.attn_qkv.shape(1);
            attn_qkv_weight_.emplace(
                qkv_in, qkv_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
            attn_qkv_weight_.weight      = weights_.attn_qkv.borrow();
            attn_qkv_weight_.bias        = {};
            attn_qkv_weight_.data_type   = this->weight_dtype_;
            attn_qkv_weight_.weight_type = this->weight_dtype_;
            attn_qkv_weight_.input_type  = this->weight_dtype_;
            attn_qkv_weight_.prepare(/*fused_moe=*/false);

            // Attention output: [q_size, hidden]
            const int attn_o_in  = weights_.attn_o.shape(0);
            const int attn_o_out = weights_.attn_o.shape(1);
            attn_o_weight_.emplace(
                attn_o_in, attn_o_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
            attn_o_weight_.weight      = weights_.attn_o.borrow();
            attn_o_weight_.bias        = {};
            attn_o_weight_.data_type   = this->weight_dtype_;
            attn_o_weight_.weight_type = this->weight_dtype_;
            attn_o_weight_.input_type  = this->weight_dtype_;
            attn_o_weight_.prepare(/*fused_moe=*/false);

            // Legacy EagleNet FC “MLP” projection: [2 * hidden, hidden]
            if (have_legacy_fc) {
                const int fc_in  = weights_.fc.shape(0);
                const int fc_out = weights_.fc.shape(1);
                fc_weight_.emplace(
                    fc_in, fc_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
                fc_weight_.weight      = weights_.fc.borrow();
                fc_weight_.bias        = {};
                fc_weight_.data_type   = this->weight_dtype_;
                fc_weight_.weight_type = this->weight_dtype_;
                fc_weight_.input_type  = this->weight_dtype_;
                fc_weight_.prepare(/*fused_moe=*/false);
            }

            // Eagle3 pre-FC over concatenated multi-layer hidden:
            // [eagle_fc_in_dim, hidden]
            if (have_eagle_fc) {
                const int eagle_fc_in  = weights_.eagle_fc.shape(0);
                const int eagle_fc_out = weights_.eagle_fc.shape(1);
                eagle_fc_weight_.emplace(
                    eagle_fc_in, eagle_fc_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
                eagle_fc_weight_.weight      = weights_.eagle_fc.borrow();
                eagle_fc_weight_.bias        = {};
                eagle_fc_weight_.data_type   = this->weight_dtype_;
                eagle_fc_weight_.weight_type = this->weight_dtype_;
                eagle_fc_weight_.input_type  = this->weight_dtype_;
                eagle_fc_weight_.prepare(/*fused_moe=*/false);
            }

            draft_block_prepared_ = true;
        }
        else {
            TM_LOG_WARNING("[EAGLE] Draft block weights missing; using RMSNorm+LM head only");
        }
    }

    // For Eagle3 drafts, additionally prepare FFN wrapper, norms, and
    // shape checks for a more faithful draft layer path using existing
    // LLaMA FFN blocks. Attention wiring (LlamaAttentionWeight) is kept
    // separate and guarded by geometry checks.
    if (eagle_mode_ == EagleMode::kEagle3 && mlp_down_ok && weights_.mlp_gate_up && weights_.output_norm) {
        eagle3_draft_layer_ = std::make_unique<Eagle3DraftLayerWeight>();

        // Attach norms.
        eagle3_draft_layer_->hidden_norm    = weights_.hidden_norm.borrow();
        eagle3_draft_layer_->input_norm     = weights_.input_norm.borrow();
        eagle3_draft_layer_->post_attn_norm = weights_.attn_norm.borrow();
        eagle3_draft_layer_->output_norm    = weights_.output_norm.borrow();

        // Populate newly added fields for geometry propagation
        eagle3_draft_layer_->base_hidden_dim  = base_hidden_units_;
        eagle3_draft_layer_->draft_hidden_dim = draft_hidden_units_;
        eagle3_draft_layer_->head_num         = head_num;
        eagle3_draft_layer_->kv_head_num      = head_num; // Assuming MHA if not specified, otherwise needs config entry
        eagle3_draft_layer_->size_per_head    = head_dim;

        // Minimal FFN wrapper: treat mlp_gate_up as a fused gating
        // projection and mlp_down as the output projection. We rely on
        // the same LlamaLinear backend used elsewhere in EagleModule.
        const int inter_size = intermediate_size;
        const int tp_size    = 1;
        const int tp_rank    = 0;
        const int group_size = 1;

        new (&eagle3_draft_layer_->ffn) LlamaFfnWeight(draft_hidden_units_,
                                                       inter_size,
                                                       /*bias=*/false,
                                                       tp_size,
                                                       tp_rank,
                                                       weight_dtype_,
                                                       weight_dtype_,
                                                       group_size,
                                                       ActivationType::kSilu,
                                                       /*fuse_silu_act=*/true);
        eagle3_draft_layer_->ffn.debug_name = "EAGLE3_FFN";

        eagle3_draft_layer_->ffn.fused_gating_intermediate.weight = weights_.mlp_gate_up.borrow();
        eagle3_draft_layer_->ffn.output.weight                    = weights_.mlp_down.borrow();
        eagle3_draft_layer_->ffn.prepare(/*fused_moe=*/false);

        // Optional FC capture projection for building the 2H QKV input
        // from concatenated captured hidden states. When present, the
        // draft layer can compute fc_out directly from the captured
        // hidden tensor instead of relying on caller-provided tiling.
        eagle3_draft_layer_->fc_weight = weights_.eagle_fc.borrow();

        // Hard geometry checks for Eagle3 draft layer. When any of these
        // invariants fail we disable the Eagle3 draft layer only and fall
        // back to the legacy shallow block, without disabling EAGLE.
        bool draft_ok = true;

        auto check_shape_2d = [&](const Tensor& t, int dim0, int dim1, const char* name) {
            if (!t || t.ndim() != 2 || t.shape(0) != dim0 || t.shape(1) != dim1) {
                TM_LOG_WARNING(
                    "[EAGLE][Eagle3DraftLayer][fallback] %s has shape [%d,%d], expected [%d,%d]; "
                    "disabling Eagle3 draft layer.",
                    name,
                    t ? t.shape(0) : -1,
                    t ? t.shape(1) : -1,
                    dim0,
                    dim1);
                draft_ok = false;
            }
        };

        // Non-fatal attention shape check: for attention we can fall back
        // to the shallow QKV path instead of disabling the Eagle3 draft
        // layer globally.
        auto check_shape_2d_attn = [&](const Tensor& t,
                                       int dim0,
                                       int dim1,
                                       const char* name,
                                       bool& ok_flag) {
            if (!t || t.ndim() != 2 || t.shape(0) != dim0 || t.shape(1) != dim1) {
                TM_LOG_WARNING(
                    "[EAGLE][Eagle3DraftLayer][fallback] %s has shape [%d,%d], expected [%d,%d]; "
                    "Eagle3 draft will fall back to shallow attention.",
                    name,
                    t ? t.shape(0) : -1,
                    t ? t.shape(1) : -1,
                    dim0,
                    dim1);
                ok_flag = false;
            }
        };

        auto check_norm = [&](const Tensor& t, const char* name) {
            const int expected = draft_hidden_units_;
            if (!t || t.ndim() != 1 || t.shape(0) != expected) {
                TM_LOG_WARNING(
                    "[EAGLE][Eagle3DraftLayer][fallback] %s norm has shape [%d], expected [%d]; "
                    "disabling Eagle3 draft layer.",
                    name,
                    t ? t.shape(0) : -1,
                    expected);
                draft_ok = false;
            }
        };

        // FFN geometry:
        //   mlp_gate_up: [2 * inter_size, draft_hidden_units_]
        //   mlp_down:    [inter_size, draft_hidden_units_]
        check_shape_2d(weights_.mlp_gate_up, inter_size * 2, draft_hidden_units_, "mlp_gate_up");
        check_shape_2d(weights_.mlp_down, inter_size, draft_hidden_units_, "mlp_down");

        // Attention geometry for Eagle3 shallow block (LLaMA-style path):
        //   attn_qkv: [2 * draft_hidden_units_, eagle_q_size_ + 2 * eagle_kv_size_]
        //   attn_o:   [eagle_q_size_, base_hidden_units_]
        bool attn_ok = true;
        if (eagle_q_size_ > 0 && eagle_kv_size_ > 0) {
            const int qkv_expected_in  = eagle_qkv_in_dim_;
            const int qkv_expected_out = eagle_q_size_ + 2 * eagle_kv_size_;
            check_shape_2d_attn(weights_.attn_qkv, qkv_expected_in, qkv_expected_out, "attn_qkv", attn_ok);

            const int attn_o_expected_in  = eagle_q_size_;
            const int attn_o_expected_out = base_hidden_units_;
            check_shape_2d_attn(weights_.attn_o, attn_o_expected_in, attn_o_expected_out, "attn_o", attn_ok);

            // For Eagle3 GPT-OSS we rely on the dedicated Eagle3Attention
            // backend instead of trying to reshape the fused QKV / Wo
            // weights into a LLaMA-style LlamaAttentionWeight. The shallow
            // path continues to use attn_qkv / attn_o via LlamaLinear, so
            // we skip constructing a draft LlamaAttentionWeight here to
            // avoid misleading shape-mismatch warnings.
        }

        // Native Eagle3 attention geometry (non‑LLaMA). When available, we
        // initialise Eagle3DraftLayerWeight::eagle3_attn so that the
        // dedicated Eagle3Attention backend can be used instead of trying
        // to reshape Eagle3 Q/K/V/O into a LLaMA layout.
        if (eagle_q_ok && eagle_k_ok && eagle_v_ok && eagle_o_ok) {
            eagle3_draft_layer_->eagle3_attn.init_from_tensors(
                weights_.eagle_q_proj.borrow(),
                weights_.eagle_k_proj.borrow(),
                weights_.eagle_v_proj.borrow(),
                weights_.eagle_o_proj.borrow());

            auto& ew = eagle3_draft_layer_->eagle3_attn;
            // Strict shape checks against Eagle3 geometry from config.yaml.
            if (ew.is_initialized) {
                if (!ew.q_proj || ew.q_proj.ndim() != 2
                    || ew.q_proj.shape(0) != eagle_q_size_
                    || ew.q_proj.shape(1) != eagle_qkv_in_dim_) {
                    TM_LOG_WARNING(
                        "[EAGLE3][EagleModule::load] q_proj shape [%d,%d] "
                        "!= [%d,%d]; disabling Eagle3 attention for this engine.",
                        ew.q_proj ? ew.q_proj.shape(0) : -1,
                        ew.q_proj ? ew.q_proj.shape(1) : -1,
                        eagle_q_size_,
                        eagle_qkv_in_dim_);
                    ew.is_initialized = false;
                }
                if (ew.is_initialized
                    && (!ew.k_proj || ew.k_proj.ndim() != 2
                        || ew.k_proj.shape(0) != eagle_kv_size_
                        || ew.k_proj.shape(1) != eagle_qkv_in_dim_)) {
                    TM_LOG_WARNING(
                        "[EAGLE3][EagleModule::load] k_proj shape [%d,%d] "
                        "!= [%d,%d]; disabling Eagle3 attention for this engine.",
                        ew.k_proj ? ew.k_proj.shape(0) : -1,
                        ew.k_proj ? ew.k_proj.shape(1) : -1,
                        eagle_kv_size_,
                        eagle_qkv_in_dim_);
                    ew.is_initialized = false;
                }
                if (ew.is_initialized
                    && (!ew.v_proj || ew.v_proj.ndim() != 2
                        || ew.v_proj.shape(0) != eagle_kv_size_
                        || ew.v_proj.shape(1) != eagle_qkv_in_dim_)) {
                    TM_LOG_WARNING(
                        "[EAGLE3][EagleModule::load] v_proj shape [%d,%d] "
                        "!= [%d,%d]; disabling Eagle3 attention for this engine.",
                        ew.v_proj ? ew.v_proj.shape(0) : -1,
                        ew.v_proj ? ew.v_proj.shape(1) : -1,
                        eagle_kv_size_,
                        eagle_qkv_in_dim_);
                    ew.is_initialized = false;
                }
                if (ew.is_initialized
                    && (!ew.o_proj || ew.o_proj.ndim() != 2
                        || ew.o_proj.shape(0) != draft_hidden_units_
                        || ew.o_proj.shape(1) != eagle_q_size_)) {
                    TM_LOG_WARNING(
                        "[EAGLE3][EagleModule::load] o_proj shape [%d,%d] "
                        "!= [%d,%d]; disabling Eagle3 attention for this engine.",
                        ew.o_proj ? ew.o_proj.shape(0) : -1,
                        ew.o_proj ? ew.o_proj.shape(1) : -1,
                        draft_hidden_units_,
                        eagle_q_size_);
                    ew.is_initialized = false;
                }

                // When geometry checks pass, derive the head layout for the
                // Eagle3 attention backend from the base model config. We
                // treat q_out as head_num * head_dim and kv_out as
                // kv_head_num * head_dim, mirroring the standard GQA
                // convention. This metadata is required by
                // Eagle3AttentionLayer to interpret the flattened Q/K/V
                // projections.
                if (ew.is_initialized) {
                    const int attn_hidden_units = head_num * head_dim;
                    if (ew.q_out == attn_hidden_units && head_dim > 0 && ew.kv_out % head_dim == 0) {
                        ew.head_dim     = head_dim;
                        ew.num_q_heads  = head_num;
                        ew.num_kv_heads = ew.kv_out / head_dim;
                        ew.rope_base    = static_cast<float>(rope_base_);
                        ew.rope_scale   = static_cast<float>(rope_scale_);
                    }
                    else {
                        TM_LOG_WARNING(
                            "[EAGLE3][EagleModule::load] unable to derive Eagle3 head layout "
                            "(q_out=%d, kv_out=%d, hidden=%d, head_num=%d, head_dim=%d); "
                            "disabling Eagle3 attention for this engine.",
                            ew.q_out,
                            ew.kv_out,
                            attn_hidden_units,
                            head_num,
                            head_dim);
                        ew.is_initialized = false;
                    }
                }
            }
        }

        // Norms:
        check_norm(eagle3_draft_layer_->input_norm, "input");
        check_norm(eagle3_draft_layer_->post_attn_norm, "post_attn");
        check_norm(eagle3_draft_layer_->output_norm, "output");

        if (!draft_ok) {
            // Disable only the Eagle3 draft layer; keep EAGLE enabled and
            // fall back to the legacy shallow block + output_norm path.
            eagle3_draft_layer_.reset();
        } else {
            logEagleProgress(40, "Prepared Eagle3DraftLayer weights (FFN + norms + native attention)");
        }
    }

    // Optionally load a model-specific EAGLE tree definition.
    const std::string tree_path = model_dir + "/eagle_tree.yaml";
    std::ifstream     tree_stream(tree_path);
    if (tree_stream.good()) {
        try {
            YAML::Node tree_cfg = YAML::LoadFile(tree_path);
            auto       choices  = tree_cfg["choices"];
            if (choices && choices.IsSequence()) {
                default_eagle_choices_.clear();
                default_eagle_choices_.resize(choices.size());
                for (size_t i = 0; i < choices.size(); ++i) {
                    auto node_children = choices[i];
                    if (!node_children || !node_children.IsSequence()) {
                        continue;
                    }
                    for (auto const& child : node_children) {
                        default_eagle_choices_[i].push_back(child.as<EagleModule::SizeType>());
                    }
                }
                TM_LOG_INFO("[EAGLE] Loaded tree choices from %s (nodes=%zu)",
                            tree_path.c_str(),
                            default_eagle_choices_.size());
            }
            else {
                TM_LOG_WARNING("[EAGLE] eagle_tree.yaml found but missing 'choices' list; keeping default tree");
            }
        }
        catch (const std::exception& e) {
            TM_LOG_WARNING("[EAGLE] Failed to parse eagle_tree.yaml at %s: %s",
                           tree_path.c_str(),
                           e.what());
        }
    }

    // For Eagle3 drafts, require the full shallow block geometry to be
    // present. When critical weights are missing, we treat the draft
    // model as unusable instead of silently falling back to identity/
    // minimal paths, so behaviour stays aligned with the exported
    // Eagle3 config.
    if (eagle_mode_ == EagleMode::kEagle3) {
        const bool block_ok = fc_ok && attn_qkv_ok && attn_o_ok && mlp_down_ok;
        if (!block_ok) {
            logEagleError(
                "Eagle3 draft block weights incomplete (fc/attn_qkv/attn_o/mlp_down); disabling EAGLE for this engine");
            success = false;
        }
    }

    if (!success) {
        logEagleError("Draft model load failed; disabling EAGLE for this engine");
        enabled_ = false;
        return;
    }

    enabled_ = true;
    logEagleProgress(50, "EagleModule draft weights loaded and validated");
    TM_LOG_INFO(
        "[EAGLE] EagleModule draft model loaded successfully: hidden_units=%d vocab_size=%d", hidden_units_, vocab_size_);
}

void EagleModule::forward_draft_tree(const Tensor& last_hidden_states,
                                     const Tensor& captured_hidden_states,
                                     const Tensor& base_logits,
                                     int           tokens_per_seq,
                                     EagleBuffers& buffers,
                                     Tensor&       draft_logits_buffer,
                                     LlamaLinear&  linear,
                                     cudaStream_t  stream)
{
    NvtxScope nvtx_scope("EagleModule::forward_draft_tree");
    if (!enabled_ || !weights_.is_initialized || tokens_per_seq <= 0) {
        return;
    }

    const int batch_size = last_hidden_states.shape(0);
    if (batch_size <= 0) {
        return;
    }

    // Reuse the main forward path to obtain per-sequence draft logits
    // for this step. We ignore input_ids here because the current draft
    // network derives its inputs from hidden + captured_hidden.
    Tensor dummy_ids;
    Tensor draft_hidden;
    forward(dummy_ids,
            last_hidden_states,
            captured_hidden_states,
            draft_logits_buffer,
            draft_hidden,
            linear,
            stream);

    if (!draft_logits_buffer) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule] forward_draft_tree received empty draft logits; "
            "leaving EagleBuffers draft_tokens untouched for this step.");
        return;
    }

    const int vocab_size_pad = draft_logits_buffer.shape(1);
    const int vocab_size     = vocab_size_ > 0 ? vocab_size_ : vocab_size_pad;
    const int total_draft    = batch_size * tokens_per_seq;

    if (total_draft <= 0) {
        return;
    }

    // Prepare FP32 draft logits for GPU top-k.
    Tensor eagle_sampling_logits{core::Layout{batch_size, vocab_size_pad}, kFloat32, kDEVICE};
    if (draft_logits_buffer.dtype() != kFloat32 && draft_logits_buffer.dtype() != kBfloat16
        && draft_logits_buffer.dtype() != kFloat16) {
        TM_LOG_WARNING(
            "[EAGLE][EagleModule] draft_logits dtype=%s is not f16/bf16/f32 in forward_draft_tree; "
            "skipping device-side top-k.",
            to_string(draft_logits_buffer.dtype()));
        return;
    }
    invokeCastFloat2D(draft_logits_buffer, eagle_sampling_logits, stream);
    sync_check_cuda_error();

    using turbomind::TopKSortFilterParams;

    const int logit_elems = batch_size * vocab_size_pad;

    Buffer_<int> draft_tokens(total_draft, kCPU);
    Buffer_<int> target_tokens(total_draft, kCPU);

    if (tokens_per_seq > 0 && tokens_per_seq <= vocab_size_pad && tokens_per_seq <= 1024) {
        Buffer_<int> topk_buf(batch_size, kDEVICE);
        Buffer_<int> kept_buf(batch_size, kDEVICE);

        std::vector<int> h_topk(batch_size, tokens_per_seq);
        std::vector<int> h_kept(batch_size, tokens_per_seq);

        core::Copy(h_topk.data(), batch_size, topk_buf.data());
        core::Copy(h_kept.data(), batch_size, kept_buf.data());

        Buffer_<int> draft_sorted_indices(logit_elems, kDEVICE);
        Buffer_<int> target_sorted_indices(logit_elems, kDEVICE);

        TopKSortFilterParams draft_params{};
        draft_params.logits            = eagle_sampling_logits.buffer().raw_data();
        draft_params.sorted_logits     = eagle_sampling_logits.buffer().raw_data();
        draft_params.sorted_indices    = draft_sorted_indices.data();
        draft_params.kept              = kept_buf.data();
        draft_params.top_ks            = topk_buf.data();
        draft_params.max_top_k         = tokens_per_seq;
        draft_params.batch_size        = batch_size;
        draft_params.vocab_size        = vocab_size;
        draft_params.vocab_size_padded = vocab_size_pad;

        invokeTopKSortFilter<float>(draft_params, stream);

        TopKSortFilterParams target_params{};
        if (base_logits) {
            // buffer().raw_data() is const in this context; TopKSortFilterParams
            // expects mutable pointers for in-place sorting, so we cast away
            // constness here, matching other call sites.
            target_params.logits            = const_cast<void*>(base_logits.buffer().raw_data());
            target_params.sorted_logits     = const_cast<void*>(base_logits.buffer().raw_data());
            target_params.sorted_indices    = target_sorted_indices.data();
            target_params.kept              = kept_buf.data();
            target_params.top_ks            = topk_buf.data();
            target_params.max_top_k         = tokens_per_seq;
            target_params.batch_size        = batch_size;
            target_params.vocab_size        = vocab_size;
            target_params.vocab_size_padded = base_logits.shape(1);

            invokeTopKSortFilter<float>(target_params, stream);
        }

        const size_t idx_width_bytes = sizeof(int) * tokens_per_seq;
        const size_t idx_src_pitch   = sizeof(int) * vocab_size_pad;

        const size_t dst_pitch_tokens =
            static_cast<size_t>(max_decoding_tokens_) * sizeof(int);

        // Draft IDs: D→D into EagleBuffers::inputs.draft_tokens.
        check_cuda_error(cudaMemcpy2DAsync(buffers.inputs.draft_tokens,
                                           dst_pitch_tokens,
                                           draft_sorted_indices.data(),
                                           idx_src_pitch,
                                           idx_width_bytes,
                                           batch_size,
                                           cudaMemcpyDeviceToDevice,
                                           stream));

        // Target IDs: D→D into EagleBuffers::inputs.target_tokens when
        // base logits are provided and target-tree decode is not going
        // to overwrite them. The caller decides when tree decode runs.
        if (base_logits && buffers.inputs.target_tokens) {
            const size_t target_pitch = sizeof(int) * base_logits.shape(1);
            check_cuda_error(cudaMemcpy2DAsync(buffers.inputs.target_tokens,
                                               dst_pitch_tokens,
                                               target_sorted_indices.data(),
                                               target_pitch,
                                               idx_width_bytes,
                                               batch_size,
                                               cudaMemcpyDeviceToDevice,
                                               stream));
        }

        // Mirror the compact [batch_size, tokens_per_seq] region back to
        // host for CPU tree construction. Layout in EagleBuffers is
        // [batch_size, max_decoding_tokens]; the first tokens_per_seq
        // entries per row are exactly the IDs we just wrote.
        check_cuda_error(cudaMemcpyAsync(draft_tokens.data(),
                                         buffers.inputs.draft_tokens,
                                         static_cast<size_t>(total_draft) * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream));

        if (base_logits && buffers.inputs.target_tokens) {
            check_cuda_error(cudaMemcpyAsync(target_tokens.data(),
                                             buffers.inputs.target_tokens,
                                             static_cast<size_t>(total_draft) * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream));
        }

        check_cuda_error(cudaStreamSynchronize(stream));
    }
    else {
        std::fill_n(draft_tokens.data(), total_draft, 0);
        std::fill_n(target_tokens.data(), total_draft, 0);
    }

    if (isEagleDebugEnabled()) {
        TM_LOG_DEBUG("[EAGLE][EagleModule] forward_draft_tree: batch=%d, tokens_per_seq=%d, num_draft_tokens=%d",
                     last_hidden_states.shape(0),
                     tokens_per_seq,
                     total_draft);
    }

    // The caller (LlamaV2) is responsible for using draft_tokens /
    // target_tokens to build the tree and mirror them into any host
    // structures it maintains. Here we only populated the device
    // buffers and host mirrors.
}

void EagleModule::forward(const Tensor& input_ids,
                          const Tensor& hidden_states,
                          Tensor&       output_logits,
                          Tensor&       output_hidden_states,
                          LlamaLinear&  linear,
                          cudaStream_t  stream)
{
    // Backwards-compatible wrapper: no captured multi-layer hidden
    // states are provided, so we pass an empty tensor for the new
    // parameter and keep behaviour identical to pre-Eagle3 wiring.
    Tensor empty_captured;
    forward(input_ids, hidden_states, empty_captured, output_logits, output_hidden_states, linear, stream);
}

void EagleModule::forward(const Tensor& input_ids,
                          const Tensor& last_hidden_states,
                          const Tensor& captured_hidden_states,
                          Tensor&       output_logits,
                          Tensor&       output_hidden_states,
                          LlamaLinear&  linear,
                          cudaStream_t  stream)
{
    NvtxScope nvtx_scope("EagleModule::forward");
    if (isEagleDebugEnabled()) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);
        // Clear any stale debug views from previous calls.
        debug_fc_out_          = Tensor{};
        debug_attn_input_      = Tensor{};
        debug_attn_out_        = Tensor{};
        debug_ffn_out_         = Tensor{};
        debug_pre_head_hidden_ = Tensor{};
        debug_logits_          = Tensor{};
    }

    if (!enabled_) {
        TM_LOG_WARNING("[EAGLE] forward() called while module is disabled; "
                       "passing through hidden states without draft logits");
        output_hidden_states = last_hidden_states;
        return;
    }

    if (!weights_.is_initialized) {
        TM_LOG_WARNING("[EAGLE] forward() called before weights are initialized");
        // Safe no-op: pass through hidden state and leave logits untouched.
        output_hidden_states = last_hidden_states;
        return;
    }

    // Basic shape / dtype checks
    const Tensor& hidden_states = last_hidden_states;
    const int     batch_size    = hidden_states.shape(0);
    const int     hidden_dim    = hidden_states.shape(1);
    const bool    is_eagle3     = eagle_mode_ == EagleMode::kEagle3;

    if (hidden_dim != hidden_units_) {
        TM_LOG_WARNING("[EAGLE] hidden_units mismatch in forward: got %d, expected %d",
                       hidden_dim,
                       hidden_units_);
    }
    // RMSNorm kernels require that the norm weights and activations
    // share the same dtype. Draft weights are currently stored in
    // weight_dtype_ (typically FP16), while some target models (e.g.
    // GPT-OSS) run with BF16 activations. If we detect a mismatch
    // here, conservatively disable EAGLE for this engine and fall
    // back to baseline decoding instead of triggering a hard check
    // failure inside invokeRMSNorm.
    const auto hidden_dtype = hidden_states.dtype();
    auto       norm_mismatch = [&](const Tensor& w, const char* name) -> bool {
        if (!w) {
            return false;
        }
        if (w.dtype() != hidden_dtype || w.shape(-1) != hidden_dim) {
            TM_LOG_WARNING(
                "[EAGLE] %s norm mismatch in forward: weight_dtype=%d hidden_dtype=%d "
                "weight_dim=%d hidden_dim=%d; disabling EAGLE for this engine.",
                name,
                static_cast<int>(w.dtype()),
                static_cast<int>(hidden_dtype),
                w.shape(-1),
                hidden_dim);
            return true;
        }
        return false;
    };
    if (norm_mismatch(weights_.input_norm, "input")
        || norm_mismatch(weights_.hidden_norm, "hidden")
        || norm_mismatch(weights_.output_norm, "output")) {
        enabled_             = false;
        output_hidden_states = hidden_states;
        return;
    }

    // Ensure LM head wrapper is ready (defensive in case load() was skipped)
    if (!lm_head_prepared_ && weights_.lm_head) {
        int lm_in  = this->hidden_units_;
        int lm_out = this->vocab_size_;
        if (weights_.lm_head.ndim() == 2) {
            lm_in  = weights_.lm_head.shape(0);
            lm_out = weights_.lm_head.shape(1);
        }
        lm_head_input_dim_ = lm_in;
        vocab_size_        = lm_out;

        lm_head_weight_.emplace(
            lm_in, lm_out, this->weight_dtype_, /*bias=*/false, this->weight_dtype_, /*group_size=*/1);
        lm_head_weight_.weight      = weights_.lm_head.borrow();
        lm_head_weight_.bias        = {};
        lm_head_weight_.data_type   = weight_dtype_;
        lm_head_weight_.weight_type = weight_dtype_;
        lm_head_weight_.input_type  = weight_dtype_;
        lm_head_weight_.prepare(/*fused_moe=*/false);
        lm_head_prepared_ = true;
    }

    // To keep the speculative decode hot path allocation-free, reuse
    // internal scratch buffers whenever possible instead of allocating
    // new temporaries on each call.

    // Use a fixed epsilon for now; can be made configurable from draft config.yaml.
    constexpr float kEps = 1e-5f;

    // When we have a prepared shallow EagleNet / Eagle3 block, run a
    // single self-attention block followed by a lightweight MLP / norm.
    // This uses the draft model’s attention weights while keeping the
    // network shallow. If the block is not prepared (e.g., older
    // checkpoints), fall back to the original RMSNorm + LM head path.
    if (draft_block_prepared_) {
        const bool eagle3 = eagle_mode_ == EagleMode::kEagle3;

        // Optional Eagle3 pre-FC over concatenated multi-layer hidden
        // states captured from UnifiedDecoder. When available and the
        // geometry matches, we use the resulting features and, when
        // possible, token embeddings to build the QKV input for Eagle3
        // attention.
        Tensor fc_out;
        bool   can_use_eagle_fc = false;
        Tensor embed_norm;
        bool   have_embed = false;

        // 1) Pre-attention RMSNorm. For Eagle3 we prefer to derive the
        // attention input from the FC-reduced multi-layer hidden when it
        // is available; otherwise we fall back to the normalized last-layer
        // hidden state as before.
        bool need_attn_in =
            !attn_input_scratch_ || attn_input_scratch_.dtype() != hidden_dtype
            || attn_input_scratch_.device().type != kDEVICE || attn_input_scratch_.shape(0) < batch_size
            || attn_input_scratch_.shape(1) != hidden_dim;
        if (need_attn_in) {
            attn_input_scratch_ = Tensor{Layout({batch_size, hidden_dim}), hidden_dtype, kDEVICE};
        }
        Tensor& attn_input = attn_input_scratch_;

        if (eagle3 && eagle_fc_weight_ && weights_.eagle_fc && eagle_fc_in_dim_ > 0
            && captured_hidden_states && captured_hidden_states.shape(0) == batch_size
            && captured_hidden_states.shape(1) == eagle_fc_in_dim_
            && captured_hidden_states.dtype() == hidden_dtype) {
            const int draft_dim = draft_hidden_units_ > 0 ? draft_hidden_units_ : hidden_dim;
            bool need_fc_out =
                !eagle_fc_out_scratch_ || eagle_fc_out_scratch_.dtype() != hidden_dtype
                || eagle_fc_out_scratch_.device().type != kDEVICE
                || eagle_fc_out_scratch_.shape(0) < batch_size
                || eagle_fc_out_scratch_.shape(1) != draft_dim;
            if (need_fc_out) {
                eagle_fc_out_scratch_ = Tensor{Layout({batch_size, draft_dim}), hidden_dtype, kDEVICE};
            }
            fc_out = eagle_fc_out_scratch_;
            linear.Forward(captured_hidden_states, eagle_fc_weight_, /*output=*/fc_out);

            // Map FC output (draft_dim) to base_hidden width with zero pad / truncate.
            check_cuda_error(cudaMemsetAsync(attn_input.raw_data(), 0, attn_input.byte_size(), stream));
            const size_t elem_bytes = byte_size(hidden_dtype, 8) / 8;
            const size_t src_stride = static_cast<size_t>(fc_out.stride(0)) * elem_bytes;
            const size_t dst_stride = static_cast<size_t>(attn_input.stride(0)) * elem_bytes;
            const int    copy_cols  = std::min(draft_dim, hidden_dim);

            for (int b = 0; b < batch_size; ++b) {
                const char* src_row = static_cast<const char*>(fc_out.raw_data()) + static_cast<size_t>(b) * src_stride;
                char*       dst_row = static_cast<char*>(attn_input.raw_data()) + static_cast<size_t>(b) * dst_stride;
                check_cuda_error(cudaMemcpyAsync(
                    dst_row, src_row, static_cast<size_t>(copy_cols) * elem_bytes, cudaMemcpyDeviceToDevice, stream));
            }

            // Normalize the base-hidden view with hidden_norm.
            invokeRMSNorm(attn_input, attn_input, weights_.hidden_norm, kEps, stream);

            if (isEagleDebugEnabled()) {
                debug_fc_out_     = fc_out;
                debug_attn_input_ = attn_input;
            }

            can_use_eagle_fc = true;

            if (isEagleDebugEnabled()) {
                TM_LOG_DEBUG(
                    "[EAGLE][EagleModule] Eagle3 FC+norm active: batch=%d in_dim=%d hidden=%d",
                    batch_size,
                    eagle_fc_in_dim_,
                    hidden_dim);
            }
        }
        else {
            // Legacy / non-Eagle3 path: fall back to normalizing the
            // last-layer hidden state with the draft input norm.
            invokeRMSNorm(attn_input, hidden_states, weights_.input_norm, kEps, stream);

            if (isEagleDebugEnabled()) {
                debug_attn_input_ = attn_input;
            }

            if (eagle3 && isEagleDebugEnabled()) {
                TM_LOG_WARNING(
                    "[EAGLE][EagleModule] Eagle3 FC path disabled for this step "
                    "(capture missing or shape/dtype mismatch)");
            }
        }

        // 2) Single-token self-attention (degenerates to a learned projection
        // for one token). For Eagle3 we build a 2×draft_hidden input composed
        // of embedding_norm (first half) + FC-normalised capture (second half),
        // mirroring TRT-LLM Eagle-3. Legacy EagleNet keeps the [hidden, 3*hidden].
        const int qkv_out_dim = weights_.attn_qkv.shape(1);
        bool      need_qkv =
            !qkv_scratch_ || qkv_scratch_.dtype() != weight_dtype_ || qkv_scratch_.device().type != kDEVICE
            || qkv_scratch_.shape(0) < batch_size || qkv_scratch_.shape(1) != qkv_out_dim;
            if (need_qkv) {
                qkv_scratch_ = Tensor{Layout({batch_size, qkv_out_dim}), this->weight_dtype_, kDEVICE};
            }
        Tensor& qkv = qkv_scratch_;

        if (eagle3) {
            const int qkv_in_dim = weights_.attn_qkv.shape(0);
            const int inferred_factor =
                hidden_dim > 0 && qkv_in_dim % hidden_dim == 0 ? qkv_in_dim / hidden_dim : 0;
            const int factor = eagle_qkv_in_factor_ > 0 ? eagle_qkv_in_factor_ : (inferred_factor > 0 ? inferred_factor : 2);

            bool need_qkv_in =
                !attn_qkv_input_scratch_ || attn_qkv_input_scratch_.dtype() != hidden_dtype
                || attn_qkv_input_scratch_.device().type != kDEVICE || attn_qkv_input_scratch_.shape(0) < batch_size
                || attn_qkv_input_scratch_.shape(1) != qkv_in_dim;
            if (need_qkv_in) {
                attn_qkv_input_scratch_ = Tensor{Layout({batch_size, qkv_in_dim}), hidden_dtype, kDEVICE};
            }
            Tensor& qkv_input = attn_qkv_input_scratch_;

            check_cuda_error(cudaMemsetAsync(qkv_input.raw_data(), 0, qkv_input.byte_size(), stream));

            const size_t elem_bytes    = byte_size(hidden_dtype, 8) / 8;
            const size_t dst_row_bytes = static_cast<size_t>(qkv_input.stride(0)) * elem_bytes;
            char*        dst_base      = static_cast<char*>(qkv_input.raw_data());

            // Build normalized embeddings.
            if (eagle3 && input_ids && weights_.embed_tokens && qkv_in_dim >= 2 * hidden_dim) {
                bool need_embed_in =
                    !embed_input_scratch_ || embed_input_scratch_.dtype() != hidden_dtype
                    || embed_input_scratch_.device().type != kDEVICE || embed_input_scratch_.shape(0) < batch_size
                    || embed_input_scratch_.shape(1) != hidden_dim;
                if (need_embed_in) {
                    embed_input_scratch_ = Tensor{Layout({batch_size, hidden_dim}), hidden_dtype, kDEVICE};
                }
                core::Buffer raw_ids(const_cast<void*>(input_ids.raw_data()), batch_size, kInt32, kDEVICE);
                core::Buffer_<int> token_ids(raw_ids);
                core::Ref<Tensor>  embed_out(embed_input_scratch_);
                invokeEmbeddingLookup(embed_out, token_ids, weights_.embed_tokens, stream);

                bool need_embed_norm =
                    !embed_norm_scratch_ || embed_norm_scratch_.dtype() != hidden_dtype
                    || embed_norm_scratch_.device().type != kDEVICE || embed_norm_scratch_.shape(0) < batch_size
                    || embed_norm_scratch_.shape(1) != hidden_dim;
                if (need_embed_norm) {
                    embed_norm_scratch_ = Tensor{Layout({batch_size, hidden_dim}), hidden_dtype, kDEVICE};
                }
                embed_norm = embed_norm_scratch_;
                invokeRMSNorm(embed_norm, embed_input_scratch_, weights_.input_norm, kEps, stream);
                have_embed = true;
            }

            const int    draft_dim  = draft_hidden_units_ > 0 ? draft_hidden_units_ : hidden_dim;
            const size_t copy_bytes = static_cast<size_t>(std::min(hidden_dim, draft_dim)) * elem_bytes;
            const size_t src_row_bytes_attn = static_cast<size_t>(attn_input.stride(0)) * elem_bytes;
            const size_t src_row_bytes_emb  = have_embed
                                                  ? static_cast<size_t>(embed_norm.stride(0)) * elem_bytes
                                                  : src_row_bytes_attn;
            char* emb_base  = have_embed ? static_cast<char*>(embed_norm.raw_data())
                                         : static_cast<char*>(attn_input.raw_data());
            char* attn_base = static_cast<char*>(attn_input.raw_data());

            for (int b = 0; b < batch_size; ++b) {
                char* dst_row = dst_base + static_cast<size_t>(b) * dst_row_bytes;
                char* emb_row = emb_base + static_cast<size_t>(b) * src_row_bytes_emb;
                char* attn_row = attn_base + static_cast<size_t>(b) * src_row_bytes_attn;

                // First half: embedding norm (or attn_input as fallback)
                check_cuda_error(cudaMemcpyAsync(dst_row, emb_row, copy_bytes, cudaMemcpyDeviceToDevice, stream));
                // Second half: FC-normalised capture / last hidden
                if (qkv_in_dim >= 2 * draft_dim) {
                    check_cuda_error(cudaMemcpyAsync(
                        dst_row + static_cast<size_t>(draft_dim) * elem_bytes,
                        attn_row,
                        copy_bytes,
                        cudaMemcpyDeviceToDevice,
                        stream));
                }
                else {
                    // Repeat as many factors as fit.
                    for (int r = 1; r < factor && (r * draft_dim) < qkv_in_dim; ++r) {
                        check_cuda_error(cudaMemcpyAsync(
                            dst_row + static_cast<size_t>(r * draft_dim) * elem_bytes,
                            attn_row,
                            copy_bytes,
                            cudaMemcpyDeviceToDevice,
                            stream));
                    }
                }
            }

            linear.Forward(qkv_input, attn_qkv_weight_, /*output=*/qkv);
        }
        else {
            const int qkv_dim = hidden_dim * 3;
            bool      need_qkv_in =
                !qkv_scratch_ || qkv_scratch_.dtype() != weight_dtype_ || qkv_scratch_.device().type != kDEVICE
                || qkv_scratch_.shape(0) < batch_size || qkv_scratch_.shape(1) != qkv_dim;
            if (need_qkv_in) {
                qkv_scratch_ = Tensor{Layout({batch_size, qkv_dim}), this->weight_dtype_, kDEVICE};
            }
            Tensor& qkv_legacy = qkv_scratch_;
            linear.Forward(attn_input, attn_qkv_weight_, /*output=*/qkv_legacy);
        }

        // Extract the value to project with Wo. For Eagle3 we use the Q slice;
        // for legacy EagleNet we keep using V as before.
        Tensor value;
        if (eagle3 && eagle_q_size_ > 0) {
            const int q_size_local = eagle_q_size_;
            value                  = qkv.slice({0, 0}, {batch_size, q_size_local});
        }
        else {
            value = qkv.slice({0, 2 * hidden_dim}, {batch_size, hidden_dim});
        }

        bool need_attn_out =
            !attn_out_scratch_ || attn_out_scratch_.dtype() != weight_dtype_
            || attn_out_scratch_.device().type != kDEVICE || attn_out_scratch_.shape(0) < batch_size
            || attn_out_scratch_.shape(1) != hidden_dim;
                    if (need_attn_out) {
                        attn_out_scratch_ = Tensor{Layout({batch_size, hidden_dim}), this->weight_dtype_, kDEVICE};
                    }        Tensor& attn_out = attn_out_scratch_;
        linear.Forward(value, attn_o_weight_, /*output=*/attn_out);
        if (isEagleDebugEnabled()) {
            debug_attn_out_ = attn_out;
        }

        // 3) Optional Eagle3 FFN + residual + output norm using
        // LLaMA-style FFN weights when available. When Eagle3
        // draft-layer weights are not prepared we fall back to the
        // original RMSNorm-only path.
        if (eagle3 && eagle3_draft_layer_) {
            const auto& mlp       = eagle3_draft_layer_->ffn;
            const int   token_num = attn_out.shape(0);
            const int   inter_sz  = mlp.inter_size;

            // Post-attention norm before FFN, mirroring the
            // layers.0.ffn_norm behaviour.
            bool need_ffn_in =
                !mlp_input_scratch_ || mlp_input_scratch_.dtype() != weight_dtype_
                || mlp_input_scratch_.device().type != kDEVICE
                || mlp_input_scratch_.shape(0) < batch_size
                || mlp_input_scratch_.shape(1) != draft_hidden_units_;
            if (need_ffn_in) {
                mlp_input_scratch_ = Tensor{Layout({batch_size, draft_hidden_units_}), this->weight_dtype_, kDEVICE};
            }
            Tensor& ffn_input = mlp_input_scratch_;

            // attn_out (base_hidden_units_) is converted to ffn_input (draft_hidden_units_)
            const int current_attn_out_dim = eagle3_draft_layer_->base_hidden_dim;
            const int target_ffn_in_dim = eagle3_draft_layer_->draft_hidden_dim;

            if (current_attn_out_dim != target_ffn_in_dim) {
                check_cuda_error(cudaMemsetAsync(ffn_input.raw_data(), 0, ffn_input.byte_size(), stream));
                const size_t elem_bytes = byte_size(weight_dtype_, 1);
                const size_t src_stride = static_cast<size_t>(attn_out.stride(0)) * elem_bytes;
                const size_t dst_stride = static_cast<size_t>(ffn_input.stride(0)) * elem_bytes;
                const int    copy_cols  = std::min(current_attn_out_dim, target_ffn_in_dim);

                for (int b = 0; b < batch_size; ++b) {
                    const char* src_row = static_cast<const char*>(attn_out.raw_data()) + static_cast<size_t>(b) * src_stride;
                    char*       dst_row = static_cast<char*>(ffn_input.raw_data()) + static_cast<size_t>(b) * dst_stride;
                    check_cuda_error(cudaMemcpyAsync(
                        dst_row, src_row, static_cast<size_t>(copy_cols) * elem_bytes, cudaMemcpyDeviceToDevice, stream));
                }
            } else {
                check_cuda_error(cudaMemcpyAsync(ffn_input.raw_data(), attn_out.raw_data(), attn_out.byte_size(), cudaMemcpyDeviceToDevice, stream));
            }

            invokeRMSNorm(ffn_input, ffn_input, eagle3_draft_layer_->post_attn_norm, kEps, stream);

            // Fused gating projection: [B, 2 * inter_sz]
            Tensor mix = linear.Forward(ffn_input, mlp.fused_gating_intermediate);
            sync_check_cuda_error();

            // Take the first inter_sz columns as gating activations and
            // the second inter_sz columns as the "up" projection.
            Tensor gating = mix.slice({0, 0}, {token_num, inter_sz});
            Tensor up     = mix.slice({0, inter_sz}, {token_num, inter_sz});

            // gate' = silu(gate) * up (or Eagle3-specific variant).
            Activation(gating, up, mlp.act_type, stream);

            // Output projection: [B, draft_hidden_units_]
            bool need_mlp_out =
                !mlp_out_scratch_ || mlp_out_scratch_.dtype() != weight_dtype_
                || mlp_out_scratch_.device().type != kDEVICE
                || mlp_out_scratch_.shape(0) < batch_size
                || mlp_out_scratch_.shape(1) != draft_hidden_units_;
            if (need_mlp_out) {
                mlp_out_scratch_ = Tensor{Layout({batch_size, draft_hidden_units_}), this->weight_dtype_, kDEVICE};
            }
            Tensor& ffn_out = mlp_out_scratch_;
            linear.Forward(gating, mlp.output, ffn_out);
            sync_check_cuda_error();
            if (isEagleDebugEnabled()) {
                debug_ffn_out_ = ffn_out;
            }

            // Apply FFN residual + RMSNorm in one step:
            //   y = RMSNorm(attn_out_converted + FFN(norm(attn_out_converted)), output_norm)
            // Here ffn_out holds FFN(norm(attn_out_converted)).
            // Convert attn_out_scratch_ (base_hidden_units_) to draft_hidden_units_ for residual.
            bool need_residual_scratch =
                !residual_to_ffn_dim_scratch_ || residual_to_ffn_dim_scratch_.dtype() != weight_dtype_
                || residual_to_ffn_dim_scratch_.device().type != kDEVICE
                || residual_to_ffn_dim_scratch_.shape(0) < batch_size
                || residual_to_ffn_dim_scratch_.shape(1) != target_ffn_in_dim; // target_ffn_in_dim is draft_hidden_units_
            if (need_residual_scratch) {
                residual_to_ffn_dim_scratch_ = Tensor{Layout({batch_size, target_ffn_in_dim}), this->weight_dtype_, kDEVICE};
            }
            Tensor& converted_residual = residual_to_ffn_dim_scratch_;

            if (current_attn_out_dim != target_ffn_in_dim) {
                check_cuda_error(cudaMemsetAsync(converted_residual.raw_data(), 0, converted_residual.byte_size(), stream));
                const size_t elem_bytes = byte_size(weight_dtype_, 1);
                const size_t src_stride = static_cast<size_t>(attn_out_scratch_.stride(0)) * elem_bytes;
                const size_t dst_stride = static_cast<size_t>(converted_residual.stride(0)) * elem_bytes;
                const int    copy_cols  = std::min(current_attn_out_dim, target_ffn_in_dim);

                for (int b = 0; b < batch_size; ++b) {
                    const char* src_row = static_cast<const char*>(attn_out_scratch_.raw_data()) + static_cast<size_t>(b) * src_stride;
                    char*       dst_row = static_cast<char*>(converted_residual.raw_data()) + static_cast<size_t>(b) * dst_stride;
                    check_cuda_error(cudaMemcpyAsync(
                        dst_row, src_row, static_cast<size_t>(copy_cols) * elem_bytes, cudaMemcpyDeviceToDevice, stream));
                }
            } else {
                check_cuda_error(cudaMemcpyAsync(converted_residual.raw_data(), attn_out_scratch_.raw_data(), attn_out_scratch_.byte_size(), cudaMemcpyDeviceToDevice, stream));
            }

            invokeResidualBiasRMSNorm(
                /*hidden_states=*/ffn_out.raw_data(),
                /*residual=*/converted_residual.raw_data(),
                /*weights=*/eagle3_draft_layer_->output_norm.raw_data(),
                /*bias=*/nullptr,
                weight_dtype_,
                eagle3_draft_layer_->draft_hidden_dim, // Corrected dimension
                batch_size,
                kEps,
                stream);
            output_hidden_states = ffn_out;
            if (isEagleDebugEnabled()) {
                debug_pre_head_hidden_ = ffn_out;
            }
        }
        else {
            // Fallback: original shallow path – normalize attention
            // output with the draft output norm.
            bool need_norm_scratch =
                !normed_hidden_scratch_ || normed_hidden_scratch_.dtype() != weight_dtype_
                || normed_hidden_scratch_.device().type != kDEVICE
                || normed_hidden_scratch_.shape(0) < batch_size
                || normed_hidden_scratch_.shape(1) != hidden_dim;
            if (need_norm_scratch) {
                normed_hidden_scratch_ = Tensor{Layout({batch_size, hidden_dim}), this->weight_dtype_, kDEVICE};
            }
            Tensor& normed_hidden = normed_hidden_scratch_;
            invokeRMSNorm(normed_hidden, attn_out_scratch_, weights_.output_norm, kEps, stream);
            output_hidden_states = normed_hidden;
            if (isEagleDebugEnabled()) {
                debug_pre_head_hidden_ = normed_hidden;
            }
        }
    }
    else {
        // Fallback: original minimal path – normalize with the output norm
        // directly on the target hidden states. This preserves the previous
        // behaviour when the shallow draft block is unavailable.
        bool need_norm =
            !normed_hidden_scratch_ || normed_hidden_scratch_.dtype() != hidden_dtype
            || normed_hidden_scratch_.device().type != kDEVICE || normed_hidden_scratch_.shape(0) < batch_size
            || normed_hidden_scratch_.shape(1) != hidden_dim;
        if (need_norm) {
            normed_hidden_scratch_ = Tensor{Layout({batch_size, hidden_dim}), hidden_dtype, kDEVICE};
        }
        Tensor& normed_hidden = normed_hidden_scratch_;
        invokeRMSNorm(normed_hidden, hidden_states, weights_.output_norm, kEps, stream);
        output_hidden_states = normed_hidden;

        // LM head below will consume `output_hidden_states` directly
        // in this fallback case (no extra RMSNorm in weight space).
    }

    // Project to vocab with LM head.
    if (!lm_head_prepared_) {
        TM_LOG_WARNING("[EAGLE] LM head not prepared; skipping logits computation");
        return;
    }

    const int vocab = this->vocab_size_ > 0 ? this->vocab_size_ : weights_.lm_head.shape(1);

    const bool need_new_logits_scratch =
        !logits_scratch_
        || logits_scratch_.dtype() != this->weight_dtype_
        || logits_scratch_.device().type != kDEVICE
        || logits_scratch_.shape(0) < batch_size
        || logits_scratch_.shape(1) != vocab;

    if (need_new_logits_scratch) {
        logits_scratch_ = Tensor{Layout({batch_size, vocab}), this->weight_dtype_, kDEVICE};
    }

    Tensor& logits = logits_scratch_;

    // Ensure LM head input geometry matches the LM head weight. When the
    // hidden width coming from the shallow Eagle3 block differs from the
    // LM head input dimension (e.g. 4096 vs 2880), convert by truncating
    // or zero‑padding to avoid invalid GEMM shapes.
    Tensor* lm_input = &output_hidden_states;
    const int lm_in  = lm_head_input_dim_ > 0 ? lm_head_input_dim_ : output_hidden_states.shape(1);

    if (output_hidden_states.ndim() != 2 || output_hidden_states.shape(0) != batch_size) {
        TM_LOG_WARNING(
            "[EAGLE][LMHead] unexpected hidden shape [%d,%d]; skipping logits computation",
            output_hidden_states.ndim() >= 1 ? output_hidden_states.shape(0) : -1,
            output_hidden_states.ndim() >= 2 ? output_hidden_states.shape(1) : -1);
        return;
    }

    if (output_hidden_states.shape(1) != lm_in) {
        const auto lm_dtype   = output_hidden_states.dtype();
        const auto lm_device  = output_hidden_states.device();
        const bool need_scratch =
            !lm_head_input_scratch_ || lm_head_input_scratch_.dtype() != lm_dtype
            || lm_head_input_scratch_.device().type != lm_device.type
            || lm_head_input_scratch_.shape(0) < batch_size
            || lm_head_input_scratch_.shape(1) != lm_in;
        if (need_scratch) {
            lm_head_input_scratch_ = Tensor{Layout({batch_size, lm_in}), lm_dtype, lm_device.type};
        }

        check_cuda_error(cudaMemsetAsync(lm_head_input_scratch_.raw_data(),
                                         0,
                                         lm_head_input_scratch_.byte_size(),
                                         stream));

        const int    src_cols   = output_hidden_states.shape(1);
        const int    copy_cols  = std::min(src_cols, lm_in);
        const size_t elem_bytes = byte_size(lm_dtype, 1);
        const size_t src_stride = static_cast<size_t>(output_hidden_states.stride(0)) * elem_bytes;
        const size_t dst_stride = static_cast<size_t>(lm_head_input_scratch_.stride(0)) * elem_bytes;

        for (int b = 0; b < batch_size; ++b) {
            const char* src_row =
                static_cast<const char*>(output_hidden_states.raw_data()) + static_cast<size_t>(b) * src_stride;
            char* dst_row =
                static_cast<char*>(lm_head_input_scratch_.raw_data()) + static_cast<size_t>(b) * dst_stride;
            check_cuda_error(cudaMemcpyAsync(
                dst_row, src_row, static_cast<size_t>(copy_cols) * elem_bytes, cudaMemcpyDeviceToDevice, stream));
        }

        lm_input = &lm_head_input_scratch_;
    }

    if (isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG")) {
        logEagleGemmShape("EAGLE3_LM_HEAD",
                          lm_input->shape(0),
                          lm_head_weight_.input_dim,
                          lm_head_weight_.output_dim,
                          static_cast<int>(lm_head_weight_.data_type),
                          "row_major");
    }
    {
        NvtxScope       nvtx_scope_lm("EAGLE_DRAFT_LM_HEAD");
        EagleGemmTagGuard guard("EAGLE3_LM_HEAD");
        linear.Forward(*lm_input, lm_head_weight_, /*output=*/logits);
    }
    output_logits = logits;

    if (isEagleDebugEnabled()) {
        debug_logits_ = logits;
    }
}

}  // namespace turbomind
