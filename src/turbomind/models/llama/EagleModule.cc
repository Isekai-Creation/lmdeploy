// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#include "src/turbomind/models/llama/EagleModule.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/eagle_debug.h"

#include <fstream>
#include <vector>
#include <string>

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

    if (hidden_units != head_num * head_dim) {
        logEagleError("hidden_units != head_num * size_per_head in config.yaml");
        return;
    }

    hidden_units_ = hidden_units;
    vocab_size_   = vocab_size;

    // Allocate weights using FP16 (matches typical EagleNet precision).
    DataType dtype = kFloat16;
    weight_dtype_  = dtype;

    weights_.embed_tokens = Tensor{{vocab_size, hidden_units}, dtype, kDEVICE};
    weights_.fc           = Tensor{{hidden_units * 2, hidden_units}, dtype, kDEVICE};

    // Layer weights (single EagleNet layer)
    weights_.input_norm  = Tensor{{hidden_units}, dtype, kDEVICE};
    weights_.hidden_norm = Tensor{{hidden_units}, dtype, kDEVICE};

    weights_.attn_qkv  = Tensor{{hidden_units, hidden_units * 3}, dtype, kDEVICE};
    weights_.attn_o    = Tensor{{hidden_units, hidden_units}, dtype, kDEVICE};
    weights_.attn_norm = Tensor{{hidden_units}, dtype, kDEVICE};

    weights_.mlp_gate_up = Tensor{{intermediate_size * 2, hidden_units}, dtype, kDEVICE};
    weights_.mlp_down    = Tensor{{intermediate_size, hidden_units}, dtype, kDEVICE};

    weights_.output_norm = Tensor{{hidden_units}, dtype, kDEVICE};
    weights_.lm_head     = Tensor{{hidden_units, vocab_size}, dtype, kDEVICE};

    weights_.is_initialized = true;

    const std::string base = model_dir + "/";

    // Model-level weights
    success &= loadTensorFromFile<half>(weights_.embed_tokens, base + "tok_embeddings.weight");
    success &= loadTensorFromFile<half>(weights_.fc, base + "fc.weight");

    // Layer 0 norms and attention
    success &= loadTensorFromFile<half>(weights_.input_norm, base + "layers.0.attention_norm.weight");
    success &= loadTensorFromFile<half>(weights_.hidden_norm, base + "layers.0.hidden_norm.weight");
    success &= loadTensorFromFile<half>(weights_.attn_qkv, base + "layers.0.attention.w_qkv.weight");
    success &= loadTensorFromFile<half>(weights_.attn_o, base + "layers.0.attention.wo.weight");
    success &= loadTensorFromFile<half>(weights_.attn_norm, base + "layers.0.ffn_norm.weight");

    // MLP gate/up: merge w1 and w3 into mlp_gate_up.
    Tensor w1{{intermediate_size, hidden_units}, dtype, kDEVICE};
    Tensor w3{{intermediate_size, hidden_units}, dtype, kDEVICE};
    bool   w1_ok = loadTensorFromFile<half>(w1, base + "layers.0.feed_forward.w1.weight");
    bool   w3_ok = loadTensorFromFile<half>(w3, base + "layers.0.feed_forward.w3.weight");
    if (w1_ok && w3_ok) {
        const size_t w_size = static_cast<size_t>(w1.size()) * sizeof(half);
        check_cuda_error(cudaMemcpy(
            weights_.mlp_gate_up.data<half>(), w1.data<half>(), w_size, cudaMemcpyDeviceToDevice));
        check_cuda_error(cudaMemcpy(
            reinterpret_cast<char*>(weights_.mlp_gate_up.data<half>()) + w_size,
            w3.data<half>(),
            w_size,
            cudaMemcpyDeviceToDevice));
    }
    else {
        success = false;
    }

    success &= loadTensorFromFile<half>(weights_.mlp_down, base + "layers.0.feed_forward.w2.weight");
    success &= loadTensorFromFile<half>(weights_.output_norm, base + "norm.weight");
    success &= loadTensorFromFile<half>(weights_.lm_head, base + "output.weight");

    // Prepare LM head wrapper for LlamaLinear.
    if (weights_.lm_head) {
        lm_head_weight_.emplace(
            hidden_units_, vocab_size_, weight_dtype_, /*bias=*/false, weight_dtype_, /*group_size=*/1);
        lm_head_weight_.weight      = weights_.lm_head.borrow();
        lm_head_weight_.bias        = {};
        lm_head_weight_.data_type   = weight_dtype_;
        lm_head_weight_.weight_type = weight_dtype_;
        lm_head_weight_.input_type  = weight_dtype_;
        lm_head_weight_.prepare(/*fused_moe=*/false);
        lm_head_prepared_ = true;
    }

    // Prepare shallow EagleNet attention + FC block wrappers for LlamaLinear.
    if (!draft_block_prepared_) {
        if (weights_.attn_qkv && weights_.attn_o && weights_.fc) {
            // QKV: [hidden, 3 * hidden]
            attn_qkv_weight_.emplace(
                hidden_units_, hidden_units_ * 3, weight_dtype_, /*bias=*/false, weight_dtype_, /*group_size=*/1);
            attn_qkv_weight_.weight      = weights_.attn_qkv.borrow();
            attn_qkv_weight_.bias        = {};
            attn_qkv_weight_.data_type   = weight_dtype_;
            attn_qkv_weight_.weight_type = weight_dtype_;
            attn_qkv_weight_.input_type  = weight_dtype_;
            attn_qkv_weight_.prepare(/*fused_moe=*/false);

            // Attention output: [hidden, hidden]
            attn_o_weight_.emplace(
                hidden_units_, hidden_units_, weight_dtype_, /*bias=*/false, weight_dtype_, /*group_size=*/1);
            attn_o_weight_.weight      = weights_.attn_o.borrow();
            attn_o_weight_.bias        = {};
            attn_o_weight_.data_type   = weight_dtype_;
            attn_o_weight_.weight_type = weight_dtype_;
            attn_o_weight_.input_type  = weight_dtype_;
            attn_o_weight_.prepare(/*fused_moe=*/false);

            // FC “MLP” projection: [2 * hidden, hidden]
            fc_weight_.emplace(
                hidden_units_ * 2, hidden_units_, weight_dtype_, /*bias=*/false, weight_dtype_, /*group_size=*/1);
            fc_weight_.weight      = weights_.fc.borrow();
            fc_weight_.bias        = {};
            fc_weight_.data_type   = weight_dtype_;
            fc_weight_.weight_type = weight_dtype_;
            fc_weight_.input_type  = weight_dtype_;
            fc_weight_.prepare(/*fused_moe=*/false);

            draft_block_prepared_ = true;
        }
        else {
            TM_LOG_WARNING("[EAGLE] Draft block weights missing; using RMSNorm+LM head only");
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

    if (!success) {
        logEagleError("Draft model load failed; disabling EAGLE for this engine");
        enabled_ = false;
        return;
    }

    enabled_ = true;
    TM_LOG_INFO(
        "[EAGLE] EagleModule draft model loaded successfully: hidden_units=%d vocab_size=%d", hidden_units_, vocab_size_);
}

void EagleModule::forward(const Tensor& input_ids,
                          const Tensor& hidden_states,
                          Tensor& output_logits,
                          Tensor& output_hidden_states,
                          LlamaLinear& linear,
                          cudaStream_t stream)
{
    NvtxScope nvtx_scope("EagleModule::forward");
    if (isEagleDebugEnabled()) {
        TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    }

    (void) input_ids;  // currently unused, reserved for future tree-aware draft nets

    if (!enabled_) {
        TM_LOG_WARNING("[EAGLE] forward() called while module is disabled; "
                       "passing through hidden states without draft logits");
        output_hidden_states = hidden_states;
        return;
    }

    if (!weights_.is_initialized) {
        TM_LOG_WARNING("[EAGLE] forward() called before weights are initialized");
        // Safe no-op: pass through hidden state and leave logits untouched.
        output_hidden_states = hidden_states;
        return;
    }

    // Basic shape checks
    const int batch_size = hidden_states.shape(0);
    const int hidden_dim = hidden_states.shape(1);

    if (hidden_dim != hidden_units_) {
        TM_LOG_WARNING("[EAGLE] hidden_units mismatch in forward: got %d, expected %d",
                       hidden_dim,
                       hidden_units_);
    }

    // Ensure LM head wrapper is ready (defensive in case load() was skipped)
    if (!lm_head_prepared_ && weights_.lm_head) {
        lm_head_weight_.emplace(
            hidden_units_, vocab_size_, weight_dtype_, /*bias=*/false, weight_dtype_, /*group_size=*/1);
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
    const auto hidden_dtype = hidden_states.dtype();

    // Use a fixed epsilon for now; can be made configurable from draft config.yaml.
    constexpr float kEps = 1e-5f;

    // When we have a prepared shallow EagleNet block, run a single
    // self-attention + FC “MLP” block followed by a final RMSNorm.
    // This uses the draft model’s attention and FC weights while
    // keeping the network lightweight. If the block is not prepared
    // (e.g., older checkpoints), fall back to the original RMSNorm +
    // LM head path.
    if (draft_block_prepared_) {
        // 1) Pre-attention RMSNorm with the draft input norm.
        bool need_attn_in =
            !attn_input_scratch_ || attn_input_scratch_.dtype() != hidden_dtype
            || attn_input_scratch_.device().type != kDEVICE || attn_input_scratch_.shape(0) < batch_size
            || attn_input_scratch_.shape(1) != hidden_dim;
        if (need_attn_in) {
            attn_input_scratch_ = Tensor{{batch_size, hidden_dim}, hidden_dtype, kDEVICE};
        }
        Tensor& attn_input = attn_input_scratch_;
        invokeRMSNorm(attn_input, hidden_states, weights_.input_norm, kEps, stream);

        // 2) Single-token self-attention (degenerates to a learned projection for one token).
        const int qkv_dim = hidden_dim * 3;
        bool      need_qkv =
            !qkv_scratch_ || qkv_scratch_.dtype() != weight_dtype_ || qkv_scratch_.device().type != kDEVICE
            || qkv_scratch_.shape(0) < batch_size || qkv_scratch_.shape(1) != qkv_dim;
        if (need_qkv) {
            qkv_scratch_ = Tensor{{batch_size, qkv_dim}, weight_dtype_, kDEVICE};
        }
        Tensor& qkv = qkv_scratch_;
        linear.Forward(attn_input, attn_qkv_weight_, /*output=*/qkv);

        // Extract V from [Q|K|V] and project with Wo. For a single token,
        // attention reduces to a learned value projection.
        Tensor v = qkv.slice({0, 2 * hidden_dim}, {batch_size, hidden_dim});

        bool need_attn_out =
            !attn_out_scratch_ || attn_out_scratch_.dtype() != weight_dtype_
            || attn_out_scratch_.device().type != kDEVICE || attn_out_scratch_.shape(0) < batch_size
            || attn_out_scratch_.shape(1) != hidden_dim;
        if (need_attn_out) {
            attn_out_scratch_ = Tensor{{batch_size, hidden_dim}, weight_dtype_, kDEVICE};
        }
        Tensor& attn_out = attn_out_scratch_;
        linear.Forward(v, attn_o_weight_, /*output=*/attn_out);

        // 3) Lightweight FC “MLP” block using the EagleNet fc weight.
        const int mlp_in_dim = hidden_dim * 2;
        bool      need_mlp_in =
            !mlp_input_scratch_ || mlp_input_scratch_.dtype() != weight_dtype_
            || mlp_input_scratch_.device().type != kDEVICE || mlp_input_scratch_.shape(0) < batch_size
            || mlp_input_scratch_.shape(1) != mlp_in_dim;
        if (need_mlp_in) {
            mlp_input_scratch_ = Tensor{{batch_size, mlp_in_dim}, weight_dtype_, kDEVICE};
        }
        Tensor& mlp_input = mlp_input_scratch_;

        // Zero the FC input, then copy the attention output into the first
        // hidden_dim channels, leaving the second half as zeros. This keeps
        // the math simple while still exercising the learned fc weights.
        check_cuda_error(cudaMemsetAsync(mlp_input.raw_data(), 0, mlp_input.byte_size(), stream));

        const size_t elem_bytes    = byte_size(weight_dtype_, 8) / 8;
        const size_t copy_bytes    = static_cast<size_t>(hidden_dim) * elem_bytes;
        const size_t src_row_bytes = static_cast<size_t>(attn_out.stride(0)) * elem_bytes;
        const size_t dst_row_bytes = static_cast<size_t>(mlp_input.stride(0)) * elem_bytes;
        char*        src_base      = static_cast<char*>(attn_out.raw_data());
        char*        dst_base      = static_cast<char*>(mlp_input.raw_data());

        for (int b = 0; b < batch_size; ++b) {
            check_cuda_error(cudaMemcpyAsync(dst_base + static_cast<size_t>(b) * dst_row_bytes,
                                             src_base + static_cast<size_t>(b) * src_row_bytes,
                                             copy_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream));
        }

        bool need_mlp_out =
            !mlp_out_scratch_ || mlp_out_scratch_.dtype() != weight_dtype_
            || mlp_out_scratch_.device().type != kDEVICE || mlp_out_scratch_.shape(0) < batch_size
            || mlp_out_scratch_.shape(1) != hidden_dim;
        if (need_mlp_out) {
            mlp_out_scratch_ = Tensor{{batch_size, hidden_dim}, weight_dtype_, kDEVICE};
        }
        Tensor& mlp_out = mlp_out_scratch_;
        linear.Forward(mlp_input, fc_weight_, /*output=*/mlp_out);

        // 4) Final RMSNorm for the shallow EagleNet block output.
        bool need_norm_scratch =
            !normed_hidden_scratch_ || normed_hidden_scratch_.dtype() != weight_dtype_
            || normed_hidden_scratch_.device().type != kDEVICE || normed_hidden_scratch_.shape(0) < batch_size
            || normed_hidden_scratch_.shape(1) != hidden_dim;
        if (need_norm_scratch) {
            normed_hidden_scratch_ = Tensor{{batch_size, hidden_dim}, weight_dtype_, kDEVICE};
        }
        Tensor& normed_hidden = normed_hidden_scratch_;
        invokeRMSNorm(normed_hidden, mlp_out_scratch_, weights_.output_norm, kEps, stream);
        output_hidden_states = normed_hidden;
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
            normed_hidden_scratch_ = Tensor{{batch_size, hidden_dim}, hidden_dtype, kDEVICE};
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

    const int vocab = vocab_size_ > 0 ? vocab_size_ : weights_.lm_head.shape(1);

    const bool need_new_logits_scratch =
        !logits_scratch_
        || logits_scratch_.dtype() != weight_dtype_
        || logits_scratch_.device().type != kDEVICE
        || logits_scratch_.shape(0) < batch_size
        || logits_scratch_.shape(1) != vocab;

    if (need_new_logits_scratch) {
        logits_scratch_ = Tensor{{batch_size, vocab}, weight_dtype_, kDEVICE};
    }

    Tensor& logits = logits_scratch_;

    linear.Forward(output_hidden_states, lm_head_weight_, /*output=*/logits);
    output_logits = logits;
}

}  // namespace turbomind
