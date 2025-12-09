/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 * Copyright (c) 2022, SK Telecom Authored by A. Dialog
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.cc

#include <algorithm>
#include <memory>
#include <sstream>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/macro.h"

#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/LlamaV2.h"
#include "src/turbomind/models/llama/LlamaWeight.h"
#include "src/turbomind/models/llama/SequenceManager.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/unified_decoder.h"

#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/sampling_topk_kernels.h"

#include "src/turbomind/models/llama/llama_kernels.h"

#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/memory_utils.h"
#include "src/turbomind/utils/eagle_debug.h"

#include "lmdeploy/turbomind/eagle_tree.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/common.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/eagle_kernels.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/packed_mask_kernels.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/target_tree_decode.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/tree_accept_kernels.h"

namespace turbomind {

using eagle::TokenIdType;

/// TODO: Padded vocab size should also be divisible by 8
inline int pad_vocab_size(int vocab_size, int tp)
{
    return (vocab_size + tp - 1) / tp * tp;
}

// Broadcast a per-slot target token ID into the first `tokens_per_seq`
// entries of the flattened [batch, max_decoding_tokens] target_tokens
// buffer. This avoids tiling the entire logits matrix when only top-1
// IDs are needed.
__global__ void broadcastTargetTokenKernel(const int*  target_ids,
                                           int         tokens_per_seq,
                                           int         max_decoding_tokens,
                                           int*        target_tokens,
                                           int         batch_size)
{
    const int b = blockIdx.x;
    const int t = threadIdx.x;
    if (b >= batch_size || t >= tokens_per_seq || tokens_per_seq <= 0 || max_decoding_tokens <= 0) {
        return;
    }
    const int id    = target_ids[b];
    const int index = b * max_decoding_tokens + t;
    target_tokens[index] = id;
}

LlamaV2::LlamaV2(DataType                     dtype,
                 const ModelParam&            model,
                 const EngineParam&           engine,
                 const AttentionParam&        attn,
                 const MoeParam&              moe,
                 const LoraParam&             lora,
                 const Context&               ctx,
                 int                          max_batch_size,
                 std::shared_ptr<LlamaWeight> weights):
    dtype_{dtype},
    param_(model),
    attn_param_(attn),
    lora_param_(lora),
    comm_(&ctx.comm),
    tp_size_(engine.attn_tp_size * engine.attn_cp_size),
    tp_rank_(engine.attn_tp_rank * engine.attn_cp_size + engine.attn_cp_rank),
    head_num_(model.head_num),
    size_per_head_(model.head_dim),
    hidden_units_(model.hidden_units),
    layer_num_(model.layer_num),
    vocab_size_(model.vocab_size),
    vocab_size_padded_(pad_vocab_size(model.vocab_size, tp_size_)),
    rmsnorm_eps_(model.norm_eps),
    local_head_num_(model.head_num / engine.attn_tp_size),
    local_kv_head_num_(model.kv_head_num / engine.attn_tp_size),
    weights_(std::move(weights)),
    stream_(ctx.stream),
    linear_(*ctx.linear),
    debug_(isDebug()),
    engine_param_(engine)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Propagate EAGLE debug/metrics flags derived from SpeculativeConfig
    // into the global helpers used by EagleModule and related kernels so
    // that all EAGLE debug logging is driven by engine configuration rather
    // than environment variables.
    setEagleDebugFlags(engine_param_.eagle_debug, engine_param_.eagle_metrics_debug);

    if (comm_->d_comm && comm_->d_comm->Query(comm::kHasAllGather2D)) {
        use_allgather_2d_ = true;
    }

    unified_decoder_ = std::make_unique<UnifiedDecoder>(model, engine, attn, moe, lora, ctx);

    // using float to avoid data overflow
    dynamic_decode_ = std::make_unique<DynamicDecodeLayer>(
        kFloat32, max_batch_size, vocab_size_, vocab_size_padded_, stream_, &ctx.device_prop);
    
    // Compute an upper bound on how many draft tokens per step the engine
    // should attempt when running in EAGLE speculative mode. This budget is
    // driven by SpeculativeConfig.num_speculative_tokens (mapped onto
    // spec_max_decoding_draft_tokens) and clamped by the per-step hardware
    // forward limit. When this budget is <= 1, EAGLE effectively runs in
    // single-token mode.
    if (engine.enable_speculative_decoding && (engine.spec_method == "eagle" || engine.spec_method == "eagle3")) {
        const int max_draft_tokens = engine.spec_max_decoding_draft_tokens;
        const int hardware_cap     = engine.max_forward_token_num;
        const int max_engine_tokens =
            std::max(1, std::min(max_draft_tokens > 0 ? max_draft_tokens : 1, hardware_cap));

        eagle_max_engine_tokens_per_step_ = max_engine_tokens;

        TM_LOG_INFO(
            "[LlamaV2][EAGLE] method=%s, spec_max_decoding_draft_tokens=%d, "
            "max_forward_token_num=%d, eagle_max_engine_tokens_per_step=%d",
            engine.spec_method.c_str(),
            max_draft_tokens,
            hardware_cap,
            eagle_max_engine_tokens_per_step_);
    }

    // Initialize EAGLE speculative decoding if enabled.
    //
    // Safety principle: EAGLE is considered active for this engine only if the
    // draft model path is provided and EagleModule::load succeeds. Otherwise
    // we explicitly disable speculative mode so TurboMind falls back to
    // baseline decoding with no partial EAGLE side effects or metrics.
    // For TurboMind, both "eagle" and "eagle3" methods share the same
    // EagleModule-based integration; "eagle3" just enables the newer
    // multi-token/tree behaviour. Treat both as enabling the EAGLE path.
    if (engine.enable_speculative_decoding
        && (engine.spec_method == "eagle" || engine.spec_method == "eagle3")) {
        TM_LOG_INFO("[LlamaV2] Initializing EAGLE speculative decoding: "
                    "max_path_len=%d, num_spec_tokens=%d, max_decoding_tokens=%d",
                    engine.spec_max_draft_path_len,
                    engine.spec_max_decoding_draft_tokens,
                    engine.spec_max_decoding_tokens);

        spec_mode_ = SpeculativeDecodingMode::Eagle();

        eagle_module_ = std::make_unique<EagleModule>(engine.spec_max_draft_path_len,
                                                      engine.spec_max_decoding_draft_tokens,
                                                      engine.spec_max_decoding_tokens,
                                                      engine.spec_max_non_leaf_nodes);

        bool eagle_ok = false;

        if (!engine.spec_draft_model_path.empty()) {
            TM_LOG_INFO("[LlamaV2] Loading EAGLE draft model from %s", engine.spec_draft_model_path.c_str());
            eagle_module_->load(engine.spec_draft_model_path, 0, stream_);
            if (!eagle_module_->isEnabled()) {
                TM_LOG_WARNING(
                    "[LlamaV2][EAGLE] Draft model load failed; disabling EAGLE and falling back to baseline decoding");
            }
            else {
                TM_LOG_INFO("[LlamaV2][EAGLE] Draft model loaded and validated");
                // Share Eagle3 draft-layer weights with the decoder when
                // available so that future Eagle3 draft paths can run
                // under UnifiedDecoder instead of the shallow block in
                // EagleModule.
                if (engine.spec_method == "eagle3" && eagle_module_->hasEagle3DraftLayer()) {
                    // Share Eagle3 draft-layer weights by aliasing the
                    // EagleModule-owned struct; no deep copy needed.
                    eagle3_draft_weight_ = eagle_module_->eagle3_draft_layer_.get();
                    if (unified_decoder_) {
                        unified_decoder_->setEagle3DraftLayer(eagle3_draft_weight_);
                    }
                }
                eagle_ok = true;
            }
        }
        else {
            TM_LOG_WARNING(
                "[LlamaV2][EAGLE] Speculative decoding requested but no draft model path provided; disabling EAGLE for "
                "this engine");
        }

        if (eagle_ok) {
            eagle_buffers_ = std::make_unique<EagleBuffers>();
            eagle_buffers_->allocate(max_batch_size, eagle_module_.get(), stream_);
            TM_LOG_INFO("[LlamaV2][EAGLE] EAGLE module initialized successfully");

            // Enable target-tree decode only when requested and when we
            // can provision dedicated hidden-state and FP32 logits
            // buffers for tree tokens. This keeps tree argmax numerics
            // stable while reusing the same BF16/MXFP4 compute path as
            // baseline decode for hidden states and weights.
            target_tree_supported_ = engine.enable_eagle_target_tree;
            if (target_tree_supported_) {
                const int64_t max_tree_tokens =
                    static_cast<int64_t>(max_batch_size)
                    * static_cast<int64_t>(engine.spec_max_decoding_tokens > 0
                                                ? engine.spec_max_decoding_tokens
                                                : 1);
                if (max_tree_tokens <= 0 || vocab_size_padded_ == 0) {
                    TM_LOG_WARNING(
                        "[LlamaV2][EAGLE][fallback] target-tree decode unavailable; "
                        "invalid max_tree_tokens=%ld or vocab_size_padded=%zu; "
                        "reverting to single-step target logits.",
                        static_cast<long>(max_tree_tokens),
                        vocab_size_padded_);
                    target_tree_supported_ = false;
                }
                else {
                    max_tree_tokens_     = static_cast<int>(max_tree_tokens);
                    const int64_t logits_size =
                        max_tree_tokens * static_cast<int64_t>(vocab_size_padded_);
                    tree_hidden_states_ = Tensor{{max_tree_tokens_, static_cast<int>(hidden_units_)}, dtype_, kDEVICE};
                    tree_logits_buffer_ = Buffer(logits_size, kFloat32, kDEVICE);

                    // Lightweight runtime dtype checks for tree decode. The
                    // hidden-state buffer must use the same dtype as the base
                    // model, and the logits buffer must be FP32 so that
                    // invokeTreeLogitsToTargetIds always sees FP32 logits and
                    // writes int32 IDs. On any mismatch we disable
                    // target-tree decode and fall back to baseline EAGLE.
                    if (!tree_hidden_states_
                        || tree_hidden_states_.dtype() != dtype_
                        || !tree_logits_buffer_
                        || tree_logits_buffer_.dtype() != kFloat32) {
                        TM_LOG_WARNING(
                            "[LlamaV2][EAGLE][fallback] target-tree decode disabled due to dtype/layout mismatch "
                            "(model=%s, tree_hidden=%s, tree_logits=%s)",
                            to_string(dtype_),
                            tree_hidden_states_ ? to_string(tree_hidden_states_.dtype()) : "nil",
                            tree_logits_buffer_ ? to_string(tree_logits_buffer_.dtype()) : "nil");
                        target_tree_supported_ = false;
                        tree_hidden_states_    = Tensor{};
                        tree_logits_buffer_    = Buffer{};
                    }
                }
            }

            // Initialize SpecPV partial-KV cache when requested and when
            // target-tree decode is available. This keeps SpecPV strictly
            // opt-in and ensures that misconfigured geometry falls back to
            // the baseline full-KV EAGLE3 pipeline.
            if (engine.enable_specpv && engine.spec_method == "eagle3" && target_tree_supported_) {
                specpv_cache_config_.block_size        = engine.specpv_block_size > 0
                                                             ? engine.specpv_block_size
                                                             : attn_param_.cache_block_seq_len;
                specpv_cache_config_.n_sink_blocks     = engine.specpv_n_sink_blocks;
                specpv_cache_config_.n_retrieval_blocks = engine.specpv_n_retrieval_blocks;
                specpv_cache_config_.n_window_blocks   = engine.specpv_n_window_blocks;
                specpv_cache_config_.n_spec_tokens_buf = engine.specpv_n_spec_tokens_buf;

                const int kv_block_len = attn_param_.cache_block_seq_len;

                bool geometry_ok = kv_block_len > 0
                                   && specpv_cache_config_.block_size == kv_block_len
                                   && specpv_cache_config_.total_budget() > 0
                                   && local_kv_head_num_ > 0
                                   && hidden_units_ > 0;

                // Hard guard: the SpecPV buffer must be large enough to hold
                // at least one step worth of EAGLE tree tokens for this
                // engine. If it is smaller than the configured
                // eagle_max_engine_tokens_per_step_ we disable SpecPV at
                // construction time and fall back to the full-KV EAGLE3
                // pipeline so that tree decode invariants are preserved.
                const int tree_step_budget = std::max(1, eagle_max_engine_tokens_per_step_);
                if (geometry_ok
                    && specpv_cache_config_.n_spec_tokens_buf > 0
                    && specpv_cache_config_.n_spec_tokens_buf < tree_step_budget) {
                    TM_LOG_WARNING(
                        "[LlamaV2][SpecPV][fallback] partial KV buffer too small for EAGLE tree "
                        "(spec_tokens_buf=%d, step_budget=%d); disabling SpecPV for this engine.",
                        specpv_cache_config_.n_spec_tokens_buf,
                        tree_step_budget);
                    geometry_ok = false;
                }

                if (!geometry_ok) {
                    TM_LOG_WARNING(
                        "[LlamaV2][SpecPV][fallback] partial KV disabled due to geometry mismatch "
                        "(cache_block_seq_len=%d, specpv_block_size=%d, total_budget=%d, local_kv_heads=%zu, "
                        "hidden_units=%zu)",
                        kv_block_len,
                        specpv_cache_config_.block_size,
                        specpv_cache_config_.total_budget(),
                        local_kv_head_num_,
                        hidden_units_);
                    specpv_cache_config_        = SpecPVCacheConfig{};
                    specpv_supported_           = false;
                    specpv_retrieval_initialized_ = false;
                    specpv_partial_steps_       = 0;
                }
                else {
                    const int kv_head_dim = static_cast<int>(size_per_head_);
                    // SpecPV maintains its own float32 KV view irrespective
                    // of the base model compute dtype. Full-KV cache contents
                    // are flattened and converted as needed when seeding.
                    specpv_kv_cache_ = std::make_unique<PartialKVCache>(specpv_cache_config_,
                                                                        max_batch_size,
                                                                        static_cast<int>(layer_num_),
                                                                        static_cast<int>(local_kv_head_num_),
                                                                        kv_head_dim,
                                                                        kFloat32);

                    if (!specpv_kv_cache_ || !specpv_kv_cache_->is_enabled()) {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] partial KV disabled due to allocation failure "
                            "(max_batch_size=%d, layers=%zu, local_kv_heads=%zu, head_dim=%d, total_budget=%d)",
                            max_batch_size,
                            layer_num_,
                            local_kv_head_num_,
                            kv_head_dim,
                            specpv_cache_config_.total_budget());
                        specpv_kv_cache_.reset();
                        specpv_cache_config_        = SpecPVCacheConfig{};
                        specpv_supported_           = false;
                        specpv_retrieval_initialized_ = false;
                        specpv_partial_steps_       = 0;
                    }
                    else {
                        specpv_supported_ = true;
                        TM_LOG_INFO(
                            "[LlamaV2][SpecPV] partial KV cache initialized "
                            "(block_size=%d, sink_blocks=%d, retrieval_blocks=%d, window_blocks=%d, "
                            "spec_tokens_buf=%d, total_budget=%d)",
                            specpv_cache_config_.block_size,
                            specpv_cache_config_.n_sink_blocks,
                            specpv_cache_config_.n_retrieval_blocks,
                            specpv_cache_config_.n_window_blocks,
                            specpv_cache_config_.n_spec_tokens_buf,
                            specpv_cache_config_.total_budget());
                    }
                }
            }
        }
        else {
            spec_mode_ = SpeculativeDecodingMode::None();
            eagle_module_.reset();
            eagle_buffers_.reset();
            TM_LOG_INFO("[LlamaV2][EAGLE] EAGLE disabled; falling back to baseline decoding for this engine");
        }
    }
}

bool LlamaV2::isSpecPVEnabled() const noexcept
{
    return engine_param_.enable_specpv && specpv_supported_ && isTargetTreeDecodeEnabled()
           && static_cast<bool>(specpv_kv_cache_);
}

bool LlamaV2::shouldUseSpecPV(int seq_len) const noexcept
{
    if (!isSpecPVEnabled() || !specpv_kv_cache_) {
        return false;
    }

    // Basic length- and capacity-based gate mirroring SpecPV's
    // `should_partial_verify` helper: only enable partial verification
    // beyond a configurable context length, and ensure we have enough
    // headroom in the partial cache budget for this step's speculative
    // tokens.
    if (seq_len <= engine_param_.specpv_partial_threshold) {
        return false;
    }

    const int current_tokens = specpv_kv_cache_->get_seq_length();
    const int max_tokens     = specpv_cache_config_.total_budget();

    if (max_tokens <= 0) {
        return false;
    }

    const int step_budget = std::max(1, eagle_max_engine_tokens_per_step_);

    return current_tokens + step_budget + 1 <= max_tokens;
}

void LlamaV2::runEagle3DraftTreeDecode(const Tensor& decoder_features,
                                       const Tensor& base_logits,
                                       int           batch_size,
                                       int           tokens_per_seq,
                                       EagleBuffers& buffers,
                                       Tensor&       draft_logits,
                                       cudaStream_t  stream)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!unified_decoder_ || !eagle3_draft_weight_) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] Eagle3 draft layer unavailable in runEagle3DraftTreeDecode; "
            "skipping speculative draft for this step.");
        return;
    }

    if (!eagle_module_ || !buffers.inputs.draft_tokens) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] Eagle buffers or module unavailable in runEagle3DraftTreeDecode; "
            "skipping speculative draft for this step.");
        return;
    }

    if (batch_size <= 0 || tokens_per_seq <= 0) {
        return;
    }

    // Attention hidden dim (full model space), e.g. 4096 for GPT-OSS-120B.
    // We derive this from the actual decoder_features tensor instead of
    // assuming it matches `hidden_units_` (draft hidden).
    if (!decoder_features || decoder_features.ndim() != 2
        || decoder_features.shape(0) != batch_size) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] decoder_features shape mismatch in runEagle3DraftTreeDecode; "
            "got=[%d,%d], expected batch=%d; skipping speculative draft for this step.",
            decoder_features ? decoder_features.shape(0) : -1,
            decoder_features ? decoder_features.shape(1) : -1,
            batch_size);
        return;
    }

    // Clamp against engine budget to ensure we never overrun the allocated
    // draft/token buffers. The host tree builder will mirror exactly the
    // first `tokens_per_seq` entries per slot.
    const int max_decoding_tokens = static_cast<int>(eagle_module_->getMaxDecodingTokens());
    if (tokens_per_seq > max_decoding_tokens) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] tokens_per_seq=%d exceeds max_decoding_tokens=%d; clamping.",
            tokens_per_seq,
            max_decoding_tokens);
        tokens_per_seq = max_decoding_tokens;
    }

    const int attn_hidden_dim  = decoder_features.shape(1);              // attention space (e.g. 4096)
    const int draft_hidden_dim = static_cast<int>(hidden_units_);        // Eagle3 draft hidden (e.g. 2880)

    if (attn_hidden_dim <= 0 || draft_hidden_dim <= 0) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] invalid dims in runEagle3DraftTreeDecode "
            "(attn_hidden_dim=%d, draft_hidden_dim=%d); skipping speculative draft.",
            attn_hidden_dim,
            draft_hidden_dim);
        return;
    }

    const int total_nodes = batch_size * tokens_per_seq;

    if (total_nodes <= 0) {
        return;
    }

    // Expand the per-slot hidden + capture buffers to a node-major layout
    // [batch * tokens_per_seq, H] so that the draft layer runs per-node
    // instead of per-slot. For now we simply replicate the last-token
    // hidden state for each node in the slot; tree-aware hidden state
    // construction will be layered on later.
    auto repeat_rows = [&](const Tensor& src, int dim, Tensor& dst) {
        if (!src || dim <= 0) {
            dst = Tensor{};
            return;
        }

        const size_t row_bytes = static_cast<size_t>(dim) * byte_size(src.dtype(), 1);
        dst                    = Tensor{{total_nodes, dim}, src.dtype(), kDEVICE};

        const char* src_base = static_cast<const char*>(src.raw_data());
        char*       dst_base = static_cast<char*>(dst.raw_data());

        for (int b = 0; b < batch_size; ++b) {
            const char* src_row = src_base + static_cast<size_t>(b) * row_bytes;
            for (int t = 0; t < tokens_per_seq; ++t) {
                char* dst_row = dst_base + static_cast<size_t>(b * tokens_per_seq + t) * row_bytes;
                check_cuda_error(cudaMemcpyAsync(dst_row, src_row, row_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }
    };

    Tensor node_input_hidden;
    repeat_rows(decoder_features, attn_hidden_dim, node_input_hidden);

    Tensor node_captured_hidden;
    if (eagle_capture_hidden_) {
        const int capture_dim = eagle_capture_hidden_.shape(1);
        repeat_rows(eagle_capture_hidden_, capture_dim, node_captured_hidden);
    }

    // Allocate draft_hidden in *draft* space [total_nodes, draft_hidden_dim].
    Tensor draft_hidden{{total_nodes, draft_hidden_dim}, decoder_features.dtype(), kDEVICE};

    // Optional packed mask for draft tree attention.
    Tensor draft_packed_mask;
    if (eagle_buffers_ && eagle_buffers_->inputs.packed_masks && eagle_module_) {
        using SizeType = EagleBuffers::SizeType;
        const SizeType num_packed =
            static_cast<SizeType>(eagle_module_->getNumPackedMasks());
        if (num_packed > 0) {
            // Build a simple (slot, token_idx) mapping for draft nodes.
            std::vector<SizeType> h_hidden_indices(static_cast<size_t>(total_nodes) * 2);
            for (int b = 0; b < batch_size; ++b) {
                for (int t = 0; t < tokens_per_seq; ++t) {
                    const size_t idx             = static_cast<size_t>(b * tokens_per_seq + t);
                    h_hidden_indices[idx * 2 + 0] = static_cast<SizeType>(b);
                    h_hidden_indices[idx * 2 + 1] = static_cast<SizeType>(t);
                }
            }
            Buffer_<SizeType> d_hidden_indices(h_hidden_indices.size(), kDEVICE);
            check_cuda_error(cudaMemcpyAsync(
                d_hidden_indices.data(),
                h_hidden_indices.data(),
                h_hidden_indices.size() * sizeof(SizeType),
                cudaMemcpyHostToDevice,
                stream));

            draft_packed_mask = Tensor{
                {total_nodes, static_cast<int>(num_packed)},
                kInt32,
                kDEVICE};

            ::lmdeploy::turbomind::kernels::speculative_decoding::invokeGatherTreePackedMask(
                reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.packed_masks),
                static_cast<SizeType>(batch_size),
                static_cast<SizeType>(tokens_per_seq),
                num_packed,
                d_hidden_indices.data(),
                static_cast<SizeType>(total_nodes),
                draft_packed_mask.data<SizeType>(),
                stream);
        }
    }

    // Optional per-node position ids (repeat slot positions) to enable RoPE offsets.
    Tensor node_position_ids;
    if (eagle_buffers_ && eagle_buffers_->inputs.eagle_net_position_ids) {
        std::vector<int> h_pos(static_cast<size_t>(batch_size * tokens_per_seq), 0);
        check_cuda_error(cudaMemcpyAsync(
            h_pos.data(),
            eagle_buffers_->inputs.eagle_net_position_ids,
            h_pos.size() * sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        check_cuda_error(cudaStreamSynchronize(stream));

        std::vector<int> h_pos_nodes(static_cast<size_t>(total_nodes), 0);
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < tokens_per_seq; ++t) {
                const size_t idx = static_cast<size_t>(b * tokens_per_seq + t);
                h_pos_nodes[idx] = h_pos[idx];
            }
        }
        node_position_ids = Tensor{{total_nodes}, kInt32, kDEVICE};
        check_cuda_error(cudaMemcpyAsync(
            node_position_ids.raw_data(),
            h_pos_nodes.data(),
            h_pos_nodes.size() * sizeof(int),
            cudaMemcpyHostToDevice,
            stream));
    }

    const int q_len = tokens_per_seq;
    const int kv_len = tokens_per_seq;
    unified_decoder_->ForwardDraft(node_input_hidden,
                                   node_captured_hidden,
                                   node_position_ids,
                                   draft_packed_mask,
                                   draft_tree_offsets,
                                   draft_runtime_offsets,
                                   draft_kv_lens_runtime,
                                   draft_successor_offsets,
                                   draft_successor_counts,
                                   q_len,
                                   kv_len,
                                   /*past_kv_len=*/0,
                                   draft_hidden,
                                   total_nodes,
                                   stream);

    // Project to vocab using the same LM head as the base model.
    const int vocab_pad = static_cast<int>(vocab_size_padded_);
    if (vocab_pad <= 0) {
        TM_LOG_WARNING("[LlamaV2][EAGLE][fallback] invalid vocab_size_padded_=%d; skipping speculative draft.",
                       vocab_pad);
        return;
    }

    Buffer local_draft_logits_buffer(static_cast<ssize_t>(total_nodes) * vocab_pad, dtype_, kDEVICE);
    draft_logits = postDecodeEmbedding(draft_hidden, local_draft_logits_buffer);

    if (!draft_logits) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] postDecodeEmbedding returned empty draft logits; "
            "skipping speculative draft.");
        return;
    }

    // Cast logits to FP32 for argmax scatter into draft_tokens.
    Tensor draft_logits_f32{{total_nodes, vocab_pad}, kFloat32, kDEVICE};
    invokeCastFloat2D(draft_logits, draft_logits_f32, stream);
    sync_check_cuda_error();

    using kernels::speculative_decoding::SizeType;
    using kernels::speculative_decoding::TreeLogitsToTargetsParams;

    // Build (slot, token_idx) mapping for each node.
    std::vector<SizeType> h_hidden_indices(static_cast<size_t>(total_nodes) * 2, 0);
    for (int idx = 0; idx < total_nodes; ++idx) {
        const int slot      = idx / tokens_per_seq;
        const int token_idx = idx % tokens_per_seq;
        h_hidden_indices[static_cast<size_t>(idx) * 2 + 0] = static_cast<SizeType>(slot);
        h_hidden_indices[static_cast<size_t>(idx) * 2 + 1] = static_cast<SizeType>(token_idx);
    }

    Buffer_<SizeType> d_hidden_indices(h_hidden_indices.size(), kDEVICE);
    check_cuda_error(cudaMemcpyAsync(d_hidden_indices.data(),
                                     h_hidden_indices.data(),
                                     h_hidden_indices.size() * sizeof(SizeType),
                                     cudaMemcpyHostToDevice,
                                     stream));

    // Argmax per node -> draft_tokens[slot, token_idx].
    TreeLogitsToTargetsParams draft_reduce{};
    draft_reduce.logits              = draft_logits_f32.data<float>();
    draft_reduce.num_tree_tokens     = static_cast<SizeType>(total_nodes);
    draft_reduce.vocab_size          = static_cast<SizeType>(vocab_pad);
    draft_reduce.hidden_indices      = d_hidden_indices.data();
    draft_reduce.max_batch_size      = static_cast<SizeType>(engine_param_.max_batch_size);
    draft_reduce.max_decoding_tokens = static_cast<SizeType>(eagle_module_->getMaxDecodingTokens());
    draft_reduce.target_tokens       = reinterpret_cast<kernels::speculative_decoding::TokenIdType*>(
        buffers.inputs.draft_tokens);
    draft_reduce.stream              = stream;

    invokeTreeLogitsToTargetIds(draft_reduce);

    // Optional: fill target_tokens from base logits (argmax) when provided
    // and target-tree decode is not enabled for this step. We compute a
    // per-slot argmax and broadcast it across tokens_per_seq to avoid
    // tiling the entire logits matrix.
    if (base_logits && buffers.inputs.target_tokens && !isTargetTreeDecodeActiveStep()) {
        const int vocab_size_pad = base_logits.shape(1);
        if (vocab_size_pad <= 0) {
            TM_LOG_WARNING("[LlamaV2][EAGLE][fallback] base_logits vocab size invalid; skipping target argmax.");
        }
        else {
            Buffer_<int> topk_buf(batch_size, kDEVICE);
            Buffer_<int> kept_buf(batch_size, kDEVICE);
            std::vector<int> h_topk(batch_size, 1);
            std::vector<int> h_kept(batch_size, 1);
            core::Copy(h_topk.data(), batch_size, topk_buf.data());
            core::Copy(h_kept.data(), batch_size, kept_buf.data());

            Buffer_<int> target_sorted_indices(batch_size * vocab_size_pad, kDEVICE);

            turbomind::TopKSortFilterParams target_params{};
            target_params.logits            = const_cast<void*>(base_logits.buffer().raw_data());
            target_params.sorted_logits     = const_cast<void*>(base_logits.buffer().raw_data());
            target_params.sorted_indices    = target_sorted_indices.data();
            target_params.kept              = kept_buf.data();
            target_params.top_ks            = topk_buf.data();
            target_params.max_top_k         = 1;
            target_params.batch_size        = batch_size;
            target_params.vocab_size        = static_cast<int>(vocab_size_);
            target_params.vocab_size_padded = vocab_size_pad;

            invokeTopKSortFilter<float>(target_params, stream);

            // The first column of sorted_indices holds top-1 per row.
            Buffer_<int> target_argmax(batch_size, kDEVICE);
            const size_t src_pitch = sizeof(int) * vocab_size_pad;
            const size_t dst_pitch = sizeof(int);
            check_cuda_error(cudaMemcpy2DAsync(target_argmax.data(),
                                               dst_pitch,
                                               target_sorted_indices.data(),
                                               src_pitch,
                                               sizeof(int),
                                               batch_size,
                                               cudaMemcpyDeviceToDevice,
                                               stream));

            const int max_decoding_tokens = static_cast<int>(eagle_module_->getMaxDecodingTokens());
            const dim3 grid(static_cast<unsigned>(batch_size));
            const dim3 block(static_cast<unsigned>(tokens_per_seq));
            broadcastTargetTokenKernel<<<grid, block, 0, stream>>>(
                target_argmax.data(), tokens_per_seq, max_decoding_tokens, buffers.inputs.target_tokens, batch_size);
        }
    }

    Buffer_<int> draft_tokens(static_cast<size_t>(total_nodes), kCPU);
    Buffer_<int> target_tokens(static_cast<size_t>(total_nodes), kCPU);

    // Record per-slot draft lengths for downstream mask/offset helpers.
    if (buffers.inputs.draft_lens) {
        std::vector<kernels::speculative_decoding::SizeType> h_draft_lens(batch_size,
                                                                          static_cast<int>(tokens_per_seq));
        check_cuda_error(cudaMemcpyAsync(buffers.inputs.draft_lens,
                                         h_draft_lens.data(),
                                         static_cast<size_t>(batch_size) * sizeof(kernels::speculative_decoding::SizeType),
                                         cudaMemcpyHostToDevice,
                                         stream));
    }

    // Draft offsets: cumulative per-slot draft positions for packed-mask /
    // offset helpers. Layout [batch+1].
    if (buffers.inputs.draft_offsets) {
        std::vector<kernels::speculative_decoding::SizeType> h_draft_offsets(
            static_cast<size_t>(batch_size) + 1, 0);
        for (int b = 0; b < batch_size; ++b) {
            h_draft_offsets[b + 1] = h_draft_offsets[b] + tokens_per_seq;
        }
        check_cuda_error(cudaMemcpyAsync(buffers.inputs.draft_offsets,
                                         h_draft_offsets.data(),
                                         h_draft_offsets.size() * sizeof(kernels::speculative_decoding::SizeType),
                                         cudaMemcpyHostToDevice,
                                         stream));
    }

    check_cuda_error(cudaMemcpyAsync(draft_tokens.data(),
                                     buffers.inputs.draft_tokens,
                                     static_cast<size_t>(total_nodes) * sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     stream));

    if (base_logits && buffers.inputs.target_tokens && !isTargetTreeDecodeActiveStep()) {
        check_cuda_error(cudaMemcpyAsync(target_tokens.data(),
                                         buffers.inputs.target_tokens,
                                         static_cast<size_t>(total_nodes) * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream));
    }

    check_cuda_error(cudaStreamSynchronize(stream));

    if (isEagleDebugEnabled()) {
        TM_LOG_DEBUG(
            "[LlamaV2][EAGLE] runEagle3DraftTreeDecode: batch=%d, tokens_per_seq=%d, num_draft_tokens=%d",
            batch_size,
            tokens_per_seq,
            total_nodes);
    }
}

void LlamaV2::runEagleTargetTreeDecode(int batch_size,
                                       const int*       d_sequence_lengths,
                                       const Sequence** sequences)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!isTargetTreeDecodeEnabled()) {
        eagle_tree_target_tokens_valid_ = false;
        return;
    }

    // Guard against any late-breaking incompatibility in model dtype or
    // vocab / geometry configuration. Tree decode requires:
    //   - a valid hidden-state buffer with the base model dtype,
    //   - a positive max_tree_tokens_ budget,
    //   - an FP32 logits buffer sized for
    //       [max_tree_tokens_, vocab_size_padded_].
    // On failure we disable target-tree decode for this engine and fall
    // back to baseline EAGLE.
    const bool hidden_ok = tree_hidden_states_
                           && tree_hidden_states_.dtype() == dtype_
                           && tree_hidden_states_.ndim() == 2
                           && tree_hidden_states_.shape(0) >= max_tree_tokens_
                           && static_cast<size_t>(tree_hidden_states_.shape(1)) == hidden_units_;

    const bool logits_ok = tree_logits_buffer_
                           && tree_logits_buffer_.dtype() == kFloat32
                           && tree_logits_buffer_.size()
                                  >= static_cast<ssize_t>(max_tree_tokens_)
                                         * static_cast<ssize_t>(vocab_size_padded_);

    if (!hidden_ok || !logits_ok || max_tree_tokens_ <= 0 || vocab_size_padded_ == 0) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] target-tree decode entry mismatch "
            "(model=%s, tree_hidden=%s, tree_logits=%s, max_tree_tokens=%d, vocab_size_padded=%zu); "
            "disabling target-tree for this engine.",
            to_string(dtype_),
            tree_hidden_states_ ? to_string(tree_hidden_states_.dtype()) : "nil",
            tree_logits_buffer_ ? to_string(tree_logits_buffer_.dtype()) : "nil",
            max_tree_tokens_,
            vocab_size_padded_);
        target_tree_supported_          = false;
        eagle_tree_target_tokens_valid_ = false;
        tree_hidden_states_             = Tensor{};
        tree_logits_buffer_             = Buffer{};
        max_tree_tokens_                = 0;
        return;
    }

    eagle_tree_target_tokens_valid_ = false;

    // Stage 1: run the staging kernel to populate per-slot tree metadata
    // (gen_lens / seq_lens / ctx_lens) and the initial per-slot flattened
    // tree token layout in eagle_net_input_ids / eagle_net_hidden_indices.
    targetTreeDecode(batch_size, d_sequence_lengths);

    if (!eagle_buffers_ || !eagle_buffers_->isAllocated() || !tree_hidden_states_ || !tree_logits_buffer_) {
        return;
    }

    using SizeType    = EagleBuffers::SizeType;
    using TokenIdType = EagleBuffers::TokenIdType;
    using namespace kernels::speculative_decoding;

    const SizeType* d_gen_lens = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.eagle_net_gen_lens);
    if (!d_gen_lens) {
        return;
    }

    // Copy per-slot speculative generation lengths to host and sum them
    // to obtain the flat tree token count for this engine step.
    std::vector<SizeType> h_gen_lens(batch_size, 0);
    check_cuda_error(cudaMemcpyAsync(
        h_gen_lens.data(), d_gen_lens, batch_size * sizeof(SizeType), cudaMemcpyDeviceToHost, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    SizeType num_tree_tokens = 0;
    for (int i = 0; i < batch_size; ++i) {
        const SizeType len = h_gen_lens[i];
        if (len > 0) {
            num_tree_tokens += len;
        }
    }

    if (num_tree_tokens <= 0) {
        return;
    }

    if (num_tree_tokens > static_cast<SizeType>(max_tree_tokens_)) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] num_tree_tokens=%d exceeds max_tree_tokens_=%d; "
            "clamping to buffer limit",
            static_cast<int>(num_tree_tokens),
            max_tree_tokens_);
        num_tree_tokens = static_cast<SizeType>(max_tree_tokens_);
    }

    const SizeType max_decoding_tokens =
        static_cast<SizeType>(eagle_module_->getMaxDecodingTokens());

    const TokenIdType* d_eagle_ids =
        reinterpret_cast<TokenIdType const*>(eagle_buffers_->inputs.eagle_net_input_ids);
    const SizeType* d_hidden_indices_flat =
        reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.eagle_net_hidden_indices);

    if (!d_eagle_ids || !d_hidden_indices_flat) {
        return;
    }

    const SizeType flat_capacity = static_cast<SizeType>(batch_size) * max_decoding_tokens;

    std::vector<TokenIdType> h_eagle_ids(flat_capacity);
    std::vector<SizeType>    h_hidden_indices_flat(static_cast<size_t>(flat_capacity) * 2);

    check_cuda_error(cudaMemcpyAsync(h_eagle_ids.data(),
                                     d_eagle_ids,
                                     static_cast<size_t>(flat_capacity) * sizeof(TokenIdType),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    check_cuda_error(cudaMemcpyAsync(h_hidden_indices_flat.data(),
                                     d_hidden_indices_flat,
                                     static_cast<size_t>(flat_capacity) * 2 * sizeof(SizeType),
                                     cudaMemcpyDeviceToHost,
                                     stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    // Compact per-slot layout [batch, max_decoding_tokens] into a contiguous
    // [num_tree_tokens] view for this step, tracking the effective per-slot
    // tree lengths after any clamping.
    std::vector<TokenIdType> h_tree_ids(static_cast<size_t>(num_tree_tokens));
    std::vector<SizeType>    h_tree_hidden_indices(static_cast<size_t>(num_tree_tokens) * 2);
    std::vector<int>         h_tree_lens(batch_size, 0);
    std::vector<int>         h_tree_lens_runtime(batch_size, 0);
    std::vector<int>         h_k_len_runtime(batch_size, 0);
    const int                tree_len_cap = static_cast<int>(engine_param_.spec_max_decoding_tokens);

    SizeType prefix = 0;
    for (int local_idx = 0; local_idx < batch_size && prefix < num_tree_tokens; ++local_idx) {
        SizeType len = h_gen_lens[local_idx];
        if (len <= 0) {
            continue;
        }
        if (prefix + len > num_tree_tokens) {
            len = num_tree_tokens - prefix;
        }

        const SizeType slot_offset = static_cast<SizeType>(local_idx) * max_decoding_tokens;
        for (SizeType t = 0; t < len; ++t) {
            const SizeType flat_orig = slot_offset + t;
            if (flat_orig >= flat_capacity) {
                break;
            }

            const SizeType dst = prefix + t;

            h_tree_ids[static_cast<size_t>(dst)] = h_eagle_ids[static_cast<size_t>(flat_orig)];

            const SizeType src_hidden = flat_orig * 2;
            const SizeType dst_hidden = dst * 2;
            h_tree_hidden_indices[static_cast<size_t>(dst_hidden) + 0] =
                h_hidden_indices_flat[static_cast<size_t>(src_hidden) + 0];
            h_tree_hidden_indices[static_cast<size_t>(dst_hidden) + 1] =
                h_hidden_indices_flat[static_cast<size_t>(src_hidden) + 1];
        }

        h_tree_lens[local_idx] = static_cast<int>(len);
        h_tree_lens_runtime[local_idx] = static_cast<int>(len > 0 ? len - 1 : 0);
        prefix += len;
    }

    for (int i = 0; i < batch_size; ++i) {
        if (tree_len_cap > 0 && h_tree_lens[i] > tree_len_cap) {
            TM_LOG_WARNING(
                "[LlamaV2][EAGLE][tree] clamping tree_len from %d to %d for slot=%d (max_decoding_tokens cap)",
                h_tree_lens[i],
                tree_len_cap,
                i);
            h_tree_lens[i] = tree_len_cap;
        }
    }

    // Cumulative tree token offsets (host) for this step; used for mask /
    // offset bookkeeping and to sanity-check total token counts.
    std::vector<int> h_cum_tree_tokens(static_cast<size_t>(batch_size) + 1, 0);
    std::vector<int> h_cum_tree_runtime(static_cast<size_t>(batch_size) + 1, 0);
    for (int i = 0; i < batch_size; ++i) {
        h_cum_tree_tokens[i + 1] = h_cum_tree_tokens[i] + h_tree_lens[i];
        h_cum_tree_runtime[i + 1] = h_cum_tree_runtime[i] + h_tree_lens_runtime[i];
    }

    // Propagate the clamped per-slot tree lengths back to device buffers
    // so downstream mask/offset helpers see consistent lengths.
    if (eagle_buffers_->inputs.eagle_net_gen_lens) {
        check_cuda_error(cudaMemcpyAsync(
            eagle_buffers_->inputs.eagle_net_gen_lens,
            h_tree_lens.data(),
            static_cast<size_t>(batch_size) * sizeof(SizeType),
            cudaMemcpyHostToDevice,
            stream_));
    }
    if (eagle_buffers_->inputs.target_offsets) {
        check_cuda_error(cudaMemcpyAsync(
            eagle_buffers_->inputs.target_offsets,
            h_cum_tree_tokens.data(),
            static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
            cudaMemcpyHostToDevice,
            stream_));
    }

    // Capture cumulative tree-length prefix for potential packed-mask /
    // offset helpers. Stored on device as an auxiliary buffer.
    Buffer_<SizeType> d_cum_tree_lens(static_cast<size_t>(batch_size) + 1, kDEVICE);
    Buffer_<SizeType> d_cum_tree_runtime(static_cast<size_t>(batch_size) + 1, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(
        d_cum_tree_lens.data(),
        h_cum_tree_tokens.data(),
        static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
        cudaMemcpyHostToDevice,
        stream_));
    check_cuda_error(cudaMemcpyAsync(
        d_cum_tree_runtime.data(),
        h_cum_tree_runtime.data(),
        static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
        cudaMemcpyHostToDevice,
        stream_));

    // Successor metadata: build per-node successor histograms from paths
    // and compact them into a flattened buffer with per-request offsets.
    Buffer_<SizeType> d_successor_offsets(static_cast<size_t>(batch_size) + 1, kDEVICE);
    Buffer_<SizeType> d_num_successors(static_cast<size_t>(batch_size) * max_decoding_tokens, kDEVICE);
    Buffer_<SizeType> d_successor_counts;
    const size_t successor_capacity =
        static_cast<size_t>(batch_size) * static_cast<size_t>(max_decoding_tokens);
    if (eagle_buffers_->inputs.successor_counts) {
        d_successor_counts = Buffer_<SizeType>{eagle_buffers_->inputs.successor_counts,
                                               static_cast<ssize_t>(successor_capacity),
                                               kDEVICE};
    }
    else {
        d_successor_counts = Buffer_<SizeType>(static_cast<ssize_t>(successor_capacity), kDEVICE);
    }

    kernels::speculative_decoding::invokeExtractSuccessorsFromPaths(
        reinterpret_cast<kernels::speculative_decoding::SizeType const*>(eagle_buffers_->inputs.draft_paths),
        static_cast<kernels::speculative_decoding::SizeType>(batch_size),
        max_decoding_tokens,
        max_path_len,
        d_successor_offsets.data(),
        d_successor_counts.data(),
        d_num_successors.data(),
        stream_);

    // Mirror successor offsets into persistent buffers when available.
    if (eagle_buffers_->inputs.successor_offsets
        && eagle_buffers_->inputs.successor_offsets != d_successor_offsets.data()) {
        check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.successor_offsets,
                                         d_successor_offsets.data(),
                                         static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
                                         cudaMemcpyDeviceToDevice,
                                         stream_));
    }
    if (eagle_buffers_->inputs.successor_counts
        && eagle_buffers_->inputs.successor_counts != d_successor_counts.data()) {
        check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.successor_counts,
                                         d_successor_counts.data(),
                                         successor_capacity * sizeof(SizeType),
                                         cudaMemcpyDeviceToDevice,
                                         stream_));
    }

    // Optional debug: verify successor offsets are monotonic and log the
    // total number of successor entries (topKs) produced this step.
    if (turbomind::isEagleDebugEnabled()) {
        std::vector<SizeType> h_succ_offsets(static_cast<size_t>(batch_size) + 1, 0);
        check_cuda_error(cudaMemcpyAsync(h_succ_offsets.data(),
                                         d_successor_offsets.data(),
                                         static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
                                         cudaMemcpyDeviceToHost,
                                         stream_));
        check_cuda_error(cudaStreamSynchronize(stream_));

        bool monotonic = true;
        for (int i = 1; i < batch_size + 1; ++i) {
            if (h_succ_offsets[i] < h_succ_offsets[i - 1]) {
                monotonic = false;
                break;
            }
        }
        if (!monotonic) {
            TM_LOG_WARNING("[LlamaV2][EAGLE][successor] non-monotonic successor_offsets detected");
        }
        else if (tp_rank_ == 0) {
            TM_LOG_INFO("[LlamaV2][EAGLE][successor] batch=%d total_topk=%d",
                        batch_size,
                        h_succ_offsets.back());
        }
    }

    if (prefix <= 0) {
        return;
    }

    num_tree_tokens = prefix;

    Buffer_<int> d_tree_ids(num_tree_tokens, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(d_tree_ids.data(),
                                     h_tree_ids.data(),
                                     static_cast<size_t>(num_tree_tokens) * sizeof(TokenIdType),
                                     cudaMemcpyHostToDevice,
                                     stream_));

    Buffer_<SizeType> d_tree_hidden_indices(static_cast<ssize_t>(num_tree_tokens) * 2, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(d_tree_hidden_indices.data(),
                                     h_tree_hidden_indices.data(),
                                     static_cast<size_t>(num_tree_tokens) * 2 * sizeof(SizeType),
                                     cudaMemcpyHostToDevice,
                                     stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    const int token_num = static_cast<int>(num_tree_tokens);

    // Stage 2: embed compact tree token ids into a [num_tree_tokens,
    // hidden_units_] buffer which will be used as decoder_input for the
    // scratch-KV tree decode pass.
    Tensor tree_input_embeds;
    if (token_num) {
        const auto& embedding_table = weights_->pre_decoder_embedding.weight;
        TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

        tree_input_embeds = Tensor{{token_num, static_cast<int>(hidden_units_)}, dtype_, kDEVICE};

        if (tp_size_ == 1) {
            invokeEmbeddingLookup(tree_input_embeds, d_tree_ids, embedding_table, stream_);
            sync_check_cuda_error();
        }
        else if (use_allgather_2d_) {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{tree_hidden_states_.buffer(),
                            {token_num, tp_size_, static_cast<int>(local_hidden_units)}};

            auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

            invokeEmbeddingLookup(local, d_tree_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather2D(local.raw_data(),
                                       temp.raw_data(),
                                       hidden_units_,
                                       local_hidden_units,
                                       local_hidden_units,
                                       token_num,
                                       local.dtype(),
                                       {true, true},
                                       comm_->d_tp_group,
                                       stream_);
            sync_check_cuda_error();

            Copy(temp.buffer(), tree_input_embeds.buffer());
        }
        else {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{tree_hidden_states_.buffer(),
                            {tp_size_, token_num, static_cast<int>(local_hidden_units)}};

            auto local = temp.slice(tp_rank_).squeeze(0);

            invokeEmbeddingLookup(local, d_tree_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather(
                local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_->d_tp_group, stream_);
            sync_check_cuda_error();

            invokeInPlaceTranspose102((uint16_t*)tree_input_embeds.raw_data(),
                                      (uint16_t*)temp.raw_data(),
                                      tp_size_,
                                      token_num,
                                      local_hidden_units,
                                      false,
                                      stream_);
            sync_check_cuda_error();
        }
    }

    // Stage 3: build per-slot tree decode lengths and a scratch KV block
    // table, then run UnifiedDecoder over the compacted tree tokens with a
    // pure prefill-style pass (no mutation of SequenceManager or live KV).
    std::vector<int> h_q_len(batch_size, 0);
    std::vector<int> h_k_len(batch_size, 0);

    // Derive per-slot prefix (history) lengths from the live sequences when
    // available so that tree decode can reuse prefix KV as read-only. When
    // sequences are not provided or SequenceManager is unavailable we fall
    // back to a history_len=0 interpretation for this step.
    const int block_seq_len = attn_param_.cache_block_seq_len;

    std::vector<int> h_prefix_len(batch_size, 0);
    if (sequences && sequence_manager_) {
        for (int i = 0; i < batch_size; ++i) {
            const Sequence* seq = sequences[i];
            if (!seq) {
                continue;
            }
            h_prefix_len[i] = std::max(0, seq->cache_len);
        }
    }

    int max_history_len = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int tree_len   = h_tree_lens[i];
        const int prefix_len = h_prefix_len[i];

        if (tree_len < 0 || prefix_len < 0) {
            TM_LOG_WARNING(
                "[LlamaV2][EAGLE][fallback] target-tree decode sees negative lengths for slot=%d "
                "(prefix_len=%d, tree_len=%d); disabling target-tree for this engine.",
                i,
                prefix_len,
                tree_len);
            target_tree_supported_          = false;
            eagle_tree_target_tokens_valid_ = false;
            return;
        }

        // Sanity check: when a live sequence is present, prefix_len must
        // not exceed the KV coverage implied by its blocks.
        if (sequences && sequence_manager_ && sequences[i]) {
            const Sequence* seq          = sequences[i];
            const int       block_tokens = static_cast<int>(seq->blocks.size()) * block_seq_len;
            if (prefix_len > block_tokens) {
                TM_LOG_WARNING(
                    "[LlamaV2][EAGLE][fallback] target-tree KV invariant violated for slot=%d "
                    "(prefix_len=%d, blocks=%zu, block_seq_len=%d); disabling target-tree.",
                    i,
                    prefix_len,
                    seq->blocks.size(),
                    block_seq_len);
                target_tree_supported_          = false;
                eagle_tree_target_tokens_valid_ = false;
                return;
            }
        }

        const int seq_len_i = prefix_len + tree_len;
        const int seq_len_rt = prefix_len + h_tree_lens_runtime[i];
        max_history_len     = std::max(max_history_len, seq_len_i);

        h_q_len[i]         = tree_len;
        h_k_len[i]         = seq_len_i;
        h_k_len_runtime[i] = seq_len_rt;

        if (eagle_buffers_) {
            if (eagle_buffers_->inputs.eagle_net_ctx_lens) {
                h_prefix_len[i] = std::max(0, h_prefix_len[i]);
            }
        }
    }

    // Sync per-slot context/sequence lengths to device so packed-mask and
    // downstream offset helpers can rely on the clamped lengths.
    if (eagle_buffers_) {
        if (eagle_buffers_->inputs.eagle_net_ctx_lens) {
            check_cuda_error(cudaMemcpyAsync(
                eagle_buffers_->inputs.eagle_net_ctx_lens,
                h_prefix_len.data(),
                static_cast<size_t>(batch_size) * sizeof(SizeType),
                cudaMemcpyHostToDevice,
                stream_));
        }
        if (eagle_buffers_->inputs.eagle_net_seq_lens) {
            std::vector<SizeType> h_seq_lens(batch_size, 0);
            for (int i = 0; i < batch_size; ++i) {
                h_seq_lens[i] = h_prefix_len[i] + h_tree_lens[i];
            }
            check_cuda_error(cudaMemcpyAsync(
                eagle_buffers_->inputs.eagle_net_seq_lens,
                h_seq_lens.data(),
                static_cast<size_t>(batch_size) * sizeof(SizeType),
                cudaMemcpyHostToDevice,
                stream_));
        }
    }

    // Decide whether to run tree decode against the SpecPV partial KV
    // view instead of the full prefix KV. In SpecPV mode we rely on the
    // partial cache geometry and verified length to bound the effective
    // history length seen by attention.
    bool use_specpv = isSpecPVEnabled()
                      && specpv_kv_cache_
                      && specpv_retrieval_initialized_
                      && shouldUseSpecPV(max_history_len);

    // Build a per-slot block table that reuses existing prefix KV blocks as
    // read-only (via SequenceManager) and allocates a scratch region for
    // any additional blocks needed to store tree tokens. The logical block
    // layout for slot `i` is:
    //   [prefix_blocks..., scratch_tree_blocks...]
    // where prefix_blocks come from Sequence::blocks and scratch blocks are
    // carved out of a dedicated pool for this call only.
    std::vector<int> h_cu_block_nums(batch_size + 1, 0);
    std::vector<int> h_prefix_blocks(batch_size, 0);
    std::vector<int> h_extra_blocks(batch_size, 0);

    // In SpecPV mode we override the prefix geometry based on the
    // partial KV budget; otherwise we keep the existing full-KV path.
    int partial_prefix_tokens = 0;
    if (use_specpv) {
        partial_prefix_tokens = specpv_cache_config_.sink_size()
                                + specpv_cache_config_.retrieval_size()
                                + specpv_cache_config_.window_size()
                                + specpv_kv_cache_->global_verified_len();
        if (partial_prefix_tokens <= 0) {
            TM_LOG_WARNING(
                "[LlamaV2][SpecPV][fallback] partial prefix tokens <= 0 in tree decode; disabling SpecPV.");
            specpv_supported_             = false;
            specpv_retrieval_initialized_ = false;
            specpv_partial_steps_         = 0;
            specpv_kv_cache_.reset();
            use_specpv = false;
        }
        else {
            // Clear any stale candidate bookkeeping at the start of a
            // new tree decode step. Candidate KV rows for the current
            // step, when implemented, will be staged explicitly after
            // decode based on the tree tokens for this step.
            if (specpv_kv_cache_) {
                specpv_kv_cache_->clear_candidates();
            }
        }
    }

    for (int i = 0; i < batch_size; ++i) {
        const int tree_len   = h_tree_lens[i];
        const int prefix_len = h_prefix_len[i];

        int prefix_blocks = 0;

        if (use_specpv) {
            const int prefix_len_i = std::min(prefix_len, partial_prefix_tokens);
            const int total_len    = prefix_len_i + tree_len;

            h_prefix_len[i] = prefix_len_i;
            h_k_len[i]      = total_len;

            prefix_blocks = (prefix_len_i > 0)
                                ? (prefix_len_i + block_seq_len - 1) / block_seq_len
                                : 0;
        }
        else {
            if (sequences && sequence_manager_) {
                const Sequence* seq = sequences[i];
                if (seq) {
                    prefix_blocks = static_cast<int>(seq->blocks.size());
                }
            }

            const int total_len = prefix_len + tree_len;
            h_k_len[i]          = total_len;
        }

        const int required_blk = (h_k_len[i] > 0)
                                     ? (h_k_len[i] + block_seq_len - 1) / block_seq_len
                                     : 0;
        const int extra_blocks = std::max(0, required_blk - prefix_blocks);

        h_prefix_blocks[i]    = std::max(prefix_blocks, 0);
        h_extra_blocks[i]     = std::max(extra_blocks, 0);
        const int blocks_i    = h_prefix_blocks[i] + h_extra_blocks[i];
        h_cu_block_nums[i + 1] = h_cu_block_nums[i] + blocks_i;
    }

    const int total_blocks = h_cu_block_nums[batch_size];
    if (total_blocks <= 0) {
        return;
    }

    Buffer_<int> d_cu_block_nums(batch_size + 1, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(d_cu_block_nums.data(),
                                     h_cu_block_nums.data(),
                                     static_cast<size_t>(batch_size + 1) * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     stream_));

    DataType kv_dtype = dtype_;
    if (param_.quant_policy & QuantPolicy::kCacheKVInt8) {
        kv_dtype = kUint8;
    }
    else if (param_.quant_policy & QuantPolicy::kCacheKVInt4) {
        kv_dtype = kUint4;
    }

    // SpecPV integration currently supports only unquantized fp16/bf16
    // KV for the tree decode path. On any other KV dtype we fall back to
    // the full-KV EAGLE3 pipeline.
    bool kv_dtype_ok = (dtype_ == kFloat16 && kv_dtype == kFloat16);
#if ENABLE_BF16
    kv_dtype_ok = kv_dtype_ok || (dtype_ == kBfloat16 && kv_dtype == kBfloat16);
#endif

    if (use_specpv && !kv_dtype_ok) {
        TM_LOG_WARNING(
            "[LlamaV2][SpecPV][fallback] partial KV tree decode only supports fp16/bf16 KV "
            "(model_dtype=%s, kv_dtype=%s); disabling SpecPV for this engine.",
            to_string(dtype_),
            to_string(kv_dtype));
        use_specpv                     = false;
        specpv_supported_              = false;
        specpv_retrieval_initialized_  = false;
        specpv_partial_steps_          = 0;
        specpv_kv_cache_.reset();
    }

    const size_t kv_block_bytes = get_cache_block_size(dtype_,
                                                       kv_dtype,
                                                       static_cast<int>(layer_num_),
                                                       static_cast<int>(local_kv_head_num_),
                                                       static_cast<int>(size_per_head_),
                                                       block_seq_len);

    // Scratch blocks are used for the tree portion of each sequence.
    // In SpecPV mode we also place the partial prefix KV into scratch
    // blocks so that tree decode can attend over the partial KV view
    // without touching SequenceManager blocks.
    int total_extra_blocks = 0;
    for (int i = 0; i < batch_size; ++i) {
        total_extra_blocks += h_extra_blocks[i];
    }

    const int scratch_blocks = use_specpv ? total_blocks : std::max(1, total_extra_blocks);

    Buffer scratch_kv_blocks(static_cast<ssize_t>(scratch_blocks)
                                 * static_cast<ssize_t>(kv_block_bytes),
                             kUint8,
                             kDEVICE);

    std::vector<uintptr_t> h_block_ptrs(static_cast<size_t>(total_blocks));
    auto*                  kv_base = static_cast<char*>(scratch_kv_blocks.raw_data());

    int scratch_block_cursor = 0;
    for (int slot = 0; slot < batch_size; ++slot) {
        const int prefix_blocks = h_prefix_blocks[slot];
        const int extra_blocks  = h_extra_blocks[slot];
        const int base_index    = h_cu_block_nums[slot];

        if (use_specpv) {
            // In SpecPV mode all prefix + tree blocks for this slot live in
            // scratch_kv_blocks; SequenceManager blocks are not used for
            // tree decode KV.
            const int blocks_i = prefix_blocks + extra_blocks;
            for (int j = 0; j < blocks_i; ++j) {
                const int global_idx = base_index + j;
                h_block_ptrs[static_cast<size_t>(global_idx)] =
                    reinterpret_cast<uintptr_t>(
                        kv_base + static_cast<size_t>(global_idx) * kv_block_bytes);
            }
            (void)scratch_block_cursor;
        }
        else {
            // Prefix KV: reuse existing block pointers from SequenceManager as
            // read-only so tree tokens can attend to committed history without
            // mutating live KV state.
            if (sequence_manager_ && sequences && sequences[slot]) {
                const Sequence* seq        = sequences[slot];
                const int       seq_blocks = static_cast<int>(seq->blocks.size());

                // Invariant: prefix_blocks must match the live sequence's block
                // count when we are reusing prefix KV; otherwise the KV layout is
                // inconsistent and we must fall back to single-step targets.
                if (prefix_blocks != seq_blocks) {
                    TM_LOG_WARNING(
                        "[LlamaV2][EAGLE][fallback] target-tree KV invariant violated for slot=%d "
                        "(prefix_blocks=%d, seq.blocks.size()=%d); disabling target-tree.",
                        slot,
                        prefix_blocks,
                        seq_blocks);
                    target_tree_supported_          = false;
                    eagle_tree_target_tokens_valid_ = false;
                    return;
                }

                for (int j = 0; j < seq_blocks; ++j) {
                    const int   block_id = seq->blocks[j];
                    void* const ptr      = sequence_manager_->GetBlockPtr(block_id);
                    h_block_ptrs[static_cast<size_t>(base_index + j)] =
                        reinterpret_cast<uintptr_t>(ptr);
                }
            }

            // Scratch KV for tree tokens beyond the existing prefix coverage.
            for (int j = 0; j < extra_blocks; ++j) {
                const int global_idx = base_index + prefix_blocks + j;
                h_block_ptrs[static_cast<size_t>(global_idx)] =
                    reinterpret_cast<uintptr_t>(kv_base
                                                + static_cast<size_t>(scratch_block_cursor + j) * kv_block_bytes);
            }

            scratch_block_cursor += extra_blocks;
        }
    }

    // In SpecPV mode, populate the scratch prefix blocks with K/V taken
    // from the partial KV cache so that tree decode attention runs over
    // the SpecPV-compressed history instead of the full prefix KV.
    if (use_specpv) {
        if (!specpv_kv_cache_) {
            TM_LOG_WARNING(
                "[LlamaV2][SpecPV][fallback] partial KV cache missing in tree decode; disabling SpecPV.");
            specpv_supported_             = false;
            specpv_retrieval_initialized_ = false;
            specpv_partial_steps_         = 0;
            specpv_kv_cache_.reset();
            use_specpv = false;
        }
        else {
            const int head_dim  = static_cast<int>(size_per_head_);
            const int kv_heads  = static_cast<int>(local_kv_head_num_);
            bool      specpv_ok = true;

            auto fill_layer_half = [&](auto head_dim_tag) {
                constexpr int kHeadDim = decltype(head_dim_tag)::value;

                block::Config<half, half, kHeadDim> cfg{kv_heads, block_seq_len};
                block::Layout<block::Config<half, half, kHeadDim>> layout{cfg};

                const int prefix_tokens = partial_prefix_tokens;

                for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
                    auto [k_prefix, v_prefix] = specpv_kv_cache_->active_prefix(layer, prefix_tokens);
                    if (!k_prefix || !v_prefix || k_prefix.dtype() != kFloat32 || v_prefix.dtype() != kFloat32) {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] invalid active_prefix for layer=%d in tree decode; "
                            "disabling SpecPV.",
                            layer);
                        specpv_ok = false;
                        return;
                    }

                    const int B = static_cast<int>(k_prefix.shape(0));
                    const int H = static_cast<int>(k_prefix.shape(1));
                    const int S = static_cast<int>(k_prefix.shape(2));
                    const int D = static_cast<int>(k_prefix.shape(3));

                    if (B < batch_size || H < kv_heads || D != kHeadDim || S <= 0) {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] active_prefix shape mismatch for layer=%d "
                            "(B=%d,H=%d,S=%d,D=%d, batch=%d, kv_heads=%d, head_dim=%d); disabling SpecPV.",
                            layer,
                            B,
                            H,
                            S,
                            D,
                            batch_size,
                            kv_heads,
                            kHeadDim);
                        specpv_ok = false;
                        return;
                    }

                    const float* k_src = k_prefix.data<float>();
                    const float* v_src = v_prefix.data<float>();

                    for (int slot = 0; slot < batch_size && specpv_ok; ++slot) {
                        const int prefix_len_i = std::min(h_prefix_len[slot], prefix_tokens);
                        const int prefix_blocks = h_prefix_blocks[slot];
                        const int base_index    = h_cu_block_nums[slot];

                        for (int b = 0; b < prefix_blocks; ++b) {
                            const int global_block = base_index + b;
                            char*     block_ptr    = kv_base
                                                  + static_cast<size_t>(global_block) * kv_block_bytes;

                            std::vector<uint8_t> host_block(static_cast<size_t>(kv_block_bytes), 0);

                            for (int head = 0; head < kv_heads; ++head) {
                                for (int t = 0; t < block_seq_len; ++t) {
                                    const int global_t = b * block_seq_len + t;
                                    if (global_t >= prefix_len_i || global_t >= S) {
                                        break;
                                    }

                                    const ssize_t src_base =
                                        (((static_cast<ssize_t>(slot) * H + head) * S + global_t)
                                         * D);

                                    const float* k_row = k_src + src_base;
                                    const float* v_row = v_src + src_base;

                                    const int k_off = layout.k_data(layer, head, t);
                                    const int v_off = layout.v_data(layer, head, t);

                                    auto* k_dst = reinterpret_cast<half*>(host_block.data() + k_off);
                                    auto* v_dst = reinterpret_cast<half*>(host_block.data() + v_off);

                                    for (int d = 0; d < D; ++d) {
                                        k_dst[d] = __float2half(k_row[d]);
                                        v_dst[d] = __float2half(v_row[d]);
                                    }
                                }
                            }

                            check_cuda_error(cudaMemcpyAsync(block_ptr,
                                                             host_block.data(),
                                                             kv_block_bytes,
                                                             cudaMemcpyHostToDevice,
                                                             stream_));
                        }
                    }
                }
            };

            auto fill_layer_bf16 = [&](auto head_dim_tag) {
#if ENABLE_BF16
                constexpr int kHeadDim = decltype(head_dim_tag)::value;

                block::Config<nv_bfloat16, nv_bfloat16, kHeadDim> cfg{kv_heads, block_seq_len};
                block::Layout<block::Config<nv_bfloat16, nv_bfloat16, kHeadDim>> layout{cfg};

                const int prefix_tokens = partial_prefix_tokens;

                for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
                    auto [k_prefix, v_prefix] = specpv_kv_cache_->active_prefix(layer, prefix_tokens);
                    if (!k_prefix || !v_prefix || k_prefix.dtype() != kFloat32 || v_prefix.dtype() != kFloat32) {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] invalid active_prefix for layer=%d in tree decode; "
                            "disabling SpecPV.",
                            layer);
                        specpv_ok = false;
                        return;
                    }

                    const int B = static_cast<int>(k_prefix.shape(0));
                    const int H = static_cast<int>(k_prefix.shape(1));
                    const int S = static_cast<int>(k_prefix.shape(2));
                    const int D = static_cast<int>(k_prefix.shape(3));

                    if (B < batch_size || H < kv_heads || D != kHeadDim || S <= 0) {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] active_prefix shape mismatch for layer=%d "
                            "(B=%d,H=%d,S=%d,D=%d, batch=%d, kv_heads=%d, head_dim=%d); disabling SpecPV.",
                            layer,
                            B,
                            H,
                            S,
                            D,
                            batch_size,
                            kv_heads,
                            kHeadDim);
                        specpv_ok = false;
                        return;
                    }

                    const float* k_src = k_prefix.data<float>();
                    const float* v_src = v_prefix.data<float>();

                    for (int slot = 0; slot < batch_size && specpv_ok; ++slot) {
                        const int prefix_len_i = std::min(h_prefix_len[slot], prefix_tokens);
                        const int prefix_blocks = h_prefix_blocks[slot];
                        const int base_index    = h_cu_block_nums[slot];

                        for (int b = 0; b < prefix_blocks; ++b) {
                            const int global_block = base_index + b;
                            char*     block_ptr    = kv_base
                                                  + static_cast<size_t>(global_block) * kv_block_bytes;

                            std::vector<uint8_t> host_block(static_cast<size_t>(kv_block_bytes), 0);

                            for (int head = 0; head < kv_heads; ++head) {
                                for (int t = 0; t < block_seq_len; ++t) {
                                    const int global_t = b * block_seq_len + t;
                                    if (global_t >= prefix_len_i || global_t >= S) {
                                        break;
                                    }

                                    const ssize_t src_base =
                                        (((static_cast<ssize_t>(slot) * H + head) * S + global_t)
                                         * D);

                                    const float* k_row = k_src + src_base;
                                    const float* v_row = v_src + src_base;

                                    const int k_off = layout.k_data(layer, head, t);
                                    const int v_off = layout.v_data(layer, head, t);

                                    auto* k_dst =
                                        reinterpret_cast<nv_bfloat16*>(host_block.data() + k_off);
                                    auto* v_dst =
                                        reinterpret_cast<nv_bfloat16*>(host_block.data() + v_off);

                                    for (int d = 0; d < D; ++d) {
                                        k_dst[d] = __float2bfloat16(k_row[d]);
                                        v_dst[d] = __float2bfloat16(v_row[d]);
                                    }
                                }
                            }

                            check_cuda_error(cudaMemcpyAsync(block_ptr,
                                                             host_block.data(),
                                                             kv_block_bytes,
                                                             cudaMemcpyHostToDevice,
                                                             stream_));
                        }
                    }
                }
#else
                (void)head_dim_tag;
                specpv_ok = false;
#endif
            };

            if (specpv_ok) {
                if (dtype_ == kFloat16) {
                    if (head_dim == 64) {
                        fill_layer_half(std::integral_constant<int, 64>{});
                    }
                    else if (head_dim == 128) {
                        fill_layer_half(std::integral_constant<int, 128>{});
                    }
                    else if (head_dim == 192) {
                        fill_layer_half(std::integral_constant<int, 192>{});
                    }
                    else {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] unsupported head_dim=%d for partial KV tree decode; "
                            "disabling SpecPV.",
                            head_dim);
                        specpv_ok = false;
                    }
                }
#if ENABLE_BF16
                else if (dtype_ == kBfloat16) {
                    if (head_dim == 64) {
                        fill_layer_bf16(std::integral_constant<int, 64>{});
                    }
                    else if (head_dim == 128) {
                        fill_layer_bf16(std::integral_constant<int, 128>{});
                    }
                    else if (head_dim == 192) {
                        fill_layer_bf16(std::integral_constant<int, 192>{});
                    }
                    else {
                        TM_LOG_WARNING(
                            "[LlamaV2][SpecPV][fallback] unsupported head_dim=%d for partial KV tree decode; "
                            "disabling SpecPV.",
                            head_dim);
                        specpv_ok = false;
                    }
                }
#endif
                else {
                    TM_LOG_WARNING(
                        "[LlamaV2][SpecPV][fallback] unsupported dtype/head_dim combination for partial KV "
                        "tree decode; disabling SpecPV.");
                    specpv_ok = false;
                }
            }

            if (!specpv_ok) {
                specpv_supported_             = false;
                specpv_retrieval_initialized_ = false;
                specpv_partial_steps_         = 0;
                specpv_kv_cache_.reset();
                use_specpv = false;
            }
        }
    }

    // Final safety check: ensure that the total KV coverage implied by the
    // block table is sufficient for the requested prefix_len + tree_len on
    // every active slot.
    for (int i = 0; i < batch_size; ++i) {
        const int prefix_len = h_prefix_len[i];
        const int tree_len   = h_tree_lens[i];
        const int blocks_i   = h_prefix_blocks[i] + h_extra_blocks[i];
        const int cover      = blocks_i * block_seq_len;
        if (prefix_len + tree_len > cover) {
            TM_LOG_WARNING(
                "[LlamaV2][EAGLE][fallback] target-tree KV coverage shortfall for slot=%d "
                "(prefix_len=%d, tree_len=%d, blocks=%d, block_seq_len=%d); disabling target-tree.",
                i,
                prefix_len,
                tree_len,
                blocks_i,
                block_seq_len);
            target_tree_supported_          = false;
            eagle_tree_target_tokens_valid_ = false;
            return;
        }
    }

    Buffer_<uintptr_t> d_block_ptrs(total_blocks, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(d_block_ptrs.data(),
                                     h_block_ptrs.data(),
                                     static_cast<size_t>(total_blocks) * sizeof(uintptr_t),
                                     cudaMemcpyHostToDevice,
                                     stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    Buffer_<float> rope_base_buf(batch_size, kDEVICE);
    std::vector<float> h_rope_base(batch_size, attn_param_.rope.base);
    check_cuda_error(cudaMemcpyAsync(rope_base_buf.data(),
                                     h_rope_base.data(),
                                     static_cast<size_t>(batch_size) * sizeof(float),
                                     cudaMemcpyHostToDevice,
                                     stream_));

    Buffer_<bool> finished_buf(batch_size, kDEVICE);
    check_cuda_error(
        cudaMemsetAsync(finished_buf.data(), 0, static_cast<size_t>(batch_size) * sizeof(bool), stream_));

    // Per-slot cumulative tree token offsets (exclusive) for offset-aware
    // helpers. Layout matches h_cum_tree_tokens computed earlier.
    Buffer_<SizeType> d_tree_offsets(static_cast<size_t>(batch_size) + 1, kDEVICE);
    check_cuda_error(cudaMemcpyAsync(
        d_tree_offsets.data(),
        h_cum_tree_tokens.data(),
        static_cast<size_t>(batch_size + 1) * sizeof(SizeType),
        cudaMemcpyHostToDevice,
        stream_));

    const int global_token_num = token_num;
    int       local_token_num  = global_token_num;

    Buffer local_token_nums_buf{&local_token_num, 1, kCPU};

    Tensor tree_hidden_view{
        tree_hidden_states_.buffer().slice(0, static_cast<ssize_t>(num_tree_tokens) * static_cast<ssize_t>(hidden_units_)),
        {static_cast<int>(num_tree_tokens), static_cast<int>(hidden_units_)},
    };

    Tensor tree_last_hidden{{batch_size, static_cast<int>(hidden_units_)}, dtype_, kDEVICE};

    Tensor partial_ML{
        {std::max(1, engine_param_.attn_cp_size), UnifiedAttentionLayer::kMaxWorkspaceTokens,
         static_cast<int>(local_head_num_), 2},
        kFloat32,
        kDEVICE};

    const int decode_num = 0;
    const int prefil_num = batch_size;

    Buffer h_q_len_buf{h_q_len.data(), batch_size, kCPU};
    Buffer h_k_len_buf{h_k_len.data(), batch_size, kCPU};
    Buffer h_k_len_runtime_buf{h_k_len_runtime.data(), batch_size, kCPU};

    TensorMap args{{"decoder_input", tree_input_embeds},
                   {"decoder_output", tree_hidden_view.borrow()},
                   {"last_token_hidden_units", tree_last_hidden},
                   {"output_norm_weight", weights_->output_norm_weight},
                   {"h_q_len", h_q_len_buf},
                   {"h_k_len", h_k_len_buf},
                   {"spec_runtime_k_len", h_k_len_runtime_buf},
                   {"finished", finished_buf},
                   {"decode_num", Buffer{const_cast<int*>(&decode_num), 1, kCPU}},
                   {"prefil_num", Buffer{const_cast<int*>(&prefil_num), 1, kCPU}},
                   {"rope_base", rope_base_buf},
                   {"partial_ML", partial_ML},
                   {"cu_block_nums", d_cu_block_nums},
                   {"kv_block_ptrs", d_block_ptrs},
                   {"local_token_nums", local_token_nums_buf},
                   {"spec_tree_offsets", d_tree_offsets},
                   {"spec_tree_runtime_offsets", d_cum_tree_runtime},
                   {"spec_successor_offsets", d_successor_offsets},
                   {"spec_successor_counts", d_successor_counts}};

    // Optional packed mask for tree decode. Prefer per-step path-based
    // masks when available; otherwise fall back to a prefix-style packed
    // mask derived from cumulative tree lengths to preserve autoregressive
    // semantics for the compact layout.
    if (eagle_module_ && eagle_buffers_) {
        const SizeType num_packed =
            static_cast<SizeType>(eagle_module_->getNumPackedMasks());

        auto gather_and_attach = [&](const int32_t* slot_packed_ptr) {
            Tensor tree_packed_mask{
                {static_cast<int>(num_tree_tokens), static_cast<int>(num_packed)},
                kInt32,
                kDEVICE};

            invokeGatherTreePackedMask(
                reinterpret_cast<SizeType const*>(slot_packed_ptr),
                static_cast<SizeType>(batch_size),
                max_decoding_tokens,
                num_packed,
                d_tree_hidden_indices.data(),
                num_tree_tokens,
                tree_packed_mask.data<SizeType>(),
                stream_);
            args.insert({"spec_packed_mask", tree_packed_mask});
        };

        if (eagle_buffers_->inputs.packed_masks) {
            gather_and_attach(reinterpret_cast<int32_t const*>(eagle_buffers_->inputs.packed_masks));

            if (isEagleDebugEnabled() && tp_rank_ == 0) {
                const SizeType module_packed =
                    static_cast<SizeType>(eagle_module_->getNumPackedMasks());
                if (module_packed != num_packed) {
                    TM_LOG_WARNING(
                        "[LlamaV2][EAGLE][tree-mask] num_packed mismatch: module=%d, gathered=%d",
                        static_cast<int>(module_packed),
                        static_cast<int>(num_packed));
                }
                else {
                    TM_LOG_INFO(
                        "[LlamaV2][EAGLE][tree-mask] step tree decode: num_tree_tokens=%d, num_packed=%d",
                        static_cast<int>(num_tree_tokens),
                        static_cast<int>(num_packed));
                }
            }
        }
        else if (d_cum_tree_runtime.size() > 0 && num_packed > 0) {
            Tensor slot_packed_mask{
                {batch_size, max_decoding_tokens, num_packed},
                kInt32,
                kDEVICE};

            ::lmdeploy::turbomind::kernels::speculative_decoding::invokeGetPackedMask(
                slot_packed_mask.data<int32_t>(),
                reinterpret_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32 const*>(
                    d_cum_tree_runtime.data()),
                static_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32>(batch_size),
                static_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32>(
                    max_decoding_tokens),
                stream_);

            gather_and_attach(slot_packed_mask.data<int32_t>());
        }
    }

    unified_decoder_->Forward(args, weights_->decoder_layer_weights);

    // Stage 4: project the tree hidden states through the LM head into
    // the dedicated FP32 tree logits buffer and reduce them to per-node
    // target_ids on device.
    (void)postDecodeEmbedding(tree_hidden_view, tree_logits_buffer_);

    const float* logits_ptr = tree_logits_buffer_.data<float>();

    TokenIdType* target_tokens =
        reinterpret_cast<TokenIdType*>(eagle_buffers_->inputs.target_tokens);

    if (!target_tokens) {
        return;
    }

    TreeLogitsToTargetsParams reduce_params{};
    reduce_params.logits              = logits_ptr;
    reduce_params.num_tree_tokens     = num_tree_tokens;
    reduce_params.vocab_size          = static_cast<SizeType>(vocab_size_padded_);
    reduce_params.hidden_indices      = d_tree_hidden_indices.data();
    reduce_params.max_batch_size      = static_cast<SizeType>(engine_param_.max_batch_size);
    reduce_params.max_decoding_tokens = static_cast<SizeType>(eagle_module_->getMaxDecodingTokens());
    reduce_params.target_tokens       = target_tokens;
    reduce_params.stream              = stream_;

    invokeTreeLogitsToTargetIds(reduce_params);

    sync_check_cuda_error();
    eagle_tree_target_tokens_valid_ = true;
}

void LlamaV2::targetTreeDecode(int batch_size, const int* d_sequence_lengths)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Guard: only meaningful when EAGLE is enabled and buffers exist.
    if (!spec_mode_.isEagle() || !eagle_module_ || !eagle_buffers_
        || !eagle_buffers_->isAllocated()
        || batch_size <= 0
        || batch_size > engine_param_.max_batch_size) {
        return;
    }

    using namespace kernels::speculative_decoding;

    const SizeType max_decoding_tokens =
        static_cast<SizeType>(engine_param_.spec_max_decoding_tokens);
    const SizeType max_path_len =
        static_cast<SizeType>(engine_param_.spec_max_draft_path_len);

    // Use the existing EagleBuffers inputs as the target-tree decode
    // staging area so that downstream kernels (and future decode
    // passes) can consume a consistent layout.
    PrepareGenTargetTreeParams params{};
    params.draft_paths             = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.draft_paths);
    params.batch_slots             = nullptr;  // identity mapping [0..batch_size)
    params.draft_tokens            = reinterpret_cast<TokenIdType const*>(eagle_buffers_->inputs.draft_tokens);
    params.base_sequence_lengths   = d_sequence_lengths
        ? reinterpret_cast<SizeType const*>(d_sequence_lengths)
        : nullptr;
    params.base_context_lengths    = nullptr;

    params.output_ids              = reinterpret_cast<TokenIdType*>(eagle_buffers_->inputs.eagle_net_input_ids);
    params.position_ids            = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_position_ids);
    params.hidden_indices          = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_hidden_indices);
    params.spec_gen_lengths        = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_gen_lens);
    params.next_sequence_lengths   = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_seq_lens);
    params.next_context_lengths    = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_ctx_lens);

    params.batch_size        = static_cast<SizeType>(batch_size);
    params.max_batch_size    = static_cast<SizeType>(engine_param_.max_batch_size);
    params.max_decoding_tokens = max_decoding_tokens;
    params.max_path_len      = max_path_len;
    params.stream            = stream_;

    invokePrepareGenTargetTreeInputs(params);
}

void LlamaV2::updateEmbedding(char*            decoder_input,
                              const int        bsz,
                              const int*       h_input_length,
                              const Sequence** sequences,
                              int              token_num,
                              int*             lora_mask,
                              bool*            have_embeddings)
{
    if (isTuning())
        return;

    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    *have_embeddings          = false;
    int*             mask_ptr = nullptr;
    std::vector<int> mask;
    if (lora_mask != nullptr) {
        mask     = std::vector<int>(token_num);
        mask_ptr = mask.data();
    }

    const size_t elem_size = byte_size(dtype_, 1);

    for (int i = 0; i < bsz; i++) {
        const auto& seq        = *sequences[i];
        const auto& embeddings = seq.input_embeddings;
        const auto& ranges     = seq.input_embedding_ranges;
        for (int j = embeddings.size() - 1; j >= 0; j--) {
            int begin = ranges[j].first;
            int end   = ranges[j].second;
            if (seq.cache_len + h_input_length[i] - 1 < begin) {
                continue;
            }
            if (end <= seq.cache_len) {
                break;
            }
            int off_dst = std::max(0, begin - seq.cache_len);
            int off_src = std::max(0, seq.cache_len - begin);
            // calculate intersection of [begin, end) and [seq.cache_len, seq.cache_len + h_input_length[i])
            begin            = std::max(begin, seq.cache_len);
            end              = std::min(end, seq.cache_len + h_input_length[i]);
            size_t byte_size = elem_size * (end - begin) * hidden_units_;
            char*  dst_ptr   = decoder_input + elem_size * off_dst * hidden_units_;
            auto   src_ptr   = embeddings[j].data() + elem_size * off_src * hidden_units_;
            check_cuda_error(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, cudaMemcpyDefault, stream_));
            if (lora_mask != nullptr) {
                std::fill_n(mask_ptr + off_dst, (end - begin), 1);
                *have_embeddings = true;
            }
        }
        decoder_input += elem_size * h_input_length[i] * hidden_units_;
        mask_ptr += h_input_length[i];
    }

    if (lora_mask != nullptr && *have_embeddings) {
        cudaMemcpyAsync(lora_mask, mask.data(), sizeof(int) * token_num, cudaMemcpyDefault, stream_);
        cudaStreamSynchronize(stream_);
    }
    sync_check_cuda_error();
}

void LlamaV2::prepareEagleContextInputs(int batch_size)
{
    if (!eagle_buffers_ || !eagle_buffers_->isAllocated() || !eagle_module_) {
        return;
    }

    using SizeType    = kernels::eagle::SizeType;
    using TokenIdType = kernels::eagle::TokenIdType;

    // Context prep consumes the previous step's acceptance outputs to seed
    // EagleNet0. We assume the caller has populated prev_* buffers.
    kernels::eagle::PrepareCtxEagleNetParams params{};
    params.eagleNetSequenceLengths = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_seq_lens);
    params.eagleNetContextLengths  = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_ctx_lens);
    params.outputIds               = reinterpret_cast<TokenIdType*>(eagle_buffers_->inputs.eagle_net_input_ids);
    params.positionIds             = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_position_ids);
    params.hiddenStatesIndices     = reinterpret_cast<SizeType*>(eagle_buffers_->inputs.eagle_net_hidden_indices);
    params.lastTokenIndices        = nullptr;  // unused in TurboMind path
    params.numLastTokenIndices     = nullptr;

    params.inputIds              = nullptr;  // TurboMind uses live decoder input elsewhere
    params.baseNetSequenceLengths = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.eagle_net_seq_lens);
    params.baseNetContextLengths  = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.eagle_net_ctx_lens);
    params.acceptedTokens         = reinterpret_cast<TokenIdType const*>(eagle_buffers_->inputs.prev_accepted_tokens);
    params.acceptedLens           = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.prev_accepted_lens);
    params.prevDraftLens          = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.prev_draft_lens);
    params.prevPaths              = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.prev_paths);
    params.bestPathIds            = reinterpret_cast<SizeType const*>(eagle_buffers_->inputs.best_path_ids);

    params.batchSize        = static_cast<SizeType>(batch_size);
    params.maxPathLen       = static_cast<SizeType>(engine_param_.spec_max_draft_path_len);
    params.maxDecodingTokens = static_cast<SizeType>(engine_param_.spec_max_decoding_tokens);
    params.stream           = stream_;

    kernels::eagle::invokePrepareCtxEagleNetInputs(params);
}

void LlamaV2::Forward(Buffer_<int>     input_ids,
                      Tensor           hidden_states_out,
                      Tensor           decoder_out,
                      Buffer           kv_block_ptrs,
                      Buffer           cu_block_nums,
                      Buffer_<int>     h_input_length,
                      Buffer_<int>     h_context_length,
                      Buffer           rope_base,
                      MropeRope*       mrope,
                      Tensor           partial_ML,
                      Buffer           finished,
                      Buffer           local_token_nums,
                      Buffer           lora_mask,
                      int              decode_num,
                      int              prefil_num,
                      const Sequence** sequences)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    Tensor input_embeds;

    const int token_num = input_ids.size();

    if (token_num) {
        const auto& embedding_table = weights_->pre_decoder_embedding.weight;
        TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units_);

        input_embeds = Tensor{{token_num, (int)hidden_units_}, dtype_, kDEVICE};

        if (tp_size_ == 1) {
            invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, stream_);
            sync_check_cuda_error();
        }
        else if (use_allgather_2d_) {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {token_num, tp_size_, local_hidden_units}};

            auto local = temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather2D(local.raw_data(),
                                       temp.raw_data(),
                                       hidden_units_,
                                       local_hidden_units,
                                       local_hidden_units,
                                       token_num,
                                       local.dtype(),
                                       {true, true},
                                       comm_->d_tp_group,
                                       stream_);
            sync_check_cuda_error();

            Copy(temp.buffer(), input_embeds.buffer());
        }
        else {
            const auto local_hidden_units = embedding_table.shape(1);
            Tensor     temp{hidden_states_out.buffer(), {tp_size_, token_num, local_hidden_units}};

            auto local = temp.slice(tp_rank_).squeeze(0);

            invokeEmbeddingLookup(local, input_ids, embedding_table, stream_);
            sync_check_cuda_error();

            comm_->d_comm->AllGather(
                local.raw_data(), temp.raw_data(), local.size(), dtype_, comm_->d_tp_group, stream_);
            sync_check_cuda_error();

            invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                      (uint16_t*)temp.raw_data(),
                                      tp_size_,
                                      token_num,
                                      local_hidden_units,
                                      false,
                                      stream_);
            sync_check_cuda_error();
        }
    }

    bool have_embeddings = false;
    if (token_num) {
        // Copy input embeddings from corresponding sequences
        updateEmbedding((char*)input_embeds.raw_data(),
                        h_input_length.size(),
                        h_input_length.data(),
                        sequences,
                        token_num,
                        lora_mask ? lora_mask.data<int>() : nullptr,
                        &have_embeddings);
        sync_check_cuda_error();
    }

    TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

    TensorMap args{{"decoder_input", input_embeds},
                   {"decoder_output", hidden_states_out.view({-1, (int)hidden_units_}).borrow()},
                   {"last_token_hidden_units", decoder_out},
                   {"output_norm_weight", weights_->output_norm_weight},
                   {"h_q_len", h_input_length},
                   {"h_k_len", h_context_length},
                   {"finished", finished},
                   {"decode_num", Buffer{&decode_num, 1, kCPU}},
                   {"prefil_num", Buffer{&prefil_num, 1, kCPU}},
                   {"rope_base", rope_base},
                   {"partial_ML", partial_ML},
                   {"cu_block_nums", cu_block_nums},
                   {"kv_block_ptrs", kv_block_ptrs},
                   {"local_token_nums", local_token_nums}};

    // When running Eagle3 speculative decoding with TurboMind we ask
    // UnifiedDecoder to capture last-token hidden states from a small
    // set of decoder layers (typically the last 3). The captured
    // hidden states are concatenated into a single
    // [batch, hidden * num_capture_layers] tensor which is written
    // back into this entry in the args map.
    if (engine_param_.enable_speculative_decoding && engine_param_.spec_method == "eagle3") {
        eagle_capture_hidden_ = Tensor{};
        args.insert({"eagle_capture_hidden", eagle_capture_hidden_});
    }

    if (mrope != nullptr && mrope->position_ids) {
        args.insert({"mrope_position_ids", mrope->position_ids});
        args.insert({"mrope_position_delta", mrope->position_delta});
        args.insert({"mrope_position_length", mrope->length});
    }

    unified_decoder_->Forward(args, weights_->decoder_layer_weights);

    // If Eagle3 capture was requested, pull the populated tensor back
    // from the args map so that eagleDraftForward can feed it into
    // EagleModule. When capture is enabled, the UnifiedDecoder will
    // ensure the buffer has shape [batch, hidden * num_capture_layers].
    if (engine_param_.enable_speculative_decoding && engine_param_.spec_method == "eagle3") {
        auto it = args.find("eagle_capture_hidden");
        if (it != args.end()) {
            eagle_capture_hidden_ = it->second;
        }
        else {
            eagle_capture_hidden_ = Tensor{};
        }

        if (eagle_capture_hidden_ && eagle_module_) {
            const int expected_width = eagle_module_->getEagleFcInDim();
            if (expected_width > 0) {
                const int bs  = decoder_out.shape(0);
                const int got = eagle_capture_hidden_.shape(1);
                if (eagle_capture_hidden_.shape(0) != bs || got != expected_width) {
                    TM_LOG_WARNING(
                        "[LlamaV2][EAGLE][fallback] Eagle3 capture buffer has shape [%d, %d], "
                        "expected [%d, %d]; Eagle3 FC path disabled for this engine.",
                        eagle_capture_hidden_.shape(0),
                        got,
                        bs,
                        expected_width);
                    // Disable Eagle3 FC path by clearing the capture tensor;
                    // draft will fall back to the legacy shallow path.
                    eagle_capture_hidden_ = Tensor{};
                }
            }
        }
    }
}

bool LlamaV2::flattenPrefixKVForLayer(int              layer_idx,
                                      int              verified_seq_len,
                                      const Sequence** sequences,
                                      int              batch_size,
                                      const int*       h_seq_len,
                                      Tensor&          out_k,
                                      Tensor&          out_v)
{
    if (!sequence_manager_ || !sequences || batch_size <= 0 || verified_seq_len <= 0) {
        return false;
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_num_)) {
        return false;
    }

    // For now we flatten only when the base KV cache uses a fp16/bf16
    // layout; quantized caches are not supported in SpecPV mode.
    if (dtype_ != kFloat16
#if ENABLE_BF16
        && dtype_ != kBfloat16
#endif
    ) {
        TM_LOG_WARNING(
            "[LlamaV2][SpecPV][fallback] flattenPrefixKVForLayer only supports fp16/bf16 KV "
            "(dtype=%s); disabling SpecPV for this engine.",
            to_string(dtype_));
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return false;
    }

    const int block_seq_len = attn_param_.cache_block_seq_len;
    if (block_seq_len <= 0) {
        return false;
    }

    // Host-side per-slot prefix lengths and block counts.
    std::vector<int> h_prefix_len(batch_size, 0);
    std::vector<int> h_cu_k_len(batch_size + 1, 0);
    std::vector<int> h_block_counts(batch_size, 0);
    std::vector<int> h_cu_block_counts(batch_size + 1, 0);

    for (int i = 0; i < batch_size; ++i) {
        const Sequence* seq = sequences[i];
        const int       cache_len =
            seq ? std::max(0, seq->cache_len) : 0;
        int len_i = verified_seq_len;
        if (h_seq_len) {
            len_i = std::min(len_i, std::max(0, h_seq_len[i]));
        }
        len_i = std::min(len_i, cache_len);
        if (len_i < 0) {
            len_i = 0;
        }
        h_prefix_len[i]    = len_i;
        h_cu_k_len[i + 1]  = h_cu_k_len[i] + len_i;
        const int blocks_i = seq ? static_cast<int>(seq->blocks.size()) : 0;
        h_block_counts[i]  = blocks_i;
        h_cu_block_counts[i + 1] = h_cu_block_counts[i] + blocks_i;
    }

    const int total_tokens = h_cu_k_len[batch_size];
    const int total_blocks = h_cu_block_counts[batch_size];
    if (total_tokens <= 0 || total_blocks <= 0) {
        return false;
    }

    // Build block pointer table for the active sequences, storing device
    // pointers as 64-bit integers so we can reinterpret them as `char**`
    // when launching the flatten kernel.
    std::vector<uint64_t> h_block_ptrs(static_cast<size_t>(total_blocks), 0);
    int                    cursor = 0;
    for (int i = 0; i < batch_size; ++i) {
        const Sequence* seq = sequences[i];
        if (!seq) {
            continue;
        }
        for (int block_id : seq->blocks) {
            if (cursor >= total_blocks) {
                break;
            }
            void* ptr = sequence_manager_->GetBlockPtr(block_id);
            h_block_ptrs[cursor++] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
        }
    }

    if (cursor != total_blocks) {
        TM_LOG_WARNING(
            "[LlamaV2][SpecPV][fallback] flattenPrefixKVForLayer saw inconsistent block counts "
            "(cursor=%d, total_blocks=%d); disabling SpecPV.",
            cursor,
            total_blocks);
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return false;
    }

    Buffer_<int>      d_cu_k_len(batch_size + 1, kDEVICE);
    Buffer_<int>      d_cu_block_num(batch_size + 1, kDEVICE);
    Buffer_<uint64_t> d_block_ptrs(total_blocks, kDEVICE);

    check_cuda_error(cudaMemcpyAsync(
        d_cu_k_len.data(), h_cu_k_len.data(), sizeof(int) * (batch_size + 1), cudaMemcpyHostToDevice, stream_));
    check_cuda_error(cudaMemcpyAsync(d_cu_block_num.data(),
                                     h_cu_block_counts.data(),
                                     sizeof(int) * (batch_size + 1),
                                     cudaMemcpyHostToDevice,
                                     stream_));
    check_cuda_error(cudaMemcpyAsync(d_block_ptrs.data(),
                                     h_block_ptrs.data(),
                                     static_cast<size_t>(total_blocks) * sizeof(uint64_t),
                                     cudaMemcpyHostToDevice,
                                     stream_));

    // Flatten per-layer KV for this prefix into a temporary fp16/bf16
    // buffer and then convert to float32 for SpecPV.
    const int head_num    = static_cast<int>(local_kv_head_num_);
    const int head_dim    = static_cast<int>(size_per_head_);
    const int max_seq_len = verified_seq_len;

    Tensor flat_k_half{{batch_size, head_num, max_seq_len, head_dim}, dtype_, kDEVICE};
    Tensor flat_v_half{{batch_size, head_num, max_seq_len, head_dim}, dtype_, kDEVICE};

    if (!flat_k_half || !flat_v_half) {
        TM_LOG_WARNING("[LlamaV2][SpecPV][fallback] flattenPrefixKVForLayer allocation failure "
                       "(batch=%d, heads=%d, len=%d, dim=%d)",
                       batch_size,
                       head_num,
                       max_seq_len,
                       head_dim);
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return false;
    }

    const int64_t stride_b = static_cast<int64_t>(head_num) * max_seq_len;
    const int64_t stride_c = 0;               // ignore global token offset
    const int64_t stride_h = max_seq_len;
    const int64_t stride_s = 1;

    RopeKernelParam rope_param{};

    const int cp_rank = engine_param_.attn_cp_rank;
    cutlass::FastDivmod cp_size(engine_param_.attn_cp_size > 0 ? engine_param_.attn_cp_size : 1);

    auto* k_ptr = flat_k_half.raw_data();
    auto* v_ptr = flat_v_half.raw_data();

    if (dtype_ == kFloat16) {
        invokeFlattenKV_v2(reinterpret_cast<half*>(k_ptr),
                           reinterpret_cast<half*>(v_ptr),
                           reinterpret_cast<char**>(d_block_ptrs.data()),
                           d_cu_k_len.data(),
                           d_cu_block_num.data(),
                           rope_param,
                           stride_b,
                           stride_c,
                           stride_h,
                           stride_s,
                           block_seq_len,
                           layer_idx,
                           cp_rank,
                           cp_size,
                           max_seq_len,
                           head_num,
                           head_dim,
                           batch_size,
                           param_.quant_policy,
                           stream_);
    }
#if ENABLE_BF16
    else if (dtype_ == kBfloat16) {
        invokeFlattenKV_v2(reinterpret_cast<nv_bfloat16*>(k_ptr),
                           reinterpret_cast<nv_bfloat16*>(v_ptr),
                           reinterpret_cast<char**>(d_block_ptrs.data()),
                           d_cu_k_len.data(),
                           d_cu_block_num.data(),
                           rope_param,
                           stride_b,
                           stride_c,
                           stride_h,
                           stride_s,
                           block_seq_len,
                           layer_idx,
                           cp_rank,
                           cp_size,
                           max_seq_len,
                           head_num,
                           head_dim,
                           batch_size,
                           param_.quant_policy,
                           stream_);
    }
#endif
    else {
        return false;
    }

    sync_check_cuda_error();

    // Convert flattened KV to float32 for SpecPV summarization.
    out_k = Tensor{{batch_size, head_num, max_seq_len, head_dim}, kFloat32, kDEVICE};
    out_v = Tensor{{batch_size, head_num, max_seq_len, head_dim}, kFloat32, kDEVICE};
    if (!out_k || !out_v) {
        TM_LOG_WARNING("[LlamaV2][SpecPV][fallback] flattenPrefixKVForLayer float32 allocation failure "
                       "(batch=%d, heads=%d, len=%d, dim=%d)",
                       batch_size,
                       head_num,
                       max_seq_len,
                       head_dim);
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return false;
    }

    core::Copy(flat_k_half, out_k);
    core::Copy(flat_v_half, out_v);

    return true;
}

void LlamaV2::initSpecPVFromFullKV(int              verified_seq_len,
                                   const Sequence** sequences,
                                   int              batch_size,
                                   const int*       h_seq_len)
{
    if (!specpv_kv_cache_ || !specpv_kv_cache_->is_enabled()) {
        return;
    }

    if (!sequence_manager_ || !sequences || batch_size <= 0) {
        TM_LOG_WARNING(
            "[LlamaV2][SpecPV][fallback] initSpecPVFromFullKV missing SequenceManager or sequences; "
            "disabling SpecPV for this engine.");
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return;
    }

    const int clamped_len =
        std::max(0, std::min(verified_seq_len, specpv_cache_config_.total_budget()));

    if (clamped_len <= 0) {
        specpv_kv_cache_->reset_buffer();
        for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
            specpv_kv_cache_->set_verified_length(layer, 0);
        }
        specpv_retrieval_initialized_ = false;
        specpv_full_prefix_len_       = 0;
        TM_LOG_INFO("[LlamaV2][SpecPV] initialized partial KV with empty prefix (verified_len=0)");
        return;
    }

    specpv_kv_cache_->reset_buffer();
    specpv_retrieval_initialized_ = false;

    for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
        Tensor full_k;
        Tensor full_v;

        if (!flattenPrefixKVForLayer(layer, clamped_len, sequences, batch_size, h_seq_len, full_k, full_v)) {
            TM_LOG_WARNING(
                "[LlamaV2][SpecPV][fallback] flattenPrefixKVForLayer failed for layer=%d; disabling SpecPV.",
                layer);
            specpv_supported_             = false;
            specpv_retrieval_initialized_ = false;
            specpv_partial_steps_         = 0;
            specpv_kv_cache_.reset();
            return;
        }

        specpv_kv_cache_->summary_key_states(layer, full_k, clamped_len);

        const int q_window = std::min(clamped_len, 4);
        if (q_window > 0) {
            std::vector<ssize_t> q_idx{0, 0, clamped_len - q_window, 0};
            std::vector<ssize_t> q_shape{
                full_k.shape(0), full_k.shape(1), q_window, full_k.shape(3)};
            Tensor query_states = full_k.slice(q_idx, q_shape);

            specpv_kv_cache_->refresh_retrieval(layer, query_states, full_k, full_v, clamped_len);
        }

        // Start with an empty speculative buffer; all verified prefix
        // tokens are represented via sink/retrieval/window slices.
        specpv_kv_cache_->set_verified_length(layer, 0);
    }

    specpv_retrieval_initialized_ = true;
    specpv_full_prefix_len_       = clamped_len;
    TM_LOG_INFO("[LlamaV2][SpecPV] partial KV seeded from full KV (verified_len=%d)", clamped_len);
}

void LlamaV2::updateSpecPVAfterAcceptance(const Buffer&        sequence_length,
                                          int                  batch_size,
                                          const Sequence**     sequences,
                                          const std::vector<int>* committed_lengths)
{
    if (!isSpecPVEnabled() || !specpv_kv_cache_ || batch_size <= 0 || !sequences) {
        return;
    }

    if (sequence_length.device() != kDEVICE || sequence_length.dtype() != kInt32) {
        TM_LOG_WARNING(
            "[LlamaV2][SpecPV][fallback] sequence_length buffer has unexpected device/dtype; "
            "disabling SpecPV for this engine.");
        specpv_supported_             = false;
        specpv_retrieval_initialized_ = false;
        specpv_partial_steps_         = 0;
        specpv_kv_cache_.reset();
        return;
    }

    std::vector<int> h_seq_len(batch_size, 0);

    const int* d_seq_len = sequence_length.data<int>();
    if (!d_seq_len) {
        return;
    }

    check_cuda_error(cudaMemcpyAsync(
        h_seq_len.data(), d_seq_len, static_cast<size_t>(batch_size) * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    check_cuda_error(cudaStreamSynchronize(stream_));

    int max_len = 0;
    for (int i = 0; i < batch_size; ++i) {
        if (h_seq_len[i] > max_len) {
            max_len = h_seq_len[i];
        }
    }

    if (max_len <= 0 || !shouldUseSpecPV(max_len)) {
        return;
    }

    // Lightweight partial-KV safety hook: validate per-slot committed
    // lengths against the configured speculative buffer budget. The
    // main SpecPV logic below still drives seeding and incremental
    // updates via initSpecPVFromFullKV / PartialKVCache::update; this
    // helper keeps PartialKVCache fail-safe in isolation.
    if (specpv_kv_cache_ && specpv_kv_cache_->is_enabled()) {
        for (int i = 0; i < batch_size; ++i) {
            int advance = 0;
            if (committed_lengths && i < static_cast<int>(committed_lengths->size())) {
                advance = std::max(0, (*committed_lengths)[i]);
            }
            else if (h_seq_len[i] > 0) {
                // Fallback: assume at least one token was committed
                // for active slots when committed_lengths are not
                // available (e.g. non-EAGLE or single-token paths).
                advance = 1;
            }

            if (advance > 0) {
                for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
                    specpv_kv_cache_->update_after_acceptance(layer, i, advance);
                }
            }
        }

        if (!specpv_kv_cache_->is_enabled()) {
            TM_LOG_WARNING(
                "[SpecPV][fallback] PartialKVCache update_after_acceptance failed; disabling SpecPV for this engine.");
            specpv_supported_             = false;
            specpv_retrieval_initialized_ = false;
            specpv_partial_steps_         = 0;
            specpv_kv_cache_.reset();
            return;
        }
    }

    // When SpecPV has not yet been seeded (or after a full-refresh),
    // build the partial KV view from the live full-KV prefix.
    if (!specpv_retrieval_initialized_) {
        initSpecPVFromFullKV(max_len, sequences, batch_size, h_seq_len.data());

        // initSpecPVFromFullKV may disable SpecPV on failure; in that
        // case we stop here and leave the engine on the full-KV path.
        if (!specpv_retrieval_initialized_ || !specpv_supported_ || !specpv_kv_cache_) {
            return;
        }
    }
    else {
        // Incremental partial KV update: append newly committed tail
        // tokens beyond the last full-prefix length into the SpecPV
        // buffer via PartialKVCache::update(...). This keeps sink /
        // retrieval / window as seeded, while the buffer tracks recent
        // verified tokens between full-refreshes.

        const int prev_full_len = std::max(0, specpv_full_prefix_len_);
        const int tail_len      = std::max(0, max_len - prev_full_len);

        if (tail_len > 0) {
            const int tail_start = max_len - tail_len;

            for (int layer = 0; layer < static_cast<int>(layer_num_); ++layer) {
                Tensor full_k;
                Tensor full_v;

                // Flatten the current full-prefix KV for this layer. On
                // any failure this helper will log and disable SpecPV.
                if (!flattenPrefixKVForLayer(layer,
                                             max_len,
                                             sequences,
                                             batch_size,
                                             h_seq_len.data(),
                                             full_k,
                                             full_v)) {
                    TM_LOG_WARNING(
                        "[LlamaV2][SpecPV][fallback] incremental flattenPrefixKVForLayer failed for "
                        "layer=%d; disabling SpecPV.",
                        layer);
                    specpv_supported_             = false;
                    specpv_retrieval_initialized_ = false;
                    specpv_partial_steps_         = 0;
                    specpv_kv_cache_.reset();
                    return;
                }

                if (!full_k || !full_v || full_k.shape(2) <= tail_start) {
                    continue;
                }

                const int D = static_cast<int>(full_k.shape(3));
                std::vector<ssize_t> tail_idx{0, 0, tail_start, 0};
                std::vector<ssize_t> tail_shape{
                    full_k.shape(0), full_k.shape(1), tail_len, D};

                // Clamp the tail view to the available tokens in case
                // some sequences are shorter than max_len.
                const int available_tokens = static_cast<int>(full_k.shape(2)) - tail_start;
                if (available_tokens <= 0) {
                    continue;
                }
                tail_shape[2] = std::min(tail_len, available_tokens);

                Tensor new_k = full_k.slice(tail_idx, tail_shape);
                Tensor new_v = full_v.slice(tail_idx, tail_shape);

                specpv_kv_cache_->update(layer, new_k, new_v);
            }

            specpv_full_prefix_len_ = max_len;
        }
    }

    if (engine_param_.specpv_full_refresh_steps > 0) {
        ++specpv_partial_steps_;

        const int buffer_tokens = specpv_cache_config_.buffer_size();
        const int current_len   = specpv_kv_cache_->global_verified_len();
        const int step_budget   = std::max(1, eagleMaxEngineTokensPerStep());
        const bool buffer_close_to_full =
            buffer_tokens > 0 && (current_len + step_budget > buffer_tokens);

        if (specpv_partial_steps_ > engine_param_.specpv_full_refresh_steps || buffer_close_to_full) {
            TM_LOG_INFO(
                "[LlamaV2][SpecPV] full-refresh trigger: partial_steps=%d, refresh_steps=%d, "
                "buffer_tokens=%d, current_len=%d, step_budget=%d",
                specpv_partial_steps_,
                engine_param_.specpv_full_refresh_steps,
                buffer_tokens,
                current_len,
                step_budget);

            specpv_kv_cache_->reset();
            specpv_retrieval_initialized_ = false;
            specpv_partial_steps_         = 0;
            specpv_full_prefix_len_       = 0;
        }
    }
}

Tensor LlamaV2::postDecodeEmbedding(const Tensor& features, Buffer local_logits)
{
    NvtxScope scope("postDecodeEmbedding");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    TM_CHECK(vocab_size_padded_ % tp_size_ == 0) << vocab_size_padded_ << " " << tp_size_;

    const int bsz              = features.shape(0);
    const int local_vocab_size = vocab_size_padded_ / tp_size_;

    if (tp_size_ == 1) {
        Tensor logits{local_logits, {bsz, (int)vocab_size_padded_}};
        linear_.Forward(features, weights_->post_decoder_embedding, logits);
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_allgather_2d_) {
        Tensor logits{local_logits, {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(1));
        sync_check_cuda_error();
        comm_->d_comm->AllGather2D(local.raw_data(),
                                   logits.raw_data(),
                                   vocab_size_padded_,
                                   local_vocab_size,
                                   local_vocab_size,
                                   bsz,
                                   logits.dtype(),
                                   {true, true},
                                   comm_->d_tp_group,
                                   stream_);
        sync_check_cuda_error();
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{local_logits, {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        linear_.Forward(features, weights_->post_decoder_embedding, local.squeeze(0));
        sync_check_cuda_error();
        comm_->d_comm->AllGather(
            local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_->d_tp_group, stream_);
        sync_check_cuda_error();
        Tensor out{{bsz, (int)vocab_size_padded_}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, stream_);
        sync_check_cuda_error();
        return out;
    }
}

void LlamaV2::dynamicDecode(Buffer token_ids,
                            Buffer finished,
                            Buffer sequence_length,
                            Tensor curand_state,
                            Tensor logits,
                            Buffer seq_limit_len,
                            Buffer init_context_length,
                            Buffer context_length,
                            Buffer prompt_length,
                            Buffer sampled_logprobs,
                            Buffer sampled_indexes,
                            Buffer sampled_nums,
                            int    step,
                            int    max_context_len)
{
    NvtxScope scope("dynamicDecode");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TensorMap args{
        {"logits", logits},
        {"step", Buffer{&step, 1, kCPU}},
        {"max_input_length", Buffer{&max_context_len, 1, kCPU}},
        {"sequence_limit_length", seq_limit_len},
        {"init_context_length", init_context_length},
        {"context_length", context_length},
        {"prompt_length", prompt_length},
        {"output_ids", token_ids},             // inout
        {"finished", finished},                // inout
        {"sequence_length", sequence_length},  // inout
        {"curand_state", curand_state},        // inout
    };

    if (sampled_logprobs) {
        args.emplace("sampled_logprobs", sampled_logprobs);
        args.emplace("sampled_indexes", sampled_indexes);
        args.emplace("sampled_nums", sampled_nums);
    }

    dynamic_decode_->Forward(args);
}

void LlamaV2::dynamicDecodeMultiStep(Buffer             token_ids,
                                     Buffer             finished,
                                     Buffer             sequence_length,
                                     Tensor             curand_state,
                                     Tensor             logits,
                                     Buffer             seq_limit_len,
                                     Buffer             init_context_length,
                                     Buffer             context_length,
                                     Buffer             prompt_length,
                                     Buffer             sampled_logprobs,
                                     Buffer             sampled_indexes,
                                     Buffer             sampled_nums,
                                     int                step,
                                     int                max_context_len,
                                     const ForcedTailContext* forced_ctx)
{
    NvtxScope scope("dynamicDecodeMultiStep");
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    TensorMap args{
        {"logits", logits},
        {"step", Buffer{&step, 1, kCPU}},
        {"max_input_length", Buffer{&max_context_len, 1, kCPU}},
        {"sequence_limit_length", seq_limit_len},
        {"init_context_length", init_context_length},
        {"context_length", context_length},
        {"prompt_length", prompt_length},
        {"output_ids", token_ids},             // inout
        {"finished", finished},                // inout
        {"sequence_length", sequence_length},  // inout
        {"curand_state", curand_state},        // inout
    };

    if (sampled_logprobs) {
        args.emplace("sampled_logprobs", sampled_logprobs);
        args.emplace("sampled_indexes", sampled_indexes);
        args.emplace("sampled_nums", sampled_nums);
    }

    dynamic_decode_->ForwardMultiStep(args, forced_ctx);
}

void LlamaV2::dynamicDecodeWithSpec(GenerationState& g,
                                    Buffer           token_ids,
                                    Buffer           finished,
                                    Buffer           sequence_length,
                                    Tensor           curand_state,
                                    const Tensor&    decoder_features,
                                    Tensor           logits,
                                    Buffer           seq_limit_len,
                                    Buffer           init_context_length,
                                    Buffer           context_length,
                                    Buffer           prompt_length,
                                    Buffer           sampled_logprobs,
                                    Buffer           sampled_indexes,
                                    Buffer           sampled_nums,
                                    int              max_context_len,
                                    const SpecContext& spec_ctx)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Reset per-step EAGLE acceptance summary.
    eagle_step_tokens_per_seq_ = 0;
    eagle_step_accepted_lens_.clear();
    eagle_step_accepted_tokens_.clear();
    eagle_step_max_extra_ = 0;

    // EAGLE-enabled engines use the fused multi-step path so that base
    // decode, tree decode, acceptance, and tail commit are handled in a
    // single place (dynamicDecodeWithSpecMulti).
    if (spec_ctx.enable_eagle && isEagleEnabled() && eagle_module_ && eagle_buffers_) {
        dynamicDecodeWithSpecMulti(g,
                                   token_ids,
                                   finished,
                                   sequence_length,
                                   curand_state,
                                   decoder_features,
                                   logits,
                                   seq_limit_len,
                                   init_context_length,
                                   context_length,
                                   prompt_length,
                                   sampled_logprobs,
                                   sampled_indexes,
                                   sampled_nums,
                                   max_context_len,
                                   spec_ctx);
        return;
    }

    // Baseline path when EAGLE is disabled or not fully initialized:
    // keep the original single-step DynamicDecode semantics.
    dynamicDecode(token_ids,
                  finished,
                  sequence_length,
                  curand_state,
                  logits,
                  seq_limit_len,
                  init_context_length,
                  context_length,
                  prompt_length,
                  sampled_logprobs,
                  sampled_indexes,
                  sampled_nums,
                  g.step,
                  max_context_len);
}

void LlamaV2::dynamicDecodeWithSpecMulti(GenerationState& g,
                                         Buffer           token_ids,
                                         Buffer           finished,
                                         Buffer           sequence_length,
                                         Tensor           curand_state,
                                         const Tensor&    decoder_features,
                                         Tensor           logits,
                                         Buffer           seq_limit_len,
                                         Buffer           init_context_length,
                                         Buffer           context_length,
                                         Buffer           prompt_length,
                                         Buffer           sampled_logprobs,
                                         Buffer           sampled_indexes,
                                         Buffer           sampled_nums,
                                         int              max_context_len,
                                         const SpecContext& spec_ctx)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    // Reset per-step EAGLE acceptance summary and extra count.
    eagle_step_tokens_per_seq_  = 0;
    eagle_step_accepted_lens_.clear();
    eagle_step_accepted_tokens_.clear();
    eagle_step_max_extra_ = 0;

    const int batch_size = logits.shape(0);
    if (batch_size <= 0) {
        // Nothing to decode; keep baseline behaviour.
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    // Baseline path when EAGLE is disabled or not fully initialized.
    if (!spec_ctx.enable_eagle || !isEagleEnabled() || !eagle_module_ || !eagle_buffers_) {
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    // If the Eagle3 draft layer under UnifiedDecoder is unavailable for
    // this engine, treat this step as a single-token decode while
    // keeping EAGLE enabled for future steps.
    if (engine_param_.spec_method == "eagle3"
        && (!unified_decoder_ || !eagle3_draft_weight_)) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] Eagle3 draft layer unavailable; "
            "treating this step as single-token decode.");
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    // ========= EAGLE draft + tree decode + acceptance (copied from dynamicDecodeWithSpec) =========

    // Prefer the per-slot plan from LlamaBatch when available so that
    // tokens_per_seq reflects SpeculativeConfig.num_speculative_tokens
    // and per-request clamping. Fall back to a simple engine-budget
    // computation only when no plan was provided.
    int tokens_per_seq = 0;
    if (spec_ctx.planned_tokens_per_seq) {
        // planned_tokens_per_seq[i] already includes per-slot clamping;
        // here we use the maximum over the active decode batch as the
        // effective tokens_per_seq for logging and tree construction.
        int max_planned = 0;
        for (int i = 0; i < batch_size; ++i) {
            const int planned = spec_ctx.planned_tokens_per_seq[i];
            if (planned > max_planned) {
                max_planned = planned;
            }
        }
        tokens_per_seq = max_planned;
    }
    else {
        auto compute_draft_tokens_per_seq = [&](int decode_batch_size) -> int {
            if (decode_batch_size <= 0) {
                return 0;
            }

            const int max_engine_tokens = eagleMaxEngineTokensPerStep();
            if (max_engine_tokens <= 0) {
                return 0;
            }

            // When the engine-side budget resolves to 1, treat this as
            // single-token EAGLE regardless of batch size.
            if (max_engine_tokens <= 1) {
                return 1;
            }

            const int per_seq_cap = engine_param_.spec_max_decoding_draft_tokens;
            if (per_seq_cap <= 0) {
                return 1;
            }

            // Hard upper bound from engine budget: do not exceed the per-step
            // engine token allowance when spreading draft tokens across the
            // active decode batch.
            const int max_by_engine = max_engine_tokens / decode_batch_size;
            if (max_by_engine <= 0) {
                return 1;
            }

            const int tps = std::max(1, std::min(per_seq_cap, max_by_engine));
            return tps;
        };

        tokens_per_seq = compute_draft_tokens_per_seq(batch_size);
    }

    if (tokens_per_seq <= 0) {
        // No speculative tokens planned for this step; fall back to
        // single-token decode without disabling EAGLE for the engine.
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE] tokens_per_seq resolved to %d in dynamicDecodeWithSpecMulti; "
            "treating this step as single-token.",
            tokens_per_seq);
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    // Run Eagle draft path + Top-K to populate EagleBuffers draft/target
    // tokens and host mirrors. For Eagle3 we route this through the
    // UnifiedDecoder draft layer; legacy Eagle engines continue to use
    // EagleModule's shallow draft head.
    Tensor draft_logits;
    if (engine_param_.spec_method == "eagle3") {
        runEagle3DraftTreeDecode(
            decoder_features, logits, batch_size, tokens_per_seq, *eagle_buffers_, draft_logits, stream_);
    }
    else {
        eagle_module_->forward_draft_tree(
            decoder_features,
            eagle_capture_hidden_,
            logits,
            tokens_per_seq,
            *eagle_buffers_,
            draft_logits,
            linear_,
            stream_);
    }

    const int total_draft = batch_size * tokens_per_seq;
    Buffer_<int> draft_tokens(total_draft, kCPU);
    Buffer_<int> target_tokens(total_draft, kCPU);

    if (tokens_per_seq > 0 && total_draft > 0) {
        check_cuda_error(cudaMemcpyAsync(draft_tokens.data(),
                                         eagle_buffers_->inputs.draft_tokens,
                                         static_cast<size_t>(total_draft) * sizeof(int),
                                         cudaMemcpyDeviceToHost,
                                         stream_));

        if (!isTargetTreeDecodeActiveStep()) {
            check_cuda_error(cudaMemcpyAsync(target_tokens.data(),
                                             eagle_buffers_->inputs.target_tokens,
                                             static_cast<size_t>(total_draft) * sizeof(int),
                                             cudaMemcpyDeviceToHost,
                                             stream_));
        }
        check_cuda_error(cudaStreamSynchronize(stream_));
    }
    else {
        std::fill_n(draft_tokens.data(), total_draft, 0);
        std::fill_n(target_tokens.data(), total_draft, 0);
    }

    TM_LOG_DEBUG("[LlamaV2][EAGLE] step=%d, batch=%d, tokens_per_seq=%d, num_draft_tokens=%d",
                 g.step,
                 batch_size,
                 tokens_per_seq,
                 total_draft);

    // ========= Step 1: Build EAGLE tree from draft tokens =========
    const int num_draft_tokens = total_draft;
    const int inferred_tokens_per_seq =
        batch_size > 0 ? std::max(1, num_draft_tokens / batch_size) : 1;
    const int expected_total = batch_size * inferred_tokens_per_seq;
    if (expected_total != num_draft_tokens) {
        TM_LOG_WARNING("[LlamaV2][EAGLE] draft_tokens size (%d) is not a multiple of batch_size (%d); "
                       "interpreting as tokens_per_seq=%d and ignoring any trailing entries",
                       num_draft_tokens,
                       batch_size,
                       inferred_tokens_per_seq);
    }

    eagle::SpeculationTree tree(
        engine_param_.spec_max_draft_path_len,
        engine_param_.spec_max_decoding_tokens);

    const auto& choices = eagle_module_->getDefaultChoices();
    if (!choices.empty()) {
        tree.buildTreeWithChoices(draft_tokens.data(), inferred_tokens_per_seq, choices);
    }
    else {
        tree.buildTree(draft_tokens.data(), inferred_tokens_per_seq);
    }
    tree.extractPaths();

    const int* paths_flat         = tree.getPathsFlat();
    const int  num_paths          = tree.getNumPaths();
    const int  max_path_len       = engine_param_.spec_max_draft_path_len;
    const int  max_decoding_tokens = engine_param_.spec_max_decoding_tokens;

    if (num_paths == 0) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] No valid paths in speculative tree in dynamicDecodeWithSpecMulti; "
            "treating this step as single-token decode.");
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    TM_LOG_DEBUG("[LlamaV2][EAGLE] Built tree with %d paths, max_depth=%d",
                 num_paths,
                 max_path_len);

    // ========= Step 2: Mirror tokens and paths into EagleBuffers =========
    cudaMemcpyAsync(eagle_buffers_->inputs.draft_tokens,
                    draft_tokens.data(),
                    num_draft_tokens * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream_);
    if (!isTargetTreeDecodeActiveStep()) {
        cudaMemcpyAsync(eagle_buffers_->inputs.target_tokens,
                        target_tokens.data(),
                        num_draft_tokens * sizeof(int),
                        cudaMemcpyHostToDevice,
                        stream_);
    }

    // Prepare per-sequence path buffers. The current tree is single-sequence,
    // so we replicate its paths across all batch slots.
    std::vector<int> host_paths(batch_size * max_decoding_tokens * max_path_len, -1);
    const int        paths_per_seq = std::min(num_paths, max_decoding_tokens);

    for (int b = 0; b < batch_size; ++b) {
        for (int p = 0; p < paths_per_seq; ++p) {
            const int* src = paths_flat + p * max_path_len;
            int*       dst = host_paths.data() + (b * max_decoding_tokens + p) * max_path_len;
            std::copy(src, src + max_path_len, dst);
        }
    }

    // Build per-slot tree/runtime offsets and successor metadata for the draft
    // pass. For now each slot contributes a contiguous tokens_per_seq range;
    // runtime offsets drop the root token to mirror kv_lens_runtime.
    std::vector<SizeType> h_tree_offsets(static_cast<size_t>(batch_size) + 1, 0);
    std::vector<SizeType> h_runtime_offsets(static_cast<size_t>(batch_size) + 1, 0);
    for (int b = 0; b < batch_size; ++b) {
        h_tree_offsets[b + 1]    = h_tree_offsets[b] + tokens_per_seq;
        h_runtime_offsets[b + 1] = h_runtime_offsets[b] + std::max(0, tokens_per_seq - 1);
    }

    // Successor counts (TopK) in node-ascending order per slot.
    std::vector<SizeType> h_succ_offsets(static_cast<size_t>(batch_size) + 1, 0);
    std::vector<SizeType> h_succ_counts;
    h_succ_counts.reserve(static_cast<size_t>(batch_size) * max_decoding_tokens);

    for (int b = 0; b < batch_size; ++b) {
        // Build adjacency for slot b.
        std::vector<uint8_t> adj(static_cast<size_t>(max_decoding_tokens) * max_decoding_tokens, 0);
        for (int p = 0; p < paths_per_seq; ++p) {
            const int* path = host_paths.data() + (b * max_decoding_tokens + p) * max_path_len;
            for (int l = 0; l + 1 < max_path_len; ++l) {
                const int from = path[l];
                const int to   = path[l + 1];
                if (from >= 0 && to >= 0 && from < max_decoding_tokens && to < max_decoding_tokens) {
                    adj[static_cast<size_t>(from) * max_decoding_tokens + to] = 1;
                }
            }
        }
        for (int node = 0; node < max_decoding_tokens; ++node) {
            SizeType cnt   = 0;
            size_t   base  = static_cast<size_t>(node) * max_decoding_tokens;
            for (int j = 0; j < max_decoding_tokens; ++j) {
                cnt += static_cast<SizeType>(adj[base + j]);
            }
            if (cnt > 0) {
                h_succ_counts.push_back(cnt);
            }
        }
        h_succ_offsets[b + 1] = static_cast<SizeType>(h_succ_counts.size());
    }

    // Debug: validate successor_offsets/counts ordering against host paths.
    if (turbomind::isEagleDebugEnabled() && eagle_buffers_->inputs.successor_offsets
        && eagle_buffers_->inputs.successor_counts) {
        const size_t offsets_bytes = static_cast<size_t>(batch_size + 1) * sizeof(SizeType);
        std::vector<SizeType> h_succ_offsets_dev(batch_size + 1, 0);
        check_cuda_error(cudaMemcpyAsync(h_succ_offsets_dev.data(),
                                         eagle_buffers_->inputs.successor_offsets,
                                         offsets_bytes,
                                         cudaMemcpyDeviceToHost,
                                         stream_));

        // Copy the flattened counts up to the reported total.
        check_cuda_error(cudaStreamSynchronize(stream_));
        const SizeType total_topk = h_succ_offsets_dev.back();
        std::vector<SizeType> h_succ_counts_dev(total_topk, 0);
        if (total_topk > 0) {
            check_cuda_error(cudaMemcpyAsync(h_succ_counts_dev.data(),
                                             eagle_buffers_->inputs.successor_counts,
                                             static_cast<size_t>(total_topk) * sizeof(SizeType),
                                             cudaMemcpyDeviceToHost,
                                             stream_));
            check_cuda_error(cudaStreamSynchronize(stream_));
        }

        bool ok = true;
        for (int b = 0; b < batch_size && ok; ++b) {
            // Build adjacency for this batch from host_paths.
            std::vector<uint8_t> adj(static_cast<size_t>(max_decoding_tokens) * max_decoding_tokens, 0);
            for (int p = 0; p < paths_per_seq; ++p) {
                const int* path = host_paths.data() + (b * max_decoding_tokens + p) * max_path_len;
                for (int l = 0; l + 1 < max_path_len; ++l) {
                    const int from = path[l];
                    const int to   = path[l + 1];
                    if (from >= 0 && to >= 0 && from < max_decoding_tokens && to < max_decoding_tokens) {
                        adj[static_cast<size_t>(from) * max_decoding_tokens + to] = 1;
                    }
                }
            }

            std::vector<SizeType> expected;
            expected.reserve(max_decoding_tokens);
            for (int node = 0; node < max_decoding_tokens; ++node) {
                SizeType cnt = 0;
                const size_t base = static_cast<size_t>(node) * max_decoding_tokens;
                for (int j = 0; j < max_decoding_tokens; ++j) {
                    cnt += static_cast<SizeType>(adj[base + j]);
                }
                if (cnt > 0) {
                    expected.push_back(cnt);
                }
            }

            const SizeType start = h_succ_offsets_dev[b];
            const SizeType end   = h_succ_offsets_dev[b + 1];
            if (end - start != static_cast<SizeType>(expected.size())) {
                TM_LOG_WARNING(
                    "[LlamaV2][EAGLE][successor] count length mismatch slot=%d expected=%zu got=%d "
                    "(start=%d end=%d)",
                    b,
                    expected.size(),
                    end - start,
                    start,
                    end);
                ok = false;
                break;
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                if (h_succ_counts_dev[start + i] != expected[i]) {
                    TM_LOG_WARNING(
                        "[LlamaV2][EAGLE][successor] mismatch slot=%d idx=%zu expected=%d got=%d",
                        b,
                        i,
                        expected[i],
                        h_succ_counts_dev[start + i]);
                    ok = false;
                    break;
                }
            }
        }

        if (ok && tp_rank_ == 0) {
            TM_LOG_INFO("[LlamaV2][EAGLE][successor] host/device successor counts match (total_topk=%d)",
                        h_succ_offsets_dev.back());
        }
    }

    cudaMemcpyAsync(eagle_buffers_->inputs.draft_paths,
                    host_paths.data(),
                    host_paths.size() * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream_);

    sync_check_cuda_error();

    // Materialize draft-side offsets / successors for attention. Shapes:
    //  - tree_offsets: [batch+1], runtime_offsets: [batch+1]
    //  - successor_offsets: [batch+1], successor_counts: [total_topk]
    Tensor draft_tree_offsets{{batch_size + 1}, kInt32, kDEVICE};
    Tensor draft_runtime_offsets{{batch_size + 1}, kInt32, kDEVICE};
    Tensor draft_kv_lens_runtime{{batch_size}, kInt32, kDEVICE};
    check_cuda_error(cudaMemcpyAsync(draft_tree_offsets.raw_data(),
                                     h_tree_offsets.data(),
                                     h_tree_offsets.size() * sizeof(SizeType),
                                     cudaMemcpyHostToDevice,
                                     stream_));
    check_cuda_error(cudaMemcpyAsync(draft_runtime_offsets.raw_data(),
                                     h_runtime_offsets.data(),
                                     h_runtime_offsets.size() * sizeof(SizeType),
                                     cudaMemcpyHostToDevice,
                                     stream_));
    // kv_lens_runtime matches TRT semantics: per-slot runtime lengths excluding extra draft tokens.
    {
        std::vector<SizeType> h_runtime_lens(static_cast<size_t>(batch_size), 0);
        for (int b = 0; b < batch_size; ++b) {
            h_runtime_lens[b] = std::max<SizeType>(0, h_runtime_offsets[b + 1] - h_runtime_offsets[b]);
        }
        check_cuda_error(cudaMemcpyAsync(draft_kv_lens_runtime.raw_data(),
                                         h_runtime_lens.data(),
                                         h_runtime_lens.size() * sizeof(SizeType),
                                         cudaMemcpyHostToDevice,
                                         stream_));
    }

    Tensor draft_successor_offsets;
    Tensor draft_successor_counts;
    if (!h_succ_counts.empty()) {
        draft_successor_offsets = Tensor{{batch_size + 1}, kInt32, kDEVICE};
        draft_successor_counts  = Tensor{{static_cast<int>(h_succ_counts.size())}, kInt32, kDEVICE};

        check_cuda_error(cudaMemcpyAsync(draft_successor_offsets.raw_data(),
                                         h_succ_offsets.data(),
                                         h_succ_offsets.size() * sizeof(SizeType),
                                         cudaMemcpyHostToDevice,
                                         stream_));
        check_cuda_error(cudaMemcpyAsync(draft_successor_counts.raw_data(),
                                         h_succ_counts.data(),
                                         h_succ_counts.size() * sizeof(SizeType),
                                         cudaMemcpyHostToDevice,
                                         stream_));
    }

    // ========= Step 3: Generate leaf mask and packed masks =========
    kernels::eagle::invokeBuildLeafMask(
        eagle_buffers_->inputs.leaf_mask,
        eagle_buffers_->inputs.draft_paths,
        batch_size,
        engine_param_.spec_max_decoding_tokens,
        engine_param_.spec_max_draft_path_len,
        stream_);

    std::vector<int> batch_slots(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_slots[i] = i;
    }

    int* d_batch_slots = nullptr;
    check_cuda_error(cudaMalloc(&d_batch_slots, batch_size * sizeof(int)));
    check_cuda_error(cudaMemcpyAsync(
        d_batch_slots,
        batch_slots.data(),
        batch_size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_));

    // Packed masks (powers-of-two compressed) using the speculative_decoding
    // helper to mirror TRT's packed tree masks. Guard overly large batches to
    // avoid pathological grid sizes.
    const int max_tokens_for_masks = static_cast<int>(engine_param_.spec_max_decoding_tokens);
    if (batch_size > 512) {
        TM_LOG_WARNING(
            "[LlamaV2][EAGLE][fallback] batch_size=%d exceeds packed-mask guard (512); "
            "packed masks may be incomplete.",
            batch_size);
    }

    ::lmdeploy::turbomind::kernels::speculative_decoding::invokeGetPackedMaskFromPath(
        reinterpret_cast<int32_t*>(eagle_buffers_->inputs.packed_masks),
        reinterpret_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32*>(d_batch_slots),
        reinterpret_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32 const*>(
            eagle_buffers_->inputs.draft_paths),
        static_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32>(batch_size),
        static_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32>(max_tokens_for_masks),
        static_cast<::lmdeploy::turbomind::kernels::speculative_decoding::SizeType32>(
            engine_param_.spec_max_draft_path_len),
        stream_);

    sync_check_cuda_error();

    TM_LOG_DEBUG("[LlamaV2][EAGLE] Generated masks for %d paths", num_paths);

    // Optional target-tree decode over the speculative tree.
    if (spec_ctx.enable_eagle_target_tree && isTargetTreeDecodeEnabled()) {
        runEagleTargetTreeDecode(batch_size, spec_ctx.d_sequence_lengths, spec_ctx.sequences);
    }

    // Posterior/typical gating (entropy-based) on draft logits if provided.
    if (draft_logits && draft_logits.dtype() == kFloat32 && spec_ctx.d_posterior_thresholds
        && spec_ctx.d_posterior_alphas && spec_ctx.d_temperatures) {
        // For now assume flat layout [batch, vocab] per slot; reuse tokens_per_seq as max_tokens.
        const int max_tokens      = tokens_per_seq;
        const int vocab_size_pad  = draft_logits.shape(1);
        float**   logits_ptrs_dev = nullptr;
        float**   out_ids_ptrs    = nullptr;
        bool*     skip_decode     = nullptr;
        float*    runtime_top_p   = nullptr;

        // Allocate small helper buffers on device.
        check_cuda_error(cudaMalloc(&logits_ptrs_dev, sizeof(float*) * batch_size * max_tokens));
        check_cuda_error(cudaMalloc(&out_ids_ptrs, sizeof(float*) * batch_size * max_tokens));
        check_cuda_error(cudaMalloc(&skip_decode, sizeof(bool) * batch_size * max_tokens));
        check_cuda_error(cudaMalloc(&runtime_top_p, sizeof(float) * batch_size * max_tokens));

        // Build logits_ptrs as contiguous rows in draft_logits.
        std::vector<float*> h_logits_ptrs(static_cast<size_t>(batch_size * max_tokens), nullptr);
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < max_tokens; ++t) {
                h_logits_ptrs[b * max_tokens + t] =
                    static_cast<float*>(draft_logits.raw_data()) + static_cast<size_t>(b) * vocab_size_pad;
            }
        }
        check_cuda_error(cudaMemcpyAsync(
            logits_ptrs_dev, h_logits_ptrs.data(), h_logits_ptrs.size() * sizeof(float*), cudaMemcpyHostToDevice, stream_));

        // Entropy buffer: allocate and compute softmax+entropy for each row.
        const size_t probs_elems = static_cast<size_t>(batch_size * max_tokens * vocab_size_pad);
        float*       probs_dev   = nullptr;
        float*       entropy_dev = nullptr;
        check_cuda_error(cudaMalloc(&probs_dev, probs_elems * sizeof(float)));
        check_cuda_error(cudaMalloc(&entropy_dev, static_cast<size_t>(batch_size * max_tokens) * sizeof(float)));

        // Simple softmax + entropy per row.
        invokeSoftmaxWithEntropy(static_cast<const float*>(draft_logits.raw_data()),
                                 probs_dev,
                                 entropy_dev,
                                 batch_size * max_tokens,
                                 vocab_size_pad,
                                 stream_);

        kernels::speculative_decoding::EntropyMaskParams mask_params{};
        mask_params.logits_ptrs         = logits_ptrs_dev;
        mask_params.output_id_ptrs      = out_ids_ptrs;
        mask_params.skip_decode         = skip_decode;
        mask_params.output_ids          = nullptr;
        mask_params.runtime_top_p       = runtime_top_p;
        mask_params.probs               = probs_dev;
        mask_params.entropies           = entropy_dev;
        mask_params.generation_lengths  = spec_ctx.d_sequence_lengths;
        mask_params.posterior_thresholds = spec_ctx.d_posterior_thresholds;
        mask_params.posterior_alphas    = spec_ctx.d_posterior_alphas;
        mask_params.temperatures        = spec_ctx.d_temperatures;
        mask_params.batch_slots         = nullptr;
        mask_params.batch_size          = batch_size;
        mask_params.max_tokens          = max_tokens;
        mask_params.vocab_size          = vocab_size_pad;
        mask_params.stream              = stream_;

        kernels::speculative_decoding::maskLogitsBasedOnEntropy(mask_params);

        cudaFree(logits_ptrs_dev);
        cudaFree(out_ids_ptrs);
        cudaFree(skip_decode);
        cudaFree(runtime_top_p);
        cudaFree(probs_dev);
        cudaFree(entropy_dev);
    }

    // ========= Step 4: Device-side acceptance over tree paths =========
    std::vector<int> h_draft(num_draft_tokens);
    std::vector<int> h_target(num_draft_tokens);
    std::copy_n(draft_tokens.data(), num_draft_tokens, h_draft.data());

    std::vector<int> h_accepted_lens(batch_size, 0);
    std::vector<int> h_accepted_tokens(batch_size * max_path_len, -1);
    std::vector<int> host_best_path_ids(batch_size, 0);

    using SpecSizeType    = kernels::speculative_decoding::SizeType;
    using SpecTokenIdType = kernels::speculative_decoding::TokenIdType;

    const SpecSizeType max_batch_size   = static_cast<SpecSizeType>(batch_size);
    const SpecSizeType max_draft_tokens = static_cast<SpecSizeType>(inferred_tokens_per_seq);
    const SpecSizeType num_paths_device = static_cast<SpecSizeType>(paths_per_seq);
    const SpecSizeType max_path_len_dev = static_cast<SpecSizeType>(max_path_len);

    kernels::speculative_decoding::invokeTreeAcceptByIdsWithPaths(
        reinterpret_cast<SpecTokenIdType const*>(eagle_buffers_->inputs.draft_tokens),
        reinterpret_cast<SpecTokenIdType const*>(eagle_buffers_->inputs.target_tokens),
        reinterpret_cast<SpecSizeType const*>(eagle_buffers_->inputs.draft_paths),
        reinterpret_cast<SpecTokenIdType const*>(spec_ctx.d_end_ids),
        reinterpret_cast<SpecSizeType const*>(d_batch_slots),
        static_cast<SpecSizeType>(batch_size),
        max_batch_size,
        num_paths_device,
        max_path_len_dev,
        max_draft_tokens,
        reinterpret_cast<SpecSizeType*>(eagle_buffers_->outputs.best_path_ids),
        reinterpret_cast<SpecSizeType*>(eagle_buffers_->outputs.accepted_lens),
        reinterpret_cast<SpecTokenIdType*>(eagle_buffers_->outputs.accepted_tokens),
        stream_);

    sync_check_cuda_error();

    check_cuda_error(cudaMemcpyAsync(
        host_best_path_ids.data(),
        eagle_buffers_->outputs.best_path_ids,
        batch_size * sizeof(SpecSizeType),
        cudaMemcpyDeviceToHost,
        stream_));

    check_cuda_error(cudaMemcpyAsync(
        h_accepted_lens.data(),
        eagle_buffers_->outputs.accepted_lens,
        batch_size * sizeof(SpecSizeType),
        cudaMemcpyDeviceToHost,
        stream_));

    check_cuda_error(cudaMemcpyAsync(
        h_accepted_tokens.data(),
        eagle_buffers_->outputs.accepted_tokens,
        batch_size * max_path_len * sizeof(SpecTokenIdType),
        cudaMemcpyDeviceToHost,
        stream_));

    check_cuda_error(cudaStreamSynchronize(stream_));

    int total_accepted = 0;
    for (int b = 0; b < batch_size; ++b) {
        if (h_accepted_lens[b] < 0) {
            h_accepted_lens[b] = 0;
        }
        total_accepted += h_accepted_lens[b];
    }

    // Host-side cumsum of accepted lengths for offset packing; mirrors TRT
    // expectation for packed acceptance metadata. Prefix size is batch+1.
    std::vector<SpecSizeType> h_accepted_cumsum(static_cast<size_t>(batch_size) + 1, 0);
    for (int b = 0; b < batch_size; ++b) {
        h_accepted_cumsum[b + 1] = h_accepted_cumsum[b] + h_accepted_lens[b];
    }

    // Optional debug logging of accepted paths/tokens.
    if (isEagleDebugEnabled() && tp_rank_ == 0) {
        if (isTargetTreeDecodeActiveStep()) {
            check_cuda_error(cudaMemcpyAsync(
                h_target.data(),
                eagle_buffers_->inputs.target_tokens,
                num_draft_tokens * sizeof(int),
                cudaMemcpyDeviceToHost,
                stream_));
            check_cuda_error(cudaStreamSynchronize(stream_));
        }
        else {
            std::copy_n(target_tokens.data(), num_draft_tokens, h_target.data());
        }

        for (int b = 0; b < batch_size; ++b) {
            const int len     = h_accepted_lens[b];
            const int path_id = host_best_path_ids[b];
            if (len <= 0) {
                TM_LOG_WARNING("[LlamaV2][EAGLE] step_spec seq=%d no accepted tokens (best_path=%d)", b, path_id);
                continue;
            }

            std::ostringstream draft_ss;
            std::ostringstream target_ss;
            std::ostringstream accepted_ss;

            for (int d = 0; d < max_path_len; ++d) {
                const int node_idx = paths_flat[path_id * max_path_len + d];
                if (node_idx <= 0) {
                    if (node_idx < 0) {
                        break;
                    }
                    continue;
                }
                const int token_idx = node_idx - 1;
                if (token_idx < 0 || token_idx >= inferred_tokens_per_seq) {
                    break;
                }
                const int global_idx = b * inferred_tokens_per_seq + token_idx;
                if (global_idx < 0 || global_idx >= num_draft_tokens) {
                    break;
                }

                const int draft_id  = h_draft[global_idx];
                const int target_id = h_target[global_idx];

                if (draft_ss.tellp() > 0) {
                    draft_ss << ',';
                    target_ss << ',';
                }
                draft_ss << draft_id;
                target_ss << target_id;
            }

            for (int t = 0; t < len; ++t) {
                if (t) {
                    accepted_ss << ',';
                }
                accepted_ss << h_accepted_tokens[b * max_path_len + t];
            }

            TM_LOG_WARNING("[LlamaV2][EAGLE] step_spec seq=%d best_path=%d accepted_len=%d "
                           "path_draft_tokens=[%s] path_target_tokens=[%s] "
                           "accepted_tokens=[%s]",
                           b,
                           path_id,
                           len,
                           draft_ss.str().c_str(),
                           target_ss.str().c_str(),
                           accepted_ss.str().c_str());
        }
    }

    if (eagle_buffers_ && eagle_buffers_->isAllocated()) {
        // Write accepted_lengths_cumsum to device for downstream offset helpers.
        if (eagle_buffers_->outputs.accepted_lengths_cumsum) {
            check_cuda_error(cudaMemcpyAsync(
                eagle_buffers_->outputs.accepted_lengths_cumsum,
                h_accepted_cumsum.data(),
                static_cast<size_t>(batch_size + 1) * sizeof(SpecSizeType),
                cudaMemcpyHostToDevice,
                stream_));
        }

        // Pack accepted paths on device for downstream KV / context helpers.
        kernels::speculative_decoding::invokePackAcceptedPaths(
            reinterpret_cast<SpecSizeType*>(eagle_buffers_->outputs.accepted_lengths_cumsum),
            reinterpret_cast<SpecSizeType*>(eagle_buffers_->outputs.accepted_path_offsets),
            reinterpret_cast<SpecSizeType const*>(eagle_buffers_->outputs.accepted_lens),
            reinterpret_cast<SpecSizeType const*>(eagle_buffers_->outputs.best_path_ids),
            reinterpret_cast<SpecSizeType const*>(eagle_buffers_->inputs.draft_paths),
            reinterpret_cast<SpecSizeType const*>(d_batch_slots),
            static_cast<SpecSizeType>(batch_size),
            static_cast<SpecSizeType>(batch_size),
            static_cast<SpecSizeType>(paths_per_seq),
            static_cast<SpecSizeType>(max_path_len),
            stream_);

        check_cuda_error(cudaFree(d_batch_slots));
    }

    // Stash acceptance results for the next step's context prep (EagleNet0).
    if (eagle_buffers_ && eagle_buffers_->isAllocated()) {
        const size_t lens_bytes   = static_cast<size_t>(batch_size) * sizeof(SpecSizeType);
        const size_t paths_bytes  = static_cast<size_t>(batch_size) * max_decoding_tokens * max_path_len
                                   * sizeof(SpecSizeType);
        const size_t tokens_bytes = static_cast<size_t>(batch_size) * max_path_len * sizeof(SpecTokenIdType);

        if (eagle_buffers_->inputs.prev_accepted_lens) {
            check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.prev_accepted_lens,
                                             eagle_buffers_->outputs.accepted_lens,
                                             lens_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        if (eagle_buffers_->inputs.prev_draft_lens) {
            Buffer_<SpecSizeType> h_prev_draft_lens(batch_size, kCPU);
            std::fill_n(h_prev_draft_lens.data(), batch_size, static_cast<SpecSizeType>(tokens_per_seq));
            check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.prev_draft_lens,
                                             h_prev_draft_lens.data(),
                                             lens_bytes,
                                             cudaMemcpyHostToDevice,
                                             stream_));
        }
        if (eagle_buffers_->inputs.prev_accepted_tokens) {
            check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.prev_accepted_tokens,
                                             eagle_buffers_->outputs.accepted_tokens,
                                             tokens_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        if (eagle_buffers_->inputs.prev_paths && eagle_buffers_->inputs.draft_paths) {
            check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.prev_paths,
                                             eagle_buffers_->inputs.draft_paths,
                                             paths_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        if (eagle_buffers_->inputs.best_path_ids && eagle_buffers_->outputs.best_path_ids) {
            check_cuda_error(cudaMemcpyAsync(eagle_buffers_->inputs.best_path_ids,
                                             eagle_buffers_->outputs.best_path_ids,
                                             lens_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }

        // Seed EagleNet0 inputs for the next step using the current accepted
        // tokens/paths. This mirrors TRT's prepareCtxEagleNetInputs.
        prepareEagleContextInputs(batch_size);
    }

    const int effective_draft_tokens = batch_size;  // one token considered per seq
    float      acceptance_rate       = (effective_draft_tokens > 0)
                                           ? static_cast<float>(total_accepted)
                                                 / static_cast<float>(effective_draft_tokens)
                                           : 0.0f;

    TM_LOG_WARNING("[LlamaV2][EAGLE] Accepted %d/%d draft tokens (%.1f%% acceptance rate)",
                   total_accepted,
                   effective_draft_tokens,
                   acceptance_rate * 100.0f);

    float h_acceptance_rate = acceptance_rate;
    cudaMemcpyAsync(
        eagle_buffers_->outputs.acceptance_rate,
        &h_acceptance_rate,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream_);

    sync_check_cuda_error();

    // Cache per-step acceptance summary for downstream metrics and
    // multi-token advancement.
    eagle_step_tokens_per_seq_  = inferred_tokens_per_seq;
    eagle_step_accepted_lens_   = std::move(h_accepted_lens);
    eagle_step_accepted_tokens_ = std::move(h_accepted_tokens);

    // ========= Build ForcedTailContext from accepted tokens (extras only) =========
    const int max_tail_len = engine_param_.spec_max_draft_path_len;
    if (max_tail_len <= 0 || eagle_step_accepted_lens_.empty()) {
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    std::vector<int> forced_lengths(batch_size, 0);
    std::vector<int> forced_tokens(static_cast<size_t>(batch_size) * max_tail_len, -1);
    std::vector<int> committed_lengths(batch_size, 0);

    int any_forced = 0;

    for (int i = 0; i < batch_size; ++i) {
        int len = (i < static_cast<int>(eagle_step_accepted_lens_.size()))
                      ? eagle_step_accepted_lens_[i]
                      : 0;
        if (len <= 1) {
            continue;
        }
        if (len > max_tail_len) {
            len = max_tail_len;
        }

        const int extra = len - 1;  // extras only; base token is committed by DynamicDecodeLayer
        if (extra <= 0) {
            continue;
        }

        const int token_offset = i * max_tail_len;
        if (token_offset + len > static_cast<int>(eagle_step_accepted_tokens_.size())) {
            continue;
        }

        forced_lengths[i] = extra;
        for (int t = 0; t < extra; ++t) {
            forced_tokens[static_cast<size_t>(i) * max_tail_len + t] =
                eagle_step_accepted_tokens_[token_offset + 1 + t];
        }

        any_forced = std::max(any_forced, extra);
    }

    if (any_forced <= 0) {
        // No usable tails; fall back to single-token decode.
        dynamicDecodeMultiStep(token_ids,
                               finished,
                               sequence_length,
                               curand_state,
                               logits,
                               seq_limit_len,
                               init_context_length,
                               context_length,
                               prompt_length,
                               sampled_logprobs,
                               sampled_indexes,
                               sampled_nums,
                               g.step,
                               max_context_len,
                               nullptr);
        return;
    }

    // Allocate a 2-D tail buffer for DynamicDecodeLayer::ForwardMultiStep.
    // Layout: [max_tail_len, batch_size]

    ForcedTailContext forced{};
    forced.max_tail_len      = max_tail_len;
    forced.forced_tokens     = forced_tokens.data();
    forced.forced_lengths    = forced_lengths.data();
    forced.committed_lengths = committed_lengths.data();

    dynamicDecodeMultiStep(token_ids,
                           finished,
                           sequence_length,
                           curand_state,
                           logits,
                           seq_limit_len,
                           init_context_length,
                           context_length,
                           prompt_length,
                           sampled_logprobs,
                           sampled_indexes,
                           sampled_nums,
                           g.step,
                           max_context_len,
                           &forced);

    // Reconcile acceptance with what was actually committed by the decode
    // layer. This ensures metrics, KV rewind, and g.step all see the same
    // effective extra lengths.
    int max_extra_committed = 0;
    for (int i = 0; i < batch_size; ++i) {
        const int committed_extra = (i < static_cast<int>(committed_lengths.size())) ? committed_lengths[i] : 0;
        if (committed_extra <= 0) {
            // If nothing was committed, treat this slot as single-token.
            if (i < static_cast<int>(eagle_step_accepted_lens_.size())) {
                eagle_step_accepted_lens_[i] = std::min(eagle_step_accepted_lens_[i], 1);
            }
            continue;
        }

        max_extra_committed = std::max(max_extra_committed, committed_extra);

        if (i < static_cast<int>(eagle_step_accepted_lens_.size())) {
            const int effective_len = 1 + committed_extra;  // base + extras
            eagle_step_accepted_lens_[i] = std::min(eagle_step_accepted_lens_[i], effective_len);
        }
    }

    eagle_step_max_extra_ = max_extra_committed;

    // Force the accepted root token for each slot to match the base token
    // actually committed by DynamicDecodeLayer, and instrument tail depth
    // statistics. This keeps accepted_tokens[0] aligned with the true
    // decode base while still using extras (indices >= 1) for tails.
    if (max_tail_len > 0 && !eagle_step_accepted_lens_.empty() && !eagle_step_accepted_tokens_.empty()) {
        // Host view of the base token committed at time index g.step for
        // each slot in this decode batch.
        std::vector<int> h_base_tokens(batch_size, -1);
        {
            int* base_ptr = token_ids.data<int>() + g.step * batch_size;
            check_cuda_error(cudaMemcpyAsync(
                h_base_tokens.data(),
                base_ptr,
                static_cast<size_t>(batch_size) * sizeof(int),
                cudaMemcpyDeviceToHost,
                stream_));
            check_cuda_error(cudaStreamSynchronize(stream_));
        }

        int max_accepted_len = 0;
        int slots_ge2        = 0;
        int mismatch_slots   = 0;

        for (int i = 0; i < batch_size; ++i) {
            const int len = (i < static_cast<int>(eagle_step_accepted_lens_.size()))
                                ? eagle_step_accepted_lens_[i]
                                : 0;
            if (len <= 0) {
                continue;
            }

            max_accepted_len = std::max(max_accepted_len, len);
            if (len >= 2) {
                ++slots_ge2;
            }

            const int token_offset = i * max_tail_len;
            if (token_offset >= static_cast<int>(eagle_step_accepted_tokens_.size())) {
                continue;
            }

            const int decode_base =
                (i < static_cast<int>(h_base_tokens.size())) ? h_base_tokens[i] : -1;
            const int eagle_base_before = eagle_step_accepted_tokens_[token_offset];

            if (decode_base != -1 && eagle_base_before != decode_base) {
                ++mismatch_slots;
            }

            // Force the accepted root token to match the DynamicDecode base
            // token so that tails are always built on top of the committed
            // base, even when the raw tree-accept root disagrees.
            if (decode_base != -1) {
                eagle_step_accepted_tokens_[token_offset] = decode_base;
            }
        }

        if (isEagleDebugEnabled() && tp_rank_ == 0) {
            TM_LOG_DEBUG(
                "[LlamaV2][EAGLE][stats] step=%d tokens_per_seq=%d max_accepted_len=%d "
                "slots_ge2=%d base_mismatch_slots=%d",
                g.step,
                eagle_step_tokens_per_seq_,
                max_accepted_len,
                slots_ge2,
                mismatch_slots);
        }
    }

    // Update SpecPV bookkeeping based on the final committed sequence
    // lengths for this step. This keeps partial-KV length tracking in
    // sync with DynamicDecodeLayer and EAGLE acceptance.
    updateSpecPVAfterAcceptance(sequence_length, batch_size, spec_ctx.sequences, &committed_lengths);
}

void LlamaV2::getEagleAcceptanceForStep(std::vector<int>& accepted_lens,
                                        std::vector<int>& accepted_tokens,
                                        int&              tokens_per_seq) const
{
    accepted_lens   = eagle_step_accepted_lens_;
    accepted_tokens = eagle_step_accepted_tokens_;
    tokens_per_seq  = eagle_step_tokens_per_seq_;
}

void LlamaV2::eagleDraftForward(const Buffer& input_ids,
                                const Tensor& hidden_states,
                                Tensor&       draft_logits,
                                Tensor&       draft_hidden)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!spec_mode_.isEagle() || !eagle_module_ || !eagle_buffers_) {
        return;
    }

    // EagleModule::forward currently does not rely on input_ids for the
    // shallow draft network, so we pass an empty Tensor here. If a future
    // draft path needs token ids (e.g. for embeddings), this helper can be
    // extended to materialize them appropriately.
    Tensor input_ids_tensor;
    const Tensor& captured_hidden = eagle_capture_hidden_;
    eagle_module_->forward(
        input_ids_tensor,   // per-slot base token ids
        hidden_states,      // last-token hidden states from target model
        captured_hidden,    // optional multi-layer capture buffer
        draft_logits,       // [batch, vocab_size_padded]
        draft_hidden,       // [batch, hidden_units]
        linear_,            // reuse LlamaLinear for LM head matmul
        stream_);
}

}  // namespace turbomind
