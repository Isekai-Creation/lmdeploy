

#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

UnifiedDecoder::UnifiedDecoder(const ModelParam&     model,
                               const EngineParam&    engine,
                               const AttentionParam& attn,
                               const MoeParam&       moe,
                               const LoraParam&      lora,
                               const Context&        ctx):
    layer_num_(model.layer_num),
    hidden_units_(model.hidden_units),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    rmsnorm_eps_(model.norm_eps),
    stream_(ctx.stream),
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(model.tune_layer_num)
{
    attn_layer_ = std::make_unique<UnifiedAttentionLayer>(model, attn, engine, lora, attn_tp_size_, ctx);
    eagle3_attn_layer_ = std::make_unique<Eagle3AttentionLayer>(&ctx.device_prop, ctx.stream);

    if (std::accumulate(moe.expert_num.begin(), moe.expert_num.end(), 0LL)) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(model, moe, engine, ctx);
    }

    if (std::accumulate(model.inter_size.begin(), model.inter_size.end(), 0LL)) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(model, ctx);
    }

    // Enable multi-layer hidden capture for Eagle3 when requested by the
    // engine. For Eagle3 we follow the TensorRT-LLM convention and capture
    // three spread-out layers when there are enough layers; otherwise we
    // fall back to the last few layers.
    if (engine.enable_speculative_decoding && engine.spec_method == "eagle3") {
        const int L = static_cast<int>(layer_num_);
        if (L > 5) {
            // Match Eagle3OneModelSpecMetadata default:
            //   (1, num_layers // 2 - 1, num_layers - 4)
            eagle_capture_layers_ = {1, L / 2 - 1, L - 4};
            eagle_capture_enabled_ = true;
        }
        else if (L >= 3) {
            // Shallow models: capture the last three layers.
            eagle_capture_layers_ = {L - 3, L - 2, L - 1};
            eagle_capture_enabled_ = true;
        }
        else if (L > 0) {
            // Degenerate case: single capture layer at the top.
            eagle_capture_layers_ = {L - 1};
            eagle_capture_enabled_ = true;
        }
    }
}

UnifiedDecoder::~UnifiedDecoder() = default;

void UnifiedDecoder::setEagle3DraftLayer(const Eagle3DraftLayerWeight* w)
{
    eagle3_draft_weight_ = w;

    if (w && attn_layer_) {
        // FFN backend is optional here; Eagle3DraftLayer will skip the
        // FFN path when ffn_layer_ is null and behave as an attention-only
        // draft layer. This is preferable to disabling the draft layer
        // outright, as it keeps Eagle3 structurally active.
        eagle3_draft_layer_ = std::make_unique<Eagle3DraftLayer>(
            w,
            attn_layer_.get(),
            eagle3_attn_layer_.get(),
            ffn_layer_ ? ffn_layer_.get() : nullptr,
            rmsnorm_eps_);

        if (!ffn_layer_) {
            TM_LOG_WARNING(
                "[UnifiedDecoder][EAGLE3][fallback] ffn_layer_ is null; "
                "Eagle3 draft will run without FFN (attention-only).");
        }
    }
    else {
        TM_LOG_WARNING(
            "[UnifiedDecoder][EAGLE3][fallback] draft layer disabled "
            "(weights=%p, attn_layer=%p, eagle3_attn_layer=%p, ffn_layer=%p)",
            static_cast<const void*>(w),
            static_cast<void*>(attn_layer_.get()),
            static_cast<void*>(eagle3_attn_layer_.get()),
            static_cast<void*>(ffn_layer_.get()));
        eagle3_draft_layer_.reset();
    }
}

void UnifiedDecoder::ForwardDraft(const Tensor& input_hidden,
                                  const Tensor& captured_hidden,
                                  const Tensor& input_ids,
                                  const Tensor& embed_tokens_weights,
                                  const Tensor& position_ids,
                                  const Tensor& packed_mask,
                                  const Tensor& tree_offsets,
                                  const Tensor& runtime_offsets,
                                  const Tensor& kv_lens_runtime,
                                  const Tensor& successor_offsets,
                                  const Tensor& successor_counts,
                                  int           q_len,
                                  int           kv_len,
                                  int           past_kv_len,
                                  Tensor&       output_hidden,
                                  int           batch_size,
                                  cudaStream_t  stream)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!eagle3_draft_layer_ || !eagle3_draft_weight_) {
        TM_LOG_WARNING(
            "[UnifiedDecoder][EAGLE3][fallback] draft layer unavailable; passing through hidden states.");
        output_hidden = input_hidden;
        return;
    }

    if (!input_hidden || input_hidden.ndim() != 2 || input_hidden.shape(0) != batch_size
        || input_hidden.shape(1) != static_cast<int>(hidden_units_)) {
        TM_LOG_WARNING(
            "[UnifiedDecoder][EAGLE3][fallback] input_hidden shape mismatch in ForwardDraft "
            "(got=[%d,%d], expected=[%d,%zu]); passing through.",
            input_hidden ? input_hidden.shape(0) : -1,
            input_hidden ? input_hidden.shape(1) : -1,
            batch_size,
            hidden_units_);
        output_hidden = input_hidden;
        return;
    }

    if (!output_hidden || output_hidden.ndim() != 2 || output_hidden.shape(0) != batch_size
        || output_hidden.shape(1) != static_cast<int>(hidden_units_)
        || output_hidden.dtype() != input_hidden.dtype()
        || output_hidden.device().type != input_hidden.device().type) {
        output_hidden = Tensor{{batch_size, static_cast<int>(hidden_units_)}, input_hidden.dtype(), kDEVICE};
    }

    Tensor mask_tensor = packed_mask;
    // Attach position ids / masks to Eagle3 attention via optional params.
    eagle3_draft_layer_->Forward(input_hidden,
                                 captured_hidden,
                                 input_ids,
                                 embed_tokens_weights,
                                 position_ids,
                                 mask_tensor,
                                 tree_offsets,
                                 runtime_offsets,
                                 kv_lens_runtime,
                                 successor_offsets,
                                 successor_counts,
                                 q_len,
                                 kv_len,
                                 past_kv_len,
                                 output_hidden,
                                 stream);
}

const Tensor& UnifiedDecoder::debug_fc_out() const
{
    static Tensor empty;
    return eagle3_draft_layer_ ? eagle3_draft_layer_->debug_fc_out() : empty;
}

const Tensor& UnifiedDecoder::debug_qkv() const
{
    static Tensor empty;
    return eagle3_draft_layer_ ? eagle3_draft_layer_->debug_qkv() : empty;
}

const Tensor& UnifiedDecoder::debug_attn_out() const
{
    static Tensor empty;
    return eagle3_draft_layer_ ? eagle3_draft_layer_->debug_attn_out() : empty;
}

const Tensor& UnifiedDecoder::debug_ffn_out() const
{
    static Tensor empty;
    return eagle3_draft_layer_ ? eagle3_draft_layer_->debug_ffn_out() : empty;
}

const Tensor& UnifiedDecoder::debug_pre_head_hidden() const
{
    static Tensor empty;
    return eagle3_draft_layer_ ? eagle3_draft_layer_->debug_pre_head_hidden() : empty;
}

void UnifiedDecoder::AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                              Tensor&       residual,
                                              const Tensor& bias,
                                              const Tensor& weight,
                                              int           token_num,
                                              int           group0,
                                              int           group1,
                                              const int*    local_token_nums)
{
    const auto dtype = hidden_states.dtype();
    if (0) {}
    else if (group0 || group1) {
        d_comm_->AllreduceResidualBiasRMSnormEx(hidden_states.raw_data(),
                                                residual.data_or((void*)nullptr),
                                                bias.data_or((void*)nullptr),
                                                weight.raw_data(),
                                                rmsnorm_eps_,
                                                hidden_units_,
                                                dtype,
                                                group0,
                                                group1,
                                                local_token_nums,
                                                stream_);
        sync_check_cuda_error();
    }
    else if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(hidden_states.raw_data(),
                                              residual.data_or((void*)nullptr),
                                              bias.data_or((void*)nullptr),
                                              weight.raw_data(),
                                              rmsnorm_eps_,
                                              hidden_units_,
                                              token_num,
                                              dtype,
                                              0,
                                              stream_);
        sync_check_cuda_error();
    }
    else {
        invokeResidualBiasRMSNorm(hidden_states.raw_data(),
                                  residual.data_or((void*)nullptr),
                                  weight.raw_data(),
                                  bias.data_or((void*)nullptr),
                                  dtype,
                                  hidden_units_,
                                  token_num,
                                  rmsnorm_eps_,
                                  stream_);
        sync_check_cuda_error();
    }
}

void UnifiedDecoder::Forward(TensorMap& args, const std::vector<WeightType*>& weights)
{
    /**
     * input tensors:
     *   \param decoder_input [token_num, hidden_units], float
     *   \param output_norm_weight [hidden_dims], float
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param last_token_hidden_units [batch_size, hidden_units]
     *   \param block_ptrs [total_block_counts], void*
     */

    const int decode_num = *args.at("decode_num").data<int>();
    const int prefil_num = *args.at("prefil_num").data<int>();
    const int batch_size = prefil_num + decode_num;

    constexpr auto device = kDEVICE;

    Tensor_<int> local_token_nums = args.at("local_token_nums");

    Tensor local_residual       = args.at("decoder_input");
    Tensor global_hidden_states = args.at("decoder_output");

    Tensor local_hidden_states = global_hidden_states;

    // Optional buffer where multi-layer Eagle3 captures will be stored.
    Tensor eagle_capture_hidden;
    int    eagle_num_capture_layers = 0;
    if (eagle_capture_enabled_ && args.find("eagle_capture_hidden") != args.end()) {
        eagle_capture_hidden = args.at("eagle_capture_hidden");
        eagle_num_capture_layers = static_cast<int>(eagle_capture_layers_.size());
        if (!eagle_capture_hidden
            || eagle_capture_hidden.shape(0) != batch_size
            || eagle_capture_hidden.shape(1) != static_cast<int>(hidden_units_ * eagle_num_capture_layers)) {
            eagle_capture_hidden = Tensor{{batch_size, static_cast<int>(hidden_units_ * eagle_num_capture_layers)},
                                          local_hidden_states.dtype(),
                                          kDEVICE};
            args.at("eagle_capture_hidden") = eagle_capture_hidden;
        }
    }

    const auto global_token_num = global_hidden_states.shape(0);
    const auto local_token_num  = local_residual.size() ? local_residual.shape(0) : 0;

    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector cumul_token_nums(attn_dp_size_ + 1, 0);
        std::inclusive_scan(
            local_token_nums.data(), local_token_nums.data() + attn_dp_size_, cumul_token_nums.begin() + 1);
        const int offset    = cumul_token_nums[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});
    }

    // Optional packed mask for speculative/tree decode. Baseline decode
    // never sets this key in `args`, so attention runs with standard
    // autoregressive masking only. A separate tree-decode path can
    // provide a flattened [token_num, packed_dim] tensor via this key.
    Tensor spec_packed_mask;
    auto   it_mask = args.find("spec_packed_mask");
    if (it_mask != args.end()) {
        spec_packed_mask = it_mask->second;
    }

    // Optional runtime kv lengths for speculative/tree decode: when present,
    // prefer these over the default h_k_len to match TRT kv_lens_runtime.
    auto it_k_rt = args.find("spec_runtime_k_len");
    if (it_k_rt != args.end()) {
        args["h_k_len"] = it_k_rt->second;
    }

    attn_layer_->Initialize(args);

    TM_DEBUG_TENSOR(local_residual, "res", 1);
    TM_DEBUG_TENSOR(weights.at(0)->self_attn_norm, "norm_weight", 2);

    invokeRMSNorm(local_hidden_states, local_residual, weights.at(0)->self_attn_norm, rmsnorm_eps_, stream_);
    sync_check_cuda_error();

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);

    for (int layer = 0; layer < layer_num_; ++layer) {

        /// TODO: do not skip the layers when they are heterogeneous
        if (isTuning() && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention
        UnifiedAttentionLayer::ForwardParam attn_param{};
        attn_param.input       = local_hidden_states;
        attn_param.output      = local_hidden_states;
        attn_param.packed_mask = spec_packed_mask;
        attn_param.weights     = weights.at(layer)->self_attn_weights.get();
        attn_param.layer_id    = layer;
        attn_layer_->Forward(attn_param);

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);

        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 weights.at(layer)->self_attn_weights->output.bias,
                                 weights.at(layer)->ffn_norm,
                                 local_token_num,
                                 attn_tp_group_,
                                 0,
                                 local_token_nums.data());

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);

        ////////////////////////////////////////////
        /// feed-forward network

        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        if (weights.at(layer)->moe_weights) {
            moe_fwd_param = MoeFfnLayer::ForwardParam{global_hidden_states,
                                                      global_hidden_states,
                                                      weights.at(layer)->moe_weights.get(),
                                                      ffn_layer_ ? 1.f : 0.f,
                                                      layer};
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (weights.at(layer)->ffn_weights) {
            ffn_layer_->forward(
                {global_hidden_states, global_hidden_states, weights.at(layer)->ffn_weights.get(), (int)layer});
        }

        if (moe_fwd_param) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);

        const bool last = layer == layer_num_ - 1;

        auto& scale_weight = !last ? weights.at(layer + 1)->self_attn_norm : args.at("output_norm_weight");

        AllreduceResidualRMSnorm(global_hidden_states,
                                 local_residual,
                                 {},
                                 scale_weight,
                                 local_token_num,
                                 0,
                                 attn_tp_group_,
                                 local_token_nums.data());
        sync_check_cuda_error();

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);

        // After the final RMSNorm for this layer, capture last-token hidden
        // states for Eagle3 when this layer is one of the configured capture
        // layers.
        if (eagle_capture_enabled_ && eagle_capture_hidden) {
            auto it = std::find(eagle_capture_layers_.begin(), eagle_capture_layers_.end(), layer);
            if (it != eagle_capture_layers_.end()) {
                const int slot        = static_cast<int>(it - eagle_capture_layers_.begin());
                const int num_capture = eagle_num_capture_layers;
                using T = uint16_t;  // matches last_token_hidden_units dtype

                T* capture_base = (T*)eagle_capture_hidden.raw_data();
                // Layout: [batch, hidden * num_capture_layers]
                // Slot `slot` occupies a contiguous [batch, hidden] chunk.
                T* layer_dst = capture_base + static_cast<size_t>(slot) * hidden_units_;

                if (decode_num) {
                    // For decode tokens, copy the last-token hidden states for
                    // each sequence directly from local_hidden_states.
                    check_cuda_error(cudaMemcpyAsync(layer_dst,
                                                     (T*)local_hidden_states.raw_data(),
                                                     sizeof(T) * decode_num * hidden_units_,
                                                     cudaMemcpyDefault,
                                                     stream_));
                }

                if (prefil_num) {
                    invokeGetFeatureOfLastToken(
                        layer_dst + static_cast<size_t>(decode_num) * hidden_units_,  //
                        (T*)local_hidden_states.raw_data(),
                        attn_layer_->d_cu_q_len() + decode_num,
                        hidden_units_,
                        prefil_num,
                        stream_);
                    sync_check_cuda_error();
                }
            }
        }
    }

    /// TODO
    using T = uint16_t;

    auto last_token_hidden_units = (T*)args.at("last_token_hidden_units").raw_data();

    if (decode_num) {
        check_cuda_error(cudaMemcpyAsync(last_token_hidden_units,
                                         (T*)local_hidden_states.raw_data(),
                                         sizeof(T) * decode_num * hidden_units_,
                                         cudaMemcpyDefault,
                                         stream_));
        // TM_DEBUG_RAW(last_token_hidden_units, decode_num * hidden_units_, "dc_out", 2);
    }

    if (prefil_num) {
        invokeGetFeatureOfLastToken(last_token_hidden_units + decode_num * hidden_units_,  //
                                    (T*)local_hidden_states.raw_data(),
                                    attn_layer_->d_cu_q_len() + decode_num,
                                    hidden_units_,
                                    prefil_num,
                                    stream_);
        sync_check_cuda_error();
        // TM_DEBUG_RAW(last_token_hidden_units + decode_num * hidden_units_, prefil_num * hidden_units_, "pf_out", 2);
    }

    Buffer out(
        (void*)last_token_hidden_units, (decode_num + prefil_num) * hidden_units_, local_residual.dtype(), kDEVICE);

    TM_DEBUG_TENSOR(out, "out", 1);

    attn_layer_->Finalize();
}

}  // namespace turbomind
