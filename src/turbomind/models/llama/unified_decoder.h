#pragma once

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/EagleDraftLayer.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

class UnifiedDecoder {
public:
    using WeightType = LlamaDecoderLayerWeight;

    UnifiedDecoder(const ModelParam&     model,
                   const EngineParam&    engine,
                   const AttentionParam& attn,
                   const MoeParam&       moe,
                   const LoraParam&      lora,
                   const Context&        ctx);

    ~UnifiedDecoder();

    void Forward(TensorMap& args, const std::vector<WeightType*>& weights);

    void setEagle3DraftLayer(const Eagle3DraftLayerWeight* w);

    void ForwardDraft(const Tensor& input_hidden,
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
                      int           num_tokens,
                      cudaStream_t  stream);

    // Debug tensor accessors for eagle3 draft layer (populated when
    // eagle_debug is enabled). These return empty tensors when the
    // draft layer is unavailable or debug is off.
    const Tensor& debug_fc_out() const;
    const Tensor& debug_qkv() const;
    const Tensor& debug_attn_out() const;
    const Tensor& debug_ffn_out() const;
    const Tensor& debug_pre_head_hidden() const;

private:
    const size_t layer_num_;
    const size_t hidden_units_;

    const int attn_tp_size_;
    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int mlp_tp_size_;

    const int attn_tp_group_;

    const float        rmsnorm_eps_;
    cudaStream_t const stream_;

    comm::DeviceCommImpl* const d_comm_;

    const int tune_layer_num_;

    std::unique_ptr<UnifiedAttentionLayer>    attn_layer_;
    std::unique_ptr<LlamaFfnLayer>            ffn_layer_;
    std::unique_ptr<MoeFfnLayer>              moe_ffn_layer_;
    // Optional dedicated Eagle‑3 attention backend for non‑LLaMA Eagle‑3
    // geometries (e.g. GPT‑OSS‑120B‑Eagle3 midlayer q/k/v/o). When null,
    // Eagle‑3 draft attention falls back to UnifiedAttentionLayer or the
    // shallow QKV path.
    std::unique_ptr<class Eagle3AttentionLayer> eagle3_attn_layer_;

    // Eagle3 multi-layer hidden capture metadata. When enabled, the
    // decoder will capture last-token hidden states from a small set
    // of layers (e.g. the last 3) into a concatenated buffer for
    // EagleModule to consume.
    std::vector<int> eagle_capture_layers_;
    bool             eagle_capture_enabled_{false};

    // Optional Eagle3 draft-layer weights shared from EagleModule /
    // LlamaV2. When non-null, UnifiedDecoder can run a dedicated
    // Eagle3 draft layer using the same attention / FFN primitives as
    // the main model.
    const Eagle3DraftLayerWeight* eagle3_draft_weight_{nullptr};
    std::unique_ptr<Eagle3DraftLayer>      eagle3_draft_layer_;

    void AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                  Tensor&       residual,
                                  const Tensor& bias,
                                  const Tensor& weight,
                                  int           token_num,
                                  int           t0,
                                  int           t1,
                                  const int*    local_token_nums);
};

}  // namespace turbomind
