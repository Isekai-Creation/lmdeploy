// Copyright (c) OpenMMLab. All rights reserved.
// Shared Eagle3 draft layer metadata used by EagleModule and UnifiedDecoder.

#pragma once

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/Eagle3AttentionWeight.h"

namespace turbomind {

struct Eagle3DraftLayerWeight {
    LlamaAttentionWeight attn;
    LlamaFfnWeight       ffn;
    // Optional dedicated Eagle3 attention weights with non‑LLaMA geometry
    // (e.g. GPT‑OSS‑120B‑Eagle3 midlayer q/k/v/o projections). When not
    // initialised, the draft layer falls back to UnifiedAttentionLayer or
    // the shallow QKV path.
        Eagle3AttentionWeight eagle3_attn;
        Tensor                fc_weight;      // keep if your converter fills it
        Tensor                hidden_norm;
        Tensor                input_norm;
        Tensor                post_attn_norm;
        Tensor                output_norm;
    
        int draft_hidden_dim;
        int base_hidden_dim;
        int head_num;
        int kv_head_num;
        int size_per_head;
    
        Eagle3DraftLayerWeight() = default;};

class Eagle3DraftLayer {
public:
    Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight,
                     UnifiedAttentionLayer*        attn_layer,
                     class Eagle3AttentionLayer*   eagle3_attn_layer,
                     LlamaFfnLayer*                ffn_layer,
                     float                         rmsnorm_eps);

    void Forward(const Tensor& input_hidden,
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
                 cudaStream_t  stream);

    const Tensor& debug_fc_out() const         { return debug_fc_out_; }
    const Tensor& debug_attn_out() const       { return debug_attn_out_; }
    const Tensor& debug_ffn_out() const        { return debug_ffn_out_; }
    const Tensor& debug_pre_head_hidden() const{ return debug_pre_head_hidden_; }
    const Tensor& debug_qkv() const            { return debug_qkv_; }

private:
    const Eagle3DraftLayerWeight* weight_{nullptr};
    LlamaFfnLayer*                ffn_layer_{nullptr};
    float                         rmsnorm_eps_{1e-5f};

    Tensor debug_fc_out_;
    Tensor debug_attn_out_;
    Tensor debug_ffn_out_;
    Tensor debug_pre_head_hidden_;
    Tensor debug_qkv_;

    UnifiedAttentionLayer* attn_layer_{nullptr};
    class Eagle3AttentionLayer* eagle3_attn_layer_{nullptr};
    int                    head_num_{0};
    int                    kv_head_num_{0};
    int                    size_per_head_{0};
    bool                   attn_geom_ok_{true};
    bool is_qkv_compatible_() const;
    int draft_hidden_dim_{0};
    int base_hidden_dim_{0};
};

}  // namespace turbomind
