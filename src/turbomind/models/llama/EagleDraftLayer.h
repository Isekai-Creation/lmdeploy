// Copyright (c) OpenMMLab. All rights reserved.
// Shared Eagle3 draft layer metadata used by EagleModule and UnifiedDecoder.

#pragma once

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

struct Eagle3DraftLayerWeight {
    LlamaAttentionWeight attn;
    LlamaFfnWeight       ffn;
    Tensor               fc_weight;      // keep if your converter fills it
    Tensor               input_norm;
    Tensor               post_attn_norm;
    Tensor               output_norm;

    Eagle3DraftLayerWeight() = default;
};

class Eagle3DraftLayer {
public:
    Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight,
                     UnifiedAttentionLayer*        attn_layer,
                     LlamaFfnLayer*                ffn_layer,
                     float                         rmsnorm_eps);

    void Forward(const Tensor& input_hidden,
                 Tensor&       output_hidden,
                 cudaStream_t  stream);

    const Tensor& debug_fc_out() const         { return debug_fc_out_; }
    const Tensor& debug_attn_out() const       { return debug_attn_out_; }
    const Tensor& debug_ffn_out() const        { return debug_ffn_out_; }
    const Tensor& debug_pre_head_hidden() const{ return debug_pre_head_hidden_; }

private:
    const Eagle3DraftLayerWeight* weight_{nullptr};
    LlamaFfnLayer*                ffn_layer_{nullptr};
    float                         rmsnorm_eps_{1e-5f};

    Tensor debug_fc_out_;
    Tensor debug_attn_out_;
    Tensor debug_ffn_out_;
    Tensor debug_pre_head_hidden_;

    UnifiedAttentionLayer* attn_layer_{nullptr};
    int                    head_num_{0};
    int                    kv_head_num_{0};
    int                    size_per_head_{0};
};

}  // namespace turbomind
