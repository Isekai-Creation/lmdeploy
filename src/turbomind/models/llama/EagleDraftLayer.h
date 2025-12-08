// Copyright (c) OpenMMLab. All rights reserved.
// Shared Eagle3 draft layer metadata used by EagleModule and UnifiedDecoder.

#pragma once

#include "src/turbomind/models/llama/LlamaFfnLayer.h"

namespace turbomind {

struct Eagle3DraftLayerWeight {
    LlamaAttentionWeight attn;
    LlamaFfnWeight       ffn;
    Tensor               input_norm;
    Tensor               post_attn_norm;
    Tensor               output_norm;

    Eagle3DraftLayerWeight() = default;
};

// Lightweight Eagle3 draft layer wrapper. For now this is a structural
// placeholder that allows UnifiedDecoder to own a dedicated Eagle3 draft
// layer object; future iterations will reuse the attention/FFN weights
// above to run a true Eagle3 decoder layer.
class Eagle3DraftLayer {
public:
    Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight, LlamaFfnLayer* ffn_layer, float rmsnorm_eps);

    void Forward(const Tensor& input_hidden, Tensor& output_hidden, cudaStream_t stream);

    const Tensor& debug_attn_out() const
    {
        return debug_attn_out_;
    }

    const Tensor& debug_ffn_out() const
    {
        return debug_ffn_out_;
    }

    const Tensor& debug_pre_head_hidden() const
    {
        return debug_pre_head_hidden_;
    }

private:
    const Eagle3DraftLayerWeight* weight_{nullptr};
    LlamaFfnLayer*                ffn_layer_{nullptr};
    float                         rmsnorm_eps_{1e-5f};

    Tensor debug_attn_out_;
    Tensor debug_ffn_out_;
    Tensor debug_pre_head_hidden_;
};

}  // namespace turbomind
