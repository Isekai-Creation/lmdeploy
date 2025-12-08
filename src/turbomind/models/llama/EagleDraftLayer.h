// Copyright (c) OpenMMLab. All rights reserved.
// Shared Eagle3 draft layer metadata used by EagleModule and UnifiedDecoder.

#pragma once

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

struct Eagle3DraftLayerWeight {
    LlamaAttentionWeight attn;
    LlamaFfnWeight       ffn;
    Tensor               fc_weight;       // optional capture/FC projection
    Tensor               input_norm;
    Tensor               post_attn_norm;
    Tensor               output_norm;

    Eagle3DraftLayerWeight() = default;
};

// Eagle3 draft layer wrapper.
//
// This layer is intended to mirror a LLaMA decoder block:
//
//   input -> RMSNorm(input_norm)
//         -> attention(attn)
//         -> RMSNorm(post_attn_norm)
//         -> FFN(ffn)
//         -> residual + RMSNorm(output_norm)
//
// For v1 we run attention via UnifiedAttentionLayer when geometry is
// valid, and fall back to a guarded shallow QKV + Wo path otherwise.
class Eagle3DraftLayer {
public:
    Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight,
                     UnifiedAttentionLayer*        attn_layer,
                     LlamaFfnLayer*                ffn_layer,
                     float                         rmsnorm_eps);

    void Forward(const Tensor& input_hidden,
                 Tensor&       output_hidden,
                 cudaStream_t  stream);

    const Tensor& debug_fc_out() const
    {
        return debug_fc_out_;
    }

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
    // Weights and helpers
    const Eagle3DraftLayerWeight* weight_{nullptr};
    UnifiedAttentionLayer*        attn_layer_{nullptr};
    LlamaFfnLayer*                ffn_layer_{nullptr};
    float                         rmsnorm_eps_{1e-5f};

    // Debug / inspection tensors
    Tensor debug_fc_out_;          // post input_norm
    Tensor debug_attn_out_;        // attention output
    Tensor debug_ffn_out_;         // FFN output (pre-residual norm)
    Tensor debug_pre_head_hidden_; // final hidden fed into LM head
};

}  // namespace turbomind
