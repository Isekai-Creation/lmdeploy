// Copyright (c) OpenMMLab. All rights reserved.
// Eagle-3 attention layer scaffolding. This backend is dedicated to
// Eagle-3 geometry (4096x2880 / 512x2880 / 2880x4096) and is wired
// into Eagle3DraftLayer, separate from UnifiedAttentionLayer.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/Eagle3AttentionWeight.h"

namespace turbomind {

struct Eagle3AttentionParam {
    // Input hidden after Eagle-3 pre-FC and norm.
    // Shape: [B, q_in] (e.g. [B, 2880]).
    Tensor input;

    // Output hidden after attention. In the initial scaffolding we keep
    // this the same shape as input; once the full Eagle-3 math is
    // ported we may choose to expose [B, q_out] instead and handle any
    // projection back to draft hidden outside.
    Tensor output;

    const Eagle3AttentionWeight* weights{nullptr};

    // Optional packed tree mask for Eagle-3 target-tree or multi-token
    // scenarios. Unused in the initial scaffolding.
    const Tensor* packed_mask{nullptr};

    int layer_id{0};
};

class Eagle3AttentionLayer {
public:
    Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream);
    ~Eagle3AttentionLayer() = default;

    // Forward currently acts as a validated pass-through; the real
    // Eagle-3 attention math (Q/K/V projections, RoPE, SDPA, KV reuse)
    // will be ported here from TensorRT-LLM.
    void Forward(Eagle3AttentionParam& param);

private:
    cudaStream_t           stream_{};
    const cudaDeviceProp*  device_prop_{nullptr};
};

}  // namespace turbomind

