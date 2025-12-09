// Copyright (c) OpenMMLab. All rights reserved.
// Eagle-3 attention layer. This backend is dedicated to Eagle-3
// geometry (e.g. 4096x2880 / 512x2880 / 2880x4096) and is wired
// into Eagle3DraftLayer, separate from UnifiedAttentionLayer.

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/Eagle3AttentionWeight.h"
#include "src/turbomind/models/llama/LlamaDecoderLayerWeight.h"

namespace turbomind {

// Lightweight GEMM helpers implemented in eagle3_attention_kernels.cu.
void launch_eagle3_matmul_rowmajor_half(const half_t* A,
                                        const half_t* B,
                                        half_t*       C,
                                        int           M,
                                        int           K,
                                        int           N,
                                        cudaStream_t  stream);

#if ENABLE_BF16
void launch_eagle3_matmul_rowmajor_bf16(const bfloat16_t* A,
                                        const bfloat16_t* B,
                                        bfloat16_t*       C,
                                        int               M,
                                        int               K,
                                        int               N,
                                        cudaStream_t      stream);
#endif

void launch_eagle3_matmul_rowmajor_float(const float* A,
                                         const float* B,
                                         float*       C,
                                         int          M,
                                         int          K,
                                         int          N,
                                         cudaStream_t stream);

struct Eagle3AttentionParam {
    // Input hidden after Eagle-3 pre-FC and norm.
    // Shape: [B, q_in] (e.g. [B, 2 * draft_hidden]).
    Tensor input;

    // Output hidden after attention. Draft layer preallocates this to
    // the base-hidden width (Wo_out) so we honour the provided shape.
    Tensor output;

    // Optional fused LLaMA-style attention weights (for Wo projection)
    // used alongside the native Eagle3 projections.
    const LlamaAttentionWeight* attn_weights{nullptr};

    const Eagle3AttentionWeight* weights{nullptr};

    // Optional per-token position ids for RoPE offsetting.
    const Tensor* position_ids{nullptr};

    // Optional packed tree mask for Eagle-3 target-tree or multi-token
    // scenarios. Unused in the current single-token implementation.
    const Tensor* packed_mask{nullptr};
    int           packed_mask_stride{0};

    // Token geometry for multi-position draft attention.
    int batch_size{0};
    int q_len{1};
    int kv_len{1};
    int past_kv_len{0};

    int layer_id{0};

    // Optional debug captures.
    Tensor* debug_qkv{nullptr};
    Tensor* debug_attn_out{nullptr};
};

class Eagle3AttentionLayer {
public:
    Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream);
    ~Eagle3AttentionLayer() = default;

    // Forward applies a simplified Eagle-3 attention block:
    //   Q = input @ q_proj^T
    //   Y = Q @ o_proj^T
    // This uses the native Eagle-3 midlayer projections without trying
    // to squeeze them into the standard LLaMA fused QKV layout. The K
    // and V projections are reserved for future full SDPA wiring.
    void Forward(Eagle3AttentionParam& param);

private:
    cudaStream_t           stream_{};
    const cudaDeviceProp*  device_prop_{nullptr};
};

}  // namespace turbomind
