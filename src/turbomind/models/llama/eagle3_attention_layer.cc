// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle3_attention_layer.h"

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

Eagle3AttentionLayer::Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream):
    stream_{stream},
    device_prop_{prop}
{
    (void)device_prop_;
}

void Eagle3AttentionLayer::Forward(Eagle3AttentionParam& param)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!param.input || !param.output || !param.weights || !param.weights->is_initialized) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid Eagle3AttentionParam; "
            "treating Eagle-3 attention as pass-through for this step.");
        if (param.input && param.output) {
            // Best-effort pass-through: copy input into output when the
            // caller has already allocated output.
            core::Copy(param.input, param.output);
        }
        return;
    }

    const int batch_size = param.input.shape(0);
    const int q_in_dim   = param.input.shape(1);

    if (batch_size <= 0 || q_in_dim <= 0) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] non-positive batch or hidden dim "
            "(batch=%d, q_in=%d); treating as pass-through.",
            batch_size,
            q_in_dim);
        if (param.input && param.output) {
            core::Copy(param.input, param.output);
        }
        return;
    }

    if (q_in_dim != param.weights->q_in) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] input dim (%d) != q_in (%d); "
            "treating Eagle-3 attention as pass-through.",
            q_in_dim,
            param.weights->q_in);
        if (param.input && param.output) {
            core::Copy(param.input, param.output);
        }
        return;
    }

    // TODO: Implement real Eagle-3 attention here by porting the math
    // from TensorRT-LLM:
    //
    // 1) Q = X @ Wq^T   : [B, q_in] @ [q_in, q_out]   -> [B, q_out]
    // 2) K = X @ Wk^T   : [B, q_in] @ [q_in, kv_out]  -> [B, kv_out]
    // 3) V = X @ Wv^T   : [B, q_in] @ [q_in, kv_out]  -> [B, kv_out]
    // 4) Reshape Q/K/V into [B, num_heads, head_dim] / [B, num_kv_heads, head_dim].
    // 5) Apply RoPE to Q/K as in GPT-OSS/Eagle-3.
    // 6) Compute SDPA(Q, K, V) with optional packed tree mask.
    // 7) Context @ o_proj^T -> [B, q_in] or [B, q_out] depending on design.
    //
    // For now, we keep behaviour safe by copying input into output.
    if (param.input && param.output) {
        core::Copy(param.input, param.output);
    }
}

}  // namespace turbomind
