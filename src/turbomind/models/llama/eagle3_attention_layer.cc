// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle3_attention_layer.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

namespace {

template<typename T>
__device__ inline float to_float(T x)
{
    return static_cast<float>(x);
}

template<>
__device__ inline float to_float<half_t>(half_t x)
{
    return __half2float(x);
}

#if ENABLE_BF16
template<>
__device__ inline float to_float<bfloat16_t>(bfloat16_t x)
{
    return __bfloat162float(x);
}
#endif

template<typename T>
__device__ inline T from_float(float x);

template<>
__device__ inline half_t from_float<half_t>(float x)
{
    return __float2half(x);
}

#if ENABLE_BF16
template<>
__device__ inline bfloat16_t from_float<bfloat16_t>(float x)
{
    return __float2bfloat16(x);
}
#endif

template<>
__device__ inline float from_float<float>(float x)
{
    return x;
}

// Apply standard RoPE to Q/K in-place. Position ids are optional; when
// absent we fall back to (past_kv_len + token_idx % q_len).
template<typename T>
__global__ void apply_rope_kernel(T*       q_ptr,
                                  T*       k_ptr,
                                  int      token_num,
                                  int      num_q_heads,
                                  int      num_kv_heads,
                                  int      head_dim,
                                  int      q_len,
                                  int      past_kv_len,
                                  const int* position_ids,
                                  float    rope_base)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q = token_num * num_q_heads * head_dim;
    if (idx >= total_q) {
        return;
    }

    const int d        = idx % head_dim;
    const int head     = (idx / head_dim) % num_q_heads;
    const int token_id = idx / (head_dim * num_q_heads);

    // Apply past_kv_len offset so RoPE aligns with absolute positions.
    const int base_pos = position_ids ? position_ids[token_id] : (q_len > 0 ? token_id % q_len : token_id);
    const int pos      = base_pos + past_kv_len;

    const float inv_freq = powf(rope_base, -2.f * (d / 2) / static_cast<float>(head_dim));
    const float angle    = static_cast<float>(pos) * inv_freq;
    const float cosv     = cosf(angle);
    const float sinv     = sinf(angle);

    auto rotate = [&](T* base, int h, int head_count) {
        T* vec = base + (token_id * head_count + h) * head_dim;
        // Rotate even/odd pairs; guard the tail when head_dim is odd.
        if ((d & 1) == 0 && d + 1 < head_dim) {
            const float x0 = to_float(vec[d]);
            const float x1 = to_float(vec[d + 1]);
            vec[d]         = from_float<T>(x0 * cosv - x1 * sinv);
            vec[d + 1]     = from_float<T>(x0 * sinv + x1 * cosv);
        }
    };

    rotate(q_ptr, head, num_q_heads);
    // Map query head to its kv head and rotate K accordingly.
    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head < num_kv_heads) {
        rotate(k_ptr, kv_head, num_kv_heads);
    }
}

// Naive SDPA for small q_len/kv_len. One thread handles one (token, head)
// pair and loops over kv tokens. This is not performance tuned but keeps
// the math faithful for Eagle-3 draft until a cutlass path is ported.
template<typename T>
__global__ void sdpa_kernel(const T* __restrict__ q_ptr,
                            const T* __restrict__ k_ptr,
                            const T* __restrict__ v_ptr,
                            T* __restrict__       ctx_ptr,
                            int                   token_num,
                            int                   batch_size,
                            int                   q_len,
                            int                   kv_len,
                            int                   num_q_heads,
                            int                   num_kv_heads,
                            int                   head_dim,
                            int                   past_kv_len,
                            const int32_t*        packed_mask,
                            int                   packed_stride)
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = token_num * num_q_heads;
    if (idx >= total) {
        return;
    }

    const int head      = idx % num_q_heads;
    const int token_idx = idx / num_q_heads;
    const int batch     = q_len > 0 ? token_idx / q_len : 0;
    const int q_pos     = q_len > 0 ? token_idx % q_len : token_idx;

    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const T* q = q_ptr + (token_idx * num_q_heads + head) * head_dim;

    // Pass 1: find max logit for numerical stability.
    float max_score = -1e20f;
    for (int j = 0; j < kv_len; ++j) {
        const int kv_token = batch * kv_len + j;
        const T*  k        = k_ptr + (kv_token * num_kv_heads + kv_head) * head_dim;
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        // Causal guard: forbid attending beyond current q position.
        if (j + past_kv_len > q_pos + past_kv_len) {
            dot = -1e20f;
        }
        // Optional packed mask per token.
        if (packed_mask && packed_stride > 0) {
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j / 32];
            const bool    allowed  = (mask_val >> (j & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        max_score = fmaxf(max_score, dot / sqrtf(static_cast<float>(head_dim)));
    }

    // Pass 2: compute softmax denominator.
    float sum_exp = 0.f;
    for (int j = 0; j < kv_len; ++j) {
        const int kv_token = batch * kv_len + j;
        const T*  k        = k_ptr + (kv_token * num_kv_heads + kv_head) * head_dim;
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        if (j + past_kv_len > q_pos + past_kv_len) {
            dot = -1e20f;
        }
        if (packed_mask && packed_stride > 0) {
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j / 32];
            const bool    allowed  = (mask_val >> (j & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        const float score = dot / sqrtf(static_cast<float>(head_dim)) - max_score;
        sum_exp += __expf(score);
    }
    const float inv_denom = sum_exp > 0.f ? (1.f / sum_exp) : 0.f;

    // Pass 3: accumulate context.
    T* ctx = ctx_ptr + (token_idx * num_q_heads + head) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        ctx[d] = from_float<T>(0.f);
    }

    for (int j = 0; j < kv_len; ++j) {
        const int kv_token = batch * kv_len + j;
        const T*  k        = k_ptr + (kv_token * num_kv_heads + kv_head) * head_dim;
        const T*  v        = v_ptr + (kv_token * num_kv_heads + kv_head) * head_dim;
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        if (j + past_kv_len > q_pos + past_kv_len) {
            dot = -1e20f;
        }
        if (packed_mask && packed_stride > 0) {
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j / 32];
            const bool    allowed  = (mask_val >> (j & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        const float score = dot / sqrtf(static_cast<float>(head_dim)) - max_score;
        const float w     = __expf(score) * inv_denom;
        for (int d = 0; d < head_dim; ++d) {
            const float acc = to_float(ctx[d]) + w * to_float(v[d]);
            ctx[d]          = from_float<T>(acc);
        }
    }
}

template<typename T>
void launch_apply_rope(T* q_ptr,
                       T* k_ptr,
                       int token_num,
                       int num_q_heads,
                       int num_kv_heads,
                       int head_dim,
                       int q_len,
                       int past_kv_len,
                       const Tensor* position_ids,
                       cudaStream_t stream)
{
    if (!q_ptr || !k_ptr || token_num <= 0 || head_dim <= 0) {
        return;
    }
    const int total = token_num * num_q_heads * head_dim;
    constexpr int kBlock = 256;
    const int grid       = (total + kBlock - 1) / kBlock;
    const int* pos_ptr   = position_ids ? position_ids->data<int>() : nullptr;
    // LLaMA-style base; TRT uses the same default for Eagle-3.
    constexpr float rope_base = 10000.f;
    apply_rope_kernel<T><<<grid, kBlock, 0, stream>>>(
        q_ptr, k_ptr, token_num, num_q_heads, num_kv_heads, head_dim, q_len, past_kv_len, pos_ptr, rope_base);
}

template<typename T>
void launch_sdpa(const T* q_ptr,
                 const T* k_ptr,
                 const T* v_ptr,
                 T*       ctx_ptr,
                 int      token_num,
                 int      batch_size,
                 int      q_len,
                 int      kv_len,
                 int      num_q_heads,
                 int      num_kv_heads,
                 int      head_dim,
                 int      past_kv_len,
                 const Tensor* packed_mask,
                 int      packed_mask_stride,
                 cudaStream_t stream)
{
    if (!q_ptr || !k_ptr || !v_ptr || !ctx_ptr || token_num <= 0 || q_len <= 0 || kv_len <= 0) {
        return;
    }
    const int packed_stride =
        packed_mask_stride > 0 ? packed_mask_stride : (packed_mask && packed_mask->ndim() >= 2 ? packed_mask->shape(1) : 0);
    const int total         = token_num * num_q_heads;
    constexpr int kBlock    = 256;
    const int grid          = (total + kBlock - 1) / kBlock;
    const int32_t* packed   = packed_mask ? packed_mask->data<int32_t>() : nullptr;
    sdpa_kernel<T><<<grid, kBlock, 0, stream>>>(q_ptr,
                                                k_ptr,
                                                v_ptr,
                                                ctx_ptr,
                                                token_num,
                                                batch_size,
                                                q_len,
                                                kv_len,
                                                num_q_heads,
                                                num_kv_heads,
                                                head_dim,
                                                past_kv_len,
                                                packed,
                                                packed_stride);
}

}  // namespace

Eagle3AttentionLayer::Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream):
    stream_{stream},
    device_prop_{prop}
{
    (void)device_prop_;
}

void Eagle3AttentionLayer::Forward(Eagle3AttentionParam& param)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!param.input || !param.weights || !param.weights->is_initialized) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid Eagle3AttentionParam; "
            "treating Eagle-3 attention as pass-through for this step.");
        if (param.input) {
            param.output = param.input;
        }
        return;
    }

    const auto& w = *param.weights;

    const int token_num = param.input.shape(0);
    const int q_in_dim  = param.input.shape(1);
    const auto dtype    = param.input.dtype();

    const int num_q_heads  = w.num_q_heads;
    const int num_kv_heads = w.num_kv_heads;
    const int head_dim     = w.head_dim;

    const int q_out_dim  = w.q_out;
    const int kv_out_dim = w.kv_out;

    if (token_num <= 0 || q_in_dim <= 0 || q_out_dim <= 0 || kv_out_dim <= 0 || num_q_heads <= 0
        || num_kv_heads <= 0 || head_dim <= 0) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid geometry (token=%d, q_in=%d, q_out=%d, kv_out=%d, "
            "q_heads=%d, kv_heads=%d, head_dim=%d); treating as pass-through.",
            token_num,
            q_in_dim,
            q_out_dim,
            kv_out_dim,
            num_q_heads,
            num_kv_heads,
            head_dim);
        param.output = param.input;
        return;
    }

    // Ensure output buffer matches the expected shape from the caller.
    if (!param.output || param.output.ndim() != 2 || param.output.shape(0) != token_num
        || param.output.shape(1) <= 0 || param.output.dtype() != dtype
        || param.output.device().type != param.input.device().type) {
        param.output = Tensor{{token_num, q_in_dim}, dtype, param.input.device()};
    }

    // Fallback when q_in does not line up with weights.
    if (q_in_dim != w.q_in || w.q_proj.dtype() != dtype || w.k_proj.dtype() != dtype || w.v_proj.dtype() != dtype) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] q_in/weight dtype mismatch (q_in=%d vs weight=%d, dtype=%s/%s); "
            "pass-through.",
            q_in_dim,
            w.q_in,
            to_string(dtype),
            to_string(w.q_proj.dtype()));
        param.output = param.input;
        return;
    }

    const int batch_size = param.batch_size > 0 ? param.batch_size : token_num;
    const int q_len      = param.q_len > 0 ? param.q_len : (batch_size > 0 ? token_num / batch_size : 1);
    const int kv_len     = param.kv_len > 0 ? param.kv_len : q_len;
    const int group_size = std::max(1, num_q_heads / num_kv_heads);

    // Temporary buffers for Q/K/V and context.
    Tensor q{{token_num, q_out_dim}, dtype, param.input.device()};
    Tensor k{{token_num, kv_out_dim}, dtype, param.input.device()};
    Tensor v{{token_num, kv_out_dim}, dtype, param.input.device()};
    Tensor ctx{{token_num, q_out_dim}, dtype, param.input.device()};

    auto do_forward = [&](auto tag) {
        using T = decltype(tag);
        const T* x_ptr  = param.input.data<T>();
        const T* wq_ptr = w.q_proj.data<T>();
        const T* wk_ptr = w.k_proj.data<T>();
        const T* wv_ptr = w.v_proj.data<T>();
        auto matmul = [&](const T* A, const T* B, T* C, int M, int K, int N) {
            if constexpr (std::is_same_v<T, half_t>) {
                launch_eagle3_matmul_rowmajor_half(
                    reinterpret_cast<const half_t*>(A),
                    reinterpret_cast<const half_t*>(B),
                    reinterpret_cast<half_t*>(C),
                    M,
                    K,
                    N,
                    stream_);
            }
#if ENABLE_BF16
            else if constexpr (std::is_same_v<T, bfloat16_t>) {
                launch_eagle3_matmul_rowmajor_bf16(
                    reinterpret_cast<const bfloat16_t*>(A),
                    reinterpret_cast<const bfloat16_t*>(B),
                    reinterpret_cast<bfloat16_t*>(C),
                    M,
                    K,
                    N,
                    stream_);
            }
#endif
#if ENABLE_FP32
            else if constexpr (std::is_same_v<T, float>) {
                launch_eagle3_matmul_rowmajor_float(
                    reinterpret_cast<const float*>(A),
                    reinterpret_cast<const float*>(B),
                    reinterpret_cast<float*>(C),
                    M,
                    K,
                    N,
                    stream_);
            }
#endif
            else {
                TM_LOG_WARNING(
                    "[EAGLE3][Attention][fallback] unsupported dtype in matmul; treating as pass-through.");
            }
        };

        // Q/K/V projections.
        matmul(x_ptr, wq_ptr, q.data<T>(), token_num, q_in_dim, q_out_dim);
        matmul(x_ptr, wk_ptr, k.data<T>(), token_num, q_in_dim, kv_out_dim);
        matmul(x_ptr, wv_ptr, v.data<T>(), token_num, q_in_dim, kv_out_dim);
        sync_check_cuda_error();

        // Optional RoPE on Q/K.
        launch_apply_rope<T>(q.data<T>(),
                             k.data<T>(),
                             token_num,
                             num_q_heads,
                             num_kv_heads,
                             head_dim,
                             q_len,
                             param.past_kv_len,
                             param.position_ids,
                             stream_);
        sync_check_cuda_error();

        // SDPA (naive) over kv_len tokens.
        launch_sdpa<T>(q.data<T>(),
                       k.data<T>(),
                       v.data<T>(),
                       ctx.data<T>(),
                       token_num,
                       batch_size,
                       q_len,
                       kv_len,
                       num_q_heads,
                       num_kv_heads,
                       head_dim,
                       param.past_kv_len,
                       param.packed_mask,
                       param.packed_mask_stride,
                       stream_);
        sync_check_cuda_error();

        // Optional debug capture of QKV/attn_out.
        if (param.debug_qkv) {
            *param.debug_qkv = Tensor{{token_num, q_out_dim + 2 * kv_out_dim}, q.dtype(), q.device()};
            // Layout: [Q | K | V] concatenated along the last dim.
            Tensor& dbg = *param.debug_qkv;
            check_cuda_error(cudaMemcpyAsync(
                dbg.raw_data(), q.raw_data(), q.byte_size(), cudaMemcpyDeviceToDevice, stream_));
            check_cuda_error(cudaMemcpyAsync(static_cast<char*>(dbg.raw_data()) + q.byte_size(),
                                             k.raw_data(),
                                             k.byte_size(),
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(static_cast<char*>(dbg.raw_data()) + q.byte_size() + k.byte_size(),
                                             v.raw_data(),
                                             v.byte_size(),
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        if (param.debug_attn_out) {
            *param.debug_attn_out = ctx;
        }

        // Final Wo projection. Prefer fused LLaMA-style Wo when present;
        // fall back to native o_proj otherwise.
        const Tensor* wo = param.attn_weights ? &param.attn_weights->output.weight : nullptr;
        if (!wo || !(*wo)) {
            wo = &w.o_proj;
        }

        if (!wo || !(*wo) || wo->ndim() != 2) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] missing Wo; returning context without projection.");
            param.output = ctx;
            return;
        }

        const int wo_in  = wo->shape(0);
        const int wo_out = wo->shape(1);
        if (wo_in != q_out_dim) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] Wo shape mismatch (Wo=[%d,%d], q_out=%d); "
                "returning context.",
                wo_in,
                wo_out,
                q_out_dim);
            param.output = ctx;
            return;
        }
        if (wo->dtype() != ctx.dtype()) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] Wo dtype mismatch (Wo=%s, ctx=%s); returning context.",
                to_string(wo->dtype()),
                to_string(ctx.dtype()));
            param.output = ctx;
            return;
        }

        if (!param.output || param.output.ndim() != 2 || param.output.shape(0) != token_num
            || param.output.shape(1) != wo_out || param.output.dtype() != ctx.dtype()
            || param.output.device().type != ctx.device().type) {
            param.output = Tensor{{token_num, wo_out}, ctx.dtype(), ctx.device()};
        }

        if constexpr (std::is_same_v<T, half_t>) {
            launch_eagle3_matmul_rowmajor_half(
                reinterpret_cast<const half_t*>(ctx.data<T>()),
                reinterpret_cast<const half_t*>(wo->data<T>()),
                reinterpret_cast<half_t*>(param.output.data<T>()),
                token_num,
                wo_in,
                wo_out,
                stream_);
        }
#if ENABLE_BF16
        else if constexpr (std::is_same_v<T, bfloat16_t>) {
            launch_eagle3_matmul_rowmajor_bf16(ctx.data<T>(), wo->data<T>(), param.output.data<T>(), token_num, wo_in, wo_out, stream_);
        }
#endif
#if ENABLE_FP32
        else if constexpr (std::is_same_v<T, float>) {
            launch_eagle3_matmul_rowmajor_float(ctx.data<T>(), wo->data<T>(), param.output.data<T>(), token_num, wo_in, wo_out, stream_);
        }
#endif
        else {
            TM_LOG_WARNING("[EAGLE3][Attention][fallback] unsupported dtype for Wo projection; returning context.");
            param.output = ctx;
        }
        sync_check_cuda_error();
    };

    if (dtype == kFloat16) {
        do_forward(half_t{});
    }
#if ENABLE_BF16
    else if (dtype == kBfloat16) {
        do_forward(bfloat16_t{});
    }
#endif
#if ENABLE_FP32
    else if (dtype == kFloat32) {
        do_forward(float{});
    }
    else
#endif
    {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] unsupported Eagle3Attention dtype=%s; treating as pass-through.",
            to_string(dtype));
        param.output = param.input;
    }
}

}  // namespace turbomind
