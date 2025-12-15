// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/core/data_type.h"

namespace turbomind {
namespace ft {

// Tunable constants for Eagle3 FMHA. These are kept modest to
// balance parallelism and resource usage while enabling multi-CTA
// tiling over long KV spans. The heavy FMHA kernels are optionally
// compiled behind TM_ENABLE_EAGLE3_FMHA to avoid shared-memory
// overuse on some toolchains while we iterate on other features
// (e.g. FP4 KV cache).
#ifndef TM_ENABLE_EAGLE3_FMHA
#define TM_ENABLE_EAGLE3_FMHA 0
#endif

constexpr int kEagle3FmhaBlockSizeMax = 256;   // max threads per CTA in FMHA kernels
constexpr int kEagle3FmhaBlockSize    = 128;   // default CTA size for tiled FMHA launch
constexpr int kEagle3FmhaMaxHeadDim   = 128;   // maximum head_dim supported by FMHA path
constexpr int kEagle3FmhaMaxTiles     = 8;     // max tiles per (token, head) for multi-CTA KV

// Optional global tile statistics for debugging / perf analysis.
// Tile index layout for span/mask/execute buckets.
enum : int {
    kEagle3FmhaTilesTotal     = 0,
    kEagle3FmhaTilesSpanEmpty = 1,
    kEagle3FmhaTilesMaskEmpty = 2,
    kEagle3FmhaTilesExecuted  = 3,
    kEagle3FmhaTilesCount     = 4,
};

// Device-side storage for the counters.
__device__ unsigned long long g_eagle3_fmha_tile_stats[kEagle3FmhaTilesCount] = {};

cudaError_t GetEagle3FmhaTileStats(void** dev_ptr)
{
    return cudaGetSymbolAddress(dev_ptr, g_eagle3_fmha_tile_stats);
}


namespace { // anonymous namespace for device helpers

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

} // anonymous namespace

// Simple row-major GEMM kernel computing C = A @ B^T.
//   A: [M, K]
//   B: [N, K]
//   C: [M, N]
template<typename T>
__global__ void MatmulRowMajorKernel(const T* __restrict__ A,
                                     const T* __restrict__ B,
                                     T* __restrict__       C,
                                     int                   M,
                                     int                   K,
                                     int                   N)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // Corrected

    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const float a = to_float(A[row * K + k]);
        const float b = to_float(B[col * K + k]);  // B is [N, K], row-major
        acc += a * b;
    }

    C[row * N + col] = from_float<T>(acc);
}


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
                                  float    rope_base,
                                  float    rope_scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_q = token_num * num_q_heads * head_dim;
    if (idx >= total_q) {
        return;
    }

    const int d        = idx % head_dim;
    const int head     = (idx / head_dim) % num_q_heads;
    const int token_id = idx / (head_dim * num_q_heads);

    const int base_pos =
        position_ids ? position_ids[token_id] : (past_kv_len + (q_len > 0 ? token_id % q_len : token_id));

    const int   pair_idx = d >> 1;
    const float inv_freq =
        rope_scale * __powf(rope_base, -static_cast<float>(pair_idx) / (static_cast<float>(head_dim) * 0.5f));
    const float angle = static_cast<float>(base_pos) * inv_freq;
    float       sinv, cosv;
    __sincosf(angle, &sinv, &cosv);

    auto rotate = [&](T* base, int h, int head_count) {
        T* vec = base + (token_id * head_count + h) * head_dim;
        if ((d & 1) == 0 && d + 1 < head_dim) {
            const float x0 = to_float(vec[d]);
            const float x1 = to_float(vec[d + 1]);
            vec[d]         = from_float<T>(x0 * cosv - x1 * sinv);
            vec[d + 1]     = from_float<T>(x0 * sinv + x1 * cosv);
        }
    };

    rotate(q_ptr, head, num_q_heads);
    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head < num_kv_heads) {
        rotate(k_ptr, kv_head, num_kv_heads);
    }
}

template<typename T, int kMaxHeadDim>
__global__ void sdpa_streaming_kernel(const T* __restrict__ q_ptr,
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
                                      const int*            position_ids,
                                      const int32_t*        packed_mask,
                                      int                   packed_stride,
                                      const int32_t*        runtime_offsets,
                                      const int32_t*        tree_offsets,
                                      const int32_t*        kv_lens_runtime,
                                      const int32_t*        successor_offsets,
                                      const int32_t*        successor_counts)
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
    const int q_token_offset = token_idx;

    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const T* q = q_ptr + (token_idx * num_q_heads + head) * head_dim;

    int kv_start = 0;
    int kv_end   = kv_len;
    auto clamp_span = [&](int start, int end) {
        kv_start = max(kv_start, start);
        kv_end   = min(kv_end, end);
    };
    if (runtime_offsets) {
        clamp_span(runtime_offsets[batch], runtime_offsets[batch + 1]);
    }
    if (tree_offsets) {
        clamp_span(tree_offsets[batch], tree_offsets[batch + 1]);
    }
    if (kv_lens_runtime) {
        clamp_span(0, kv_lens_runtime[batch]);
    }
    if (packed_mask && packed_stride > 0) {
        clamp_span(kv_start, min(kv_end, packed_stride * 32));
    }

    kv_start        = max(0, min(kv_start, kv_len));
    kv_end          = max(0, min(kv_end, kv_len));
    const int kv_len_slot = kv_end - kv_start;

    if (kv_len_slot <= 0) {
        return;
    }

    const int   pos_idx_q  = position_ids ? min(q_token_offset, token_num - 1) : q_token_offset;
    const int   q_position = position_ids ? position_ids[pos_idx_q] : (past_kv_len + kv_start + q_pos);
    const float inv_sqrt   = rsqrtf(static_cast<float>(head_dim));

    if (head_dim > kMaxHeadDim) {
        return;
    }

    // Cache Q in registers to avoid reloading it from global memory for
    // every KV position. head_dim is small (e.g. 64), so this is cheap.
    float q_reg[kMaxHeadDim];
    float ctx_acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        q_reg[d]  = to_float(q[d]);
        ctx_acc[d] = 0.f;
    }

    float m     = -1e20f;
    float sum_w = 0.f;

    for (int j = 0; j < kv_len_slot; ++j) {
        const int k_token_offset = batch * kv_len + (kv_start + j);
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;

        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_reg[d] * to_float(k[d]);
        }

        const int   pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int   kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
        bool        allowed_pos = kv_position <= q_position;
        if (!allowed_pos) {
            continue;
        }

        if (packed_mask && packed_stride > 0) {
            const int     j_off_mask = kv_start + j;
            const int32_t mask_val   = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed    = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                continue;
            }
        }

        const float score = dot * inv_sqrt;

        if (sum_w == 0.f && m <= -1e19f) {
            // First valid element.
            const int v_token_offset = batch * kv_len + (kv_start + j);
            const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;

            m     = score;
            sum_w = 1.f;
            for (int d = 0; d < head_dim; ++d) {
                ctx_acc[d] = to_float(v[d]);
            }
            continue;
        }

        if (score <= m) {
            const float e = __expf(score - m);
            sum_w += e;

            const int v_token_offset = batch * kv_len + (kv_start + j);
            const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                ctx_acc[d] += e * to_float(v[d]);
            }
        }
        else {
            const float e = __expf(m - score);
            sum_w = sum_w * e + 1.f;

            const int v_token_offset = batch * kv_len + (kv_start + j);
            const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                ctx_acc[d] = ctx_acc[d] * e + to_float(v[d]);
            }
            m = score;
        }
    }

    T* ctx = ctx_ptr + (token_idx * num_q_heads + head) * head_dim;
    const float inv_sum = sum_w > 0.f ? (1.f / sum_w) : 0.f;
    for (int d = 0; d < head_dim; ++d) {
        ctx[d] = from_float<T>(ctx_acc[d] * inv_sum);
    }
}

// Tree-aware Eagle3 FMHA-style kernel that consumes precomputed per-token
// KV windows [kv_start_per_token[t], kv_start_per_token[t] +
// kv_len_per_token[t]) and the packed mask. Compared to the scalar
// streaming SDPA kernel, this version parallelizes over the KV span
// within a CTA and uses shared-memory reductions for the softmax stats
// and context, avoiding per-thread serial walks over all KV tokens.
template<typename T, int kMaxHeadDim>
__global__ void eagle3_fmha_kernel(const T* __restrict__ q_ptr,
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
                                   const int*            position_ids,
                                   const int32_t*        packed_mask,
                                   int                   packed_stride,
                                   const int32_t*        kv_start_per_token,
                                   const int32_t*        kv_len_per_token)
{
    const int linear    = blockIdx.x;
    const int head      = linear % num_q_heads;
    const int token_idx = linear / num_q_heads;
    if (token_idx >= token_num) {
        return;
    }

    const int batch     = q_len > 0 ? token_idx / q_len : 0;
    const int q_pos     = q_len > 0 ? token_idx % q_len : token_idx;
    const int q_token_offset = token_idx;

    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const T* q = q_ptr + (token_idx * num_q_heads + head) * head_dim;

    if (!kv_start_per_token || !kv_len_per_token || head_dim <= 0 || head_dim > kMaxHeadDim) {
        return;
    }

    const int kv_start    = kv_start_per_token[token_idx];
    const int kv_len_slot = kv_len_per_token[token_idx];

    if (kv_len_slot <= 0 || kv_start < 0 || kv_start >= kv_len) {
        return;
    }

    const int   pos_idx_q  = position_ids ? min(q_token_offset, token_num - 1) : q_token_offset;
    const int   q_position = position_ids ? position_ids[pos_idx_q] : (past_kv_len + kv_start + q_pos);
    const float inv_sqrt   = rsqrtf(static_cast<float>(head_dim));

    const int kBlockLocal = blockDim.x;

    __shared__ float s_max[kEagle3FmhaBlockSizeMax];
    __shared__ float s_sum[kEagle3FmhaBlockSizeMax];
    __shared__ float s_ctx[kMaxHeadDim][kEagle3FmhaBlockSizeMax];

    const int tid = threadIdx.x;
    if (tid >= kBlockLocal) {
        return;
    }

    // Cache Q in registers.
    float q_reg[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        q_reg[d] = to_float(q[d]);
    }

    // Phase 1: local max over assigned KV positions.
    float local_max = -1e20f;
    for (int j = tid; j < kv_len_slot; j += kBlockLocal) {
        const int k_token_offset = batch * kv_len + (kv_start + j);
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;

        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_reg[d] * to_float(k[d]);
        }

        const int   pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int   kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
        if (kv_position > q_position) {
            continue;
        }

        if (packed_mask && packed_stride > 0) {
            const int     j_off_mask = kv_start + j;
            const int32_t mask_val   = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed    = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                continue;
            }
        }

        const float score = dot * inv_sqrt;
        local_max         = fmaxf(local_max, score);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Reduce max across threads in the CTA.
    for (int stride = kBlockLocal / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    const float m = s_max[0];

    // Phase 2: accumulate weighted values and sum of weights.
    float local_sum = 0.f;
    float local_ctx[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        local_ctx[d] = 0.f;
    }

    if (m > -1e19f) {
        for (int j = tid; j < kv_len_slot; j += kBlockLocal) {
            const int k_token_offset = batch * kv_len + (kv_start + j);
            const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;

            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_reg[d] * to_float(k[d]);
            }

            const int   pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
            const int   kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
            if (kv_position > q_position) {
                continue;
            }

            if (packed_mask && packed_stride > 0) {
                const int     j_off_mask = kv_start + j;
                const int32_t mask_val   = packed_mask[token_idx * packed_stride + j_off_mask / 32];
                const bool    allowed    = (mask_val >> (j_off_mask & 31)) & 1;
                if (!allowed) {
                    continue;
                }
            }

            const float score = dot * inv_sqrt - m;
            const float e     = __expf(score);
            if (e == 0.f) {
                continue;
            }

            local_sum += e;

            const int v_token_offset = batch * kv_len + (kv_start + j);
            const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                local_ctx[d] += e * to_float(v[d]);
            }
        }
    }

    // Store per-thread ctx partials to shared memory (D-major).
    for (int d = 0; d < head_dim; ++d) {
        s_ctx[d][tid] = local_ctx[d];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum of weights.
    for (int stride = kBlockLocal / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    const float sum_w = s_sum[0];
    const float inv_sum = sum_w > 0.f ? (1.f / sum_w) : 0.f;

    // Reduce ctx across threads for each dimension and write out.
    T* ctx = ctx_ptr + (token_idx * num_q_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += kBlockLocal) {
        float acc = 0.f;
        for (int t = 0; t < kBlockLocal; ++t) {
            acc += s_ctx[d][t];
        }
        ctx[d] = from_float<T>(acc * inv_sum);
    }
}

// Multi-CTA, tile-based Eagle3 FMHA kernel (phase 1).
// Each CTA processes a subset ("tile") of the KV span for a given
// (token, head) pair and writes partial max/sum/context into global
// scratch buffers. A second kernel reduces across tiles.
template<typename T, int kMaxHeadDim>
__global__ void eagle3_fmha_multi_cta_kernel1(const T* __restrict__ q_ptr,
                                              const T* __restrict__ k_ptr,
                                              const T* __restrict__ v_ptr,
                                              int                   token_num,
                                              int                   batch_size,
                                              int                   q_len,
                                              int                   kv_len,
                                              int                   num_q_heads,
                                              int                   num_kv_heads,
                                              int                   head_dim,
                                              int                   past_kv_len,
                                              const int*            position_ids,
                                              const int32_t*        packed_mask,
                                              int                   packed_stride,
                                              const int32_t*        kv_start_per_token,
                                              const int32_t*        kv_len_per_token,
                                              float* __restrict__   partial_o,   // [total * num_tiles, head_dim]
                                              float* __restrict__   partial_m,   // [total * num_tiles]
                                              float* __restrict__   partial_l,   // [total * num_tiles]
                                              int                   num_tiles)
{
    const int linear = blockIdx.x;
    const int tile   = blockIdx.y;

    const int total = token_num * num_q_heads;
    if (linear >= total || tile >= num_tiles) {
        return;
    }

    const int     kBlockLocal = blockDim.x;
    const int     tid         = threadIdx.x;
    if (tid >= kBlockLocal) {
        return;
    }

    const int head      = linear % num_q_heads;
    const int token_idx = linear / num_q_heads;
    const int batch     = q_len > 0 ? token_idx / q_len : 0;
    const int q_pos     = q_len > 0 ? token_idx % q_len : token_idx;
    const int q_token_offset = token_idx;

    if (batch < 0 || batch >= batch_size) {
        return;
    }

    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const T* q = q_ptr + (token_idx * num_q_heads + head) * head_dim;

    const int idx_tile = linear * num_tiles + tile;
    if (!kv_start_per_token || !kv_len_per_token || head_dim <= 0 || head_dim > kMaxHeadDim) {
        // Invalid metadata for this token/head: neutral-fill this tile.
        if (tid == 0) {
            partial_m[idx_tile] = -1e20f;
            partial_l[idx_tile] = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = 0.f;
            }
        }
        return;
    }

    const int kv_start_base = kv_start_per_token[token_idx];
    const int kv_len_slot   = kv_len_per_token[token_idx];

    if (kv_len_slot <= 0 || kv_start_base < 0 || kv_start_base >= kv_len) {
        if (tid == 0) {
            partial_m[idx_tile] = -1e20f;
            partial_l[idx_tile] = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = 0.f;
            }
        }
        return;
    }

    const int   pos_idx_q  = position_ids ? min(q_token_offset, token_num - 1) : q_token_offset;
    const int   q_position = position_ids ? position_ids[pos_idx_q] : (past_kv_len + kv_start_base + q_pos);
    const float inv_sqrt   = rsqrtf(static_cast<float>(head_dim));

    const int tile_len      = (kv_len_slot + num_tiles - 1) / num_tiles;
    const int off_in_slot   = tile * tile_len;
    if (off_in_slot >= kv_len_slot) {
        if (tid == 0) {
            partial_m[idx_tile] = -1e20f;
            partial_l[idx_tile] = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = 0.f;
            }
        }
        return;
    }

    const int tile_start      = kv_start_base + off_in_slot;
    const int tile_end        = min(kv_start_base + kv_len_slot, tile_start + tile_len);
    const int tile_len_actual = max(0, tile_end - tile_start);
    if (tile_len_actual <= 0) {
        if (tid == 0) {
            atomicAdd(&g_eagle3_fmha_tile_stats[kEagle3FmhaTilesSpanEmpty], 1ull);
            partial_m[idx_tile] = -1e20f;
            partial_l[idx_tile] = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = 0.f;
            }
        }
        return;
    }

    // Shared buffers for this CTA.
    __shared__ float         s_max[kEagle3FmhaBlockSizeMax];
    __shared__ float         s_sum[kEagle3FmhaBlockSizeMax];
    __shared__ float         s_ctx[kMaxHeadDim][kEagle3FmhaBlockSizeMax];
    __shared__ int           s_tile_has_any;

    // Count this tile for tile statistics; host decides whether to read.
    if (tid == 0) {
        atomicAdd(&g_eagle3_fmha_tile_stats[kEagle3FmhaTilesTotal], 1ull);
    }

    // If a packed mask is present, cheaply detect tiles that have no
    // allowed KV positions at all and neutral-fill them so that phase-2
    // can safely ignore them. This avoids doing dot products for tiles
    // that are fully masked out.
    if (packed_mask && packed_stride > 0) {
        if (tid == 0) {
            int any      = 0;
            const int row = token_idx * packed_stride;

            const int tile_first = tile_start;
            const int tile_last  = tile_start + tile_len_actual; // exclusive

            const int word_start = tile_first / 32;
            const int word_end   = (tile_last + 31) / 32;

            for (int w = word_start; w < word_end && !any; ++w) {
                const int32_t mask_val = packed_mask[row + w];
                if (mask_val == 0) {
                    continue;
                }

                // Restrict to the bits that fall within [tile_first, tile_last).
                const int bit_lo = (w == word_start) ? (tile_first & 31) : 0;
                const int bit_hi =
                    (w == word_end - 1) ? ((tile_last - 1) & 31) : 31;  // inclusive
                const int  bit_count = bit_hi - bit_lo + 1;
                const uint32_t mask  = (bit_count >= 32)
                                           ? 0xFFFFFFFFu
                                           : ((static_cast<uint32_t>(1u) << bit_count) - 1u) << bit_lo;

                if ((static_cast<uint32_t>(mask_val) & mask) != 0u) {
                    any = 1;
                }
            }

            s_tile_has_any = any;
            if (!any) {
                atomicAdd(&g_eagle3_fmha_tile_stats[kEagle3FmhaTilesMaskEmpty], 1ull);
                partial_m[idx_tile] = -1e20f;
                partial_l[idx_tile] = 0.f;
                for (int d = 0; d < head_dim; ++d) {
                    partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = 0.f;
                }
            }
        }
        __syncthreads();
        if (!s_tile_has_any) {
            return;
        }
    }

    // Cache Q in registers.
    float q_reg[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        q_reg[d] = to_float(q[d]);
    }

    // Phase 1: local max over this tile.
    float local_max = -1e20f;
    for (int j = tid; j < tile_len_actual; j += kBlockLocal) {
        const int kv_index       = tile_start + j;
        const int k_token_offset = batch * kv_len + kv_index;
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;

        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_reg[d] * to_float(k[d]);
        }

        const int   pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int   kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_index);
        if (kv_position > q_position) {
            continue;
        }

        if (packed_mask && packed_stride > 0) {
            const int     j_off_mask = kv_index;
            const int32_t mask_val   = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed    = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                continue;
            }
        }

        const float score = dot * inv_sqrt;
        local_max         = fmaxf(local_max, score);
    }

    s_max[tid] = local_max;
    __syncthreads();

    // Reduce max across threads in the CTA.
    for (int stride = kBlockLocal / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    const float m_tile = s_max[0];

    // Phase 2: accumulate weighted values and sum of weights for this tile.
    float local_sum = 0.f;
    float local_ctx[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        local_ctx[d] = 0.f;
    }

    if (m_tile > -1e19f) {
        if (tid == 0) {
            atomicAdd(&g_eagle3_fmha_tile_stats[kEagle3FmhaTilesExecuted], 1ull);
        }
        for (int j = tid; j < tile_len_actual; j += kBlockLocal) {
            const int kv_index       = tile_start + j;
            const int k_token_offset = batch * kv_len + kv_index;
            const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;

            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_reg[d] * to_float(k[d]);
            }

            const int   pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
            const int   kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_index);
            if (kv_position > q_position) {
                continue;
            }

            if (packed_mask && packed_stride > 0) {
                const int     j_off_mask = kv_index;
                const int32_t mask_val   = packed_mask[token_idx * packed_stride + j_off_mask / 32];
                const bool    allowed    = (mask_val >> (j_off_mask & 31)) & 1;
                if (!allowed) {
                    continue;
                }
            }

            const float score = dot * inv_sqrt - m_tile;
            const float e     = __expf(score);
            if (e == 0.f) {
                continue;
            }

            local_sum += e;

            const int v_token_offset = batch * kv_len + kv_index;
            const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                local_ctx[d] += e * to_float(v[d]);
            }
        }
    }

    for (int d = 0; d < head_dim; ++d) {
        s_ctx[d][tid] = local_ctx[d];
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum and context across threads.
    for (int stride = kBlockLocal / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    const float tile_sum = s_sum[0];

    if (tid == 0) {
        partial_m[idx_tile] = m_tile;
        partial_l[idx_tile] = tile_sum;
    }

    // Reduce ctx across threads for each dimension and store to partial_o.
    for (int d = 0; d < head_dim; ++d) {
        float acc = 0.f;
        for (int t = 0; t < kBlockLocal; ++t) {
            acc += s_ctx[d][t];
        }
        if (tid == 0) {
            partial_o[static_cast<size_t>(idx_tile) * head_dim + d] = acc;
        }
    }
}

// Multi-CTA, tile-based Eagle3 FMHA kernel (phase 2).
// Reduces partial outputs across tiles and writes the final
// normalized context vectors into ctx_ptr.
template<typename T>
__global__ void eagle3_fmha_multi_cta_kernel2(const float* __restrict__ partial_o,
                                              const float* __restrict__ partial_m,
                                              const float* __restrict__ partial_l,
                                              T* __restrict__       ctx_ptr,
                                              int                   total,
                                              int                   num_tiles,
                                              int                   head_dim)
{
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }

    const int base_idx = linear * num_tiles;

    float m_global = -1e20f;
    for (int t = 0; t < num_tiles; ++t) {
        const float m_tile = partial_m[base_idx + t];
        m_global           = fmaxf(m_global, m_tile);
    }

    if (m_global <= -1e19f) {
        // No valid contributions; zero out the context.
        T* ctx = ctx_ptr + static_cast<size_t>(linear) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            ctx[d] = from_float<T>(0.f);
        }
        return;
    }

    float sum_global = 0.f;
    for (int t = 0; t < num_tiles; ++t) {
        const float m_tile = partial_m[base_idx + t];
        const float l_tile = partial_l[base_idx + t];
        if (l_tile <= 0.f) {
            continue;
        }
        sum_global += __expf(m_tile - m_global) * l_tile;
    }

    const float inv_sum = sum_global > 0.f ? (1.f / sum_global) : 0.f;

    T* ctx = ctx_ptr + static_cast<size_t>(linear) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        float o = 0.f;
        for (int t = 0; t < num_tiles; ++t) {
            const float m_tile = partial_m[base_idx + t];
            const float scale  = __expf(m_tile - m_global);
            const float val =
                partial_o[(static_cast<size_t>(base_idx + t)) * head_dim + d];
            o += scale * val;
        }
        ctx[d] = from_float<T>(o * inv_sum);
    }
}

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
                            const int*            position_ids,
                            const int32_t*        packed_mask,
                            int                   packed_stride,
                            const int32_t*        runtime_offsets,
                            const int32_t*        tree_offsets,
                            const int32_t*        kv_lens_runtime,
                            const int32_t*        successor_offsets,
                            const int32_t*        successor_counts)
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
    const int q_token_offset = token_idx;

    const int group_size = num_kv_heads > 0 ? (num_q_heads + num_kv_heads - 1) / num_kv_heads : 1;
    const int kv_head    = head / group_size;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const T* q = q_ptr + (token_idx * num_q_heads + head) * head_dim;

    int kv_start = 0;
    int kv_end   = kv_len;
    auto clamp_span = [&](int start, int end) {
        kv_start = max(kv_start, start);
        kv_end   = min(kv_end, end);
    };
    if (runtime_offsets) {
        clamp_span(runtime_offsets[batch], runtime_offsets[batch + 1]);
    }
    if (tree_offsets) {
        clamp_span(tree_offsets[batch], tree_offsets[batch + 1]);
    }
    if (kv_lens_runtime) {
        clamp_span(0, kv_lens_runtime[batch]);
    }
    if (packed_mask && packed_stride > 0) {
        clamp_span(kv_start, min(kv_end, packed_stride * 32));
    }

    kv_start        = max(0, min(kv_start, kv_len));
    kv_end          = max(0, min(kv_end, kv_len));
    const int kv_len_slot = kv_end - kv_start;

    if (kv_len_slot <= 0) {
        return;
    }

    const int   pos_idx_q  = position_ids ? min(q_token_offset, token_num - 1) : q_token_offset;
    const int   q_position = position_ids ? position_ids[pos_idx_q] : (past_kv_len + kv_start + q_pos);
    const float inv_sqrt   = rsqrtf(static_cast<float>(head_dim));

    // Three-pass implementation: max-score, sum-exp, then context accumulation.
    float max_score = -1e20f;
    for (int j = 0; j < kv_len_slot; ++j) {
        const int k_token_offset = batch * kv_len + (kv_start + j);
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;
        float     dot            = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
        if (kv_position > q_position) {
            dot = -1e20f;
        }
        if (packed_mask && packed_stride > 0) {
            const int j_off_mask   = kv_start + j;
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed  = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        max_score = fmaxf(max_score, dot * inv_sqrt);
    }

    float sum_exp = 0.f;
    for (int j = 0; j < kv_len_slot; ++j) {
        const int k_token_offset = batch * kv_len + (kv_start + j);
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;
        float     dot            = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
        if (kv_position > q_position) {
            dot = -1e20f;
        }
        if (packed_mask && packed_stride > 0) {
            const int j_off_mask   = kv_start + j;
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed  = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        const float score = dot * inv_sqrt - max_score;
        sum_exp += __expf(score);
    }
    const float inv_denom = sum_exp > 0.f ? (1.f / sum_exp) : 0.f;

    T* ctx = ctx_ptr + (token_idx * num_q_heads + head) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
        ctx[d] = from_float<T>(0.f);
    }

    for (int j = 0; j < kv_len_slot; ++j) {
        const int k_token_offset = batch * kv_len + (kv_start + j);
        const int v_token_offset = batch * kv_len + (kv_start + j); // Assuming V has same indexing as K
        const T*  k              = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;
        const T*  v              = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
        float     dot            = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset;
        const int kv_position = position_ids ? position_ids[pos_idx_k] : (past_kv_len + kv_start + j);
        if (kv_position > q_position) {
            dot = -1e20f;
        }
        if (packed_mask && packed_stride > 0) {
            const int j_off_mask   = kv_start + j;
            const int32_t mask_val = packed_mask[token_idx * packed_stride + j_off_mask / 32];
            const bool    allowed  = (mask_val >> (j_off_mask & 31)) & 1;
            if (!allowed) {
                dot = -1e20f;
            }
        }
        const float score = dot * inv_sqrt - max_score;
        const float w     = __expf(score) * inv_denom;
        for (int d = 0; d < head_dim; ++d) {
            const float acc = to_float(ctx[d]) + w * to_float(v[d]);
            ctx[d]          = from_float<T>(acc);
        }
    }
}

template<typename T>
__global__ void expand_kv_to_q_kernel(const T* __restrict__ kv,
                                      T* __restrict__ q,
                                      int batch,
                                      int kv_heads,
                                      int head_dim,
                                      int group_size)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx >= total) {
        return;
    }
    int d      = idx % head_dim;
    int kv_idx = (idx / head_dim) % kv_heads;
    int b      = idx / (head_dim * kv_heads);

    const T* src = kv + ((b * kv_heads + kv_idx) * head_dim + d);
    T*       dst = q + ((b * kv_heads * group_size + kv_idx * group_size) * head_dim + d);
    for (int g = 0; g < group_size; ++g) {
        dst[g * head_dim] = *src;
    }
}

template<typename T>
void launch_matmul_rowmajor_impl(const T* A,
                                 const T* B,
                                 T*       C,
                                 int      M,
                                 int      K,
                                 int      N,
                                 cudaStream_t stream)
{
    if (M == 0 || K == 0 || N == 0) {
        return;
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    MatmulRowMajorKernel<T><<<grid, block, 0, stream>>>(A, B, C, M, K, N);
}

template<typename T>
void launch_apply_rope_kernel(T* q_ptr,
                              T* k_ptr,
                              int token_num,
                              int num_q_heads,
                              int num_kv_heads,
                              int head_dim,
                              int q_len,
                              int past_kv_len,
                              const Tensor* position_ids,
                              float rope_base,
                              float rope_scale,
                              cudaStream_t stream)
{
    if (!q_ptr || !k_ptr || token_num <= 0 || head_dim <= 0) {
        return;
    }
    const int total = token_num * num_q_heads * head_dim;
    constexpr int kBlock = 256;
    const int grid       = (total + kBlock - 1) / kBlock;
    const int* pos_ptr   = position_ids ? position_ids->data<int>() : nullptr;
    apply_rope_kernel<T><<<grid, kBlock, 0, stream>>>(
        q_ptr, k_ptr, token_num, num_q_heads, num_kv_heads, head_dim, q_len, past_kv_len, pos_ptr, rope_base, rope_scale);
}

template<typename T>
void launch_sdpa_kernel(const T* q_ptr,
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
                        const Tensor* position_ids,
                        const Tensor* packed_mask,
                        int      packed_mask_stride,
                        const Tensor* runtime_offsets,
                        const Tensor* tree_offsets,
                        const Tensor* kv_lens_runtime,
                        const Tensor* successor_offsets,
                        const Tensor* successor_counts,
                        cudaStream_t stream)
{
    if (!q_ptr || !k_ptr || !v_ptr || !ctx_ptr || token_num <= 0 || q_len <= 0 || kv_len <= 0) {
        return;
    }
    const int packed_stride_val =
        packed_mask_stride > 0 ? packed_mask_stride : (packed_mask && packed_mask->ndim() >= 2 ? packed_mask->shape(1) : 0);
    const int total         = token_num * num_q_heads;
    constexpr int kBlock    = 256;
    const int grid          = (total + kBlock - 1) / kBlock;
    const int32_t* packed   = packed_mask ? packed_mask->data<int32_t>() : nullptr;
    const int* pos_ptr      = position_ids ? position_ids->data<int>() : nullptr;
    const int32_t* runtime  = (runtime_offsets && runtime_offsets->dtype() == kInt32)
                                 ? runtime_offsets->data<int32_t>()
                                 : nullptr;
    const int32_t* tree = (tree_offsets && tree_offsets->dtype() == kInt32) ? tree_offsets->data<int32_t>() : nullptr;
    const int32_t* kv_runtime =
        (kv_lens_runtime && kv_lens_runtime->dtype() == kInt32) ? kv_lens_runtime->data<int32_t>() : nullptr;
    const int32_t* succ_off =
        (successor_offsets && successor_offsets->dtype() == kInt32) ? successor_offsets->data<int32_t>() : nullptr;
    const int32_t* succ_cnt =
        (successor_counts && successor_counts->dtype() == kInt32) ? successor_counts->data<int32_t>() : nullptr;

    constexpr int kMaxHeadDim = 256;
    const char*   env         = std::getenv("TM_ENABLE_EAGLE3_SDPA_STREAMING");
    const bool    use_streaming =
        (env && env[0] != '\0' && head_dim <= kMaxHeadDim && head_dim > 0 && head_dim <= kMaxHeadDim);

    if (use_streaming) {
        sdpa_streaming_kernel<T, kMaxHeadDim><<<grid, kBlock, 0, stream>>>(q_ptr,
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
                                                                           pos_ptr,
                                                                           packed,
                                                                           packed_stride_val,
                                                                           runtime,
                                                                           tree,
                                                                           kv_runtime,
                                                                           succ_off,
                                                                           succ_cnt);
    }
    else {
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
                                                    pos_ptr,
                                                    packed,
                                                    packed_stride_val,
                                                    runtime,
                                                    tree,
                                                    kv_runtime,
                                                    succ_off,
                                                    succ_cnt);
    }
}

template<typename T>
void launch_eagle3_fmha_kernel(const T*       q_ptr,
                               const T*       k_ptr,
                               const T*       v_ptr,
                               T*             ctx_ptr,
                               int            token_num,
                               int            batch_size,
                               int            q_len,
                               int            kv_len,
                               int            num_q_heads,
                               int            num_kv_heads,
                               int            head_dim,
                               int            past_kv_len,
                               const Tensor*  position_ids,
                               const Tensor*  packed_mask,
                               int            packed_mask_stride,
                               const int32_t* kv_start_per_token,
                               const int32_t* kv_len_per_token,
                               float*         partial_o,
                               float*         partial_m,
                               float*         partial_l,
                               int            partial_tiles,
                               cudaStream_t   stream)
{
#if TM_ENABLE_EAGLE3_FMHA
    if (!q_ptr || !k_ptr || !v_ptr || !ctx_ptr || token_num <= 0 || q_len <= 0 || kv_len <= 0
        || !kv_start_per_token || !kv_len_per_token) {
        return;
    }

    const int total = token_num * num_q_heads;

    // Decide whether to use the tiled multi-CTA FMHA path. For Eagle3 we
    // expect head_dim=64 with small q_len (num_spec_tokens) and large KV,
    // so enable the tiled path by default in that regime and keep env
    // overrides for debug/tuning:
    //   TM_ENABLE_EAGLE3_FMHA_TILED  -> force tiled on
    //   TM_DISABLE_EAGLE3_FMHA_TILED -> force tiled off
    const char* tiled_env_on  = std::getenv("TM_ENABLE_EAGLE3_FMHA_TILED");
    const char* tiled_env_off = std::getenv("TM_DISABLE_EAGLE3_FMHA_TILED");
    const bool  force_tiled   = (tiled_env_on && tiled_env_on[0] != '\0');
    const bool  force_scalar  = (tiled_env_off && tiled_env_off[0] != '\0');
    const bool  default_tiled = (head_dim > 0 && head_dim <= kEagle3FmhaMaxHeadDim
                                && head_dim == 64 && kv_len >= 512);
    const bool  use_tiled     = !force_scalar && (force_tiled || default_tiled);

    const int32_t* packed = nullptr;
    int            stride = 0;
    if (packed_mask) {
        if (packed_mask->ndim() >= 2) {
            stride = static_cast<int>(packed_mask->shape(1));
        }
        packed = packed_mask->data<int32_t>();
    }
    if (packed_mask_stride > 0) {
        stride = packed_mask_stride;
    }

    const int* pos_ptr = position_ids ? position_ids->data<int>() : nullptr;

    bool run_tiled = use_tiled;

    auto get_block_size = []() {
        static int cached = 0;
        if (cached > 0) {
            return cached;
        }
        int block = kEagle3FmhaBlockSizeMax;
        if (const char* env = std::getenv("TM_EAGLE3_FMHA_BLOCK")) {
            int v = std::atoi(env);
            if (v >= 32 && v <= kEagle3FmhaBlockSizeMax && (v % 32) == 0) {
                block = v;
            }
        }
        cached = block;
        return cached;
    };

    if (run_tiled) {
        // Multi-CTA tiled path: split each (token, head) KV span into a fixed
        // number of tiles and reduce partial outputs across tiles. To avoid
        // per-step cudaMalloc/cudaFree churn in the hot decode loop, the
        // caller (Eagle3AttentionLayer) is responsible for providing
        // adequately sized scratch buffers partial_o/m/l.
        const int num_tiles = (partial_tiles > 0) ? min(partial_tiles, kEagle3FmhaMaxTiles) : kEagle3FmhaMaxTiles;

        if (!partial_o || !partial_m || !partial_l) {
            TM_LOG_ERROR("[EAGLE3][FMHA] tiled path selected but scratch buffers are null");
            if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_PERF_MODE")) {
                std::abort();
            }
            run_tiled = false;
        }
        else {
            const int block_threads = get_block_size();
            dim3 grid1(total, num_tiles);
            dim3 block1(block_threads);

        eagle3_fmha_multi_cta_kernel1<T, kEagle3FmhaMaxHeadDim><<<grid1, block1, 0, stream>>>(q_ptr,
                                                                                              k_ptr,
                                                                                              v_ptr,
                                                                                              token_num,
                                                                                              batch_size,
                                                                                              q_len,
                                                                                              kv_len,
                                                                                              num_q_heads,
                                                                                              num_kv_heads,
                                                                                              head_dim,
                                                                                              past_kv_len,
                                                                                              pos_ptr,
                                                                                              packed,
                                                                                              stride,
                                                                                              kv_start_per_token,
                                                                                              kv_len_per_token,
                                                                                              partial_o,
                                                                                              partial_m,
                                                                                              partial_l,
                                                                                              num_tiles);
        sync_check_cuda_error();

        const int kBlockReduce = 128;
        const int grid2        = (total + kBlockReduce - 1) / kBlockReduce;

            eagle3_fmha_multi_cta_kernel2<T><<<grid2, kBlockReduce, 0, stream>>>(partial_o,
                                                                             partial_m,
                                                                             partial_l,
                                                                             ctx_ptr,
                                                                             total,
                                                                             num_tiles,
                                                                             head_dim);
        sync_check_cuda_error();
        }
    }

    if (!run_tiled) {
        // Single-CTA fallback over the full KV span.
        const int kBlock = get_block_size();
        const int grid   = (total + kBlock - 1) / kBlock;

        eagle3_fmha_kernel<T, kEagle3FmhaMaxHeadDim><<<grid, kBlock, 0, stream>>>(q_ptr,
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
                                                                                  pos_ptr,
                                                                                  packed,
                                                                                  stride,
                                                                                  kv_start_per_token,
                                                                                  kv_len_per_token);
        sync_check_cuda_error();
    }
    if (!run_tiled) {
        // Single-CTA fallback over the full KV span.
        const int kBlock = get_block_size();
        const int grid   = (total + kBlock - 1) / kBlock;

        eagle3_fmha_kernel<T, kEagle3FmhaMaxHeadDim><<<grid, kBlock, 0, stream>>>(q_ptr,
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
                                                                                  pos_ptr,
                                                                                  packed,
                                                                                  stride,
                                                                                  kv_start_per_token,
                                                                                  kv_len_per_token);
        sync_check_cuda_error();
    }
#else
    (void)q_ptr;
    (void)k_ptr;
    (void)v_ptr;
    (void)ctx_ptr;
    (void)token_num;
    (void)batch_size;
    (void)q_len;
    (void)kv_len;
    (void)num_q_heads;
    (void)num_kv_heads;
    (void)head_dim;
    (void)past_kv_len;
    (void)position_ids;
    (void)packed_mask;
    (void)packed_mask_stride;
    (void)kv_start_per_token;
    (void)kv_len_per_token;
    (void)partial_o;
    (void)partial_m;
    (void)partial_l;
    (void)partial_tiles;
    (void)stream;
    // When TM_ENABLE_EAGLE3_FMHA is disabled we treat the FMHA
    // path as unavailable and rely on the higher-level Eagle3
    // attention layer to fall back to its non-FMHA path.
    TM_LOG_WARNING("[EAGLE3][FMHA] launch_eagle3_fmha_kernel is disabled at compile time "
                   "(TM_ENABLE_EAGLE3_FMHA=0); falling back to non-FMHA Eagle3 attention.");
#endif
}

// Type-specialized wrappers so that external translation units can call
// the Eagle3 FMHA path without instantiating the template themselves.
void launch_eagle3_fmha_kernel_fp16(const half_t*       q_ptr,
                                    const half_t*       k_ptr,
                                    const half_t*       v_ptr,
                                    half_t*             ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream)
{
    launch_eagle3_fmha_kernel<half_t>(q_ptr,
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
                                      position_ids,
                                      packed_mask,
                                      packed_mask_stride,
                                      kv_start_per_token,
                                      kv_len_per_token,
                                      partial_o,
                                      partial_m,
                                      partial_l,
                                      partial_tiles,
                                      stream);
}

#if ENABLE_BF16
void launch_eagle3_fmha_kernel_bf16(const bfloat16_t*   q_ptr,
                                    const bfloat16_t*   k_ptr,
                                    const bfloat16_t*   v_ptr,
                                    bfloat16_t*         ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream)
{
    launch_eagle3_fmha_kernel<bfloat16_t>(q_ptr,
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
                                          position_ids,
                                          packed_mask,
                                          packed_mask_stride,
                                          kv_start_per_token,
                                          kv_len_per_token,
                                          partial_o,
                                          partial_m,
                                          partial_l,
                                          partial_tiles,
                                          stream);
}
#endif

#if ENABLE_FP32
void launch_eagle3_fmha_kernel_fp32(const float*        q_ptr,
                                    const float*        k_ptr,
                                    const float*        v_ptr,
                                    float*              ctx_ptr,
                                    int                 token_num,
                                    int                 batch_size,
                                    int                 q_len,
                                    int                 kv_len,
                                    int                 num_q_heads,
                                    int                 num_kv_heads,
                                    int                 head_dim,
                                    int                 past_kv_len,
                                    const Tensor*       position_ids,
                                    const Tensor*       packed_mask,
                                    int                 packed_mask_stride,
                                    const int32_t*      kv_start_per_token,
                                    const int32_t*      kv_len_per_token,
                                    float*              partial_o,
                                    float*              partial_m,
                                    float*              partial_l,
                                    int                 partial_tiles,
                                    cudaStream_t        stream)
{
    launch_eagle3_fmha_kernel<float>(q_ptr,
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
                                     position_ids,
                                     packed_mask,
                                     packed_mask_stride,
                                     kv_start_per_token,
                                     kv_len_per_token,
                                     partial_o,
                                     partial_m,
                                     partial_l,
                                     partial_tiles,
                                     stream);
}
#endif

template<typename T>
void launch_expand_kv_to_q_kernel(const Tensor& kv, Tensor& q_expanded, int kv_heads, int head_dim, int group_size, cudaStream_t stream)
{
    const int batch = kv.shape(0);
    const int elems = batch * kv_heads * head_dim;
    const int threads = 256;
    const int blocks  = (elems + threads - 1) / threads;
    expand_kv_to_q_kernel<T><<<blocks, threads, 0, stream>>>(
        kv.data<T>(), q_expanded.data<T>(), batch, kv_heads, head_dim, group_size);
}

// Global launchers
namespace {

struct Eagle3GemmWorkspace {
    gemm::Gemm     gemm;
    gemm::Workspace workspace{};
    bool           initialized{false};
};

inline Eagle3GemmWorkspace& get_eagle3_gemm_workspace()
{
    static Eagle3GemmWorkspace ws;
    return ws;
}

template<typename T>
bool launch_eagle3_matmul_gemm(const T* A,
                               const T* B,
                               T*       C,
                               int      M,
                               int      K,
                               int      N,
                               cudaStream_t stream)
{
    using namespace gemm;

    if (M == 0 || K == 0 || N == 0) {
        return true;
    }

    const int sm = getSMVersion();
    if (!(std::is_same_v<T, half_t>
#if ENABLE_BF16
          || std::is_same_v<T, bfloat16_t>
#endif
          )
        || sm < 120) {
        return false;
    }

    auto& ctx = get_eagle3_gemm_workspace();
    if (!ctx.initialized) {
        ctx.workspace.barriers_size   = Gemm::kBarriersSize;
        ctx.workspace.partials_size   = Gemm::kPartialsSize;
        ctx.workspace.tensormaps_size = 8192 * 128;

        check_cuda_error(cudaMalloc(&ctx.workspace.barriers, ctx.workspace.barriers_size));
        check_cuda_error(cudaMalloc(&ctx.workspace.partials, ctx.workspace.partials_size));
        check_cuda_error(cudaMalloc(&ctx.workspace.tensormaps, ctx.workspace.tensormaps_size));
        check_cuda_error(cudaMemsetAsync(ctx.workspace.barriers, 0, ctx.workspace.barriers_size, stream));
        check_cuda_error(cudaMalloc(&ctx.workspace.flags, sizeof(int)));

        ctx.initialized = true;
    }

    Operation op{};
    op.dispatch        = DispatchPolicy::kReuse;
    op.epilogue        = Epilogue::kNone;
    op.quant_a         = QuantDesc{QuantType::kNone, 0};
    op.quant_b         = QuantDesc{QuantType::kNone, 0};
    op.batch_dim       = 0;

    MatrixLayout Adesc{};
    Adesc.type   = data_type_v<T>;
    Adesc.order  = kRowMajor;
    Adesc.rows   = M;
    Adesc.cols   = K;
    Adesc.ld     = K;
    Adesc.pack   = 0;
    Adesc.num    = 0;
    Adesc.offsets = nullptr;
    Adesc.idxs    = nullptr;

    MatrixLayout Bdesc{};
    Bdesc.type   = data_type_v<T>;
    Bdesc.order  = kColMajor;
    Bdesc.rows   = K;
    Bdesc.cols   = N;
    Bdesc.ld     = K;
    Bdesc.pack   = 0;
    Bdesc.num    = 0;
    Bdesc.offsets = nullptr;
    Bdesc.idxs    = nullptr;

    MatrixLayout Cdesc{};
    Cdesc.type   = data_type_v<T>;
    Cdesc.order  = kRowMajor;
    Cdesc.rows   = M;
    Cdesc.cols   = N;
    Cdesc.ld     = N;
    Cdesc.pack   = 0;
    Cdesc.num    = 0;
    Cdesc.offsets = nullptr;
    Cdesc.idxs    = nullptr;

    const void* A_ptr = static_cast<const void*>(A);
    const void* B_ptr = static_cast<const void*>(B);
    void*       D_ptr = static_cast<void*>(C);

    const void* U_ptr = nullptr;
    const void* V_ptr = nullptr;
    MatrixLayout Udesc{};
    MatrixLayout Vdesc{};

    const void* C_in_ptr = nullptr;

    const int ec = ctx.gemm.Run(op,
                                1.0f,
                                A_ptr,
                                Adesc,
                                U_ptr,
                                Udesc,
                                B_ptr,
                                Bdesc,
                                V_ptr,
                                Vdesc,
                                0.0f,
                                C_in_ptr,
                                Cdesc,
                                D_ptr,
                                Cdesc,
                                ctx.workspace,
                                stream);

    if (ec != 0) {
        TM_LOG_ERROR(
            "[EAGLE3][GEMM][fallback] Gemm::Run failed with error=%d for shape M=%d K=%d N=%d; "
            "falling back to naive row-major kernel.",
            ec,
            M,
            K,
            N);
        return false;
    }

    return true;
}

}  // namespace

template<typename T>
void launch_eagle3_matmul_rowmajor_dispatch(const T* A,
                                            const T* B,
                                            T*       C,
                                            int      M,
                                            int      K,
                                            int      N,
                                            cudaStream_t stream)
{
    if (!launch_eagle3_matmul_gemm<T>(A, B, C, M, K, N, stream)) {
        launch_matmul_rowmajor_impl<T>(A, B, C, M, K, N, stream);
    }
}

namespace {

__global__ void build_eagle3_kv_spans_kernel(int           token_num,
                                             int           batch_size,
                                             int           q_len,
                                             int           kv_len,
                                             const int32_t* runtime_offsets,
                                             const int32_t* tree_offsets,
                                             const int32_t* kv_lens_runtime,
                                             const int32_t* successor_offsets,
                                             const int32_t* successor_counts,
                                             int32_t*       kv_start_per_token,
                                             int32_t*       kv_len_per_token)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= token_num) {
        return;
    }

    const int batch = q_len > 0 ? idx / q_len : 0;
    if (batch < 0 || batch >= batch_size) {
        return;
    }

    int kv_start = 0;
    int kv_end   = kv_len;

    auto clamp_span = [&](int start, int end) {
        kv_start = max(kv_start, start);
        kv_end   = min(kv_end, end);
    };

    if (runtime_offsets) {
        clamp_span(runtime_offsets[batch], runtime_offsets[batch + 1]);
    }
    if (tree_offsets) {
        clamp_span(tree_offsets[batch], tree_offsets[batch + 1]);
    }
    if (kv_lens_runtime) {
        clamp_span(0, kv_lens_runtime[batch]);
    }

    kv_start = max(0, min(kv_start, kv_len));
    kv_end   = max(0, min(kv_end, kv_len));
    const int kv_len_slot = kv_end - kv_start;

    if (kv_len_slot <= 0) {
        kv_start_per_token[idx] = 0;
        kv_len_per_token[idx]   = 0;
        return;
    }

    kv_start_per_token[idx] = kv_start;
    kv_len_per_token[idx]   = kv_len_slot;
}

}  // namespace

void launch_build_eagle3_kv_spans(int           token_num,
                                  int           batch_size,
                                  int           q_len,
                                  int           kv_len,
                                  const int32_t* runtime_offsets,
                                  const int32_t* tree_offsets,
                                  const int32_t* kv_lens_runtime,
                                  const int32_t* successor_offsets,
                                  const int32_t* successor_counts,
                                  int32_t*       kv_start_per_token,
                                  int32_t*       kv_len_per_token,
                                  cudaStream_t   stream)
{
    if (token_num <= 0 || batch_size <= 0 || q_len <= 0 || kv_len <= 0 || !kv_start_per_token || !kv_len_per_token) {
        return;
    }

    const int block = 256;
    const int grid  = (token_num + block - 1) / block;

    build_eagle3_kv_spans_kernel<<<grid, block, 0, stream>>>(token_num,
                                                             batch_size,
                                                             q_len,
                                                             kv_len,
                                                             runtime_offsets,
                                                             tree_offsets,
                                                             kv_lens_runtime,
                                                             successor_offsets,
                                                             successor_counts,
                                                             kv_start_per_token,
                                                             kv_len_per_token);
    sync_check_cuda_error();
}


// Explicitly instantiate launchers for various types
template void launch_apply_rope_kernel<half_t>(half_t*, half_t*, int, int, int, int, int, int, const Tensor*, float, float, cudaStream_t);
template void launch_sdpa_kernel<half_t>(const half_t*, const half_t*, const half_t*, half_t*, int, int, int, int, int, int, int, int, const Tensor*, const Tensor*, int, const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*, cudaStream_t);
template void launch_expand_kv_to_q_kernel<half_t>(const Tensor&, Tensor&, int, int, int, cudaStream_t);
template void launch_eagle3_matmul_rowmajor_dispatch<half_t>(const half_t*, const half_t*, half_t*, int, int, int, cudaStream_t);


#if ENABLE_BF16
template void launch_apply_rope_kernel<bfloat16_t>(bfloat16_t*, bfloat16_t*, int, int, int, int, int, int, const Tensor*, float, float, cudaStream_t);
template void launch_sdpa_kernel<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, const bfloat16_t*, bfloat16_t*, int, int, int, int, int, int, int, int, const Tensor*, const Tensor*, int, const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*, cudaStream_t);
template void launch_expand_kv_to_q_kernel<bfloat16_t>(const Tensor&, Tensor&, int, int, int, cudaStream_t);
template void launch_eagle3_matmul_rowmajor_dispatch<bfloat16_t>(const bfloat16_t*, const bfloat16_t*, bfloat16_t*, int, int, int, cudaStream_t);
#endif

#if ENABLE_FP32
template void launch_apply_rope_kernel<float>(float*, float*, int, int, int, int, int, int, const Tensor*, float, float, cudaStream_t);
template void launch_sdpa_kernel<float>(const float*, const float*, const float*, float*, int, int, int, int, int, int, int, int, const Tensor*, const Tensor*, int, const Tensor*, const Tensor*, const Tensor*, const Tensor*, const Tensor*, cudaStream_t);
template void launch_expand_kv_to_q_kernel<float>(const Tensor&, Tensor&, int, int, int, cudaStream_t);
template void launch_eagle3_matmul_rowmajor_dispatch<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
#endif

} // namespace ft
} // namespace turbomind
