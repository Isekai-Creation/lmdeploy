// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace ft {

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
    if (successor_offsets) {
        const int succ_start = successor_offsets[batch];
        const int succ_end   = successor_counts ? (succ_start + successor_counts[batch])
                                                : successor_offsets[batch + 1];
        clamp_span(succ_start, succ_end);
    }
    if (kv_lens_runtime) {
        clamp_span(0, kv_lens_runtime[batch]);
    }
    if (packed_mask && packed_stride > 0) {
        clamp_span(kv_start, min(kv_end, packed_stride * 32));
    }

    kv_start    = max(0, min(kv_start, kv_len));
    kv_end      = max(0, min(kv_end, kv_len)); // Corrected max(kv_start, ...) to max(0, ...)
    int kv_len_slot = kv_end - kv_start;

    if (kv_len_slot <= 0) { // kv_start >= kv_len implies kv_len_slot <= 0
        return;
    }

    const int pos_idx_q  = position_ids ? min(token_idx, token_num - 1) : token_idx;
    const int q_position = position_ids ? position_ids[pos_idx_q] : (past_kv_len + kv_start + q_pos);
    const float inv_sqrt = rsqrtf(static_cast<float>(head_dim));

    float max_score = -1e20f;
    for (int j = 0; j < kv_len_slot; ++j) {
        const int k_token_offset = batch * kv_len + (kv_start + j); // Offset into k_ptr's linear memory
        const T*  k        = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim; // Corrected K pointer calculation
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(kv_start + j, token_num - 1) : (kv_start + j);
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
        const T*  k        = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(k_token_offset, token_num - 1) : k_token_offset; // Corrected kv_token to k_token_offset
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
        const T*  k        = k_ptr + (k_token_offset * num_kv_heads + kv_head) * head_dim;
        const T*  v        = v_ptr + (v_token_offset * num_kv_heads + kv_head) * head_dim;
        float     dot      = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += to_float(q[d]) * to_float(k[d]);
        }
        const int pos_idx_k   = position_ids ? min(kv_start + j, token_num - 1) : (kv_start + j);
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
template<typename T>
void launch_eagle3_matmul_rowmajor_dispatch(const T* A,
                                            const T* B,
                                            T*       C,
                                            int      M,
                                            int      K,
                                            int      N,
                                            cudaStream_t stream)
{
    launch_matmul_rowmajor_impl<T>(A, B, C, M, K, N, stream);
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