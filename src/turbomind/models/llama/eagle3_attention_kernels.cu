// Copyright (c) OpenMMLab. All rights reserved.
// Lightweight GEMM helpers for Eagle-3 attention. These implement a
// row-major matmul C = A @ B^T where:
//   A: [M, K]
//   B: [N, K]
//   C: [M, N]
//
// Used by Eagle3AttentionLayer to apply the native Eagle-3
// midlayer.self_attn.{q,o}_proj weights without going through the
// general LlamaLinear path. This keeps the dependency surface small
// while we iterate on the Eagle-3 backend.

#include <cuda_fp16.h>
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include "src/turbomind/core/data_type.h"

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
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

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

}  // namespace

void launch_eagle3_matmul_rowmajor_half(const half_t* A,
                                        const half_t* B,
                                        half_t*       C,
                                        int           M,
                                        int           K,
                                        int           N,
                                        cudaStream_t  stream)
{
    launch_matmul_rowmajor_impl<half_t>(A, B, C, M, K, N, stream);
}

#if ENABLE_BF16
void launch_eagle3_matmul_rowmajor_bf16(const bfloat16_t* A,
                                        const bfloat16_t* B,
                                        bfloat16_t*       C,
                                        int               M,
                                        int               K,
                                        int               N,
                                        cudaStream_t      stream)
{
    launch_matmul_rowmajor_impl<bfloat16_t>(A, B, C, M, K, N, stream);
}
#endif

void launch_eagle3_matmul_rowmajor_float(const float* A,
                                         const float* B,
                                         float*       C,
                                         int          M,
                                         int          K,
                                         int          N,
                                         cudaStream_t stream)
{
    launch_matmul_rowmajor_impl<float>(A, B, C, M, K, N, stream);
}

}  // namespace turbomind

