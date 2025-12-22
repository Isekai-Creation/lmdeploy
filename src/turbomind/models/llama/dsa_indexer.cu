// Copyright (c) OpenMMLab. All rights reserved.
// Ported from vLLM cache_kernels.cu - indexer_k_quant_and_cache_kernel
// DSA (Dynamic Sparse Attention) Indexer for DeepSeek V32 MLA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// FP8 E4M3 conversion with scale
template<typename T>
__device__ __forceinline__ uint8_t fp8_e4m3_scaled_convert(T val, float scale) {
    float fval = static_cast<float>(val) / scale;
    // Clamp to FP8 E4M3 range: [-448, 448]
    fval = fmaxf(fminf(fval, 448.0f), -448.0f);
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(fval, __NV_SATFINITE, __NV_E4M3));
}

/**
 * @brief Quantize indexer K values to FP8 and store in cache
 * 
 * This kernel:
 * 1. Loads K values for the indexer (per-token, single head)
 * 2. Computes per-block amax via warp reduction
 * 3. Computes ue8m0 scale: scale = exp2(ceil(log2(amax / 448)))
 * 4. Quantizes to FP8 E4M3 and stores in paged cache
 * 
 * @tparam T Input data type (half/bfloat16)
 * @tparam VEC_SIZE Vector load size (typically 4)
 * 
 * Reference: vLLM cache_kernels.cu:792-861
 */
template<typename T, int VEC_SIZE = 4>
__global__ void indexer_k_quant_and_cache_kernel(
    const T* __restrict__ k,        // [num_tokens, head_dim]
    uint8_t* __restrict__ kv_cache, // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping, // [num_tokens]
    const int head_dim,             // dimension of indexer head (e.g., 128)
    const int quant_block_size,     // quantization block size (e.g., 128)
    const int cache_block_size,     // tokens per cache block (e.g., 64)
    const int cache_stride,         // stride per token in cache (head_dim + scale_bytes)
    const bool use_ue8m0            // use ue8m0 scale format (log2-based)
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                  threadIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
    const int64_t slot_idx = slot_mapping[token_idx];
    
    // slot_idx can be -1 for padded tokens
    if (slot_idx < 0 || head_dim_idx >= head_dim) {
        return;
    }
    
    const int64_t block_idx = slot_idx / cache_block_size;
    const int64_t block_offset = slot_idx % cache_block_size;
    
    // Load K values (vectorized)
    float k_vals[VEC_SIZE];
    const T* k_src = k + token_idx * head_dim + head_dim_idx;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        k_vals[i] = static_cast<float>(k_src[i]);
    }
    
    // Per-thread amax
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        amax = fmaxf(amax, fabsf(k_vals[i]));
    }
    
    __syncwarp();
    
    // Warp reduction for per-quantization-block amax
    // Assuming quant_block_size / VEC_SIZE threads per group = 32 (one warp)
    for (int mask = 16; mask > 0; mask /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask));
    }
    
    __syncwarp();
    
    // Compute scale
    float scale = fmaxf(amax, 1e-4f) / 448.0f;
    if (use_ue8m0) {
        // ue8m0 format: scale = 2^ceil(log2(scale))
        scale = exp2f(ceilf(log2f(scale)));
    }
    
    // Quantize and store FP8 values
    const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                               block_offset * head_dim + head_dim_idx;
    
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        kv_cache[dst_offset + i] = fp8_e4m3_scaled_convert(k_vals[i], scale);
    }
    
    // Store scale (one per quant_block_size elements, stored as float)
    if (threadIdx.x == 0) {
        const int64_t scale_offset = 
            block_idx * cache_block_size * cache_stride +
            cache_block_size * head_dim +  // After all FP8 values
            (block_offset * head_dim + head_dim_idx) * sizeof(float) / quant_block_size;
        
        reinterpret_cast<float*>(kv_cache)[scale_offset / sizeof(float)] = scale;
    }
}

/**
 * @brief Gate function for DSA indexer cache kernel dispatch
 */
template<typename T>
void invokeIndexerKQuantAndCache(
    const T* k,
    uint8_t* kv_cache,
    const int64_t* slot_mapping,
    int num_tokens,
    int head_dim,
    int quant_block_size,
    int cache_block_size,
    int cache_stride,
    bool use_ue8m0,
    cudaStream_t stream
) {
    constexpr int VEC_SIZE = 4;
    const int threads_per_block = 32;  // One warp handles one quantization block
    
    dim3 grid(num_tokens, (head_dim + VEC_SIZE * threads_per_block - 1) / (VEC_SIZE * threads_per_block));
    dim3 block(threads_per_block, 1);
    
    indexer_k_quant_and_cache_kernel<T, VEC_SIZE><<<grid, block, 0, stream>>>(
        k, kv_cache, slot_mapping,
        head_dim, quant_block_size, cache_block_size, cache_stride,
        use_ue8m0
    );
}

// Explicit instantiations
template void invokeIndexerKQuantAndCache<half>(
    const half*, uint8_t*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);

template void invokeIndexerKQuantAndCache<__nv_bfloat16>(
    const __nv_bfloat16*, uint8_t*, const int64_t*, int, int, int, int, int, bool, cudaStream_t);

}  // namespace turbomind
