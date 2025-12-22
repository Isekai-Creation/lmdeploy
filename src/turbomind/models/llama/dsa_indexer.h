// Copyright (c) OpenMMLab. All rights reserved.
// DSA (Dynamic Sparse Attention) Indexer for DeepSeek V32 MLA

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace turbomind {

/**
 * @brief Quantize indexer K values to FP8 E4M3 and store in paged cache
 *
 * This is the core kernel for DeepSeek V3.2's Dynamic Sparse Attention (DSA)
 * indexer. It:
 * 1. Loads K values for sparse token selection
 * 2. Quantizes to FP8 E4M3 with per-block scales
 * 3. Stores in paged KV cache format
 *
 * @tparam T Input data type (half or bfloat16)
 * @param k Input K tensor [num_tokens, head_dim]
 * @param kv_cache Output cache [num_blocks, block_size, cache_stride]
 * @param slot_mapping Token to slot mapping [num_tokens]
 * @param num_tokens Number of input tokens
 * @param head_dim Indexer head dimension (typically 128)
 * @param quant_block_size Elements per quantization block (typically 128)
 * @param cache_block_size Tokens per cache block (typically 64)
 * @param cache_stride Stride per token in cache
 * @param use_ue8m0 Use ue8m0 scale format (log2-based)
 * @param stream CUDA stream
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
);

}  // namespace turbomind
