/*
 * Packed Mask Kernels for EAGLE3 Speculative Decoding
 * Ported from TensorRT-LLM for LMDeploy
 * 
 * Provides 32x memory compression for attention masks by packing
 * boolean values into int32 bit fields.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace lmdeploy {
namespace turbomind {
namespace kernels {
namespace speculative_decoding {

// Type aliases for compatibility
using SizeType32 = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Extract packed mask from cumulative generation lengths.
 * 
 * Compresses boolean attention mask into int32 bit fields, achieving
 * 32x memory reduction (32 bools -> 1 int32).
 * 
 * @param packedMask Output buffer [batchSize, maxDecodingTokens, ceil(maxDecodingTokens/32)]
 *                   Packed int32 masks
 * @param cumGenerationLengths Input buffer [batchSize + 1]
 *                             Cumulative sum of generation lengths
 * @param batchSize Number of sequences in batch
 * @param maxDecodingTokens Maximum number of decoding tokens per sequence
 * @param stream CUDA stream for kernel execution
 */
void invokeGetPackedMask(
    int32_t* packedMask,
    SizeType32 const* cumGenerationLengths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    cudaStream_t stream
);

/**
 * @brief Extract packed mask from draft paths.
 * 
 * Generates packed attention masks from tree paths, enabling efficient
 * GPU memory usage for tree-based speculation.
 * 
 * @param packedMask Output buffer [batchSize, maxDecodingTokens, ceil(maxDecodingTokens/32)]
 *                   Packed int32 masks at batch slots
 * @param batchSlots Input buffer [batchSize]
 *                   Mapping from local to global batch indices
 * @param nextDraftPaths Input buffer [batchSize, maxDecodingTokens, maxPathLen]
 *                       Tree paths for draft tokens
 * @param batchSize Number of sequences in batch
 * @param maxDecodingTokens Maximum number of decoding tokens per sequence
 * @param maxPathLen Maximum path length in tree
 * @param stream CUDA stream for kernel execution
 */
void invokeGetPackedMaskFromPath(
    int32_t* packedMask,
    SizeType32 const* batchSlots,
    SizeType32 const* nextDraftPaths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen,
    cudaStream_t stream
);

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
} // namespace lmdeploy
