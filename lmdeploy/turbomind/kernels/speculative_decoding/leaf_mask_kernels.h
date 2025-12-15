/*
 * Leaf Mask Kernel for EAGLE3 Tree Traversal
 * Ported from TensorRT-LLM for LMDeploy
 * 
 * Identifies leaf vs non-leaf nodes in speculation tree
 * for efficient path selection.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace lmdeploy {
namespace turbomind {
namespace kernels {
namespace speculative_decoding {

using SizeType32 = int32_t;

/**
 * @brief Build leaf mask for tree nodes.
 * 
 * Identifies which nodes in the speculation tree are leaves (no children)
 * vs non-leaves (have children). Critical for efficient tree traversal.
 * 
 * @param isLeafMask Output buffer [batchSize, maxDecodingTokens]
 *                   1 = leaf node, 0 = non-leaf node
 * @param nextPaths Input buffer [batchSize, maxDecodingTokens, maxPathLen]
 *                  Tree paths for each token
 * @param batchSize Number of sequences in batch
 * @param maxDecodingTokens Maximum number of tokens per sequence
 * @param maxPathLen Maximum path length in tree
 * @param stream CUDA stream for kernel execution
 */
void invokeBuildLeafMask(
    int8_t* isLeafMask,
    SizeType32 const* nextPaths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen,
    cudaStream_t stream
);

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
} // namespace lmdeploy
