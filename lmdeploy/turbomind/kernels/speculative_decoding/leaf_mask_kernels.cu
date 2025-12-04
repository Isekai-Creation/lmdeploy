/*
 * Leaf Mask Kernel for EAGLE3 Tree Traversal
 * Ported from TensorRT-LLM for LMDeploy
 */

#include "leaf_mask_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace lmdeploy {
namespace turbomind {
namespace kernels {
namespace speculative_decoding {

namespace {

/**
 * @brief CUDA kernel to build leaf mask from tree paths.
 * 
 * A node is a leaf if no other node has it as a parent in their path.
 * 
 * Grid: (batchSize, maxPathLen - 1)
 * Block: 512 threads
 */
__global__ void buildLeafMaskKernel(
    int8_t* __restrict__ isLeafMask,
    SizeType32 const* __restrict__ nextPaths,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen
)
{
    // Batch index
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    // Level in tree (0 to maxPathLen-2)
    auto const levelIdx = static_cast<SizeType32>(blockIdx.y);
    
    // Get path offset for this batch
    auto const pathOffset = batchIdx * maxDecodingTokens * maxPathLen;
    
    // Initialize leaf mask to 1 (assume all are leaves)
    for (auto tokenIdx = static_cast<SizeType32>(threadIdx.x);
         tokenIdx < maxDecodingTokens;
         tokenIdx += static_cast<SizeType32>(blockDim.x))
    {
        if (levelIdx == 0) {
            // First level: initialize
            isLeafMask[batchIdx * maxDecodingTokens + tokenIdx] = 1;
        }
    }
    
    __syncthreads();
    
    // Mark non-leaves
    // A node is non-leaf if it appears in any path at position < maxPathLen-1
    for (auto tokenIdx = static_cast<SizeType32>(threadIdx.x);
         tokenIdx < maxDecodingTokens;
         tokenIdx += static_cast<SizeType32>(blockDim.x))
    {
        // Check if this token has children at this level
        auto const curPathOffset = pathOffset + tokenIdx * maxPathLen;
        
        // Get the node at this level in the path
        if (levelIdx < maxPathLen - 1) {
            auto const nodeAtLevel = nextPaths[curPathOffset + levelIdx];
            
            // If valid node and has next level
            if (nodeAtLevel >= 0 && nodeAtLevel < maxDecodingTokens) {
                auto const nextNode = nextPaths[curPathOffset + levelIdx + 1];
                
                // If next level exists and is valid, current node is not a leaf
                if (nextNode >= 0 && nextNode < maxDecodingTokens) {
                    isLeafMask[batchIdx * maxDecodingTokens + nodeAtLevel] = 0;
                }
            }
        }
    }
}

} // anonymous namespace

void invokeBuildLeafMask(
    int8_t* isLeafMask,
    SizeType32 const* nextPaths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen,
    cudaStream_t stream
)
{
    // Grid: (batchSize, maxPathLen - 1)
    // Block: 512 threads
    dim3 grid(batchSize, maxPathLen - 1);
    dim3 block(512);
    
    buildLeafMaskKernel<<<grid, block, 0, stream>>>(
        isLeafMask,
        nextPaths,
        maxDecodingTokens,
        maxPathLen
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "buildLeafMask kernel launch failed: %s\n",
                cudaGetErrorString(err));
    }
}

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
} // namespace lmdeploy
