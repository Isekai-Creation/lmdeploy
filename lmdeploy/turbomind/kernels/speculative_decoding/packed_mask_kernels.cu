/*
 * Packed Mask Kernels for EAGLE3 Speculative Decoding
 * Ported from TensorRT-LLM for LMDeploy
 * 
 * Achieves 32x memory compression for attention masks.
 */

#include "packed_mask_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace lmdeploy {
namespace turbomind {
namespace kernels {
namespace speculative_decoding {

namespace {

// Constants
constexpr int32_t BITS_PER_INT32 = 32;
constexpr int32_t WARP_SIZE = 32;

/**
 * @brief CUDA kernel to generate packed masks from cumulative lengths.
 * 
 * Each thread processes one position in the attention mask, packing
 * 32 boolean values into a single int32.
 * 
 * Grid: (batchSize, maxDecodingTokens, ceil(maxDecodingTokens/32))
 * Block: 32 threads (one warp)
 */
__global__ void getPackedMaskKernel(
    int32_t* __restrict__ packedMask,
    SizeType32 const* __restrict__ cumGenerationLengths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens
)
{
    // Grid dimensions: (batchSize, maxDecodingTokens, packedSize)
    int32_t const batchIdx = blockIdx.x;
    int32_t const tokenIdx = blockIdx.y;
    int32_t const packedIdx = blockIdx.z;
    int32_t const laneIdx = threadIdx.x;

    if (batchIdx >= batchSize) return;

    // Get generation length for this batch item
    int32_t const genLength = cumGenerationLengths[batchIdx + 1] 
                             - cumGenerationLengths[batchIdx];

    // Calculate which bit position this thread handles
    int32_t const bitPos = packedIdx * BITS_PER_INT32 + laneIdx;

    // Determine if this position should be masked
    // Mask is 1 if bitPos < genLength and bitPos <= tokenIdx
    int32_t mask = 0;
    if (bitPos < maxDecodingTokens && bitPos < genLength && bitPos <= tokenIdx) {
        mask = 1;
    }

    // Pack 32 bits from warp into single int32
    // Use warp-level ballot to collect bits
    uint32_t packed = __ballot_sync(0xFFFFFFFF, mask);

    // First thread in warp writes the packed result
    if (laneIdx == 0) {
        int32_t const outputIdx = batchIdx * maxDecodingTokens * 
                                  ((maxDecodingTokens + BITS_PER_INT32 - 1) / BITS_PER_INT32)
                                + tokenIdx * ((maxDecodingTokens + BITS_PER_INT32 - 1) / BITS_PER_INT32)
                                + packedIdx;
        packedMask[outputIdx] = static_cast<int32_t>(packed);
    }
}

/**
 * @brief CUDA kernel to generate packed masks from tree paths.
 * 
 * Processes tree paths to generate attention masks, then packs them
 * into int32 bit fields for efficient memory usage.
 * 
 * Grid: (batchSize, maxDecodingTokens, ceil(maxDecodingTokens/32))
 * Block: 32 threads (one warp)
 */
__global__ void getPackedMaskFromPathKernel(
    int32_t* __restrict__ packedMask,
    SizeType32 const* __restrict__ batchSlots,
    SizeType32 const* __restrict__ nextDraftPaths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen
)
{
    // Grid dimensions: (batchSize, maxDecodingTokens, packedSize)
    int32_t const localBatchIdx = blockIdx.x;
    int32_t const tokenIdx = blockIdx.y;
    int32_t const packedIdx = blockIdx.z;
    int32_t const laneIdx = threadIdx.x;

    if (localBatchIdx >= batchSize) return;

    // Get global batch slot
    int32_t const globalBatchIdx = batchSlots ? batchSlots[localBatchIdx] : localBatchIdx;

    // Calculate which bit position this thread handles
    int32_t const bitPos = packedIdx * BITS_PER_INT32 + laneIdx;

    // Determine if this position should be masked based on path
    int32_t mask = 0;
    
    if (bitPos < maxDecodingTokens) {
        // Check if bitPos is in the path of tokenIdx
        // Path format: nextDraftPaths[batch, token, pathLevel]
        // A position is masked if it's an ancestor in the tree
        
        bool isAncestor = false;
        
        // tokenIdx's path
        int32_t const pathOffset = globalBatchIdx * maxDecodingTokens * maxPathLen 
                                  + tokenIdx * maxPathLen;
        
        // bitPos's path
        int32_t const bitPathOffset = globalBatchIdx * maxDecodingTokens * maxPathLen 
                                     + bitPos * maxPathLen;
        
        // Check if bitPos is an ancestor of tokenIdx
        if (bitPos == tokenIdx) {
            isAncestor = true;  // Self is always in mask
        } else if (bitPos < tokenIdx) {
            // Check if bitPos appears in tokenIdx's path
            for (int32_t level = 0; level < maxPathLen; ++level) {
                if (nextDraftPaths[pathOffset + level] == bitPos) {
                    isAncestor = true;
                    break;
                }
            }
        }
        
        mask = isAncestor ? 1 : 0;
    }

    // Pack 32 bits from warp into single int32
    uint32_t packed = __ballot_sync(0xFFFFFFFF, mask);

    // First thread in warp writes the packed result
    if (laneIdx == 0) {
        int32_t const packedSize = (maxDecodingTokens + BITS_PER_INT32 - 1) / BITS_PER_INT32;
        int32_t const outputIdx = globalBatchIdx * maxDecodingTokens * packedSize
                                + tokenIdx * packedSize
                                + packedIdx;
        packedMask[outputIdx] = static_cast<int32_t>(packed);
    }
}

} // anonymous namespace

// Host functions

void invokeGetPackedMask(
    int32_t* packedMask,
    SizeType32 const* cumGenerationLengths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    cudaStream_t stream
)
{
    // Calculate packed size (how many int32s needed for maxDecodingTokens bits)
    int32_t const packedSize = (maxDecodingTokens + BITS_PER_INT32 - 1) / BITS_PER_INT32;

    // Grid: (batchSize, maxDecodingTokens, packedSize)
    // Block: 32 threads (one warp per packed int32)
    dim3 grid(batchSize, maxDecodingTokens, packedSize);
    dim3 block(WARP_SIZE);

    getPackedMaskKernel<<<grid, block, 0, stream>>>(
        packedMask,
        cumGenerationLengths,
        batchSize,
        maxDecodingTokens
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "getPackedMask kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

void invokeGetPackedMaskFromPath(
    int32_t* packedMask,
    SizeType32 const* batchSlots,
    SizeType32 const* nextDraftPaths,
    SizeType32 batchSize,
    SizeType32 maxDecodingTokens,
    SizeType32 maxPathLen,
    cudaStream_t stream
)
{
    // Calculate packed size
    int32_t const packedSize = (maxDecodingTokens + BITS_PER_INT32 - 1) / BITS_PER_INT32;

    // Grid: (batchSize, maxDecodingTokens, packedSize)
    // Block: 32 threads (one warp per packed int32)
    dim3 grid(batchSize, maxDecodingTokens, packedSize);
    dim3 block(WARP_SIZE);

    getPackedMaskFromPathKernel<<<grid, block, 0, stream>>>(
        packedMask,
        batchSlots,
        nextDraftPaths,
        batchSize,
        maxDecodingTokens,
        maxPathLen
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "getPackedMaskFromPath kernel launch failed: %s\n", 
                cudaGetErrorString(err));
    }
}

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
} // namespace lmdeploy
