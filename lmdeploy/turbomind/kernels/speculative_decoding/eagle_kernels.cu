/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * Based on TensorRT-LLM's EAGLE implementation
 *
 * EAGLE3 CUDA Kernel Implementations
 */

#include "eagle_kernels.h"
#include <cuda_runtime.h>
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {
namespace kernels {
namespace eagle {

namespace {

inline __device__ void eagleKernelsDeviceAssert(bool cond, const char* msg)
{
    if (!cond) {
        printf("[EAGLE][kernels][device_assert] %s\n", msg);
    }
}

// CUB-style block scan for computing offsets
template <int BLOCK_SIZE>
__device__ void blockExclusiveSum(SizeType value, SizeType& result) {
    __shared__ SizeType temp[BLOCK_SIZE];
    int tid = threadIdx.x;
    
    temp[tid] = value;
    __syncthreads();
    
    // Simple sequential scan (can be optimized with parallel scan)
    if (tid == 0) {
        SizeType sum = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            SizeType val = temp[i];
            temp[i] = sum;
            sum += val;
        }
    }
    __syncthreads();
    
    result = temp[tid];
}

/**
 * @brief Kernel: Prepare context-phase EagleNet inputs
 */
template <int BLOCK_SIZE>
__global__ void prepareCtxEagleNetInputsKernel(
    // Outputs
    SizeType* eagleNetSequenceLengths,
    SizeType* eagleNetContextLengths,
    TokenIdType* outputIds,
    SizeType* positionIds,
    SizeType* hiddenStatesIndices,
    SizeType* lastTokenIndices,
    SizeType* numLastTokenIndices,
    
    // Inputs
    TokenIdType const* inputIds,
    SizeType const* baseNetSequenceLengths,
    SizeType const* baseNetContextLengths,
    TokenIdType const* acceptedTokens,
    SizeType const* acceptedLens,
    SizeType const* prevDraftLens,
    SizeType const* prevPaths,
    SizeType const* bestPathIds,
    
    // Dimensions
    SizeType batchSize,
    SizeType maxPathLen,
    SizeType maxDecodingTokens
) {
    auto const bid = static_cast<SizeType>(threadIdx.x);
    
    if (bid >= batchSize) return;
    
    // Determine if this is a context or generation request
    SizeType prevDraftLen = prevDraftLens[bid];
    bool isContextRequest = (prevDraftLen == 0);
    
    SizeType numDecodingTokens;
    SizeType numInputTokens;
    
    if (isContextRequest) {
        // Context: process all prompt tokens
        numInputTokens = baseNetContextLengths[bid];
        numDecodingTokens = numInputTokens;
    } else {
        // Generation: process accepted tokens
        numInputTokens = prevDraftLen + 1;
        numDecodingTokens = acceptedLens[bid];
    }
    
    // Compute output offset using block scan
    SizeType outputStartPos;
    SizeType inputIndexBase;
    blockExclusiveSum<BLOCK_SIZE>(numDecodingTokens, outputStartPos);
    blockExclusiveSum<BLOCK_SIZE>(numInputTokens, inputIndexBase);
    
    // Extract tokens and build indices
    auto const oldSequenceLength = baseNetSequenceLengths[bid] - numInputTokens;
    
    for (SizeType ti = 0; ti < numDecodingTokens; ++ti) {
        TokenIdType token;
        SizeType hiddenStateIdx;
        
        if (isContextRequest) {
            // Context: use input tokens
            token = inputIds[inputIndexBase + ti];
            hiddenStateIdx = inputIndexBase + ti;
        } else {
            // Generation: use accepted tokens from best path
            token = acceptedTokens[bid * maxPathLen + ti];
            
            // Get hidden state index from path
            auto const bestPathId = bestPathIds[bid];
            auto const pathIdx = prevPaths[bid * maxDecodingTokens * maxPathLen + 
                                          bestPathId * maxPathLen + ti];
            hiddenStateIdx = inputIndexBase + pathIdx;
        }
        
        outputIds[outputStartPos + ti] = token;
        positionIds[outputStartPos + ti] = oldSequenceLength + ti;
        hiddenStatesIndices[outputStartPos + ti] = hiddenStateIdx;
    }
    
    // Set metadata. Some integration paths (e.g. TurboMind) may pass
    // nullptr for lastTokenIndices/numLastTokenIndices when they only
    // need the per-batch sequence/context lengths and not the explicit
    // last-token index array. Guard these stores accordingly to avoid
    // illegal memory access when those outputs are unused.
    eagleNetContextLengths[bid]  = numDecodingTokens;
    eagleNetSequenceLengths[bid] = oldSequenceLength + numDecodingTokens;
    if (lastTokenIndices) {
        lastTokenIndices[bid] = outputStartPos + numDecodingTokens;
    }
    
    // Last thread writes total count
    if (numLastTokenIndices && bid == batchSize - 1) {
        numLastTokenIndices[0] = batchSize;
    }
}

/**
 * @brief Kernel: Build leaf mask
 */
__global__ void buildLeafMaskKernel(
    int8_t* isLeafMask,
    SizeType const* paths,
    SizeType maxDecodingTokens,
    SizeType maxPathLen
) {
    auto const bid = static_cast<SizeType>(blockIdx.x);
    auto const level = static_cast<SizeType>(blockIdx.y);
    
    // Initialize all as leaves (1)
    for (auto pathIdx = static_cast<SizeType>(threadIdx.x); 
         pathIdx < maxDecodingTokens; 
         pathIdx += static_cast<SizeType>(blockDim.x)) {
        
        // Get current and next level offsets
        auto const curLevelOffset = bid * maxDecodingTokens * maxPathLen + 
                                    pathIdx * maxPathLen + level;
        auto const nextLevelOffset = curLevelOffset + 1;
        
        auto const curNodeTokenIdx = paths[curLevelOffset];
        
        // If current node exists and has child, mark as non-leaf (0)
        if (curNodeTokenIdx != -1 && level + 1 < maxPathLen && 
            paths[nextLevelOffset] != -1) {
            isLeafMask[bid * maxDecodingTokens + curNodeTokenIdx] = 0;
        }
    }
}

/**
 * @brief Device helper: Pack boolean mask into int32
 */
__device__ void maskToPackedMask(
    SizeType* outputPtr,
    char const* shMask,
    SizeType maxDecodingTokens,
    SizeType numPackedMasks
) {
    for (SizeType maskId = 0; maskId < numPackedMasks; ++maskId) {
        SizeType packed = 0;
        
        for (int bit = 0; bit < 32; ++bit) {
            int idx = maskId * 32 + bit;
            if (idx < maxDecodingTokens && shMask[idx] == '1') {
                packed |= (1 << bit);
            }
        }
        
        outputPtr[maskId] = packed;
    }
}

/**
 * @brief Kernel: Pack attention mask from boolean array
 */
__global__ void getPackedMaskKernel(
    SizeType* packedMask,
    bool const* mask,
    SizeType maxDecodingTokens
) {
    auto const batchIdx = static_cast<SizeType>(blockIdx.y);
    auto const tokenIdx = static_cast<SizeType>(blockIdx.x);
    
    auto const numPackedMasks = (maxDecodingTokens + 31) / 32;
    
    extern __shared__ char shMask[];
    
    // Load mask row into shared memory
    bool const* maskPtr = mask + batchIdx * maxDecodingTokens * maxDecodingTokens + 
                          tokenIdx * maxDecodingTokens;
    
    for (auto ti = static_cast<SizeType>(threadIdx.x); 
         ti < maxDecodingTokens; 
         ti += static_cast<SizeType>(blockDim.x)) {
        shMask[ti] = maskPtr[ti] ? '1' : '0';
    }
    __syncthreads();
    
    // Pack into int32 (thread 0 only)
    if (threadIdx.x == 0) {
        auto* outputPtr = packedMask + batchIdx * maxDecodingTokens * numPackedMasks + 
                          tokenIdx * numPackedMasks;
        maskToPackedMask(outputPtr, shMask, maxDecodingTokens, numPackedMasks);
    }
}

/**
 * @brief Kernel: Pack attention mask from path representation
 */
__global__ void getPackedMaskFromPathKernel(
    SizeType* packedMask,
    SizeType const* batchSlots,
    SizeType const* paths,
    SizeType maxDecodingTokens,
    SizeType maxPathLen
) {
    extern __shared__ char adjacencyMatrix[];
    
    auto const batchIdx = static_cast<SizeType>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];
    
    if (batchSlot < 0) return;
    
    // Initialize adjacency matrix
    for (auto tix = static_cast<SizeType>(threadIdx.x); 
         tix < maxDecodingTokens * maxDecodingTokens; 
         tix += static_cast<SizeType>(blockDim.x)) {
        adjacencyMatrix[tix] = '0';
    }
    
    // Set root token
    if (threadIdx.x == 0) {
        adjacencyMatrix[0 * maxDecodingTokens + (maxDecodingTokens - 1)] = '1';
    }
    __syncthreads();
    
    // Build adjacency matrix from paths
    auto const curPath = paths + batchSlot * maxDecodingTokens * maxPathLen;
    
    for (auto tix = static_cast<SizeType>(threadIdx.x); 
         tix < maxDecodingTokens; 
         tix += static_cast<SizeType>(blockDim.x)) {
        
        for (SizeType ti = 1; ti < maxPathLen; ++ti) {
            auto const pathOffset = tix * maxPathLen;
            auto const toIndex = curPath[pathOffset + ti];
            
            if (toIndex == -1) break;
            
            // Mark this node
            adjacencyMatrix[toIndex * maxDecodingTokens + (maxDecodingTokens - 1 - toIndex)] = '1';
            
            // Mark edges to ancestors
            for (SizeType fi = 0; fi < ti; ++fi) {
                auto const fromIndex = maxDecodingTokens - 1 - curPath[pathOffset + fi];
                adjacencyMatrix[toIndex * maxDecodingTokens + fromIndex] = '1';
            }
        }
    }
    __syncthreads();
    
    // Pack adjacency matrix
    auto const numPackedMasks = (maxDecodingTokens + 31) / 32;
    for (auto ti = static_cast<SizeType>(threadIdx.x); 
         ti < maxDecodingTokens; 
         ti += static_cast<SizeType>(blockDim.x)) {
        
        auto outputPtr = packedMask + batchSlot * maxDecodingTokens * numPackedMasks + 
                        ti * numPackedMasks;
        maskToPackedMask(outputPtr, adjacencyMatrix + ti * maxDecodingTokens, 
                        maxDecodingTokens, numPackedMasks);
    }
}

} // anonymous namespace

// Public API implementations

static inline void EagleKernelsCudaCheckAt(const char* site)
{
    if (!::turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_INVARIANTS_DEBUG")) {
        return;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TM_LOG_ERROR("[EAGLE][kernels][invariants] CUDA error %d (%s) at %s",
                     static_cast<int>(err),
                     cudaGetErrorString(err),
                     site);
        std::abort();
    }
}

void invokePrepareCtxEagleNetInputs(PrepareCtxEagleNetParams const& params) {
    constexpr int BLOCK_SIZE = 512;
    
    prepareCtxEagleNetInputsKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, params.stream>>>(
        params.eagleNetSequenceLengths,
        params.eagleNetContextLengths,
        params.outputIds,
        params.positionIds,
        params.hiddenStatesIndices,
        params.lastTokenIndices,
        params.numLastTokenIndices,
        params.inputIds,
        params.baseNetSequenceLengths,
        params.baseNetContextLengths,
        params.acceptedTokens,
        params.acceptedLens,
        params.prevDraftLens,
        params.prevPaths,
        params.bestPathIds,
        params.batchSize,
        params.maxPathLen,
        params.maxDecodingTokens
    );

    EagleKernelsCudaCheckAt("eagle_kernels::invokePrepareCtxEagleNetInputs");
}

void invokeBuildLeafMask(
    int8_t* isLeafMask,
    SizeType const* paths,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    SizeType maxPathLen,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(batchSize, maxPathLen - 1);
    
    buildLeafMaskKernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        isLeafMask, paths, maxDecodingTokens, maxPathLen
    );

    EagleKernelsCudaCheckAt("eagle_kernels::invokeBuildLeafMask");
}

void invokeGetPackedMask(
    SizeType* packedMask,
    bool const* mask,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    dim3 grid(maxDecodingTokens, batchSize);
    size_t sharedMemSize = maxDecodingTokens * sizeof(char);
    
    getPackedMaskKernel<<<grid, BLOCK_SIZE, sharedMemSize, stream>>>(
        packedMask, mask, maxDecodingTokens
    );

    EagleKernelsCudaCheckAt("eagle_kernels::invokeGetPackedMask");
}

void invokeGetPackedMaskFromPath(
    SizeType* packedMask,
    SizeType const* batchSlots,
    SizeType const* paths,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    SizeType maxPathLen,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    size_t sharedMemSize = maxDecodingTokens * maxDecodingTokens * sizeof(char);
    
    // One block per batch item
    getPackedMaskFromPathKernel<<<batchSize, BLOCK_SIZE, sharedMemSize, stream>>>(
        packedMask, batchSlots, paths, maxDecodingTokens, maxPathLen
    );

    EagleKernelsCudaCheckAt("eagle_kernels::invokeGetPackedMaskFromPath");
}

} // namespace eagle
} // namespace kernels
} // namespace turbomind
