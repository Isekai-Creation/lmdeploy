/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * Based on TensorRT-LLM's EAGLE implementation
 *
 * EAGLE3 CUDA Kernels
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace turbomind {
namespace kernels {
namespace eagle {

using SizeType = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Parameters for context-phase EagleNet preparation
 */
struct PrepareCtxEagleNetParams {
    // Outputs
    SizeType* eagleNetSequenceLengths;    // [batchSize]
    SizeType* eagleNetContextLengths;     // [batchSize]
    TokenIdType* outputIds;               // [numOutputTokens]
    SizeType* positionIds;                // [numOutputTokens]
    SizeType* hiddenStatesIndices;        // [numOutputTokens]
    SizeType* lastTokenIndices;           // [batchSize]
    SizeType* numLastTokenIndices;        // [1]
    
    // Inputs
    TokenIdType const* inputIds;          // [numInputTokens]
    SizeType const* baseNetSequenceLengths;  // [batchSize]
    SizeType const* baseNetContextLengths;   // [batchSize]
    TokenIdType const* acceptedTokens;    // [batchSize, maxPathLen]
    SizeType const* acceptedLens;         // [batchSize]
    SizeType const* prevDraftLens;        // [batchSize]
    SizeType const* prevPaths;            // [batchSize, maxDecodingTokens, maxPathLen]
    SizeType const* bestPathIds;          // [batchSize]
    
    // Dimensions
    SizeType batchSize;
    SizeType maxPathLen;
    SizeType maxDecodingTokens;
    
    cudaStream_t stream;
};

/**
 * @brief Prepare inputs for context-phase EagleNet (EagleNet0)
 * 
 * Handles both context requests (initial prompt) and generation requests
 * (accepted tokens from previous iteration).
 */
void invokePrepareCtxEagleNetInputs(PrepareCtxEagleNetParams const& params);

/**
 * @brief Build leaf mask to distinguish leaf vs non-leaf nodes
 * 
 * A node is a leaf if it has no children in the tree. Only non-leaf
 * nodes are expanded in the next iteration.
 * 
 * @param isLeafMask output [batchSize, maxDecodingTokens], 1=leaf, 0=non-leaf
 * @param paths input [batchSize, maxDecodingTokens, maxPathLen]
 * @param batchSize number of requests
 * @param maxDecodingTokens maximum tokens per request
 * @param maxPathLen maximum path length
 * @param stream CUDA stream
 */
void invokeBuildLeafMask(
    int8_t* isLeafMask,
    SizeType const* paths,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    SizeType maxPathLen,
    cudaStream_t stream
);

/**
 * @brief Pack boolean attention masks into int32 for 32x compression
 * 
 * @param packedMask output [batchSize, maxDecodingTokens, ceil(maxDecodingTokens/32)]
 * @param mask input [batchSize, maxDecodingTokens, maxDecodingTokens]
 * @param batchSize number of requests
 * @param maxDecodingTokens maximum tokens per request
 * @param stream CUDA stream
 */
void invokeGetPackedMask(
    SizeType* packedMask,
    bool const* mask,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    cudaStream_t stream
);

/**
 * @brief Pack attention masks from path representation
 * 
 * Builds adjacency matrix from tree paths and packs into int32.
 * 
 * @param packedMask output [batchSize, maxDecodingTokens, numPackedMasks]
 * @param batchSlots batch slot mapping [batchSize]
 * @param paths tree paths [batchSize, maxDecodingTokens, maxPathLen]
 * @param batchSize number of active requests
 * @param maxDecodingTokens maximum tokens per request
 * @param maxPathLen maximum path length
 * @param stream CUDA stream
 */
void invokeGetPackedMaskFromPath(
    SizeType* packedMask,
    SizeType const* batchSlots,
    SizeType const* paths,
    SizeType batchSize,
    SizeType maxDecodingTokens,
    SizeType maxPathLen,
    cudaStream_t stream
);

} // namespace eagle
} // namespace kernels
} // namespace turbomind
