/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * Adapted from TensorRT-LLM's speculative decoding kernels
 *
 * Common utilities for speculative decoding CUDA kernels
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

// Type aliases for clarity
using SizeType = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Parameters for accepting draft tokens
 * 
 * This kernel compares draft tokens with target model outputs
 * and accepts matching tokens, updating output sequences.
 */
template <typename T>
struct AcceptDraftTokensParams {
    // Output: accepted tokens added to output_ids
    TokenIdType* output_ids;              // [maxBatchSize, maxSeqLen]
    
    // Input: draft and target tokens
    TokenIdType const* draft_ids;         // [maxBatchSize, maxDraftTokens]
    TokenIdType const* target_ids;        // [maxBatchSize, maxDraftTokens]
    
    // Output: acceptance metadata
    SizeType* accepted_lengths;           // [maxBatchSize]
    SizeType* sequence_lengths;           // [maxBatchSize]
    
    // Input: paths for tree-based verification
    SizeType const* paths;                // [maxBatchSize, maxDecodingTokens, maxPathLen]
    SizeType const* best_path_ids;        // [maxBatchSize]
    
    // Optional: EOS checking
    TokenIdType const* end_ids;           // [maxBatchSize]
    bool* finished_states;                // [maxBatchSize]
    
    // Batch management
    SizeType const* batch_slots;          // [batchSize] -> [maxBatchSize] mapping
    
    SizeType batch_size;
    SizeType max_batch_size;
    SizeType max_seq_len;
    SizeType max_draft_tokens;
    SizeType max_path_len;
    
    cudaStream_t stream;
};

/**
 * @brief Accept draft tokens by comparing with target model outputs
 * 
 * Implements greedy verification: accepts tokens that match exactly.
 * Stops at first mismatch and includes the correct target token.
 */
template <typename T>
void acceptDraftTokens(AcceptDraftTokensParams<T> const& params);

/**
 * @brief Pack accepted paths for efficient memory layout
 * 
 * Linearly packs accepted paths in memory according to acceptance lengths.
 * Used for efficient KV cache updates.
 */
void invokePackAcceptedPaths(
    SizeType* accepted_lengths_cumsum,
    SizeType* paths_offsets,
    SizeType const* accepted_lengths,
    SizeType const* best_path_ids,
    SizeType const* paths,
    SizeType const* batch_slots,
    SizeType batch_size,
    SizeType max_batch_size,
    SizeType num_paths,
    SizeType max_path_len,
    cudaStream_t stream
);

/**
 * @brief Parameters for KV cache rewind
 * 
 * Manages freeing KV cache blocks for rejected draft tokens.
 */
struct KVCacheRewindParams {
    // KV cache block pointers (model-specific layout)
    void** kv_cache_blocks;               // [num_layers][num_blocks]
    
    // Rewind metadata
    SizeType const* rewind_lengths;       // [maxBatchSize] - tokens to rewind
    SizeType const* batch_slots;          // [batchSize]
    SizeType const* block_tables;         // [maxBatchSize, maxBlocksPerSeq]
    
    SizeType batch_size;
    SizeType max_batch_size;
    SizeType num_layers;
    SizeType block_size;                  // tokens per block
    SizeType max_blocks_per_seq;
    
    cudaStream_t stream;
};

/**
 * @brief Rewind KV cache by marking blocks as free
 * 
 * Critical for memory efficiency - avoids complex page reuse logic.
 * Blocks are marked as available without actual memory operations.
 */
void invokeKVCacheRewind(KVCacheRewindParams const& params);

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
