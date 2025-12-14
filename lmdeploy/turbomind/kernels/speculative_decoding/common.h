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
 * @brief Lightweight host wrapper for the acceptance kernel.
 *
 * This helper is intended for test and debugging bindings. It builds an
 * AcceptDraftTokensParams instance from raw pointers and launches the
 * device kernel on the provided CUDA stream.
 */
void launchAcceptDraftTokensKernel(
    TokenIdType*       output_ids,
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType*          accepted_lengths,
    SizeType*          sequence_lengths,
    SizeType const*    paths,
    SizeType const*    best_path_ids,
    TokenIdType const* end_ids,
    bool*              finished_states,
    SizeType const*    batch_slots,
    SizeType           batch_size,
    SizeType           max_batch_size,
    SizeType           max_seq_len,
    SizeType           max_draft_tokens,
    SizeType           max_path_len,
    cudaStream_t       stream);

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
 * @brief Lightweight host wrapper for the path packing kernel.
 *
 * This helper mirrors invokePackAcceptedPaths but is structured for
 * direct use from pybind11 bindings and tests.
 */
void launchPackAcceptedPathsKernel(
    SizeType*       accepted_lengths_cumsum,
    SizeType*       paths_offsets,
    SizeType const* accepted_lengths,
    SizeType const* best_path_ids,
    SizeType const* paths,
    SizeType const* batch_slots,
    SizeType        batch_size,
    SizeType        max_batch_size,
    SizeType        num_paths,
    SizeType        max_path_len,
    cudaStream_t    stream);

/**
 * @brief Compute successor offsets/counts from cumulative runtime lengths.
 */
void invokeComputeSuccessorMeta(SizeType const* runtime_offsets,
                                SizeType        batch_size,
                                SizeType*       successor_offsets,
                                SizeType*       successor_counts,
                                cudaStream_t    stream);

/**
 * @brief Extract per-node successor counts and flattened TopK offsets from
 *        tree paths.
 *
 * This mirrors TensorRT-LLM's successor histogram logic: for each request,
 * it builds an adjacency matrix from the draft paths, counts successors per
 * node, and compacts the non-zero counts into a flat array. The offsets
 * array is length batch_size+1; successor_offsets[i] marks the starting
 * index of request i inside successor_counts, and successor_offsets[end]
 * stores the total number of nodes with successors.
 *
 * @param paths                Flattened tree paths [batch, max_decoding_tokens, max_path_len]
 * @param batch_size           Active batch size
 * @param max_decoding_tokens  Max draft tokens per sequence (tree width)
 * @param max_path_len         Max path length (tree depth)
 * @param successor_offsets    Output offsets [batch_size + 1]
 * @param successor_counts     Output flattened successor counts; sized at least batch_size*max_decoding_tokens
 * @param num_successors       Optional per-node successor histogram [batch_size, max_decoding_tokens]
 * @param stream               CUDA stream
 */
void invokeExtractSuccessorsFromPaths(SizeType const* paths,
                                      SizeType        batch_size,
                                      SizeType        max_decoding_tokens,
                                      SizeType        max_path_len,
                                      SizeType*       successor_offsets,
                                      SizeType*       successor_counts,
                                      SizeType*       num_successors,
                                      cudaStream_t    stream);

/**
 * @brief Replicate a single-spec tree (paths_flat) across all batch slots.
 *
 * paths_flat is laid out as [num_paths, max_path_len] and represents the
 * speculative tree discovered for a single sequence. This helper expands
 * it into a per-slot draft_paths layout:
 *
 *   draft_paths[slot, path, level] = paths_flat[path, level]
 *
 * for all slot in [0, batch_size) and path in [0, max_decoding_tokens),
 * filling unused entries with -1 when path >= num_paths.
 */
void invokeReplicatePathsFromFlat(SizeType const* paths_flat,
                                  SizeType        num_paths,
                                  SizeType        max_path_len,
                                  SizeType*       draft_paths,
                                  SizeType        batch_size,
                                  SizeType        max_decoding_tokens,
                                  cudaStream_t    stream);

/**
 * @brief Build a per-token linked-list style tree representation from
 *        draft_paths for EAGLE3.
 *
 * For each slot and draft token, this helper populates:
 *   - positions: approximate depth / position in the speculative tree,
 *   - retrive_index: canonical index per token (slot-local),
 *   - retrive_next_token: first child in the tree for this token (or -1),
 *   - retrive_next_sibling: sibling linkage (or -1).
 *
 * The current implementation assumes a single-sequence tree cloned
 * across batch slots and processes each slot independently on device.
 */
void invokeBuildLinkedTreeFromDraftPaths(SizeType const* draft_paths,
                                         SizeType        batch_size,
                                         SizeType        max_decoding_tokens,
                                         SizeType        max_path_len,
                                         SizeType        tokens_per_seq,
                                         SizeType*       positions,
                                         SizeType*       retrive_index,
                                         SizeType*       retrive_next_token,
                                         SizeType*       retrive_next_sibling,
                                         cudaStream_t    stream);

/**
 * @brief Build per-slot forced tail tokens/lengths from accepted tokens.
 *
 * For each slot i:
 *   - Let len = accepted_lens[i].
 *   - Extras = max(min(len, max_tail_len) - 1, 0).
 *   - forced_lengths[i] = Extras.
 *   - forced_tokens[i, t] = accepted_tokens[i, 1 + t] for t in [0, Extras),
 *     and -1 for remaining positions up to max_tail_len.
 *
 * This mirrors LlamaV2's host-side tail construction but runs entirely
 * on device, enabling a GPU-only tail path when combined with
 * DynamicDecodeLayer's TM_ENABLE_GPU_TAIL branch.
 */
void invokeBuildForcedTailsFromAccepted(TokenIdType const* accepted_tokens,
                                        SizeType const*    accepted_lens,
                                        SizeType           batch_size,
                                        SizeType           max_path_len,
                                        SizeType           max_tail_len,
                                        TokenIdType*       forced_tokens,
                                        SizeType*          forced_lengths,
                                        cudaStream_t       stream);


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

void invokeKVCacheRewind(KVCacheRewindParams const& params);

struct ActiveSlotsParams {
    const bool* finished{nullptr};        // [max_batch_size]
    const int*  sequence_lengths{nullptr}; // [max_batch_size]
    const int*  seq_limit_len{nullptr};    // [max_batch_size]
    int         batch_size{0};            // logical decode batch for this step
    int         max_batch_size{0};        // engine_param_.max_batch_size
    int*        active_slots{nullptr};    // [max_batch_size] compact list of active slots
    int*        active_inverse{nullptr};  // [max_batch_size] slot -> compact index or -1
    int*        active_count{nullptr};    // [1] scalar count of active slots
    cudaStream_t stream{};                // CUDA stream
};

// Compute the compact set of active slots for an EAGLE step.
//
// A slot is considered active when:
//   - finished[slot] == false
//   - sequence_lengths[slot] < seq_limit_len[slot] (when seq_limit_len provided)
//
// The kernel writes:
//   active_slots[0..active_count-1]  = slot indices
//   active_inverse[slot]             = compact index in [0, active_count) or -1
//   *active_count                    = number of active slots
//
// active_inverse is optional and may be null when not needed.
void launchComputeActiveSlots(const ActiveSlotsParams& params);

struct EntropyMaskParams {
    float const* logits{nullptr};             // [rows, cols]
    float*       logits_out{nullptr};         // optional out (if null, in-place)
    float const* probs{nullptr};              // [rows, cols]
    float const* entropies{nullptr};          // [rows]
    float const* posterior_thresholds{nullptr}; // [batch]
    float const* posterior_alphas{nullptr};     // [batch]
    float const* temperatures{nullptr};         // [batch]
    const bool*  skip_decode{nullptr};          // [rows]
    float*       runtime_top_p{nullptr};        // [rows] optional
    const int*   generation_lengths{nullptr};   // [batch]
    int          rows{0};                       // batch_size * max_tokens
    int          cols{0};                       // vocab_size_pad
    int          max_tokens{0};                 // tokens_per_seq
    int          batch_size{0};
    cudaStream_t stream{};
};

// Apply posterior/typical entropy-based masking to logits.
// - logits/probs/entropies are flat [rows, cols]/[rows].
// - Uses posterior_thresholds/posterior_alphas/temperatures per batch_slot.
// - Respects skip_decode and generation_lengths, writes -inf into masked logits.
// - If runtime_top_p != nullptr, writes per-row top-p marker (TRT-style 0/1).
void maskLogitsBasedOnEntropy(const EntropyMaskParams& params);

// Compute softmax + entropy per row. logits/probs layout: [rows, cols].
void invokeSoftmaxWithEntropy(const float* logits,
                              float*       probs,
                              float*       entropy,
                              int          rows,
                              int          cols,
                              cudaStream_t stream);

// Argmax per row of a [rows, cols] matrix. Optional skip_decode masks rows
// (out[row] = -1 when skip_decode[row] is true).
void launch_argmax_rows(const float* logits,
                        int          rows,
                        int          cols,
                        const bool*  skip_decode,
                        int*         out,
                        cudaStream_t stream);

// Derive skip_decode mask from generation_lengths.
// - rows = batch_size * max_tokens.
// - skip_decode[r] = true when token_idx >= generation_lengths[batch_slot]
//   (i.e. padded positions).
void launch_set_skip_decode(const int* generation_lengths,
                            int        batch_size,
                            int        max_tokens,
                            bool*      skip_decode,
                            cudaStream_t stream);

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
