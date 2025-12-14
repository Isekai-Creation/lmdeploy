/*
 * Copyright (c) 2024, LMDeploy Contributors.
 *
 * Target-tree decode helpers for TurboMind EAGLE.
 *
 * These kernels prepare flattened generation inputs for running the
 * target model over an EAGLE speculation tree. The initial version
 * focuses on defining stable layouts and a conservative execution
 * shape; more advanced masking / KV reuse can be layered on top.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

using SizeType = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Parameters for preparing generation inputs for target-tree decode.
 *
 * The goal is to map from the EAGLE speculation tree (stored in
 * [max_batch_size, max_decoding_tokens, max_path_len] draft_paths) to
 * a flat token list that the base model can consume in a single
 * generation pass. For this first iteration we keep the layout
 * intentionally simple:
 *
 *  - We select up to max_decoding_tokens nodes per active slot.
 *  - For each selected node we emit one entry in output_ids /
 *    position_ids / hidden_indices.
 *
 * More advanced tree-shaping (e.g. per-depth grouping) can be layered
 * on top of this interface without changing its contract.
 */
struct PrepareGenTargetTreeParams {
    // EAGLE tree inputs
    SizeType const* draft_paths;    // [max_batch_size, max_decoding_tokens, max_path_len]
    SizeType const* batch_slots;    // [batch_size] -> [max_batch_size] mapping
    TokenIdType const* draft_tokens; // [max_batch_size, max_decoding_tokens]

    // Base-model sequence state for each slot. These are typically the
    // current decode-time sequence / context lengths before the
    // target-tree decode pass is applied. Either pointer may be null,
    // in which case a base length of zero is assumed.
    SizeType const* base_sequence_lengths;  // [max_batch_size] or nullptr
    SizeType const* base_context_lengths;   // [max_batch_size] or nullptr

    // Output buffers (device):
    //
    //  - output_ids:       flattened draft token ids for all selected nodes
    //  - position_ids:     position ids for each tree token
    //  - hidden_indices:   mapping from flat index -> (slot, token_idx)
    //
    // All three buffers are sized for a conservative upper bound
    // num_tree_tokens <= batch_size * max_decoding_tokens.
    TokenIdType* output_ids;        // [batch_size * max_decoding_tokens]
    SizeType*    position_ids;      // [batch_size * max_decoding_tokens]
    SizeType*    hidden_indices;    // [batch_size * max_decoding_tokens * 2], packed (slot, token_idx)

    // Per-slot metadata describing how many tree tokens are emitted
    // for each active sequence and what the effective sequence/context
    // lengths are for the speculative pass.
    SizeType* spec_gen_lengths;       // [batch_size]
    SizeType* next_sequence_lengths;  // [batch_size]
    SizeType* next_context_lengths;   // [batch_size]

    // Dimensions.
    SizeType batch_size;
    SizeType max_batch_size;
    SizeType max_decoding_tokens;
    SizeType max_path_len;

    cudaStream_t stream;
};

/**
 * @brief Prepare flattened generation inputs for target-tree decode.
 *
 * This helper walks the speculation tree for each active slot and
 * emits at most max_decoding_tokens nodes per slot into the flat
 * buffers. For now we simply select the first max_decoding_tokens
 * non-root nodes encountered along each path; future revisions can
 * adopt a richer strategy (e.g. depth-wise grouping) without changing
 * the signature.
 */
void invokePrepareGenTargetTreeInputs(PrepareGenTargetTreeParams const& params);

/**
 * @brief Reduce tree logits to per-node target_ids and scatter into target_tokens.
 *
 * Given a logits matrix [num_tree_tokens, vocab_size_padded] and a
 * hidden_indices buffer [num_tree_tokens, 2] containing (slot, token_idx)
 * pairs, this helper computes an argmax over the vocabulary dimension for
 * each row and writes the resulting token id into:
 *
 *   target_tokens[slot * max_decoding_tokens + token_idx]
 *
 * The target_tokens buffer is laid out as
 * [max_batch_size, max_decoding_tokens] in row-major order and is shared
 * with the existing EAGLE acceptance kernels.
 */
void invokeTreeLogitsToTargetTokens(const float*     logits,
                                    SizeType         num_tree_tokens,
                                    SizeType         vocab_size_padded,
                                    const SizeType*  hidden_indices,
                                    SizeType         max_batch_size,
                                    SizeType         max_decoding_tokens,
                                    TokenIdType*     target_tokens,
                                    cudaStream_t     stream);

/**
 * @brief Parameters for reducing tree logits to per-node target IDs.
 *
 * This helper takes a dense logits buffer for all tree tokens and
 * produces per-node top-1 target token IDs in the flattened
 * EAGLE `target_tokens` layout using the (slot, token_idx) mapping
 * captured in `hidden_indices`.
 */
struct TreeLogitsToTargetsParams {
    const float* logits;            // [num_tree_tokens, vocab_size]
    SizeType     num_tree_tokens;   // number of valid tree token rows
    SizeType     vocab_size;        // padded vocab size

    const SizeType* hidden_indices; // [num_tree_tokens, 2] packed (slot, token_idx)
    SizeType        max_batch_size;
    SizeType        max_decoding_tokens;

    TokenIdType* target_tokens;     // [max_batch_size, max_decoding_tokens]

    // Optional draft-vocab remap: when non-null, logits are over a
    // reduced draft vocab and each argmax id must be mapped to the
    // full target vocab using draft_id_to_target[id].
    const TokenIdType* draft_id_to_target{nullptr}; // [vocab_size] or null

    cudaStream_t stream;
};

/**
 * @brief Reduce tree logits to per-node target IDs on device.
 *
 * For each tree token row this computes an argmax over the vocabulary
 * dimension and scatters the resulting ID into the flattened
 * `target_tokens[slot * max_decoding_tokens + token_idx]` array using
 * the (slot, token_idx) mapping provided in `hidden_indices`.
 */
void invokeTreeLogitsToTargetIds(TreeLogitsToTargetsParams const& params);

/**
 * @brief Gather per-node packed masks for tree tokens.
 *
 * Given the per-step packed masks buffer laid out as
 *   packed_masks[batch_size, max_decoding_tokens, num_packed]
 * and a compact (slot, token_idx) mapping for each tree token row,
 * this helper produces a flattened [num_tree_tokens, num_packed]
 * mask buffer aligned with the compact tree token order used by
 * target-tree decode.
 */
void invokeGatherTreePackedMask(const SizeType*  packed_masks,
                                SizeType         batch_size,
                                SizeType         max_decoding_tokens,
                                SizeType         num_packed,
                                const SizeType*  compact_hidden_indices,
                                SizeType         num_tree_tokens,
                                SizeType*        out_packed_masks,
                                cudaStream_t     stream);

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind
