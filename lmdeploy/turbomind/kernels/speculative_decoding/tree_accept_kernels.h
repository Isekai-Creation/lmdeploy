/*
 * Tree-based acceptance kernels for EAGLE speculation.
 *
 * EAGLE A31: prototype tree-accept kernel – see EAGLE_TODO.md
 * (🧪, GPU/CI validation pending). These kernels evaluate acceptance
 * lengths along SpeculationTree paths directly on device and select
 * the best path per sequence. They are currently used via test/
 * benchmark bindings and are not yet wired into LlamaV2_eagle;
 * GPU CI coverage is required before promoting them to production
 * decode paths.
 */

#pragma once

#include <cuda_runtime.h>

#include "common.h"

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

/**
 * @brief Device-side tree-based acceptance by ids and paths.
 *
 * For each active sequence, this helper walks all candidate paths in the
 * SpeculationTree (as encoded in @p paths) and counts how many tokens
 * would be accepted under the "accept while draft==target, include the
 * first mismatching target token" rule. It then selects the path with
 * the longest accepted length per sequence and materializes both the
 * best path index and accepted tokens.
 *
 * Layout assumptions:
 *   - draft_ids / target_ids: [maxBatchSize, maxDraftTokens]
 *   - paths: [maxBatchSize, numPaths, maxPathLen]
 *       where each entry is a node index (0 = root, -1 = terminator,
 *       >0 encodes token_idx+1).
 *   - batch_slots: optional [batchSize] mapping from local batch index
 *       to global slot in [0, maxBatchSize).
 *   - best_path_ids: [maxBatchSize]
 *   - accepted_lens: [maxBatchSize]
 *   - accepted_tokens: [maxBatchSize, maxPathLen]
 */
void invokeTreeAcceptByIdsWithPaths(
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType const*    paths,
    SizeType const*    batch_slots,
    SizeType           batch_size,
    SizeType           max_batch_size,
    SizeType           num_paths,
    SizeType           max_path_len,
    SizeType           max_draft_tokens,
    SizeType*          best_path_ids,
    SizeType*          accepted_lens,
    TokenIdType*       accepted_tokens,
    cudaStream_t       stream);

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind
