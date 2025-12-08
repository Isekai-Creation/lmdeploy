/*
 * CUDA kernels for tree-based EAGLE acceptance by ids and paths.
 *
 * EAGLE A31: prototype tree-accept kernel – see EAGLE_TODO.md
 * (🧪, GPU/CI validation pending). These kernels are intended for
 * A-scope testing/benchmarking and are not yet wired into LlamaV2_eagle.
 * They require GPU CI coverage before being considered production-ready.
 */

#include "tree_accept_kernels.h"

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

namespace {

__global__ void treeAcceptByIdsWithPathsKernel(
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
    TokenIdType*       accepted_tokens)
{
    const SizeType local_idx = static_cast<SizeType>(blockIdx.x);
    if (local_idx >= batch_size) {
        return;
    }

    const SizeType slot = batch_slots ? batch_slots[local_idx] : local_idx;
    if (slot < 0 || slot >= max_batch_size) {
        return;
    }

    int best_path    = 0;
    int best_accepts = 0;

    // For each path, walk its nodes and count how many tokens would be
    // accepted under a strict ID-equality rule (accept while draft_id ==
    // target_id; stop on the first mismatch without including the
    // mismatching token). This mirrors the host-side implementation in
    // LlamaV2_eagle when tree-accept is enabled.
    for (SizeType path_idx = 0; path_idx < num_paths; ++path_idx) {
        int accepted = 0;

        for (SizeType d = 0; d < max_path_len; ++d) {
            const SizeType node_idx = paths[
                (static_cast<size_t>(slot) * num_paths + path_idx) * max_path_len + d];

            // node_idx == 0: root; skip but keep walking.
            if (node_idx <= 0) {
                if (node_idx < 0) {
                    // Terminator.
                    break;
                }
                continue;
            }

            const int token_idx = static_cast<int>(node_idx) - 1;
            if (token_idx < 0 || token_idx >= max_draft_tokens) {
                break;
            }

            const TokenIdType draft_id =
                draft_ids[static_cast<size_t>(slot) * max_draft_tokens + token_idx];
            const TokenIdType target_id =
                target_ids[static_cast<size_t>(slot) * max_draft_tokens + token_idx];

            if (draft_id != target_id) {
                break;
            }

            accepted += 1;
        }

        if (accepted > best_accepts) {
            best_accepts = accepted;
            best_path    = static_cast<int>(path_idx);
        }
    }

    best_path_ids[slot]   = static_cast<SizeType>(best_path);
    accepted_lens[slot]   = static_cast<SizeType>(best_accepts);

    // Materialize accepted tokens for the best path using target_ids.
    if (accepted_tokens && best_accepts > 0) {
        int written = 0;
        for (SizeType d = 0; d < max_path_len && written < best_accepts; ++d) {
            const SizeType node_idx = paths[
                (static_cast<size_t>(slot) * num_paths + static_cast<SizeType>(best_path)) * max_path_len + d];

            if (node_idx <= 0) {
                if (node_idx < 0) {
                    break;
                }
                continue;
            }
            const int token_idx = static_cast<int>(node_idx) - 1;
            if (token_idx < 0 || token_idx >= max_draft_tokens) {
                break;
            }

            const TokenIdType target_id
                = target_ids[static_cast<size_t>(slot) * max_draft_tokens + token_idx];
            accepted_tokens[static_cast<size_t>(slot) * max_path_len + written] = target_id;
            ++written;
        }
    }
}

}  // namespace

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
    cudaStream_t       stream)
{
    if (batch_size <= 0 || max_batch_size <= 0 || num_paths <= 0 || max_path_len <= 0
        || max_draft_tokens <= 0) {
        return;
    }

    const dim3 grid(static_cast<unsigned int>(batch_size));
    const dim3 block(1);

    treeAcceptByIdsWithPathsKernel<<<grid, block, 0, stream>>>(
        draft_ids,
        target_ids,
        paths,
        batch_slots,
        batch_size,
        max_batch_size,
        num_paths,
        max_path_len,
        max_draft_tokens,
        best_path_ids,
        accepted_lens,
        accepted_tokens);
}

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind
