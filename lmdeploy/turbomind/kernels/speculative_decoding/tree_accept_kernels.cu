/*
 * CUDA kernels for tree-based EAGLE acceptance by ids and paths.
 *
 * Parity-oriented port of TRT-LLM's acceptDraftTokensByIdsWithPaths:
 * - Longest-prefix acceptance per path (draft==target), stop on first mismatch.
 * - Root (0) is skipped, -1 terminates a path.
 * - Materializes accepted tokens from target_ids for the winning path.
 * End-id handling and Medusa logits can be added later if needed.
 */

#include "tree_accept_kernels.h"

#include <algorithm>
#include <cub/cub.cuh>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

namespace {

struct Int4Max {
    __device__ __forceinline__ int4 operator()(const int4& a, const int4& b) const
    {
        return a.x >= b.x ? a : b;
    }
};

template <typename SizeT>
__device__ __forceinline__ SizeT flat_index3(SizeT i, SizeT j, SizeT k, SizeT dim1, SizeT dim2)
{
    return (i * dim1 + j) * dim2 + k;
}

template <typename SizeT>
__global__ void treeAcceptByIdsWithPathsKernel(
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeT const*       paths,
    TokenIdType const* end_ids,
    SizeT const*       batch_slots,
    SizeT              batch_size,
    SizeT              max_batch_size,
    SizeT              num_paths,
    SizeT              max_path_len,
    SizeT              max_draft_tokens,
    SizeT*             best_path_ids,
    SizeT*             accepted_lens,
    TokenIdType*       accepted_tokens)
{
    const SizeT local_idx = static_cast<SizeT>(blockIdx.x);
    if (local_idx >= batch_size) {
        return;
    }

    const SizeT slot = batch_slots ? batch_slots[local_idx] : local_idx;
    if (slot < 0 || slot >= max_batch_size) {
        return;
    }

    int4 thread_best{-1, -1, 0, 0};  // len, path_idx, has_end, last_idx

    // Evaluate each path owned by this thread.
    for (SizeT path_idx = static_cast<SizeT>(threadIdx.x); path_idx < num_paths; path_idx += blockDim.x) {
        int accepted_len     = 0;
        SizeT best_next_idx  = 0;
        const SizeT path_off = flat_index3(slot, path_idx, 0, num_paths, max_path_len);

        for (SizeT ti = 0; ti < max_path_len; ++ti) {
            const SizeT node_idx = paths[path_off + ti];
            if (node_idx == static_cast<SizeT>(-1)) {
                break;  // terminator
            }
            if (node_idx == 0) {
                // Root sentinel; skip.
                continue;
            }

            const int token_idx = static_cast<int>(node_idx) - 1;
            if (token_idx < 0 || token_idx >= static_cast<int>(max_draft_tokens)) {
                break;
            }

            const TokenIdType draft_tok
                = draft_ids[static_cast<size_t>(slot) * max_draft_tokens + static_cast<SizeT>(token_idx)];
            const TokenIdType target_tok
                = target_ids[static_cast<size_t>(slot) * max_draft_tokens + static_cast<SizeT>(token_idx)];

            best_next_idx = node_idx;

            const bool is_eos = (end_ids != nullptr && target_tok == end_ids[slot]);
            if (draft_tok != target_tok || is_eos) {
                break;
            }

            accepted_len += 1;
        }

        if (thread_best.x < accepted_len) {
            thread_best.x = accepted_len;
            thread_best.y = static_cast<int>(path_idx);
            thread_best.w = static_cast<int>(best_next_idx);
        }
    }

    // Reduce across threads in the block to find the best path.
    using BlockReduce = cub::BlockReduce<int4, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int4 block_best = BlockReduce(temp_storage).Reduce(thread_best, Int4Max{});

    __shared__ int4 total;
    if (threadIdx.x == 0) {
        total = block_best;
    }
    __syncthreads();

    const int best_path    = total.y < 0 ? 0 : total.y;
    const int accepted_len = std::max(0, total.x);

    if (accepted_tokens && threadIdx.x == 0 && accepted_len > 0) {
        const SizeT path_off = flat_index3(slot, static_cast<SizeT>(best_path), 0, num_paths, max_path_len);
        int         written  = 0;
        for (SizeT ti = 0; ti < max_path_len && written < accepted_len; ++ti) {
            const SizeT node_idx = paths[path_off + ti];
            if (node_idx == static_cast<SizeT>(-1)) {
                break;
            }
            if (node_idx == 0) {
                continue;
            }
            const int token_idx = static_cast<int>(node_idx) - 1;
            if (token_idx < 0 || token_idx >= static_cast<int>(max_draft_tokens)) {
                break;
            }
            accepted_tokens[static_cast<size_t>(slot) * max_path_len + written]
                = target_ids[static_cast<size_t>(slot) * max_draft_tokens + static_cast<SizeT>(token_idx)];
            ++written;
        }
    }

    if (threadIdx.x == 0) {
        best_path_ids[slot] = static_cast<SizeT>(best_path);
        accepted_lens[slot] = static_cast<SizeT>(accepted_len);
    }
}

}  // namespace

void invokeTreeAcceptByIdsWithPaths(
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType const*    paths,
    TokenIdType const* end_ids,
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

    constexpr int BLOCK = 256;
    const dim3    grid(static_cast<unsigned int>(batch_size));
    const dim3    block(BLOCK);

    treeAcceptByIdsWithPathsKernel<<<grid, block, 0, stream>>>(
        draft_ids,
        target_ids,
        paths,
        end_ids,
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
