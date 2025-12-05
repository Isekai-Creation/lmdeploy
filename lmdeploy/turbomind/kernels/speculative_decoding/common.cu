/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * 
 * Implementation of common speculative decoding CUDA kernels
 */

#include "common.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

/**
 * @brief Kernel: Accept draft tokens by comparing with target
 * 
 * Each thread handles one batch item. Walks the best path and accepts
 * matching tokens until a mismatch is found.
 */
__global__ void acceptDraftTokensKernel(
    TokenIdType* output_ids,
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType* accepted_lengths,
    SizeType* sequence_lengths,
    SizeType const* paths,
    SizeType const* best_path_ids,
    SizeType const* batch_slots,
    SizeType batch_size,
    SizeType max_seq_len,
    SizeType max_draft_tokens,
    SizeType max_path_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Map to global batch slot
    int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    int seq_len = sequence_lengths[slot];
    int best_path = best_path_ids[slot];
    
    // Walk the path and accept matching tokens
    int accepted = 0;
    for (int i = 0; i < max_draft_tokens; ++i) {
        // Get path index for this position
        int path_idx = paths[slot * max_draft_tokens * max_path_len + 
                            best_path * max_path_len + i];
        
        if (path_idx < 0) break;  // End of path marker
        
        // Get draft and target tokens at this path position
        int draft_token = draft_ids[slot * max_draft_tokens + path_idx];
        int target_token = target_ids[slot * max_draft_tokens + path_idx];
        
        if (draft_token == target_token) {
            // Accept token
            if (seq_len + accepted < max_seq_len) {
                output_ids[slot * max_seq_len + seq_len + accepted] = draft_token;
                accepted++;
            }
        } else {
            // Rejection: stop here, but add the correct target token
            if (seq_len + accepted < max_seq_len) {
                output_ids[slot * max_seq_len + seq_len + accepted] = target_token;
                accepted++;
            }
            break;
        }
    }
    
    // Update metadata
    accepted_lengths[slot] = accepted;
    sequence_lengths[slot] += accepted;
}

template <typename T>
void acceptDraftTokens(AcceptDraftTokensParams<T> const& params) {
    // One block per batch item, single thread per block for simplicity
    // TODO: Optimize with multiple threads per block for large paths
    dim3 grid(params.batch_size);
    dim3 block(1);
    
    acceptDraftTokensKernel<<<grid, block, 0, params.stream>>>(
        params.output_ids,
        params.draft_ids,
        params.target_ids,
        params.accepted_lengths,
        params.sequence_lengths,
        params.paths,
        params.best_path_ids,
        params.batch_slots,
        params.batch_size,
        params.max_seq_len,
        params.max_draft_tokens,
        params.max_path_len
    );
}

// Explicit template instantiation for common types
template void acceptDraftTokens<float>(AcceptDraftTokensParams<float> const&);
template void acceptDraftTokens<half>(AcceptDraftTokensParams<half> const&);

/**
 * @brief Kernel: Pack accepted paths into linear memory
 */
__global__ void packAcceptedPathsKernel(
    SizeType* accepted_lengths_cumsum,
    SizeType* paths_offsets,
    SizeType const* accepted_lengths,
    SizeType const* best_path_ids,
    SizeType const* paths,
    SizeType const* batch_slots,
    SizeType batch_size,
    SizeType num_paths,
    SizeType max_path_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    int best_path = best_path_ids[slot];
    int accepted_len = accepted_lengths[slot];
    
    // Compute cumulative sum offset
    int offset = (batch_idx == 0) ? 0 : accepted_lengths_cumsum[batch_idx - 1];
    
    // Pack accepted path
    for (int i = 0; i < accepted_len && i < max_path_len; ++i) {
        int path_idx = paths[slot * num_paths * max_path_len + 
                            best_path * max_path_len + i];
        paths_offsets[offset + i] = path_idx;
    }
}

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
) {
    dim3 grid(batch_size);
    dim3 block(1);
    
    packAcceptedPathsKernel<<<grid, block, 0, stream>>>(
        accepted_lengths_cumsum,
        paths_offsets,
        accepted_lengths,
        best_path_ids,
        paths,
        batch_slots,
        batch_size,
        num_paths,
        max_path_len
    );
}

/**
 * @brief Kernel: Rewind KV cache for rejected tokens
 *
 * Marks tail KV blocks as free in a per‑sequence block table. The caller is
 * responsible for mapping these logical blocks back to the engine’s actual
 * cache manager (e.g. TurboMind’s SequenceManager / BlockManager).
 */
__global__ void kvCacheRewindKernel(
    void** kv_cache_blocks,
    SizeType const* rewind_lengths,
    SizeType const* batch_slots,
    SizeType const* block_tables,
    SizeType batch_size,
    SizeType num_layers,
    SizeType block_size,
    SizeType max_blocks_per_seq
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    int rewind_len = rewind_lengths ? rewind_lengths[slot] : 0;
    
    if (rewind_len <= 0) return;
    
    // Calculate how many whole blocks from the tail we should consider
    // rewinding for this sequence.
    int blocks_to_free = (rewind_len + block_size - 1) / block_size;

    if (!block_tables || blocks_to_free <= 0) {
        return;
    }

    // `block_tables` is laid out as [max_batch_size, max_blocks_per_seq]
    // where each row holds the logical block IDs for a given sequence. For
    // now we simply mark the last `blocks_to_free` entries in the row as -1
    // so higher‑level code can treat them as free / invalidated.
    SizeType* table = const_cast<SizeType*>(block_tables);
    SizeType* row   = table + static_cast<size_t>(slot) * max_blocks_per_seq;

    for (int i = 0; i < blocks_to_free && i < max_blocks_per_seq; ++i) {
        int idx = max_blocks_per_seq - 1 - i;
        row[idx] = -1;
    }

    // Optionally clear the corresponding kv_cache_blocks pointers when
    // provided. The layout of `kv_cache_blocks` is engine‑specific; we
    // assume a simple [num_layers, max_blocks_per_seq] layout here.
    if (kv_cache_blocks) {
        for (int layer = 0; layer < num_layers; ++layer) {
            void** layer_blocks = kv_cache_blocks + static_cast<size_t>(layer) * max_blocks_per_seq;
            for (int i = 0; i < blocks_to_free && i < max_blocks_per_seq; ++i) {
                int idx = max_blocks_per_seq - 1 - i;
                layer_blocks[idx] = nullptr;
            }
        }
    }
}

void invokeKVCacheRewind(KVCacheRewindParams const& params) {
    dim3 grid(params.batch_size);
    dim3 block(1);
    
    kvCacheRewindKernel<<<grid, block, 0, params.stream>>>(
        params.kv_cache_blocks,
        params.rewind_lengths,
        params.batch_slots,
        params.block_tables,
        params.batch_size,
        params.num_layers,
        params.block_size,
        params.max_blocks_per_seq
    );
}

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
