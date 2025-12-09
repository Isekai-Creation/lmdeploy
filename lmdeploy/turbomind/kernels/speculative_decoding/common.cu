/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * 
 * Implementation of common speculative decoding CUDA kernels
 */

#include "common.h"
#include "optimized_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cub/cub.cuh>

#include "src/turbomind/utils/eagle_debug.h"

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
    TokenIdType*       output_ids,
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType*          accepted_lengths,
    SizeType*          sequence_lengths,
    SizeType const*    paths,
    SizeType const*    best_path_ids,
    SizeType const*    batch_slots,
    SizeType           batch_size,
    SizeType           max_batch_size,
    SizeType           max_seq_len,
    SizeType           max_draft_tokens,
    SizeType           max_path_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    
    // Map to global batch slot
    const int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    if (slot < 0 || slot >= max_batch_size) {
        // Out-of-range or inactive slot; leave metadata untouched.
        return;
    }

    int seq_len   = sequence_lengths[slot];
    int best_path = best_path_ids[slot];

    // A negative best_path_id conventionally means "no best path".
    if (best_path < 0 || best_path >= max_draft_tokens) {
        accepted_lengths[slot] = 0;
        return;
    }
    
    // Walk the path and accept matching tokens
    int accepted = 0;
    for (int i = 0; i < max_draft_tokens; ++i) {
        // Get path index for this position
        const int path_idx = paths[static_cast<size_t>(slot) * max_draft_tokens * max_path_len
                                  + static_cast<size_t>(best_path) * max_path_len + i];
        
        if (path_idx < 0) {
            // End-of-path sentinel.
            break;
        }
        if (path_idx >= max_draft_tokens) {
            // Out-of-range path index – treat as terminator to avoid
            // reading past the end of draft/target arrays.
            break;
        }
        
        // Get draft and target tokens at this path position
        const int draft_token
            = draft_ids[static_cast<size_t>(slot) * max_draft_tokens + path_idx];
        const int target_token
            = target_ids[static_cast<size_t>(slot) * max_draft_tokens + path_idx];
        
        if (draft_token == target_token) {
            // Accept token
            if (seq_len + accepted < max_seq_len) {
                output_ids[static_cast<size_t>(slot) * max_seq_len + seq_len + accepted]
                    = draft_token;
                accepted++;
            }
        } else {
            // Rejection: stop here, but add the correct target token
            if (seq_len + accepted < max_seq_len) {
                output_ids[static_cast<size_t>(slot) * max_seq_len + seq_len + accepted]
                    = target_token;
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
    // For small paths, a single-threaded kernel is sufficient and avoids
    // launch overhead; for larger paths, dispatch to the optimized
    // multi-threaded version defined in optimized_kernels.cu.
    if (params.max_draft_tokens <= 8) {
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
            params.max_batch_size,
            params.max_seq_len,
            params.max_draft_tokens,
            params.max_path_len);
    }
    else {
        acceptDraftTokensOptimized<T>(params);
    }
}

// Explicit template instantiation for common types
template void acceptDraftTokens<float>(AcceptDraftTokensParams<float> const&);
template void acceptDraftTokens<half>(AcceptDraftTokensParams<half> const&);

void launchAcceptDraftTokensKernel(
    TokenIdType*       output_ids,
    TokenIdType const* draft_ids,
    TokenIdType const* target_ids,
    SizeType*          accepted_lengths,
    SizeType*          sequence_lengths,
    SizeType const*    paths,
    SizeType const*    best_path_ids,
    SizeType const*    batch_slots,
    SizeType           batch_size,
    SizeType           max_batch_size,
    SizeType           max_seq_len,
    SizeType           max_draft_tokens,
    SizeType           max_path_len,
    cudaStream_t       stream)
{
    AcceptDraftTokensParams<float> params{};
    params.output_ids       = const_cast<TokenIdType*>(output_ids);
    params.draft_ids        = draft_ids;
    params.target_ids       = target_ids;
    params.accepted_lengths = accepted_lengths;
    params.sequence_lengths = sequence_lengths;
    params.paths            = paths;
    params.best_path_ids    = best_path_ids;
    params.end_ids          = nullptr;
    params.finished_states  = nullptr;
    params.batch_slots      = batch_slots;
    params.batch_size       = batch_size;
    params.max_batch_size   = max_batch_size;
    params.max_seq_len      = max_seq_len;
    params.max_draft_tokens = max_draft_tokens;
    params.max_path_len     = max_path_len;
    params.stream           = stream;

    if (batch_size <= 0 || max_batch_size <= 0 || max_draft_tokens <= 0 || max_seq_len <= 0
        || max_path_len <= 0) {
        if (::turbomind::isEagleDebugEnabled()) {
            std::fprintf(stderr,
                         "[EAGLE][accept] early-return: batch_size=%d max_batch_size=%d "
                         "max_seq_len=%d max_draft_tokens=%d max_path_len=%d\n",
                         static_cast<int>(batch_size),
                         static_cast<int>(max_batch_size),
                         static_cast<int>(max_seq_len),
                         static_cast<int>(max_draft_tokens),
                         static_cast<int>(max_path_len));
        }
        return;
    }

    // Reuse the main acceptance helper, which will choose between the
    // scalar and optimized kernels based on max_draft_tokens.
    acceptDraftTokens<float>(params);
}

/**
 * @brief Kernel: Pack accepted paths into linear memory
 */
__global__ void packAcceptedPathsKernel(
    SizeType*       accepted_lengths_cumsum,
    SizeType*       paths_offsets,
    SizeType const* accepted_lengths,
    SizeType const* best_path_ids,
    SizeType const* paths,
    SizeType const* batch_slots,
    SizeType        batch_size,
    SizeType        max_batch_size,
    SizeType        num_paths,
    SizeType        max_path_len
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    
    const int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    if (slot < 0 || slot >= max_batch_size) {
        return;
    }

    const int best_path = best_path_ids[slot];
    const int accepted_len = accepted_lengths[slot];

    if (best_path < 0 || best_path >= num_paths || accepted_len <= 0) {
        return;
    }
    
    // Compute cumulative sum offset
    const SizeType offset = (batch_idx == 0) ? 0 : accepted_lengths_cumsum[batch_idx - 1];
    const SizeType total_accepted = accepted_lengths_cumsum[batch_size - 1];
    if (offset >= total_accepted) {
        return;
    }

    const SizeType max_writable = total_accepted - offset;
    const SizeType len = static_cast<SizeType>(accepted_len) < max_writable
                             ? static_cast<SizeType>(accepted_len)
                             : max_writable;
    
    // Pack accepted path
    for (SizeType i = 0; i < len && i < max_path_len; ++i) {
        const SizeType path_idx = paths[static_cast<size_t>(slot) * num_paths * max_path_len
                                       + static_cast<size_t>(best_path) * max_path_len + i];
        if (path_idx < 0) {
            break;
        }
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
        max_batch_size,
        num_paths,
        max_path_len);
}

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
    cudaStream_t    stream)
{
    if (batch_size <= 0 || max_batch_size <= 0 || num_paths <= 0 || max_path_len <= 0) {
        if (::turbomind::isEagleDebugEnabled()) {
            std::fprintf(stderr,
                         "[EAGLE][pack] early-return: batch_size=%d max_batch_size=%d "
                         "num_paths=%d max_path_len=%d\n",
                         static_cast<int>(batch_size),
                         static_cast<int>(max_batch_size),
                         static_cast<int>(num_paths),
                         static_cast<int>(max_path_len));
        }
        return;
    }

    invokePackAcceptedPaths(
        accepted_lengths_cumsum,
        paths_offsets,
        accepted_lengths,
        best_path_ids,
        paths,
        batch_slots,
        batch_size,
        max_batch_size,
        num_paths,
        max_path_len,
        stream);
}

namespace {

__global__ void computeSuccessorMetaKernel(SizeType const* runtime_offsets,
                                           SizeType        batch_size,
                                           SizeType*       successor_offsets,
                                           SizeType*       successor_counts)
{
    const SizeType idx = static_cast<SizeType>(blockIdx.x);
    if (idx >= batch_size) {
        return;
    }
    if (successor_offsets) {
        successor_offsets[idx] = runtime_offsets ? runtime_offsets[idx] : 0;
    }
    if (successor_counts && runtime_offsets) {
        successor_counts[idx] = runtime_offsets[idx + 1] - runtime_offsets[idx];
    }
    if (idx == batch_size - 1 && successor_offsets && runtime_offsets) {
        successor_offsets[idx + 1] = runtime_offsets[idx + 1];
    }
}

}  // namespace

void invokeComputeSuccessorMeta(SizeType const* runtime_offsets,
                                SizeType        batch_size,
                                SizeType*       successor_offsets,
                                SizeType*       successor_counts,
                                cudaStream_t    stream)
{
    if (!runtime_offsets || batch_size <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(batch_size));
    const dim3 block(1);
    computeSuccessorMetaKernel<<<grid, block, 0, stream>>>(
        runtime_offsets, batch_size, successor_offsets, successor_counts);
}

namespace {

__global__ void computeSuccessorCountsKernel(SizeType const* paths,
                                             SizeType        batch_size,
                                             SizeType        max_decoding_tokens,
                                             SizeType        max_path_len,
                                             SizeType*       num_successors)
{
    const SizeType batch_idx = static_cast<SizeType>(blockIdx.x);
    if (batch_idx >= batch_size) {
        return;
    }

    extern __shared__ uint8_t adj[];
    const SizeType adj_size = max_decoding_tokens * max_decoding_tokens;

    // Zero adjacency matrix
    for (SizeType idx = static_cast<SizeType>(threadIdx.x); idx < adj_size; idx += blockDim.x) {
        adj[idx] = 0;
    }
    __syncthreads();

    // Populate adjacency from paths: for each edge (level -> level+1) set adj[from,to]=1
    const SizeType path_base = batch_idx * max_decoding_tokens * max_path_len;
    for (SizeType path_idx = static_cast<SizeType>(threadIdx.x); path_idx < max_decoding_tokens;
         path_idx += blockDim.x) {
        const SizeType* path = paths + path_base + path_idx * max_path_len;
        for (SizeType level = 0; level + 1 < max_path_len; ++level) {
            const SizeType from = path[level];
            const SizeType to   = path[level + 1];
            if (from >= 0 && to >= 0 && from < max_decoding_tokens && to < max_decoding_tokens) {
                adj[from * max_decoding_tokens + to] = 1;
            }
        }
    }
    __syncthreads();

    // Count successors per node
    for (SizeType node = static_cast<SizeType>(threadIdx.x); node < max_decoding_tokens; node += blockDim.x) {
        SizeType count      = 0;
        const SizeType base = node * max_decoding_tokens;
        for (SizeType j = 0; j < max_decoding_tokens; ++j) {
            count += static_cast<SizeType>(adj[base + j]);
        }
        num_successors[batch_idx * max_decoding_tokens + node] = count;
    }
}

template<int BLOCK_SIZE>
__global__ void compactSuccessorsKernel(SizeType const* num_successors,
                                        SizeType        batch_size,
                                        SizeType        max_decoding_tokens,
                                        SizeType*       successor_offsets,
                                        SizeType*       successor_counts)
{
    using BlockScan   = cub::BlockScan<SizeType, BLOCK_SIZE>;
    using BlockReduce = cub::BlockReduce<SizeType, BLOCK_SIZE>;

    __shared__ typename BlockScan::TempStorage   scan_storage;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ SizeType                          total_nodes_with_successors;

    const SizeType tid = static_cast<SizeType>(threadIdx.x);

    // Count how many nodes in this request have successors.
    SizeType nodes_with_successors = 0;
    if (tid < batch_size) {
        const SizeType* row = num_successors + tid * max_decoding_tokens;
        for (SizeType i = 0; i < max_decoding_tokens; ++i) {
            nodes_with_successors += (row[i] > 0 ? 1 : 0);
        }
    }

    SizeType offset = 0;
    BlockScan(scan_storage).ExclusiveSum(nodes_with_successors, offset);

    const SizeType total = BlockReduce(reduce_storage).Sum(nodes_with_successors);
    if (tid == 0) {
        total_nodes_with_successors = total;
    }
    __syncthreads();

    if (tid < batch_size) {
        successor_offsets[tid] = offset;

        const SizeType* row = num_successors + tid * max_decoding_tokens;
        SizeType        cursor = offset;
        for (SizeType i = 0; i < max_decoding_tokens; ++i) {
            const SizeType count = row[i];
            if (count > 0) {
                successor_counts[cursor++] = count;
            }
        }
    }

    if (tid == 0) {
        successor_offsets[batch_size] = total_nodes_with_successors;
    }
}

}  // namespace

void invokeExtractSuccessorsFromPaths(SizeType const* paths,
                                      SizeType        batch_size,
                                      SizeType        max_decoding_tokens,
                                      SizeType        max_path_len,
                                      SizeType*       successor_offsets,
                                      SizeType*       successor_counts,
                                      SizeType*       num_successors,
                                      cudaStream_t    stream)
{
    if (!paths || batch_size <= 0 || max_decoding_tokens <= 0 || max_path_len <= 0
        || !successor_offsets || !successor_counts) {
        return;
    }

    // First pass: build per-node successor histogram from paths.
    const SizeType adj_bytes = max_decoding_tokens * max_decoding_tokens * sizeof(uint8_t);
    constexpr int  BLOCK_HIST = 256;
    computeSuccessorCountsKernel<<<batch_size, BLOCK_HIST, adj_bytes, stream>>>(
        paths, batch_size, max_decoding_tokens, max_path_len, num_successors);

    // Second pass: compact histograms into flat successor_counts with offsets.
    constexpr int BLOCK_COMPACT = 512;
    if (batch_size > BLOCK_COMPACT) {
        // Constraint inherited from TRT: batch size for EAGLE tree decode
        // must be <= 512. Exceeding this would require a multi-block scan.
        if (::turbomind::isEagleDebugEnabled()) {
            std::fprintf(stderr,
                         "[EAGLE][successor][fallback] batch_size=%d exceeds BLOCK_COMPACT=%d; "
                         "successor metadata not updated.\n",
                         static_cast<int>(batch_size),
                         BLOCK_COMPACT);
        }
        return;
    }

    compactSuccessorsKernel<BLOCK_COMPACT><<<1, BLOCK_COMPACT, 0, stream>>>(
        num_successors, batch_size, max_decoding_tokens, successor_offsets, successor_counts);
}

/**
 * @brief Kernel: Rewind KV cache for rejected tokens
 *
 * Marks tail KV blocks as free in a per‑sequence block table. The caller is
 * responsible for mapping these logical blocks back to the engine’s actual
 * cache manager (e.g. TurboMind’s SequenceManager / BlockManager).
 */
__global__ void kvCacheRewindKernel(
    void**           kv_cache_blocks,
    SizeType const*  rewind_lengths,
    SizeType const*  batch_slots,
    SizeType const*  block_tables,
    SizeType         batch_size,
    SizeType         max_batch_size,
    SizeType         num_layers,
    SizeType         block_size,
    SizeType         max_blocks_per_seq
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }
    
    const int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    if (slot < 0 || slot >= max_batch_size) {
        return;
    }

    const int rewind_len = rewind_lengths ? rewind_lengths[slot] : 0;
    
    if (rewind_len <= 0) {
        return;
    }
    
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
    if (params.batch_size <= 0 || params.max_batch_size <= 0 || params.block_size <= 0
        || params.max_blocks_per_seq <= 0) {
        if (::turbomind::isEagleKVDebugEnabled()) {
            std::fprintf(stderr,
                         "[EAGLE][kv_rewind] early-return: batch_size=%d max_batch_size=%d "
                         "block_size=%d max_blocks_per_seq=%d\n",
                         static_cast<int>(params.batch_size),
                         static_cast<int>(params.max_batch_size),
                         static_cast<int>(params.block_size),
                         static_cast<int>(params.max_blocks_per_seq));
        }
        return;
    }

    dim3 grid(params.batch_size);
    dim3 block(1);
    
    kvCacheRewindKernel<<<grid, block, 0, params.stream>>>(
        params.kv_cache_blocks,
        params.rewind_lengths,
        params.batch_slots,
        params.block_tables,
        params.batch_size,
        params.max_batch_size,
        params.num_layers,
        params.block_size,
        params.max_blocks_per_seq);
}

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
