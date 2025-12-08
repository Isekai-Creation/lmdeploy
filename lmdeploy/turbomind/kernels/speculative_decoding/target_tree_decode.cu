/*
 * Copyright (c) 2024, LMDeploy Contributors.
 *
 * Target-tree decode helpers for TurboMind EAGLE.
 *
 * Initial implementation: conservative flattening of EAGLE draft paths
 * into per-step generation inputs. This intentionally keeps the
 * execution model simple; future work can extend it to more advanced
 * reshaping or masking while preserving the current interface.
 */

#include "target_tree_decode.h"

#include <cfloat>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

namespace {

/**
 * @brief Kernel to flatten per-slot draft paths into generation tokens.
 *
 * For each active batch index we walk up to max_decoding_tokens
 * entries in draft_paths and emit one tree token per non-root node.
 * The flat index layout is:
 *
 *   flat_idx = slot_local_offset + local_tree_idx
 *
 * where slot_local_offset is derived from the per-slot prefix of
 * max_decoding_tokens. This keeps the output contiguous per slot and
 * bounded by batch_size * max_decoding_tokens.
 */
__global__ void prepareGenTargetTreeInputsKernel(PrepareGenTargetTreeParams params)
{
    const SizeType local_idx = static_cast<SizeType>(blockIdx.x);
    if (local_idx >= params.batch_size) {
        return;
    }

    const SizeType slot = params.batch_slots
                              ? params.batch_slots[local_idx]
                              : local_idx;
    if (slot < 0 || slot >= params.max_batch_size) {
        return;
    }

    // Per-slot base sequence / context lengths; used as the anchor for
    // position ids and next-length metadata.
    const SizeType base_seq_len =
        params.base_sequence_lengths ? params.base_sequence_lengths[slot] : 0;
    const SizeType base_ctx_len =
        params.base_context_lengths ? params.base_context_lengths[slot] : base_seq_len;

    const SizeType max_tokens = params.max_decoding_tokens;
    const SizeType max_path_len = params.max_path_len;

    SizeType emitted = 0;

    // Walk per-slot speculative positions 0..max_tokens-1 and emit a
    // single node per position when we find a non-root node index.
    for (SizeType t = 0; t < max_tokens; ++t) {
        const SizeType path_base =
            (static_cast<size_t>(slot) * max_tokens + t) * max_path_len;

        // Find the first non-root node for this speculative position.
        SizeType node_idx = -1;
        for (SizeType d = 0; d < max_path_len; ++d) {
            const SizeType raw = params.draft_paths[path_base + d];
            if (raw < 0) {
                break;  // terminator
            }
            if (raw == 0) {
                continue;  // root
            }
            node_idx = raw - 1;  // convert to 0-based token_idx
            break;
        }

        if (node_idx < 0 || node_idx >= max_tokens) {
            continue;
        }

        if (emitted >= max_tokens) {
            break;
        }

        const SizeType flat_idx =
            local_idx * max_tokens + emitted;

        // Emit draft token id for this node. The corresponding target
        // logits / ids will be produced by the base model decode pass.
        const TokenIdType draft_id =
            params.draft_tokens[static_cast<size_t>(slot) * max_tokens + node_idx];

        params.output_ids[flat_idx] = draft_id;

        // Simple position id: base_seq_len + emitted. Future revisions
        // may prefer base_seq_len + tree_depth, but the only strict
        // requirement is monotonicity per sequence.
        params.position_ids[flat_idx] = base_seq_len + emitted;

        // Pack (slot, token_idx) into hidden_indices so that the
        // caller can map back from flat logits to tree nodes.
        const SizeType hidden_base = flat_idx * 2;
        params.hidden_indices[hidden_base + 0] = slot;
        params.hidden_indices[hidden_base + 1] = node_idx;

        ++emitted;
    }

    // Record per-slot speculative decode lengths and next sequence/context
    // lengths when the corresponding buffers are provided.
    if (params.spec_gen_lengths) {
        params.spec_gen_lengths[local_idx] = emitted;
    }
    if (params.next_sequence_lengths) {
        params.next_sequence_lengths[local_idx] = base_seq_len + emitted;
    }
    if (params.next_context_lengths) {
        // Tree decode does not extend the context window; keep context
        // length unchanged for this pass.
        params.next_context_lengths[local_idx] = base_ctx_len;
    }
}

}  // namespace

void invokePrepareGenTargetTreeInputs(PrepareGenTargetTreeParams const& params)
{
    if (params.batch_size <= 0 || params.max_batch_size <= 0
        || params.max_decoding_tokens <= 0 || params.max_path_len <= 0
        || !params.draft_paths || !params.draft_tokens
        || !params.output_ids || !params.position_ids
        || !params.hidden_indices) {
        return;
    }

    dim3 grid(static_cast<unsigned>(params.batch_size));
    dim3 block(1);

    prepareGenTargetTreeInputsKernel<<<grid, block, 0, params.stream>>>(params);
}

__global__ void treeLogitsToTargetTokensKernel(const float*    logits,
                                               SizeType        num_tree_tokens,
                                               SizeType        vocab_size_padded,
                                               const SizeType* hidden_indices,
                                               SizeType        max_batch_size,
                                               SizeType        max_decoding_tokens,
                                               TokenIdType*    target_tokens)
{
    const SizeType idx = static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tree_tokens) {
        return;
    }

    const float* row = logits + static_cast<size_t>(idx) * vocab_size_padded;

    float best_val = -FLT_MAX;
    int   best_id  = 0;

    for (SizeType j = 0; j < vocab_size_padded; ++j) {
        const float v = row[j];
        if (v > best_val) {
            best_val = v;
            best_id  = static_cast<int>(j);
        }
    }

    const SizeType slot      = hidden_indices[2 * idx + 0];
    const SizeType token_idx = hidden_indices[2 * idx + 1];

    if (slot < 0 || slot >= max_batch_size || token_idx < 0 || token_idx >= max_decoding_tokens) {
        return;
    }

    const SizeType offset = slot * max_decoding_tokens + token_idx;
    target_tokens[offset] = static_cast<TokenIdType>(best_id);
}

void invokeTreeLogitsToTargetTokens(const float*     logits,
                                    SizeType         num_tree_tokens,
                                    SizeType         vocab_size_padded,
                                    const SizeType*  hidden_indices,
                                    SizeType         max_batch_size,
                                    SizeType         max_decoding_tokens,
                                    TokenIdType*     target_tokens,
                                    cudaStream_t     stream)
{
    if (!logits || !hidden_indices || !target_tokens || num_tree_tokens <= 0 || vocab_size_padded <= 0
        || max_batch_size <= 0 || max_decoding_tokens <= 0) {
        return;
    }

    const int threads = 128;
    const int blocks  = static_cast<int>((num_tree_tokens + threads - 1) / threads);

    treeLogitsToTargetTokensKernel<<<blocks, threads, 0, stream>>>(
        logits, num_tree_tokens, vocab_size_padded, hidden_indices, max_batch_size, max_decoding_tokens, target_tokens);
}

namespace {

__global__ void treeLogitsToTargetIdsKernel(TreeLogitsToTargetsParams params)
{
    const SizeType idx = static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= params.num_tree_tokens) {
        return;
    }

    const float* row = params.logits + static_cast<size_t>(idx) * params.vocab_size;

    float   best_val = -CUDART_INF_F;
    SizeType best_id = 0;

    for (SizeType j = 0; j < params.vocab_size; ++j) {
        const float v = row[j];
        if (v > best_val) {
            best_val = v;
            best_id  = j;
        }
    }

    const SizeType hidden_base = idx * 2;
    const SizeType slot        = params.hidden_indices[hidden_base + 0];
    const SizeType token_idx   = params.hidden_indices[hidden_base + 1];

    if (slot < 0 || slot >= params.max_batch_size) {
        return;
    }
    if (token_idx < 0 || token_idx >= params.max_decoding_tokens) {
        return;
    }

    const SizeType offset = slot * params.max_decoding_tokens + token_idx;
    params.target_tokens[offset] = static_cast<TokenIdType>(best_id);
}

__global__ void gatherTreePackedMaskKernel(const SizeType* packed_masks,
                                           SizeType        batch_size,
                                           SizeType        max_decoding_tokens,
                                           SizeType        num_packed,
                                           const SizeType* compact_hidden_indices,
                                           SizeType        num_tree_tokens,
                                           SizeType*       out_packed_masks)
{
    const SizeType idx = static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tree_tokens) {
        return;
    }

    const SizeType slot      = compact_hidden_indices[2 * idx + 0];
    const SizeType token_idx = compact_hidden_indices[2 * idx + 1];

    if (slot < 0 || slot >= batch_size || token_idx < 0 || token_idx >= max_decoding_tokens) {
        return;
    }

    const SizeType src_row = slot * max_decoding_tokens + token_idx;
    const SizeType dst_row = idx;

    const SizeType* src = packed_masks + static_cast<size_t>(src_row) * num_packed;
    SizeType*       dst = out_packed_masks + static_cast<size_t>(dst_row) * num_packed;

    for (SizeType j = 0; j < num_packed; ++j) {
        dst[j] = src[j];
    }
}

}  // namespace

void invokeTreeLogitsToTargetIds(TreeLogitsToTargetsParams const& params)
{
    if (!params.logits || !params.hidden_indices || !params.target_tokens
        || params.num_tree_tokens <= 0 || params.vocab_size <= 0) {
        return;
    }

    constexpr int kBlockSize = 256;
    const int     grid       = (params.num_tree_tokens + kBlockSize - 1) / kBlockSize;

    treeLogitsToTargetIdsKernel<<<grid, kBlockSize, 0, params.stream>>>(params);
}

void invokeGatherTreePackedMask(const SizeType*  packed_masks,
                                SizeType         batch_size,
                                SizeType         max_decoding_tokens,
                                SizeType         num_packed,
                                const SizeType*  compact_hidden_indices,
                                SizeType         num_tree_tokens,
                                SizeType*        out_packed_masks,
                                cudaStream_t     stream)
{
    if (!packed_masks || !compact_hidden_indices || !out_packed_masks
        || batch_size <= 0 || max_decoding_tokens <= 0
        || num_packed <= 0 || num_tree_tokens <= 0) {
        return;
    }

    constexpr int kBlockSize = 256;
    const int     grid       = (num_tree_tokens + kBlockSize - 1) / kBlockSize;

    gatherTreePackedMaskKernel<<<grid, kBlockSize, 0, stream>>>(
        packed_masks,
        batch_size,
        max_decoding_tokens,
        num_packed,
        compact_hidden_indices,
        num_tree_tokens,
        out_packed_masks);
}

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind
