/*
 * Host-side helper for EAGLE KV cache rewind.
 *
 * This translates per-sequence draft/accepted token lengths into
 * KVCacheRewindParams and calls invokeKVCacheRewind, so higher-level
 * TurboMind code (e.g. LlamaBatch) can avoid re-implementing the
 * token-to-block mapping logic for each backend.
 */

#pragma once

#include <cuda_runtime.h>

#include "common.h"

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

struct EagleKVRewindConfig {
    SizeType block_size;          // tokens per KV block
    SizeType max_batch_size;      // rows in block_tables / rewind_lengths
    SizeType max_blocks_per_seq;  // columns in block_tables
    SizeType num_layers;          // KV layers (for kv_cache_blocks)
};

/**
 * @brief Compute rewind lengths from draft/accepted tokens and invoke KV cache rewind.
 *
 * This helper operates purely on per-sequence token counts. For each slot:
 *
 *   rewind_lengths[slot] = max(0, draft_lengths[slot] - accepted_lengths[slot])
 *
 * The caller is responsible for providing device-resident buffers for
 * `d_rewind_lengths`, `d_batch_slots`, `d_block_tables`, and
 * `d_kv_cache_blocks` that match the layout expected by KVCacheRewindParams.
 *
 * All *_lengths and batch_slots arrays passed here are host-visible; they
 * are copied to the device before invoking the CUDA kernel.
 */
void computeAndInvokeKVCacheRewind(
    EagleKVRewindConfig const& cfg,
    SizeType const*            draft_lengths,      // [max_batch_size] host
    SizeType const*            accepted_lengths,   // [max_batch_size] host
    SizeType const*            batch_slots,        // [batch_size] host (may be nullptr)
    SizeType                   batch_size,
    SizeType*                  d_rewind_lengths,   // [max_batch_size] device
    SizeType const*            d_batch_slots,      // [max_batch_size] device (may be nullptr)
    SizeType*                  d_block_tables,     // [max_batch_size, max_blocks_per_seq] device
    void**                     d_kv_cache_blocks,  // [num_layers, max_blocks_per_seq] device (may be nullptr)
    cudaStream_t               stream);

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind

