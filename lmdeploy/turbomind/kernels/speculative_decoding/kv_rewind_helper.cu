/*
 * Host-side helper for EAGLE KV cache rewind.
 *
 * This implementation computes per-sequence rewind lengths from draft and
 * accepted token counts, copies them to the device, and wires up a
 * KVCacheRewindParams call. It deliberately stays agnostic of higher-level
 * TurboMind structures such as SequenceManager so it can be reused from
 * different integration points.
 */

#include "kv_rewind_helper.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/eagle_debug.h"

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

void computeAndInvokeKVCacheRewind(
    EagleKVRewindConfig const& cfg,
    SizeType const*            draft_lengths,
    SizeType const*            accepted_lengths,
    SizeType const*            batch_slots,
    SizeType                   batch_size,
    SizeType*                  d_rewind_lengths,
    SizeType const*            d_batch_slots,
    SizeType*                  d_block_tables,
    void**                     d_kv_cache_blocks,
    cudaStream_t               stream)
{
    NvtxScope nvtx_scope("EAGLE_KVRewind");
    if (!d_rewind_lengths || !d_block_tables) {
        // Nothing to do without destination buffers.
        return;
    }

    if (batch_size <= 0 || cfg.max_batch_size <= 0) {
        return;
    }

    const SizeType n = cfg.max_batch_size;

    // Host-side buffer for rewind lengths; initialise to zero for all slots.
    std::vector<SizeType> rewind_lengths(n, 0);

    // Compute per-slot rewind lengths as (draft_len - accepted_len)+.
    for (SizeType i = 0; i < batch_size; ++i) {
        SizeType slot = batch_slots ? batch_slots[i] : i;
        if (slot < 0 || slot >= n) {
            continue;
        }

        SizeType draft_len    = draft_lengths ? draft_lengths[slot] : 0;
        SizeType accepted_len = accepted_lengths ? accepted_lengths[slot] : 0;

        draft_len    = std::max<SizeType>(0, draft_len);
        accepted_len = std::max<SizeType>(0, accepted_len);

        if (draft_len > accepted_len) {
            rewind_lengths[slot] = draft_len - accepted_len;
        }
        else {
            rewind_lengths[slot] = 0;
        }
    }

    if (::turbomind::isEagleKVDebugEnabled()) {
        SizeType non_zero_slots = 0;
        SizeType total_rewound_tokens = 0;
        for (SizeType slot = 0; slot < n; ++slot) {
            if (rewind_lengths[slot] > 0) {
                ++non_zero_slots;
                total_rewound_tokens += rewind_lengths[slot];
            }
        }
        std::fprintf(stderr,
                     "[EAGLE][KVRewind] block_size=%d batch_size=%d max_batch_size=%d "
                     "non_zero_slots=%d total_rewound_tokens=%d\n",
                     static_cast<int>(cfg.block_size),
                     static_cast<int>(batch_size),
                     static_cast<int>(cfg.max_batch_size),
                     static_cast<int>(non_zero_slots),
                     static_cast<int>(total_rewound_tokens));
    }

    // Copy host rewind lengths to device buffer.
    const size_t bytes = static_cast<size_t>(n) * sizeof(SizeType);
    cudaMemcpyAsync(d_rewind_lengths, rewind_lengths.data(), bytes, cudaMemcpyHostToDevice, stream);

    KVCacheRewindParams params{};
    params.kv_cache_blocks   = d_kv_cache_blocks;
    params.rewind_lengths    = d_rewind_lengths;
    params.batch_slots       = d_batch_slots;
    params.block_tables      = d_block_tables;
    params.batch_size        = batch_size;
    params.max_batch_size    = cfg.max_batch_size;
    params.num_layers        = cfg.num_layers;
    params.block_size        = cfg.block_size;
    params.max_blocks_per_seq = cfg.max_blocks_per_seq;
    params.stream            = stream;

    invokeKVCacheRewind(params);
}

}  // namespace speculative_decoding
}  // namespace kernels
}  // namespace turbomind
