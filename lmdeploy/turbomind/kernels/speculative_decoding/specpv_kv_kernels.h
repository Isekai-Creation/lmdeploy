/*
 * Copyright (c) 2024, LMDeploy Contributors.
 *
 * SpecPV-style partial KV cache CUDA helpers for TurboMind.
 *
 * The initial implementation focuses on float32 summaries and retrieval
 * scoring/gather. Callers are expected to guard on dtype/layout and fall
 * back to the existing host paths when geometry is incompatible.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace turbomind {
namespace kernels {
namespace specpv {

using SizeType = int32_t;

// Compute per-block Kmax/Kmin summaries over key_states[B,H,L,D] for
// blocks beyond sink_tokens. existing_blocks/max_summary_blocks control
// which block indices are populated.
void invokeSummaryKeyStates(const float* key_states,  // [B,H,L,D]
                            int          B,
                            int          H,
                            int          L,
                            int          D,
                            int          sink_tokens,
                            int          block_size,
                            int          existing_blocks,
                            int          max_summary_blocks,
                            float*       kmax,         // [B,H,max_summary_blocks,D]
                            float*       kmin,         // [B,H,max_summary_blocks,D]
                            cudaStream_t stream);

// Score blocks via max(q·Kmax, q·Kmin), select top_blocks per [B,H],
// and gather retrieval/window K/V from full KV into compact slices.
void invokeRefreshRetrieval(const float* query_states,   // [B,H,Q,D]
                            const float* key_states,     // [B,H,L,D]
                            const float* value_states,   // [B,H,L,D]
                            const float* kmax,           // [B,H,N,D]
                            const float* kmin,           // [B,H,N,D]
                            int          B,
                            int          H,
                            int          Q,
                            int          L,
                            int          D,
                            int          N,
                            int          top_blocks,
                            int          block_size,
                            int          window_tokens,
                            float*       retrieval_k,    // [B,H,R,D], R=top_blocks*block_size
                            float*       retrieval_v,    // [B,H,R,D]
                            float*       window_k,       // [B,H,W,D]
                            float*       window_v,       // [B,H,W,D]
                            cudaStream_t stream);

}  // namespace specpv
}  // namespace kernels
}  // namespace turbomind

