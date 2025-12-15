/*
 * SpecPV-style partial KV cache CUDA helpers for TurboMind.
 *
 * NOTE: This first pass is intentionally conservative: we implement
 * straightforward kernels for float32 summaries and retrieval scoring/
 * gather. Higher-performance tiling can be added later without changing
 * the public interface.
 */

#include "specpv_kv_kernels.h"

#include <cuda_runtime_api.h>
#include <cfloat>

namespace turbomind {
namespace kernels {
namespace specpv {

namespace {

__global__ void summaryKeyStatesKernel(const float* __restrict__ key_states,
                                       int                      B,
                                       int                      H,
                                       int                      L,
                                       int                      D,
                                       int                      sink_tokens,
                                       int                      block_size,
                                       int                      existing_blocks,
                                       int                      max_summary_blocks,
                                       float* __restrict__      kmax,
                                       float* __restrict__      kmin)
{
    const int block_idx = blockIdx.x + existing_blocks;
    if (block_idx >= max_summary_blocks) {
        return;
    }

    const int bh  = blockIdx.y;
    const int dim = threadIdx.x;

    if (bh >= B * H || dim >= D) {
        return;
    }

    const int b = bh / H;
    const int h = bh % H;

    const int start = sink_tokens + block_idx * block_size;
    if (start >= L) {
        return;
    }
    const int end = min(start + block_size, L);

    float vmax = -FLT_MAX;
    float vmin = FLT_MAX;

    for (int t = start; t < end; ++t) {
        const size_t idx = (((size_t)b * H + h) * L + static_cast<size_t>(t)) * D + dim;
        const float  v   = key_states[idx];
        vmax             = v > vmax ? v : vmax;
        vmin             = v < vmin ? v : vmin;
    }

    const int N = max_summary_blocks;
    const size_t out_idx = (((size_t)b * H + h) * N + static_cast<size_t>(block_idx)) * D + dim;
    kmax[out_idx]        = vmax;
    kmin[out_idx]        = vmin;
}

__global__ void refreshRetrievalKernel(const float* __restrict__ query_states,   // [B,H,Q,D]
                                       const float* __restrict__ key_states,     // [B,H,L,D]
                                       const float* __restrict__ value_states,   // [B,H,L,D]
                                       const float* __restrict__ kmax,           // [B,H,N,D]
                                       const float* __restrict__ kmin,           // [B,H,N,D]
                                       int                      B,
                                       int                      H,
                                       int                      Q,
                                       int                      L,
                                       int                      D,
                                       int                      N,
                                       int                      top_blocks,
                                       int                      block_size,
                                       int                      window_tokens,
                                       float* __restrict__      retrieval_k,     // [B,H,R,D]
                                       float* __restrict__      retrieval_v,     // [B,H,R,D]
                                       float* __restrict__      window_k,        // [B,H,W,D]
                                       float* __restrict__      window_v)        // [B,H,W,D]
{
    const int bh = blockIdx.x;
    if (bh >= B * H) {
        return;
    }
    const int b = bh / H;
    const int h = bh % H;

    // Local buffers on stack: scores and indices per block.
    extern __shared__ float shared[];
    float* scores = shared;
    int*   indices = reinterpret_cast<int*>(scores + N);

    // Compute scores for each block.
    for (int blk = threadIdx.x; blk < N; blk += blockDim.x) {
        float best = -FLT_MAX;

        for (int q = 0; q < Q; ++q) {
            float dot_max = 0.f;
            float dot_min = 0.f;

            for (int d = 0; d < D; ++d) {
                const size_t q_idx =
                    (((size_t)b * H + h) * Q + static_cast<size_t>(q)) * D + d;
                const size_t s_idx =
                    (((size_t)b * H + h) * N + static_cast<size_t>(blk)) * D + d;

                const float qv = query_states[q_idx];
                dot_max += qv * kmax[s_idx];
                dot_min += qv * kmin[s_idx];
            }

            const float s = dot_max > dot_min ? dot_max : dot_min;
            best          = s > best ? s : best;
        }

        scores[blk]  = best;
        indices[blk] = blk;
    }

    __syncthreads();

    // Simple partial selection on a single thread per (B,H).
    if (threadIdx.x == 0) {
        const int K = min(top_blocks, N);
        for (int i = 0; i < N - 1; ++i) {
            int   best_idx = i;
            float best_val = scores[i];
            for (int j = i + 1; j < N; ++j) {
                if (scores[j] > best_val) {
                    best_val = scores[j];
                    best_idx = j;
                }
            }
            if (i != best_idx) {
                float tmp_score   = scores[i];
                scores[i]         = scores[best_idx];
                scores[best_idx]  = tmp_score;
                int tmp_idx       = indices[i];
                indices[i]        = indices[best_idx];
                indices[best_idx] = tmp_idx;
            }
            if (i + 1 >= K) {
                break;
            }
        }

        const int R = top_blocks * block_size;

        // Gather retrieval tokens.
        for (int r = 0; r < K; ++r) {
            const int blk       = indices[r];
            const int src_base  = blk * block_size;
            const int dst_block = r;

            for (int t = 0; t < block_size; ++t) {
                const int src_token = src_base + t;
                const int dst_token = dst_block * block_size + t;
                if (src_token >= L || dst_token >= R) {
                    break;
                }

                for (int d = 0; d < D; ++d) {
                    const size_t src_idx =
                        (((size_t)b * H + h) * L + static_cast<size_t>(src_token)) * D + d;
                    const size_t dst_idx =
                        (((size_t)b * H + h) * R + static_cast<size_t>(dst_token)) * D + d;
                    retrieval_k[dst_idx] = key_states[src_idx];
                    retrieval_v[dst_idx] = value_states[src_idx];
                }
            }
        }

        // Gather window tokens from the tail of [0..L).
        if (window_tokens > 0) {
            const int W         = window_tokens;
            const int win_start = max(0, L - W);
            for (int t = 0; t < W; ++t) {
                const int src_token = win_start + t;
                const int dst_token = t;
                if (src_token >= L) {
                    break;
                }
                for (int d = 0; d < D; ++d) {
                    const size_t src_idx =
                        (((size_t)b * H + h) * L + static_cast<size_t>(src_token)) * D + d;
                    const size_t dst_idx =
                        (((size_t)b * H + h) * W + static_cast<size_t>(dst_token)) * D + d;
                    window_k[dst_idx] = key_states[src_idx];
                    window_v[dst_idx] = value_states[src_idx];
                }
            }
        }
    }
}

}  // namespace

void invokeSummaryKeyStates(const float* key_states,
                            int          B,
                            int          H,
                            int          L,
                            int          D,
                            int          sink_tokens,
                            int          block_size,
                            int          existing_blocks,
                            int          max_summary_blocks,
                            float*       kmax,
                            float*       kmin,
                            cudaStream_t stream)
{
    if (!key_states || !kmax || !kmin || B <= 0 || H <= 0 || L <= sink_tokens
        || D <= 0 || block_size <= 0 || existing_blocks < 0
        || max_summary_blocks <= existing_blocks) {
        return;
    }

    const int total_tokens = L - sink_tokens;
    const int num_blocks   = max(0, total_tokens / block_size);
    const int blocks_to_fill =
        max(0, min(num_blocks, max_summary_blocks) - existing_blocks);
    if (blocks_to_fill <= 0) {
        return;
    }

    dim3 grid(blocks_to_fill, B * H);
    dim3 block(min(D, 256));

    summaryKeyStatesKernel<<<grid, block, 0, stream>>>(
        key_states,
        B,
        H,
        L,
        D,
        sink_tokens,
        block_size,
        existing_blocks,
        max_summary_blocks,
        kmax,
        kmin);
}

void invokeRefreshRetrieval(const float* query_states,
                            const float* key_states,
                            const float* value_states,
                            const float* kmax,
                            const float* kmin,
                            int          B,
                            int          H,
                            int          Q,
                            int          L,
                            int          D,
                            int          N,
                            int          top_blocks,
                            int          block_size,
                            int          window_tokens,
                            float*       retrieval_k,
                            float*       retrieval_v,
                            float*       window_k,
                            float*       window_v,
                            cudaStream_t stream)
{
    if (!query_states || !key_states || !value_states || !kmax || !kmin
        || !retrieval_k || !retrieval_v || !window_k || !window_v) {
        return;
    }
    if (B <= 0 || H <= 0 || Q <= 0 || L <= 0 || D <= 0 || N <= 0
        || top_blocks <= 0 || block_size <= 0) {
        return;
    }

    const int R = top_blocks * block_size;
    if (R <= 0) {
        return;
    }

    dim3 grid(B * H);
    const int threads = 128;
    const size_t shared_bytes =
        static_cast<size_t>(N) * (sizeof(float) + sizeof(int));

    refreshRetrievalKernel<<<grid, threads, shared_bytes, stream>>>(
        query_states,
        key_states,
        value_states,
        kmax,
        kmin,
        B,
        H,
        Q,
        L,
        D,
        N,
        top_blocks,
        block_size,
        window_tokens,
        retrieval_k,
        retrieval_v,
        window_k,
        window_v);
}

}  // namespace specpv
}  // namespace kernels
}  // namespace turbomind
