/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * 
 * Optimized CUDA kernels for speculative decoding
 */

#include "common.h"
#include <cuda_runtime.h>

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

/**
 * @brief Optimized multi-threaded acceptance kernel
 * 
 * Uses multiple threads per batch item to parallelize path walking.
 * Each thread processes a subset of draft tokens.
 */
__global__ void acceptDraftTokensKernelOptimized(
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
    // Block per batch item, multiple threads per block
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    int seq_len = sequence_lengths[slot];
    int best_path = best_path_ids[slot];
    
    // Shared memory for accepted count
    __shared__ int shared_accepted;
    __shared__ bool shared_stop;
    
    if (threadIdx.x == 0) {
        shared_accepted = 0;
        shared_stop = false;
    }
    __syncthreads();
    
    // Each thread processes a subset of tokens
    for (int i = threadIdx.x; i < max_draft_tokens && !shared_stop; i += blockDim.x) {
        // Get path index for this position
        int path_idx = paths[slot * max_draft_tokens * max_path_len + 
                            best_path * max_path_len + i];
        
        if (path_idx < 0) break;  // End of path
        
        // Get draft and target tokens
        int draft_token = draft_ids[slot * max_draft_tokens + path_idx];
        int target_token = target_ids[slot * max_draft_tokens + path_idx];
        
        // Check if this is the first mismatch
        bool is_match = (draft_token == target_token);
        
        // Atomic update for accepted count
        if (is_match && !shared_stop) {
            int pos = atomicAdd(&shared_accepted, 1);
            if (seq_len + pos < max_seq_len) {
                output_ids[slot * max_seq_len + seq_len + pos] = draft_token;
            }
        } else if (!is_match && !shared_stop) {
            // First mismatch: add correct token and signal stop
            if (atomicCAS((int*)&shared_stop, 0, 1) == 0) {
                int pos = atomicAdd(&shared_accepted, 1);
                if (seq_len + pos < max_seq_len) {
                    output_ids[slot * max_seq_len + seq_len + pos] = target_token;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Thread 0 updates metadata
    if (threadIdx.x == 0) {
        accepted_lengths[slot] = shared_accepted;
        sequence_lengths[slot] += shared_accepted;
    }
}

/**
 * @brief Optimized acceptance with configurable thread count
 */
template <typename T>
void acceptDraftTokensOptimized(AcceptDraftTokensParams<T> const& params) {
    // Use 32 threads per block for better parallelism
    // Can be tuned based on max_draft_tokens
    int threads_per_block = min(32, params.max_draft_tokens);
    dim3 grid(params.batch_size);
    dim3 block(threads_per_block);
    
    acceptDraftTokensKernelOptimized<<<grid, block, 0, params.stream>>>(
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

// Explicit template instantiation
template void acceptDraftTokensOptimized<float>(AcceptDraftTokensParams<float> const&);
template void acceptDraftTokensOptimized<half>(AcceptDraftTokensParams<half> const&);

/**
 * @brief Kernel: Compute acceptance statistics
 * 
 * Useful for metrics tracking and debugging.
 */
__global__ void computeAcceptanceStatsKernel(
    float* acceptance_rates,
    SizeType const* accepted_lengths,
    SizeType const* draft_lengths,
    SizeType const* batch_slots,
    SizeType batch_size
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    
    int slot = batch_slots ? batch_slots[batch_idx] : batch_idx;
    int accepted = accepted_lengths[slot];
    int drafted = draft_lengths[slot];
    
    if (drafted > 0) {
        acceptance_rates[slot] = static_cast<float>(accepted) / static_cast<float>(drafted);
    } else {
        acceptance_rates[slot] = 0.0f;
    }
}

void invokeComputeAcceptanceStats(
    float* acceptance_rates,
    SizeType const* accepted_lengths,
    SizeType const* draft_lengths,
    SizeType const* batch_slots,
    SizeType batch_size,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    
    computeAcceptanceStatsKernel<<<blocks, threads, 0, stream>>>(
        acceptance_rates,
        accepted_lengths,
        draft_lengths,
        batch_slots,
        batch_size
    );
}

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
