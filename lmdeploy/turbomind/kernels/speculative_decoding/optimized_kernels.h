/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * 
 * Optimized kernel declarations
 */

#pragma once

#include "common.h"

namespace turbomind {
namespace kernels {
namespace speculative_decoding {

/**
 * @brief Optimized multi-threaded acceptance kernel
 * 
 * Uses 32 threads per batch item for parallel path processing.
 * Significantly faster than single-threaded version for long paths.
 */
template <typename T>
void acceptDraftTokensOptimized(AcceptDraftTokensParams<T> const& params);

/**
 * @brief Compute acceptance rate statistics
 * 
 * Calculates per-request acceptance rates for metrics tracking.
 * 
 * @param acceptance_rates output buffer [maxBatchSize]
 * @param accepted_lengths input buffer [maxBatchSize]
 * @param draft_lengths input buffer [maxBatchSize]
 * @param batch_slots optional batch slot mapping
 * @param batch_size number of requests
 * @param stream CUDA stream
 */
void invokeComputeAcceptanceStats(
    float* acceptance_rates,
    SizeType const* accepted_lengths,
    SizeType const* draft_lengths,
    SizeType const* batch_slots,
    SizeType batch_size,
    cudaStream_t stream
);

} // namespace speculative_decoding
} // namespace kernels
} // namespace turbomind
