// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#include "EagleBuffers.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <cuda_runtime.h>

namespace turbomind {

void EagleBuffers::allocate(SizeType batch_size, const EagleModule* module, cudaStream_t stream) {
    if (allocated_) {
        TM_LOG_WARNING("EagleBuffers already allocated, freeing first");
        free();
    }
    
    batch_size_ = batch_size;
    max_decoding_tokens_ = module->getMaxDecodingTokens();
    max_path_len_ = module->getMaxDraftPathLen();
    num_packed_masks_ = module->getNumPackedMasks();
    
    TM_LOG_INFO("Allocating EagleBuffers: batch_size=%d, max_decoding_tokens=%d, "
                "max_path_len=%d, num_packed_masks=%d",
                batch_size_, max_decoding_tokens_, max_path_len_, num_packed_masks_);
    
    // Allocate input buffers
    cudaMalloc(&inputs.draft_tokens, batch_size_ * max_decoding_tokens_ * sizeof(TokenIdType));
    cudaMalloc(&inputs.draft_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.target_tokens, batch_size_ * max_decoding_tokens_ * sizeof(TokenIdType));
    cudaMalloc(&inputs.draft_paths, batch_size_ * max_decoding_tokens_ * max_path_len_ * sizeof(SizeType));
    cudaMalloc(&inputs.packed_masks, batch_size_ * max_decoding_tokens_ * num_packed_masks_ * sizeof(SizeType));
    cudaMalloc(&inputs.leaf_mask, batch_size_ * max_decoding_tokens_ * sizeof(int8_t));
    
    cudaMalloc(&inputs.eagle_net_ctx_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.eagle_net_gen_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.eagle_net_seq_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.draft_offsets, (static_cast<size_t>(batch_size_) + 1) * sizeof(SizeType));
    cudaMalloc(&inputs.target_offsets, (static_cast<size_t>(batch_size_) + 1) * sizeof(SizeType));
    // Successor metadata is tracked per tree node. Allocate counts at
    // [batch_size, max_decoding_tokens] to mirror TRT's per-node
    // successor histograms, even if downstream consumers only need a
    // flattened view.
    cudaMalloc(&inputs.successor_offsets, (static_cast<size_t>(batch_size_) + 1) * sizeof(SizeType));
    cudaMalloc(&inputs.successor_counts,
               static_cast<size_t>(batch_size_) * static_cast<size_t>(max_decoding_tokens_) * sizeof(SizeType));
    
    // Allocate larger buffers for EagleNet inputs (conservative estimate)
    SizeType max_eagle_tokens = batch_size_ * max_decoding_tokens_ * 2;
    cudaMalloc(&inputs.eagle_net_input_ids, max_eagle_tokens * sizeof(TokenIdType));
    cudaMalloc(&inputs.eagle_net_position_ids, max_eagle_tokens * sizeof(SizeType));
    cudaMalloc(&inputs.eagle_net_hidden_indices, max_eagle_tokens * sizeof(SizeType));
    
    cudaMalloc(&inputs.prev_accepted_tokens, batch_size_ * max_path_len_ * sizeof(TokenIdType));
    cudaMalloc(&inputs.prev_accepted_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.prev_draft_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&inputs.prev_paths, batch_size_ * max_decoding_tokens_ * max_path_len_ * sizeof(SizeType));
    cudaMalloc(&inputs.best_path_ids, batch_size_ * sizeof(SizeType));
    
    // Allocate output buffers
    cudaMalloc(&outputs.next_draft_tokens, batch_size_ * max_decoding_tokens_ * sizeof(TokenIdType));
    cudaMalloc(&outputs.next_draft_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&outputs.next_draft_paths, batch_size_ * max_decoding_tokens_ * max_path_len_ * sizeof(SizeType));
    
    cudaMalloc(&outputs.accepted_tokens, batch_size_ * max_path_len_ * sizeof(TokenIdType));
    cudaMalloc(&outputs.accepted_lens, batch_size_ * sizeof(SizeType));
    cudaMalloc(&outputs.best_path_ids, batch_size_ * sizeof(SizeType));
    cudaMalloc(&outputs.accepted_lengths_cumsum, (static_cast<size_t>(batch_size_) + 1) * sizeof(SizeType));
    cudaMalloc(&outputs.accepted_path_offsets, batch_size_ * max_decoding_tokens_ * sizeof(SizeType));
    cudaMalloc(&outputs.acceptance_rate, batch_size_ * sizeof(float));
    
    // Initialize buffers to zero
    cudaMemsetAsync(inputs.draft_tokens, 0, batch_size_ * max_decoding_tokens_ * sizeof(TokenIdType), stream);
    cudaMemsetAsync(inputs.draft_lens, 0, batch_size_ * sizeof(SizeType), stream);
    cudaMemsetAsync(inputs.target_tokens, 0, batch_size_ * max_decoding_tokens_ * sizeof(TokenIdType), stream);
    cudaMemsetAsync(inputs.draft_paths, -1, batch_size_ * max_decoding_tokens_ * max_path_len_ * sizeof(SizeType), stream);
    cudaMemsetAsync(outputs.accepted_lens, 0, batch_size_ * sizeof(SizeType), stream);
    
    check_cuda_error(cudaStreamSynchronize(stream));
    
    allocated_ = true;
    TM_LOG_INFO("EagleBuffers allocated successfully");
}

void EagleBuffers::free() {
    if (!allocated_) {
        return;
    }
    
    TM_LOG_INFO("Freeing EagleBuffers");
    
    // Free input buffers
    cudaFree(inputs.draft_tokens);
    cudaFree(inputs.draft_lens);
    cudaFree(inputs.target_tokens);
    cudaFree(inputs.draft_paths);
    cudaFree(inputs.packed_masks);
    cudaFree(inputs.leaf_mask);
    cudaFree(inputs.eagle_net_ctx_lens);
    cudaFree(inputs.eagle_net_gen_lens);
    cudaFree(inputs.eagle_net_seq_lens);
    cudaFree(inputs.draft_offsets);
    cudaFree(inputs.target_offsets);
    cudaFree(inputs.successor_offsets);
    cudaFree(inputs.successor_counts);
    cudaFree(inputs.eagle_net_input_ids);
    cudaFree(inputs.eagle_net_position_ids);
    cudaFree(inputs.eagle_net_hidden_indices);
    cudaFree(inputs.prev_accepted_tokens);
    cudaFree(inputs.prev_accepted_lens);
    cudaFree(inputs.prev_draft_lens);
    cudaFree(inputs.prev_paths);
    cudaFree(inputs.best_path_ids);
    
    // Free output buffers
    cudaFree(outputs.next_draft_tokens);
    cudaFree(outputs.next_draft_lens);
    cudaFree(outputs.next_draft_paths);
    cudaFree(outputs.accepted_tokens);
    cudaFree(outputs.accepted_lens);
    cudaFree(outputs.best_path_ids);
    cudaFree(outputs.accepted_lengths_cumsum);
    cudaFree(outputs.accepted_path_offsets);
    cudaFree(outputs.acceptance_rate);
    
    inputs.zero();
    outputs.zero();
    
    allocated_ = false;
    TM_LOG_INFO("EagleBuffers freed");
}

} // namespace turbomind
