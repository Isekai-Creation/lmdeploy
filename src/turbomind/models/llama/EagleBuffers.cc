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
    
    cudaMalloc(&inputs.cu_block_nums, (static_cast<size_t>(batch_size_) + 1) * sizeof(SizeType));

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
    
    cudaMemsetAsync(outputs.accepted_lens, 0, batch_size_ * sizeof(SizeType), stream);
    
    // Allocate persistent KV scratch 
    // Size estimate: batch_size * max_decoding_tokens * elem_size_per_token
    // Elem size depends on model config which we don't have here directly.
    // However, LlamaV2 usually allocates this. 
    // Let's assume a safe upper bound or require external sizing?
    // For now, let's allocate a large enough buffer assuming max hidden size etc.
    // Better: LlamaV2 should resize it or we pass config.
    // Given the constraints, we might let LlamaV2 manage the specific size 
    // but store the pointer here? 
    // Or we allocate a fixed 256MB scratch per request?
    // Let's try to infer from module or just allocate a detailed size if possible.
    // For now, initializing to null and letting LlamaV2 allocate/assign it might be safer 
    // if we don't have model params.
    // BUT the goal was "Pre-allocate all EAGLE workspace buffers".
    // Let's postpone allocation to LlamaV2 which has the params, but use this pointer.
    inputs.kv_scratch = nullptr; 

    check_cuda_error(cudaStreamSynchronize(stream));
    
    allocated_ = true;
    TM_LOG_INFO("EagleBuffers allocated successfully");
}

void EagleBuffers::free() {
    if (!allocated_) {
        return;
    }
    
    TM_LOG_INFO("Freeing EagleBuffers");
    
    // Helper macro for safe free - checks null, frees, then sets to nullptr
    #define SAFE_CUDA_FREE(ptr) do { if (ptr) { cudaFree(ptr); ptr = nullptr; } } while(0)
    
    // Free input buffers
    SAFE_CUDA_FREE(inputs.draft_tokens);
    SAFE_CUDA_FREE(inputs.draft_lens);
    SAFE_CUDA_FREE(inputs.target_tokens);
    SAFE_CUDA_FREE(inputs.draft_paths);
    SAFE_CUDA_FREE(inputs.packed_masks);
    SAFE_CUDA_FREE(inputs.leaf_mask);
    SAFE_CUDA_FREE(inputs.eagle_net_ctx_lens);
    SAFE_CUDA_FREE(inputs.eagle_net_gen_lens);
    SAFE_CUDA_FREE(inputs.eagle_net_seq_lens);
    SAFE_CUDA_FREE(inputs.draft_offsets);
    SAFE_CUDA_FREE(inputs.target_offsets);
    SAFE_CUDA_FREE(inputs.successor_offsets);
    SAFE_CUDA_FREE(inputs.successor_counts);
    SAFE_CUDA_FREE(inputs.eagle_net_input_ids);
    SAFE_CUDA_FREE(inputs.eagle_net_position_ids);
    SAFE_CUDA_FREE(inputs.eagle_net_hidden_indices);
    SAFE_CUDA_FREE(inputs.prev_accepted_tokens);
    SAFE_CUDA_FREE(inputs.prev_accepted_lens);
    SAFE_CUDA_FREE(inputs.prev_draft_lens);
    SAFE_CUDA_FREE(inputs.prev_paths);
    SAFE_CUDA_FREE(inputs.best_path_ids);
    
    SAFE_CUDA_FREE(inputs.cu_block_nums);

    // Free output buffers
    SAFE_CUDA_FREE(outputs.next_draft_tokens);
    SAFE_CUDA_FREE(outputs.next_draft_lens);
    SAFE_CUDA_FREE(outputs.next_draft_paths);
    SAFE_CUDA_FREE(outputs.accepted_tokens);
    SAFE_CUDA_FREE(outputs.accepted_lens);
    SAFE_CUDA_FREE(outputs.best_path_ids);
    SAFE_CUDA_FREE(outputs.accepted_lengths_cumsum);
    SAFE_CUDA_FREE(outputs.accepted_path_offsets);
    SAFE_CUDA_FREE(outputs.acceptance_rate);
    
    #undef SAFE_CUDA_FREE
    
    allocated_ = false;
    TM_LOG_INFO("EagleBuffers freed");
}

} // namespace turbomind
