// Copyright (c) OpenMMLab. All rights reserved.
// Adapted from TensorRT-LLM's EAGLE implementation

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "EagleModule.h"

namespace turbomind {

/**
 * @brief Runtime buffers for EAGLE speculative decoding
 * 
 * Manages all GPU tensors needed for EAGLE draft generation and verification.
 * Mirrors TensorRT-LLM's EagleBuffers but simplified for TurboMind.
 */
struct EagleBuffers {
    using SizeType = int32_t;
    using TokenIdType = int32_t;
    
    /**
     * @brief Input tensors for EAGLE step
     */
    struct Inputs {
        // Draft tokens from previous iteration
        TokenIdType* draft_tokens;           // [batch_size, max_decoding_tokens]
        SizeType* draft_lens;                // [batch_size]
        // Target tokens for device-side acceptance
        TokenIdType* target_tokens;          // [batch_size, max_decoding_tokens]
        
        // Tree structure
        SizeType* draft_paths;               // [batch_size, max_decoding_tokens, max_path_len]
        SizeType* packed_masks;              // [batch_size, max_decoding_tokens, num_packed]
        int8_t* leaf_mask;                   // [batch_size, max_decoding_tokens]
        
        // EagleNet metadata
        SizeType* eagle_net_ctx_lens;        // [batch_size]
        SizeType* eagle_net_gen_lens;        // [batch_size]
        SizeType* eagle_net_seq_lens;        // [batch_size]
        SizeType* draft_offsets;             // [batch_size + 1]
        SizeType* target_offsets;            // [batch_size + 1]
        SizeType* successor_offsets;         // [batch_size + 1]
        // Flattened successor TopK counts per request. Entries are dense but
        // only the nodes with successors are populated; offsets points into
        // this flattened array. Sized to an upper bound of
        // [batch_size * max_decoding_tokens].
        SizeType* successor_counts;
        TokenIdType* eagle_net_input_ids;    // [total_eagle_tokens]
        SizeType* eagle_net_position_ids;    // [total_eagle_tokens]
        SizeType* eagle_net_hidden_indices;  // [total_eagle_tokens]
        
        // Acceptance metadata from previous step
        TokenIdType* prev_accepted_tokens;   // [batch_size, max_path_len]
        SizeType* prev_accepted_lens;        // [batch_size]
        SizeType* prev_draft_lens;           // [batch_size]
        SizeType* prev_paths;                // [batch_size, max_decoding_tokens, max_path_len]
        SizeType* best_path_ids;             // [batch_size]
        
        // KV Cache Compaction
        void* kv_scratch;                    // [max_persistent_scratch_bytes]
        SizeType* cu_block_nums;             // [batch_size + 1]

        Inputs() { zero(); }
        
        void zero() {
            draft_tokens = nullptr;
            draft_lens = nullptr;
            target_tokens = nullptr;
            draft_paths = nullptr;
            packed_masks = nullptr;
            leaf_mask = nullptr;
            eagle_net_ctx_lens = nullptr;
            eagle_net_gen_lens = nullptr;
            eagle_net_seq_lens = nullptr;
            draft_offsets = nullptr;
            target_offsets = nullptr;
            successor_offsets = nullptr;
            successor_counts = nullptr;
            eagle_net_input_ids = nullptr;
            eagle_net_position_ids = nullptr;
            eagle_net_hidden_indices = nullptr;
            prev_accepted_tokens = nullptr;
            prev_accepted_lens = nullptr;
            prev_draft_lens = nullptr;
            prev_paths = nullptr;
            best_path_ids = nullptr;
            kv_scratch = nullptr;
            cu_block_nums = nullptr;
        }
    };
    
    /**
     * @brief Output tensors from EAGLE step
     */
    struct Outputs {
        // Next iteration draft tokens
        TokenIdType* next_draft_tokens;      // [batch_size, max_decoding_tokens]
        SizeType* next_draft_lens;           // [batch_size]
        SizeType* next_draft_paths;          // [batch_size, max_decoding_tokens, max_path_len]
        
        // Acceptance results
        TokenIdType* accepted_tokens;        // [batch_size, max_path_len]
        SizeType* accepted_lens;             // [batch_size]
        SizeType* best_path_ids;             // [batch_size]

        // Packed acceptance metadata for downstream KV / decode updates
        SizeType* accepted_lengths_cumsum;   // [batch_size]
        SizeType* accepted_path_offsets;     // [batch_size, max_decoding_tokens]
        
        // Statistics
        float* acceptance_rate;              // [batch_size]
        
        Outputs() { zero(); }
        
        void zero() {
            next_draft_tokens = nullptr;
            next_draft_lens = nullptr;
            next_draft_paths = nullptr;
            accepted_tokens = nullptr;
            accepted_lens = nullptr;
            best_path_ids = nullptr;
            accepted_lengths_cumsum = nullptr;
            accepted_path_offsets = nullptr;
            acceptance_rate = nullptr;
        }
    };
    
    Inputs inputs;
    Outputs outputs;

    EagleBuffers() = default;
    ~EagleBuffers() { free(); }
    
    // Disable copy
    EagleBuffers(const EagleBuffers&) = delete;
    EagleBuffers& operator=(const EagleBuffers&) = delete;
    
    /**
     * @brief Allocate all buffers on GPU
     * 
     * @param batch_size Maximum batch size
     * @param module EAGLE module with configuration
     * @param stream CUDA stream for allocation
     */
    void allocate(SizeType batch_size, const EagleModule* module, cudaStream_t stream);
    
    /**
     * @brief Free all GPU buffers
     */
    void free();
    
    /**
     * @brief Check if buffers are allocated
     */
    bool isAllocated() const { return allocated_; }

private:
    bool allocated_ = false;
    SizeType batch_size_ = 0;
    SizeType max_decoding_tokens_ = 0;
    SizeType max_path_len_ = 0;
    SizeType num_packed_masks_ = 0;
};

} // namespace turbomind
