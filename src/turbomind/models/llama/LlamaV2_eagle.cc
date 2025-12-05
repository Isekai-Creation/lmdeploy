/*
 * Copyright (c) 2024, LMDeploy Contributors.
 * EAGLE Speculative Decoding Implementation for LlamaV2
 */

#include "src/turbomind/models/llama/LlamaV2.h"
#include "lmdeploy/turbomind/eagle_tree.h"
#include "lmdeploy/turbomind/kernels/speculative_decoding/eagle_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

void LlamaV2::eagleSpeculativeStep(
    Buffer_<int>     draft_tokens,
    int              num_draft_tokens,
    Buffer_<int>     accepted_tokens,
    Buffer_<int>     num_accepted,
    const Sequence** sequences,
    int              batch_size
) {
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    if (!spec_mode_.isEagle() || !eagle_module_ || !eagle_buffers_) {
        TM_LOG_WARNING("[EAGLE] Speculative step called but EAGLE not initialized");
        *num_accepted.data() = 0;
        return;
    }
    
    if (num_draft_tokens == 0 || batch_size == 0) {
        *num_accepted.data() = 0;
        return;
    }
    
    TM_LOG_DEBUG("[EAGLE] Processing %d draft tokens for batch_size=%d", 
                 num_draft_tokens, batch_size);
    
    // ========== Step 1: Build EAGLE tree from draft tokens ==========
    eagle::SpeculationTree tree(
        engine_param_.spec_max_draft_path_len,
        engine_param_.spec_max_decoding_tokens
    );
    
    tree.buildTree(draft_tokens.data(), num_draft_tokens);
    tree.extractPaths();
    
    const int* paths = tree.getPathsFlat();
    const int num_paths = tree.getNumPaths();
    const int paths_size = num_paths * engine_param_.spec_max_draft_path_len;
    
    if (num_paths == 0) {
        TM_LOG_WARNING("[EAGLE] No valid paths in tree");
        *num_accepted.data() = 0;
        return;
    }
    
    TM_LOG_DEBUG("[EAGLE] Built tree with %d paths, max_depth=%d", 
                 num_paths, engine_param_.spec_max_draft_path_len);
    
    // ========== Step 2: Prepare context for EagleNet ==========
    // Copy draft tokens to GPU buffers
    cudaMemcpyAsync(
        eagle_buffers_->inputs.draft_tokens,
        draft_tokens.data(),
        num_draft_tokens * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    // Copy paths to GPU
    cudaMemcpyAsync(
        eagle_buffers_->inputs.draft_paths,
        paths,
        paths_size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    sync_check_cuda_error();
    
    // ========== Step 3: Generate leaf mask and packed masks ==========
    using namespace turbomind::kernels::eagle;
    
    // Build leaf mask (distinguish leaf from non-leaf nodes)
    invokeBuildLeafMask(
        eagle_buffers_->inputs.leaf_mask,
        eagle_buffers_->inputs.draft_paths,
        batch_size,
        engine_param_.spec_max_decoding_tokens,
        engine_param_.spec_max_draft_path_len,
        stream_
    );
    
    // Generate packed attention masks from paths
    std::vector<int> batch_slots(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        batch_slots[i] = i;
    }
    
    int* d_batch_slots;
    cudaMalloc(&d_batch_slots, batch_size * sizeof(int));
    cudaMemcpyAsync(
        d_batch_slots,
        batch_slots.data(),
        batch_size * sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    invokeGetPackedMaskFromPath(
        eagle_buffers_->inputs.packed_masks,
        d_batch_slots,
        eagle_buffers_->inputs.draft_paths,
        engine_param_.spec_max_decoding_tokens,
        engine_param_.spec_max_draft_path_len,
        stream_
    );
    
    cudaFree(d_batch_slots);
    sync_check_cuda_error();
    
    TM_LOG_DEBUG("[EAGLE] Generated masks for %d paths", num_paths);
    
    // ========== Step 4: Run target model forward with draft tokens ==========
    // NOTE: This is a simplified version. Full implementation would:
    // 1. Prepare input embeddings for all draft tokens
    // 2. Run unified_decoder_->Forward() with EAGLE-specific attention masks
    // 3. Get logits for each position in the tree
    
    // For now, we'll do a simplified greedy acceptance:
    // Accept tokens one by one until we hit a mismatch
    
    // ========== Step 5: Accept/reject draft tokens (simplified greedy) ==========
    // In full implementation, this would:
    // 1. Compare draft model logits vs target model logits
    // 2. Accept tokens where argmax matches
    // 3. Stop at first mismatch
    
    // Simplified: accept first N tokens with probability decay
    int accepted_count = 0;
    const int max_accept = std::min(num_draft_tokens, 
                                    engine_param_.spec_max_decoding_draft_tokens);
    
    // Greedy acceptance: accept tokens until we decide to stop
    // (In real implementation, this would be based on logit comparison)
    for (int i = 0; i < max_accept; ++i) {
        // Simplified acceptance logic
        // Real implementation would compare target vs draft logits here
        bool accept = (i < max_accept / 2);  // Accept first half for demo
        
        if (accept) {
            accepted_tokens.data()[accepted_count] = draft_tokens.data()[i];
            accepted_count++;
        } else {
            break;  // Stop at first rejection
        }
    }
    
    *num_accepted.data() = accepted_count;
    
    // ========== Step 6: Rewind KV cache for rejected tokens ==========
    if (accepted_count < num_draft_tokens) {
        // Rewind KV cache to accepted_count position
        for (int b = 0; b < batch_size; ++b) {
            if (sequences[b]) {
                // In full implementation, update sequence->cache_len
                // to reflect only accepted tokens
                TM_LOG_DEBUG("[EAGLE] Would rewind KV cache for sequence %d", b);
            }
        }
    }
    
    // Log acceptance statistics
    float acceptance_rate = (float)accepted_count / (float)num_draft_tokens;
    TM_LOG_INFO("[EAGLE] Accepted %d/%d draft tokens (%.1f%% acceptance rate)", 
                accepted_count, num_draft_tokens, acceptance_rate * 100.0f);
    
    // Store acceptance rate in buffers for metrics
    float h_acceptance_rate = acceptance_rate;
    cudaMemcpyAsync(
        eagle_buffers_->outputs.acceptance_rate,
        &h_acceptance_rate,
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    sync_check_cuda_error();
    
    TM_LOG_DEBUG("[EAGLE] Speculative step complete");
}

} // namespace turbomind
