// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <vector>

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/models/llama/EagleBuffers.h"
#include "src/turbomind/models/llama/EagleSlotManager.h"

namespace turbomind {

class LlamaLinear;

/**
 * @brief Manages EAGLE tree building and metadata preparation entirely on GPU.
 * 
 * Replaces the CPU-bound eagle::SpeculationTree and manual loop expansion
 * found in legacy implementations. This class is responsible for invoking
 * the orchestration kernels that prepare inputs for the draft and target
 * models.
 */
class EagleOrchestrator {
public:
    EagleOrchestrator(int max_batch_size,
                      int max_decoding_tokens,
                      int max_path_len,
                      int max_non_leaf_nodes);
    
    ~EagleOrchestrator();

    void allocate(cudaStream_t stream);
    void free();

    // Reset state for a new request or full refresh
    void reset(cudaStream_t stream);

    /**
     * @brief Build the speculative tree on GPU from draft logits/tokens.
     * 
     * This corresponds to the "Step 1 & 2" of the legacy loop:
     * 1. Extract TopK from draft logits (if not already done).
     * 2. Update paths and scores based on draft history.
     * 3. Compute offsets and masks for the next draft step.
     */
    /**
     * @brief Set the static tree choices/topology.
     * 
     * Pre-computes path indices and successor info on GPU to avoid
     * dynamic tree building at runtime.
     * 
     * @param choices Adjacency list (parent -> children) or similar structure.
     */
    void setChoices(const std::vector<std::vector<int>>& choices);


    void updateDraftTree(const EagleBuffers& buffers,
                         int batch_size,
                         int tokens_per_seq,
                         cudaStream_t stream);

    /**
     * @brief Prepare inputs for the Target model verification (Tree Attention).
     * 
     * Generates:
     * - Packed masks
     * - Position IDs
     * - Tree offsets
     * - Leaf masks for verification
     */
    void prepareTargetInputs(const EagleBuffers& buffers,
                             int batch_size,
                             int tokens_per_seq,
                             cudaStream_t stream);

    /**
     * @brief Finalize acceptance and move valid tokens to output.
     * 
     * Uses the verification results to:
     * 1. Scatter accepted tokens to output_ids.
     * 2. Manage KV cache compaction/rewind indices.
     */
    void finalizeAcceptance(const EagleBuffers& buffers,
                            int batch_size,
                            int* output_ids,
                            int* sequence_lengths,
                            cudaStream_t stream);

    void updateSlotMapping(const int* d_batch_slots, int batch_size, cudaStream_t stream);

private:
    int max_batch_size_;
    int max_decoding_tokens_;
    int max_path_len_;
    int max_non_leaf_nodes_;

    bool allocated_{false};
    
    // Internal workspace buffers if needed (e.g., for scan auxiliary memory)
    void* workspace_{nullptr};
    size_t workspace_size_{0};

    // Fixed tree topology pre-computed data
    int* d_path_indices_{nullptr};        // [max_decoding_tokens, max_path_len]
    int* d_successor_offsets_{nullptr};   // [batch_size + 1] (template, replicated)
    int* d_successor_counts_{nullptr};    // [total_successors] (template)
    int  total_successors_{0};
    
    // Host copy for replication
    std::vector<int> h_successor_offsets_;
    std::vector<int> h_successor_counts_;

    std::unique_ptr<EagleSlotManager> slot_manager_;
};

} // namespace turbomind
