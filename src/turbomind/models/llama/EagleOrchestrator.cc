// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/EagleOrchestrator.h"
#include "src/turbomind/models/llama/eagle_orchestration_kernels.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

EagleOrchestrator::EagleOrchestrator(int max_batch_size,
                                     int max_decoding_tokens,
                                     int max_path_len,
                                     int max_non_leaf_nodes):
    max_batch_size_(max_batch_size),
    max_decoding_tokens_(max_decoding_tokens),
    max_path_len_(max_path_len),
    max_non_leaf_nodes_(max_non_leaf_nodes)
{
    slot_manager_ = std::make_unique<EagleSlotManager>(max_batch_size);
}

EagleOrchestrator::~EagleOrchestrator()
{
    free();
}

void EagleOrchestrator::allocate(cudaStream_t stream)
{
    if (allocated_) {
        return;
    }
    // Workspace allocation logic will go here
    allocated_ = true;
    TM_LOG_INFO("[EagleOrchestrator] Allocated GPU resources");
}

void EagleOrchestrator::free()
{
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
    }
    // Free buffers allocated in setChoices()
    if (d_path_indices_) {
        cudaFree(d_path_indices_);
        d_path_indices_ = nullptr;
    }
    if (d_successor_offsets_) {
        cudaFree(d_successor_offsets_);
        d_successor_offsets_ = nullptr;
    }
    if (d_successor_counts_) {
        cudaFree(d_successor_counts_);
        d_successor_counts_ = nullptr;
    }
    allocated_ = false;
}

// Helper to compute paths from adjacency list
void EagleOrchestrator::setChoices(const std::vector<std::vector<int>>& choices) {
    if (choices.empty()) return;

    // Parse choices to build path indices and successor info (Host Side)
    // Assumption: choices is adjacency list where index corresponds to node order.
    // Node 0 is the root of the speculation tree (connected to context).
    
    int num_nodes = static_cast<int>(choices.size()); // Assuming all nodes are keys
    if (num_nodes > max_decoding_tokens_) {
        TM_LOG_WARNING("[EagleOrchestrator] choices size %d > max_decoding_tokens %d. Truncating.", 
                       num_nodes, max_decoding_tokens_);
        num_nodes = max_decoding_tokens_;
    }

    std::vector<int> h_path_indices(num_nodes * max_path_len_, -1);
    std::vector<int> h_succ_offsets(num_nodes + 1, 0);
    std::vector<int> h_succ_counts; 

    // BFS Queue: (node_idx, current_path)
    // Actually, simply iterating 0..num_nodes works if order is topological/BFS.
    // Let's assume standard index 0, 1, 2...
    
    // Path for root (0)
    h_path_indices[0] = 0; // The token at index 0 is valid for node 0

    for (int i = 0; i < num_nodes; ++i) {
        // Path construction: copy parent's path + self?
        // Wait, 'choices' topology maps parent->children.
        // We need 'parent' pointer for each node to build path efficiently, or propagate paths.
        // Propagating paths is easier in BFS.
        
        // Successors: direct copy from choices
        const auto& children = choices[i];
        for (int child : children) {
            if (child >= num_nodes) continue;
            // Propagate path from i to child
            const int* parent_path = &h_path_indices[i * max_path_len_];
            int* child_path = &h_path_indices[child * max_path_len_];
            
            // Find length of parent path
            int parent_len = 0;
            while (parent_len < max_path_len_ && parent_path[parent_len] != -1) {
                child_path[parent_len] = parent_path[parent_len];
                parent_len++;
            }
            if (parent_len < max_path_len_) {
                child_path[parent_len] = child; // Append child index
            }
        }
        
        // Successor info (for assembly kernel)
        // Note: For the kernel, we usually flatten by BATCH. 
        // Here we prepare the TEMPLATE for one tree.
        int count = static_cast<int>(children.size());
        h_succ_counts.insert(h_succ_counts.end(), children.begin(), children.end());
        h_succ_offsets[i+1] = h_succ_offsets[i] + count;
    }
    
    total_successors_ = static_cast<int>(h_succ_counts.size());
    
    // Free existing buffers if they were previously allocated
    if (d_path_indices_) {
        cudaFree(d_path_indices_);
        d_path_indices_ = nullptr;
    }
    if (d_successor_offsets_) {
        cudaFree(d_successor_offsets_);
        d_successor_offsets_ = nullptr;
    }
    if (d_successor_counts_) {
        cudaFree(d_successor_counts_);
        d_successor_counts_ = nullptr;
    }

    cudaMalloc(&d_path_indices_, h_path_indices.size() * sizeof(int));
    cudaMemcpy(d_path_indices_, h_path_indices.data(), h_path_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // We only upload the counts template. Offsets need replication per batch on runtime?
    // Actually, invokeAssembleDraftLogitsOffsets logic can handle "template + shift".
    // We upload the TEMPLATE offsets.
    cudaMalloc(&d_successor_offsets_, h_succ_offsets.size() * sizeof(int));
    cudaMemcpy(d_successor_offsets_, h_succ_offsets.data(), h_succ_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_successor_counts_, h_succ_counts.size() * sizeof(int));
    cudaMemcpy(d_successor_counts_, h_succ_counts.data(), h_succ_counts.size() * sizeof(int), cudaMemcpyHostToDevice);
}

void EagleOrchestrator::reset(cudaStream_t stream)
{
    // Reset internal counters/state
}

void EagleOrchestrator::updateDraftTree(const EagleBuffers& buffers,
                                        int batch_size,
                                        int tokens_per_seq,
                                        cudaStream_t stream)
{
    // Step 1: Prepare EagleNet inputs for the *next* token generation
    // This replaces the CPU logic that calculates ctx_lens/seq_lens
    kernels::eagle::invokePrepareEagleNetInputs(batch_size, tokens_per_seq, const_cast<EagleBuffers&>(buffers), stream);

    // Step 2: Assemble offsets for the draft logits (if using tree)
    kernels::eagle::invokeAssembleDraftLogitsOffsets(batch_size, 
                                                     const_cast<EagleBuffers&>(buffers),
                                                     d_path_indices_,
                                                     d_successor_offsets_,
                                                     d_successor_counts_,
                                                     max_decoding_tokens_, // nodes_per_req (approx, or use separate member)
                                                     max_path_len_,
                                                     stream);
}

void EagleOrchestrator::prepareTargetInputs(const EagleBuffers& buffers,
                                            int batch_size,
                                            int tokens_per_seq,
                                            cudaStream_t stream)
{
    // Assemble offsets for target verification (tree attention)
    kernels::eagle::invokeAssembleTargetLogitsOffsets(batch_size, tokens_per_seq, const_cast<EagleBuffers&>(buffers), stream);
    
    // Note: Packed Mask generation is currently handled by the separate 
    // invokeGetPackedMaskFromPath kernel which might be called here or externally.
    // For full orchestration, we should move it here eventually.
}

void EagleOrchestrator::finalizeAcceptance(const EagleBuffers& buffers,
                                           int batch_size,
                                           int* output_ids,
                                           int* sequence_lengths,
                                           cudaStream_t stream)
{
    // Scatter accepted tokens to final output buffer and update sequence lengths
    kernels::eagle::invokeCopyOutputTokensIds(batch_size, const_cast<EagleBuffers&>(buffers), output_ids, sequence_lengths, stream);
}

void EagleOrchestrator::updateSlotMapping(const int* d_batch_slots, int batch_size, cudaStream_t stream)
{
    if (slot_manager_) {
        slot_manager_->update(d_batch_slots, batch_size, stream);
    }
}

} // namespace turbomind
