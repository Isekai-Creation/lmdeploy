// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle_orchestration_kernels.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <cub/cub.cuh>

namespace turbomind {
namespace kernels {
namespace eagle {

__global__ void prepareEagleNetInputsKernel(int batch_size,
                                            int tokens_per_seq,
                                            EagleBuffers::Inputs inputs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Calculate context and sequence lengths for the draft network
    // Current draft length for this slot is tokens_per_seq (simplified for fixed drafing optimization)
    // In dynamic tree, this would come from the previous iteration's acceptance.
    
    // For EagleNet, the input sequence length includes the prefix + current draft tokens
    int prefix_len = 0;
    if (inputs.prev_accepted_lens) {
        prefix_len = inputs.prev_accepted_lens[idx]; // Length of confirmed sequence
    }
    
    // Set up next step's context length (input for EagleNet)
    // Logic: EagleNet takes the *last* hidden state + new draft tokens.
    // For now, mirroring TRT-LLM's basic fixed-step logic:
    if (inputs.eagle_net_ctx_lens) {
        inputs.eagle_net_ctx_lens[idx] = prefix_len + tokens_per_seq; 
    }
    
    if (inputs.eagle_net_seq_lens) {
        inputs.eagle_net_seq_lens[idx] = prefix_len + tokens_per_seq;
    }
    
    // Position IDs for the draft tokens
    if (inputs.eagle_net_position_ids) {
        for (int i = 0; i < tokens_per_seq; ++i) {
            // This loop is small (e.g. 4-8), so inline is fine
            int pos = prefix_len + i;
            // Where to write in the flattened buffer?
            // Assuming linear layout for now: [batch, tokens_per_seq]
            // int write_idx = idx * tokens_per_seq + i; // Needs offset logic if flattened differently
            // inputs.eagle_net_position_ids[write_idx] = pos;
        }
    }
}

void invokePrepareEagleNetInputs(int batch_size,
                                 int tokens_per_seq,
                                 EagleBuffers& buffers,
                                 cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    prepareEagleNetInputsKernel<<<gridSize, blockSize, 0, stream>>>(batch_size, tokens_per_seq, buffers.inputs);
}

__global__ void assembleDraftLogitsOffsetsKernel(int batch_size,
                                                 EagleBuffers::Inputs inputs,
                                                 const int* template_paths,
                                                 const int* template_succ_offsets,
                                                 const int* template_succ_counts,
                                                 int nodes_per_req,
                                                 int max_path_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // 1. Set up Draft Paths
    // draft_paths contains the tree TOPOLOGY (node indices), NOT token values.
    // We copy template_paths (the static tree structure) to draft_paths for this batch slot.
    // The acceptance kernel uses these node indices to look up tokens in draft_tokens/target_tokens.
    if (inputs.draft_paths && template_paths) {
        for (int node = 0; node < nodes_per_req; ++node) {
            for (int l = 0; l < max_path_len; ++l) {
                int node_idx = template_paths[node * max_path_len + l];
                // Copy the node index directly (tree topology)
                inputs.draft_paths[(idx * nodes_per_req + node) * max_path_len + l] = node_idx;
            }
        }
    }
    
    // 2. Set up Successor Offsets & Counts
    // Successor offsets need to be shifted by the total successors per batch
    if (inputs.successor_offsets && template_succ_offsets) {
        // template_succ_offsets has size [nodes_per_req + 1]
        // We need to write to inputs.successor_offsets [batch_size + 1] ??
        // Wait, EagleBuffers says "successor_offsets: [batch_size + 1]". 
        // This implies offsets into FLATTENED successor_counts.
        // If the topology is fixed, each request has exactly `total_successors` successors.
        // So offsets are just linear: idx * total_successors.
        int total_succ = template_succ_offsets[nodes_per_req];
        inputs.successor_offsets[idx] = idx * total_succ;
        if (idx == batch_size - 1) {
            inputs.successor_offsets[batch_size] = batch_size * total_succ;
        }
        
        // Successor counts: copy template to device buffer [batch * total_successors] ?
        // Actually inputs.successor_counts is [total_successors_pool].
        // If topology fixed, we just replicate `template_succ_counts` batch_size times.
        if (inputs.successor_counts && template_succ_counts) {
            for (int i = 0; i < total_succ; ++i) {
                inputs.successor_counts[idx * total_succ + i] = template_succ_counts[i];
            }
        }
    }
}

void invokeAssembleDraftLogitsOffsets(int batch_size,
                                      EagleBuffers& buffers,
                                      const int* d_path_indices,
                                      const int* d_successor_offsets,
                                      const int* d_successor_counts,
                                      int nodes_per_req,
                                      int max_path_len,
                                      cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    
    // Safety check
    if (!d_path_indices || !d_successor_offsets) return;

    assembleDraftLogitsOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(
        batch_size, 
        buffers.inputs,
        d_path_indices,
        d_successor_offsets,
        d_successor_counts,
        nodes_per_req,
        max_path_len);
}

__global__ void assembleTargetLogitsOffsetsKernel(int batch_size,
                                                  int tokens_per_seq,
                                                  EagleBuffers::Inputs inputs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // tree_offsets: start index of tokens for this request in the flattened tree buffer
    if (inputs.target_offsets) {
        inputs.target_offsets[idx] = idx * tokens_per_seq; 
        if (idx == batch_size - 1) { // Fill the last + 1 entry
             inputs.target_offsets[batch_size] = batch_size * tokens_per_seq;
        }
    }
    
    // runtime_offsets: mirrors tree_offsets but might skip the root or padded tokens
    if (inputs.draft_offsets) {
         inputs.draft_offsets[idx] = idx * tokens_per_seq; 
         if (idx == batch_size - 1) {
             inputs.draft_offsets[batch_size] = batch_size * tokens_per_seq;
         }
    }
}

void invokeAssembleTargetLogitsOffsets(int batch_size,
                                       int tokens_per_seq,
                                       EagleBuffers& buffers,
                                       cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    assembleTargetLogitsOffsetsKernel<<<gridSize, blockSize, 0, stream>>>(batch_size, tokens_per_seq, buffers.inputs);
}


__global__ void copyOutputTokensIdsKernel(int batch_size,
                                          EagleBuffers::Outputs outputs,
                                          int* final_output_ids,
                                          int* sequence_lengths)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Scatter accepted tokens to final_output_ids
    // Note: In LlamaV2 engine, accepted_tokens are already on GPU.
    // We update the sequence lengths by adding the number of newly accepted tokens.
    if (outputs.accepted_lens && sequence_lengths) {
        int accepted = outputs.accepted_lens[idx];
        if (accepted > 0) {
            sequence_lengths[idx] += accepted;
        }
    }
}

void invokeCopyOutputTokensIds(int batch_size,
                               EagleBuffers& buffers,
                               int* output_ids,
                               int* sequence_lengths,
                               cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    copyOutputTokensIdsKernel<<<gridSize, blockSize, 0, stream>>>(batch_size, buffers.outputs, output_ids, sequence_lengths);
}

__global__ void rewindManagedKvCacheKernel(int batch_size,
                                           int* sequence_lengths,
                                           const int* accepted_lens)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // GPU-side "Rewind" logic:
    // If the sequence length was Speculatively advanced by MAX_DRAFT_TOKENS,
    // we rewind it to (prefix_len + accepted_lens).
    // In our current integration, sequence_lengths is advanced by 'accepted' in copyOutputTokensIdsKernel.
    // So 'rewind' might be a no-op if we are careful, or it handles the case where 
    // sequence_lengths was already at max.
}

void invokeRewindManagedKvCache(int batch_size,
                                int* sequence_lengths,
                                const int* accepted_lens,
                                cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    rewindManagedKvCacheKernel<<<gridSize, blockSize, 0, stream>>>(batch_size, sequence_lengths, accepted_lens);
}

__global__ void prepareSlotMappingKernel(int batch_size,
                                         const int* d_batch_slots,
                                         int* d_slot_mapping)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Bridge local batch_idx [0, BS) to global engine slot indices.
    if (d_batch_slots && d_slot_mapping) {
        d_slot_mapping[idx] = d_batch_slots[idx];
    }
}

void invokePrepareSlotMapping(int batch_size,
                              const int* d_batch_slots,
                              int* d_slot_mapping,
                              cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    prepareSlotMappingKernel<<<gridSize, blockSize, 0, stream>>>(batch_size, d_batch_slots, d_slot_mapping);
}

// TODO: Verify elem_size logic matches get_cache_block_size
__global__ void moveNewTokenBlocksKernel(int batch_size,
                                         const int* best_path_ids,
                                         const int* d_path_indices,
                                         const int* accepted_lens,
                                         const int* prev_seq_lens,
                                         const int* cu_block_nums,
                                         void** src_block_ptrs,
                                         const int* dst_block_tables,
                                         void** dst_block_base_ptrs,
                                         int max_blocks_per_seq,
                                         int block_size,
                                         int elem_size_per_token,
                                         int max_path_len)
{
    const int b = blockIdx.x;
    if (b >= batch_size) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int len = accepted_lens ? accepted_lens[b] : 0;
    if (len <= 0) return;

    const int path_id = best_path_ids ? best_path_ids[b] : 0;
    const int prefix_len = prev_seq_lens ? prev_seq_lens[b] : 0;
    
    // Per-token copy loop
    for (int k = 0; k < len; ++k) {
        int draft_token_idx = -1;
        if (d_path_indices && path_id >= 0) {
             draft_token_idx = d_path_indices[path_id * max_path_len + k];
        }
        if (draft_token_idx < 0) continue;

        // --- Source Address Calculation ---
        // src_block_ptrs is a flat array of pointers.
        // The start index for this batch slot is at cu_block_nums[b].
        // The draft tokens are appended after the prefix blocks.
        // src_global_pos relative to the START of the sequence (prefix start):
        // Note: LlamaV2 assigns blocks for the whole sequence (prefix + draft).
        // So the token at 'prefix_len + draft_token_idx' is at that logical position.
        int src_global_pos = prefix_len + draft_token_idx;
        
        int src_batch_base = cu_block_nums ? cu_block_nums[b] : 0;
        int src_block_logical = src_global_pos / block_size;
        int src_offset = (src_global_pos % block_size) * elem_size_per_token;
        
        // src_block_ptrs[src_batch_base + src_block_logical] gives the block pointer
        char* src_base = (char*)src_block_ptrs[src_batch_base + src_block_logical];
        char* src_ptr = src_base + src_offset;

        // --- Destination Address Calculation ---
        // Destination is sequence 'prefix_len + k' (contiguous growth)
        int dst_global_pos = prefix_len + k;
        int dst_block_logical = dst_global_pos / block_size;
        int dst_offset = (dst_global_pos % block_size) * elem_size_per_token;
        
        int dst_block_id = -1;
        if (dst_block_tables) {
            dst_block_id = dst_block_tables[b * max_blocks_per_seq + dst_block_logical];
        }
        
        if (dst_block_id < 0) continue; // Should not happen if Materialize worked

        // Lookup physical address of the destination block
        char* dst_base = (char*)dst_block_base_ptrs[dst_block_id];
        char* dst_ptr = dst_base + dst_offset;

        // --- Copy ---
        // Parallel copy using all threads in the block
        for (int i = tid; i < elem_size_per_token; i += stride) {
            dst_ptr[i] = src_ptr[i];
        }
    }
}

void invokeMoveNewTokenBlocks(int batch_size,
                              const int* best_path_ids,
                              const int* d_path_indices,
                              const int* accepted_lens,
                              const int* prev_seq_lens,
                              const int* cu_block_nums,
                              void** src_block_ptrs,
                              const int* dst_block_tables,
                              void** dst_block_base_ptrs,
                              int max_blocks_per_seq,
                              int block_size,
                              int elem_size_per_token,
                              int max_path_len,
                              cudaStream_t stream)
{
    // One block per request to handle the loop over accepted tokens
    // Each block uses 256 threads to copy data in parallel
    moveNewTokenBlocksKernel<<<batch_size, 256, 0, stream>>>(
        batch_size,
        best_path_ids,
        d_path_indices,
        accepted_lens,
        prev_seq_lens,
        cu_block_nums,
        src_block_ptrs,
        dst_block_tables,
        dst_block_base_ptrs,
        max_blocks_per_seq,
        block_size,
        elem_size_per_token,
        max_path_len
    );
}

// ============================================================================
// FUSED EAGLE INITIALIZATION KERNEL
// ============================================================================
// Combines PrepareEagleNetInputs, AssembleDraftLogitsOffsets, and
// AssembleTargetLogitsOffsets into a single kernel to minimize launch overhead.
// This is critical for TensorRT-LLM parity and CUDA Graph compatibility.

__global__ void fusedEagleInitKernel(int batch_size,
                                     int tokens_per_seq,
                                     EagleBuffers::Inputs inputs,
                                     const int* template_paths,
                                     const int* template_succ_offsets,
                                     const int* template_succ_counts,
                                     int nodes_per_req,
                                     int max_path_len)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // ========== Part 1: PrepareEagleNetInputs ==========
    int prefix_len = 0;
    if (inputs.prev_accepted_lens) {
        prefix_len = inputs.prev_accepted_lens[idx];
    }
    
    if (inputs.eagle_net_ctx_lens) {
        inputs.eagle_net_ctx_lens[idx] = prefix_len + tokens_per_seq;
    }
    
    if (inputs.eagle_net_seq_lens) {
        inputs.eagle_net_seq_lens[idx] = prefix_len + tokens_per_seq;
    }

    // ========== Part 2: AssembleTargetLogitsOffsets ==========
    if (inputs.target_offsets) {
        inputs.target_offsets[idx] = idx * tokens_per_seq;
        if (idx == batch_size - 1) {
            inputs.target_offsets[batch_size] = batch_size * tokens_per_seq;
        }
    }
    
    if (inputs.draft_offsets) {
        inputs.draft_offsets[idx] = idx * tokens_per_seq;
        if (idx == batch_size - 1) {
            inputs.draft_offsets[batch_size] = batch_size * tokens_per_seq;
        }
    }

    // ========== Part 3: AssembleDraftLogitsOffsets ==========
    // Successor offsets (linear for fixed topology)
    if (inputs.successor_offsets && template_succ_offsets) {
        int total_succ = template_succ_offsets[nodes_per_req];
        inputs.successor_offsets[idx] = idx * total_succ;
        if (idx == batch_size - 1) {
            inputs.successor_offsets[batch_size] = batch_size * total_succ;
        }
        
        // Successor counts: replicate template
        if (inputs.successor_counts && template_succ_counts) {
            for (int i = 0; i < total_succ; ++i) {
                inputs.successor_counts[idx * total_succ + i] = template_succ_counts[i];
            }
        }
    }

    // Draft paths: copy tree topology (node indices) from template, NOT token values
    if (inputs.draft_paths && template_paths) {
        for (int node = 0; node < nodes_per_req; ++node) {
            for (int l = 0; l < max_path_len; ++l) {
                int node_idx = template_paths[node * max_path_len + l];
                // Copy the node index directly (tree topology)
                inputs.draft_paths[(idx * nodes_per_req + node) * max_path_len + l] = node_idx;
            }
        }
    }
}

void invokeFusedEagleInit(int batch_size,
                          int tokens_per_seq,
                          EagleBuffers& buffers,
                          const int* d_path_indices,
                          const int* d_successor_offsets,
                          const int* d_successor_counts,
                          int nodes_per_req,
                          int max_path_len,
                          cudaStream_t stream)
{
    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;
    fusedEagleInitKernel<<<gridSize, blockSize, 0, stream>>>(
        batch_size,
        tokens_per_seq,
        buffers.inputs,
        d_path_indices,
        d_successor_offsets,
        d_successor_counts,
        nodes_per_req,
        max_path_len);
}

// ============================================================================
// CUDA GRAPH EXECUTOR FOR EAGLE DRAFT LOOP
// ============================================================================

void EagleCudaGraphExecutor::capture(cudaStream_t stream) {
    if (captured_) {
        reset();
    }
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // Note: All kernel launches between BeginCapture and EndCapture
    // will be recorded into the graph. The caller should invoke
    // the EAGLE draft loop kernels between this call and launch().
}

void EagleCudaGraphExecutor::launch(cudaStream_t stream) {
    if (!captured_) {
        // First launch after capture: finalize the graph
        cudaStreamEndCapture(stream, &graph_);
        cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0);
        captured_ = true;
    }
    
    if (graphExec_) {
        cudaGraphLaunch(graphExec_, stream);
    }
}

void EagleCudaGraphExecutor::reset() {
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    captured_ = false;
}

} // namespace eagle
} // namespace kernels
} // namespace turbomind
