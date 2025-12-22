// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/models/llama/EagleBuffers.h"

namespace turbomind {
namespace kernels {
namespace eagle {

using SizeType = int32_t;
using TokenIdType = int32_t;

/**
 * @brief Prepares inputs for the EagleNet draft model.
 * 
 * Replaces the CPU loop that prepares context/sequence lengths and
 * gathers hidden state indices for the next draft step.
 * 
 * @param batch_size Current batch size
 * @param tokens_per_seq Number of draft tokens per sequence
 * @param buffers GPU-resident EAGLE buffers
 * @param stream CUDA stream
 */
void invokePrepareEagleNetInputs(int batch_size,
                                 int tokens_per_seq,
                                 EagleBuffers& buffers,
                                 cudaStream_t stream);

/**
 * @brief Assumes draft logits are ready, computes offsets for the tree.
 * 
 * Replaces the CPU-based prefix sum / offset calculation for successor nodes.
 * 
 * @param batch_size Current batch size
 * @param buffers GPU-resident EAGLE buffers
 * @param stream CUDA stream
 */
void invokeAssembleDraftLogitsOffsets(int batch_size,
                                      EagleBuffers& buffers,
                                      const int* d_path_indices,
                                      const int* d_successor_offsets,
                                      const int* d_successor_counts,
                                      int nodes_per_req,
                                      int max_path_len,
                                      cudaStream_t stream);

/**
 * @brief Prepares target tree verification offsets.
 * 
 * @param batch_size Current batch size
 * @param buffers GPU-resident EAGLE buffers
 * @param stream CUDA stream
 */

void invokeAssembleTargetLogitsOffsets(int batch_size,
                                       int tokens_per_seq,
                                       EagleBuffers& buffers,
                                       cudaStream_t stream);

void invokeCopyOutputTokensIds(int batch_size,
                               EagleBuffers& buffers,
                               int* output_ids,
                               int* sequence_lengths,
                               cudaStream_t stream);

/**
 * @brief Moves KV blocks for newly accepted tokens between temporary draft space
 * and the main paged KV cache.
 */
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
                              cudaStream_t stream);

/**
 * @brief Performs a GPU-side rewind of the KV cache for rejected draft tokens.
 */
void invokeRewindManagedKvCache(int batch_size,
                                int* sequence_lengths,
                                const int* accepted_lens,
                                cudaStream_t stream);

/**
 * @brief Prepares the slot mapping on GPU to bridge local batch indices [0, BS)
 * to global engine slot indices.
 */
void invokePrepareSlotMapping(int batch_size,
                              const int* d_batch_slots,
                              int* d_slot_mapping,
                              cudaStream_t stream);

/**
 * @brief Fused EAGLE initialization kernel.
 * 
 * Combines PrepareEagleNetInputs, AssembleDraftLogitsOffsets, and 
 * AssembleTargetLogitsOffsets into a single kernel launch for reduced
 * kernel launch overhead. This is critical for CUDA Graph compatibility
 * and maximum performance parity with TensorRT-LLM.
 */
void invokeFusedEagleInit(int batch_size,
                          int tokens_per_seq,
                          EagleBuffers& buffers,
                          const int* d_path_indices,
                          const int* d_successor_offsets,
                          const int* d_successor_counts,
                          int nodes_per_req,
                          int max_path_len,
                          cudaStream_t stream);

/**
 * @brief CUDA Graph capture helper for the EAGLE draft loop.
 * 
 * Captures a sequence of EAGLE operations into a CUDA graph for
 * reduced CPU overhead during execution. The graph can be replayed
 * for subsequent iterations with different input data.
 */
struct EagleCudaGraphExecutor {
    cudaGraph_t graph_{nullptr};
    cudaGraphExec_t graphExec_{nullptr};
    bool captured_{false};
    
    void capture(cudaStream_t stream);
    void launch(cudaStream_t stream);
    void reset();
    ~EagleCudaGraphExecutor() { reset(); }
};

} // namespace eagle
} // namespace kernels
} // namespace turbomind
