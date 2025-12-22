// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

/**
 * @brief Manages the mapping between linear batch indices and engine slot indices.
 * 
 * Provides GPU-resident mapping data used by orchestrated EAGLE kernels
 * to access per-request metadata and KV cache blocks.
 */
class EagleSlotManager {
public:
    explicit EagleSlotManager(int max_batch_size);
    ~EagleSlotManager();

    /**
     * @brief Update the mapping for the current batch.
     * 
     * @param d_batch_slots Device pointer to the array of slot IDs for the current batch.
     * @param batch_size Current batch size.
     * @param stream CUDA stream for the copy.
     */
    void update(const int* d_batch_slots, int batch_size, cudaStream_t stream);

    const int* get_slot_mapping() const { return d_slot_mapping_; }

private:
    int max_batch_size_;
    int* d_slot_mapping_{nullptr}; // [max_batch_size] mapping linear index -> slot ID
};

} // namespace turbomind
