// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/EagleSlotManager.h"

namespace turbomind {

EagleSlotManager::EagleSlotManager(int max_batch_size)
    : max_batch_size_(max_batch_size)
{
    check_cuda_error(cudaMalloc(&d_slot_mapping_, max_batch_size * sizeof(int)));
}

EagleSlotManager::~EagleSlotManager()
{
    if (d_slot_mapping_) {
        cudaFree(d_slot_mapping_);
    }
}

void EagleSlotManager::update(const int* d_batch_slots, int batch_size, cudaStream_t stream)
{
    if (batch_size > 0 && d_batch_slots) {
        // Copy the slot IDs for the active participants in this batch
        check_cuda_error(cudaMemcpyAsync(d_slot_mapping_,
                                         d_batch_slots,
                                         batch_size * sizeof(int),
                                         cudaMemcpyDeviceToDevice,
                                         stream));
    }
}

} // namespace turbomind
