// Copyright (c) OpenMMLab. All rights reserved.
// MXFP4 Dequantization Kernel for DeepSeek_V32 MLA Weight Loading

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// FP4 E2M1 lookup table values (signed, symmetric)
// Matches TransMLA's FP4_LUT
__constant__ float FP4_LUT[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, +1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/**
 * Dequantize MXFP4 (E2M1) blocks + ue8m0 scales to BF16/FP16.
 *
 * MXFP4 Format:
 *   - blocks: uint8 packed 2x4bit (low nibble = even idx, high = odd)
 *   - scales: uint8 exponent with 127 bias (ue8m0 format)
 *   - group_size: elements per scale (typically 32)
 *
 * @param out       Output BF16 tensor [rows, cols]
 * @param blocks    Input packed blocks [rows, groups, group_size/2]
 * @param scales    Input scales [rows, groups]
 * @param rows      Number of rows
 * @param cols      Number of columns (must be multiple of group_size)
 * @param group_size Elements per scaling group (typically 32)
 */
template<typename T>
__global__ void dequant_mxfp4_kernel(
    T* __restrict__ out,
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    int rows,
    int cols,
    int group_size)
{
    const int packed_group = group_size / 2;  // bytes per group in blocks
    const int num_groups = cols / group_size;
    
    const int row = blockIdx.x;
    const int col_base = threadIdx.x * 2;  // each thread handles 2 elements
    
    if (row >= rows) return;
    
    // Process groups
    for (int g = blockIdx.y; g < num_groups; g += gridDim.y) {
        const int col = g * group_size + col_base;
        if (col >= cols) continue;
        
        // Get scale for this group: exponent + 127 bias
        const int scale_idx = row * num_groups + g;
        const int exponent = static_cast<int>(scales[scale_idx]) - 127;
        const float scale = ldexpf(1.0f, exponent);  // 2^exponent
        
        // Get packed byte containing 2 FP4 values
        const int block_idx = row * num_groups * packed_group + g * packed_group + (col_base / 2);
        const uint8_t packed = blocks[block_idx];
        
        // Unpack: low nibble = even index, high nibble = odd index
        const int idx_lo = packed & 0x0F;
        const int idx_hi = (packed >> 4) & 0x0F;
        
        // Lookup FP4 values and scale
        const float val_lo = FP4_LUT[idx_lo] * scale;
        const float val_hi = FP4_LUT[idx_hi] * scale;
        
        // Write output
        const int out_idx = row * cols + col;
        out[out_idx] = static_cast<T>(val_lo);
        out[out_idx + 1] = static_cast<T>(val_hi);
    }
}

/**
 * Launch MXFP4 dequantization kernel.
 */
template<typename T>
void dequant_mxfp4(
    T* out,
    const uint8_t* blocks,
    const uint8_t* scales,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream)
{
    const int threads_per_row = group_size / 2;  // Each thread handles 2 elements
    const int num_groups = cols / group_size;
    
    dim3 grid(rows, num_groups);
    dim3 block(threads_per_row);
    
    dequant_mxfp4_kernel<T><<<grid, block, 0, stream>>>(
        out, blocks, scales, rows, cols, group_size);
}

}  // namespace turbomind
