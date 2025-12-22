// Copyright (c) OpenMMLab. All rights reserved.
// MXFP4 Dequantization Implementation

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

// FP4 E2M1 lookup table values (signed, symmetric)
// Matches TransMLA's FP4_LUT
__constant__ float FP4_LUT_DEVICE[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

template<typename T>
__global__ void dequant_mxfp4_kernel(
    T* __restrict__ out,
    const uint8_t* __restrict__ blocks,
    const uint8_t* __restrict__ scales,
    int rows,
    int cols,
    int group_size)
{
    const int packed_group = group_size / 2;
    const int num_groups = cols / group_size;
    
    const int row = blockIdx.x;
    const int col_base = threadIdx.x * 2;
    
    if (row >= rows) return;
    
    for (int g = blockIdx.y; g < num_groups; g += gridDim.y) {
        const int col = g * group_size + col_base;
        if (col >= cols) continue;
        
        const int scale_idx = row * num_groups + g;
        const int exponent = static_cast<int>(scales[scale_idx]) - 127;
        const float scale = ldexpf(1.0f, exponent);
        
        const int block_idx = row * num_groups * packed_group + g * packed_group + (col_base / 2);
        const uint8_t packed = blocks[block_idx];
        
        const int idx_lo = packed & 0x0F;
        const int idx_hi = (packed >> 4) & 0x0F;
        
        const float val_lo = FP4_LUT_DEVICE[idx_lo] * scale;
        const float val_hi = FP4_LUT_DEVICE[idx_hi] * scale;
        
        const int out_idx = row * cols + col;
        out[out_idx] = static_cast<T>(val_lo);
        out[out_idx + 1] = static_cast<T>(val_hi);
    }
}

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
    const int threads_per_row = group_size / 2;
    const int num_groups = cols / group_size;
    
    dim3 grid(rows, num_groups);
    dim3 block(threads_per_row);
    
    dequant_mxfp4_kernel<T><<<grid, block, 0, stream>>>(
        out, blocks, scales, rows, cols, group_size);
}

// Explicit template instantiations
template void dequant_mxfp4<__nv_bfloat16>(
    __nv_bfloat16*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

template void dequant_mxfp4<__half>(
    __half*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

}  // namespace turbomind
