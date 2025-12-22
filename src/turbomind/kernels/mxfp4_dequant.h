// Copyright (c) OpenMMLab. All rights reserved.
// MXFP4 Dequantization Kernel for DeepSeek_V32 MLA Weight Loading

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace turbomind {

/**
 * Dequantize MXFP4 (E2M1) blocks + ue8m0 scales to BF16/FP16.
 *
 * MXFP4 Format:
 *   - blocks: uint8 packed 2x4bit (low nibble = even idx, high = odd)
 *   - scales: uint8 exponent with 127 bias (ue8m0 format)
 *   - group_size: elements per scale (typically 32)
 *
 * @param out       Output BF16/FP16 tensor [rows, cols]
 * @param blocks    Input packed blocks [rows, groups, group_size/2]
 * @param scales    Input scales [rows, groups]
 * @param rows      Number of rows
 * @param cols      Number of columns (must be multiple of group_size)
 * @param group_size Elements per scaling group (typically 32)
 * @param stream    CUDA stream
 */
template<typename T>
void dequant_mxfp4(
    T* out,
    const uint8_t* blocks,
    const uint8_t* scales,
    int rows,
    int cols,
    int group_size,
    cudaStream_t stream);

// Explicit instantiation declarations
extern template void dequant_mxfp4<__nv_bfloat16>(
    __nv_bfloat16*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

extern template void dequant_mxfp4<__half>(
    __half*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

}  // namespace turbomind
