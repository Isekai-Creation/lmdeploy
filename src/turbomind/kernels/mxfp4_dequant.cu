// Copyright (c) OpenMMLab. All rights reserved.
// MXFP4 Dequantization Implementation

#include "src/turbomind/kernels/mxfp4_dequant.h"

namespace turbomind {

// FP4 E2M1 lookup table values (signed, symmetric)
// Note: The LUT is defined in the header as __constant__

// Explicit template instantiation for the kernel launch functions
template void dequant_mxfp4<__nv_bfloat16>(
    __nv_bfloat16*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

template void dequant_mxfp4<__half>(
    __half*, const uint8_t*, const uint8_t*, int, int, int, cudaStream_t);

}  // namespace turbomind
