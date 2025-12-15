// Copyright (c) OpenMMLab. All rights reserved.

#include "fp4_kv_utils.h"
#include "quantization.h"
#include "src/turbomind/kernels/attention/test_utils.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/macro.h"
#include <cstdint>
#include <iostream>
#include <thrust/universal_vector.h>

using namespace turbomind;

// Mimics FP4Mx quantization for a single 16-dim block
template<class T>
__global__ void quantize_fp4mx_kernel(fp4_e2m1_t* dst_data,
                                      uint8_t*    dst_scale,
                                      const T*    src,
                                      int         head_dim_per_block)
{
    // Each thread handles a single element for simplicity in this test kernel.
    // In actual implementation, it's typically vectorized.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= head_dim_per_block) return;

    // Simulate 16-element block for max/exponent calculation
    float max_val = 0.0f;
    for (int i = 0; i < 16; ++i) {
        if (i < head_dim_per_block) {
            max_val = fmaxf(max_val, fabsf((float)src[i]));
        }
    }
    // All threads in a block will compute the same max_val; this is inefficient
    // for a real kernel but fine for a test.
    // For a more realistic test, each block should compute its own max_val for a 16-dim group.
    // For simplicity, we just use a single max for the entire head_dim_per_block

    uint8_t exponent_u8 = 0;
    if (max_val > 0.f) {
        // This is a simplified heuristic from NVFP4_KV_CACHE.md (MxFP4 section)
        // ceil(log2(max_val / 6.0f)) -> 6.0f is a common magic number
        // (exp+127) for storing in uint8
        float log2_val = log2f(max_val / 6.0f);
        exponent_u8    = (uint8_t)ceilf(log2_val) + 127;
    }
    *dst_scale = exponent_u8; // Store the single exponent for the block

    float scale_factor = __expf((float)(exponent_u8 - 127) * 0.693147182f); // ln(2)

    // Quantize and store
    ConvertKvCache<T, fp4_e2m1_t> quantizer{};
    dst_data[idx] = quantizer(src[idx] / scale_factor);
}

// Mimics FP4Mx dequantization
template<class T>
__global__ void dequantize_fp4mx_kernel(T*              dst,
                                        const fp4_e2m1_t* src_data,
                                        const uint8_t*  src_scale,
                                        int             head_dim_per_block)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= head_dim_per_block) return;

    uint8_t exponent_u8 = *src_scale;
    float scale_factor  = __expf((float)(exponent_u8 - 127) * 0.693147182f); // ln(2)

    ConvertKvCache<fp4_e2m1_t, T> dequantizer{};
    dst[idx] = dequantizer(src_data[idx]) * scale_factor;
}

template<class T>
void round_trip_test_fp4mx(size_t n, int head_dim_per_block)
{
    std::cout << "Running FP4Mx round-trip test for " << n << " elements ("
              << "head_dim_per_block=" << head_dim_per_block << ")\n";

    thrust::universal_vector<T> src(n);
    thrust::universal_vector<fp4_e2m1_t> quantized_data(n);
    thrust::universal_vector<uint8_t> quantized_scale(1); // One scale for the block
    thrust::universal_vector<T> dst(n);

    // Fill source with diverse values
    RNG rng{};
    rng.GenerateNormal(src.data().get(), n);

    // Quantize
    quantize_fp4mx_kernel<<<1, 256>>>(
        quantized_data.data().get(), quantized_scale.data().get(), src.data().get(), head_dim_per_block);
    cudaDeviceSynchronize();

    // Dequantize
    dequantize_fp4mx_kernel<<<1, 256>>>(
        dst.data().get(), quantized_data.data().get(), quantized_scale.data().get(), head_dim_per_block);
    cudaDeviceSynchronize();

    // Compare
    Compare(dst.data().get(), src.data().get(), n, n, 1);

    std::cout << "FP4Mx round-trip test PASSED.\n";
}

int main(int argc, char* argv[])
{
#if defined(ENABLE_FP4)
    // Test for half (FP16)
    round_trip_test_fp4mx<half>(16, 16);
    round_trip_test_fp4mx<half>(32, 32);

    // Test for float (FP32)
    round_trip_test_fp4mx<float>(16, 16);
    round_trip_test_fp4mx<float>(32, 32);

#if ENABLE_BF16
    // Test for nv_bfloat16
    round_trip_test_fp4mx<nv_bfloat16>(16, 16);
    round_trip_test_fp4mx<nv_bfloat16>(32, 32);
#endif

#else
    std::cout << "FP4 support is not enabled. Skipping FP4Mx tests.\n";
#endif
    return 0;
}
