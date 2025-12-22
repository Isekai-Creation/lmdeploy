
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace turbomind;

__global__ void test_fp4_conversion_kernel(float* src, half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 8 >= n) return;

    Array<float, 8> input_floats;
    #pragma unroll
    for(int i=0; i<8; ++i) {
        if (idx*8 + i < n) input_floats[i] = src[idx*8 + i];
        else input_floats[i] = 0.0f;
    }

    // Scale = 1.0, Zero = 0.0
    ConvertKvCache<float, fp4_e2m1_t> converter_write(1.0f, 0.0f);
    
    // Write (Quantize)
    auto packed_fp4 = converter_write(input_floats);

    // Read (Dequantize)
    ConvertKvCache<fp4_e2m1_t, half> converter_read(half(1.0f), half(0.0f));
    auto output_halfs = converter_read(packed_fp4);

    #pragma unroll
    for(int i=0; i<8; ++i) {
        if (idx*8 + i < n) dst[idx*8 + i] = output_halfs[i];
    }
}

int main() {
    int n = 64; 
    std::vector<float> input(n);
    std::vector<half> output(n);
    
    // E2M1 Values (approx): 0, 0.5, 1, 1.5, 2, 3, 4, 6
    // Inputs to trigger these:
    float input_vals[] = {
        0.05f, // -> 0
        0.4f,  // -> 0.5 (cutoff 0.25-0.75)
        0.9f,  // -> 1.0 (cutoff 0.75-1.25)
        1.4f,  // -> 1.5 (cutoff 1.25-1.75)
        2.2f,  // -> 2.0 (cutoff 1.75-2.5)
        3.0f,  // -> 3.0 (cutoff 2.5-3.5)
        4.5f,  // -> 4.0 (cutoff 3.5-5.0)
        5.5f,  // -> 6.0 (>5.0)
        -0.4f, // -> -0.5
        -5.5f  // -> -6.0
    };
    
    for(int i=0; i<n; ++i) {
        input[i] = input_vals[i % 10];
    }

    float* d_in;
    half* d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(half));

    cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    test_fp4_conversion_kernel<<< 1, 32 >>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemcpy(output.data(), d_out, n * sizeof(half), cudaMemcpyDeviceToHost);

    bool pass = true;
    for(int i=0; i<10; ++i) { // Check first 10
        float val = (float)output[i];
        std::cout << "In: " << input[i] << " Out: " << val << std::endl;
        
        // Simple verification of expected rounding
        float expected = 0.0f;
        float abs_in = fabsf(input[i]);
        if (abs_in < 0.25f) expected = 0.0f;
        else if (abs_in < 0.75f) expected = 0.5f;
        else if (abs_in < 1.25f) expected = 1.0f;
        else if (abs_in < 1.75f) expected = 1.5f;
        else if (abs_in < 2.5f) expected = 2.0f;
        else if (abs_in < 3.5f) expected = 3.0f;
        else if (abs_in < 5.0f) expected = 4.0f;
        else expected = 6.0f;
        
        if (input[i] < 0) expected = -expected;

        if (fabsf(val - expected) > 1e-3) {
            std::cout << "MISMATCH! Expected " << expected << std::endl;
            pass = false;
        }
    }

    if (pass) std::cout << "TEST PASSED" << std::endl;
    else std::cout << "TEST FAILED" << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return pass ? 0 : 1;
}
