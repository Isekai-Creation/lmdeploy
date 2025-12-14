// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/fp4_kv_utils.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

// Minimal Config type compatible with block::Layout, mirroring the
// test_attention.cu Config so we can parameterize head_dim at runtime.
template<class T, class Tkv>
struct Fp4ProbeConfig {
    int head_dim_;
    int head_num_;
    int block_len_;

    TM_HOST_DEVICE constexpr int t_bits() const
    {
        if constexpr (std::is_same_v<T, Tkv>) {
            return 0;
        }
        else {
            return bitsof<T>;
        }
    }

    TM_HOST_DEVICE constexpr int q_bits() const
    {
        return bitsof<Tkv>;
    }

    TM_HOST_DEVICE constexpr int head_dim() const
    {
        return head_dim_;
    }

    TM_HOST_DEVICE int head_num() const
    {
        return head_num_;
    }

    TM_HOST_DEVICE constexpr int block_len() const
    {
        return block_len_;
    }
};

template<class T, class BlockLayout>
__global__ void Fp4KvProbeKernel(Fp4KvProbeResult* out,
                                 char**            blocks,
                                 char**            scale_blocks,
                                 int               layer_id,
                                 int               head_idx,
                                 int               head_num,
                                 int               block_len,
                                 int               head_dim,
                                 int               local_ti)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    // Compute base scale pointer for this (layer, head, token).
    uint8_t* scale_base =
        get_fp4_mx_scale_base(scale_blocks, layer_id, head_idx, head_num, block_len, head_dim, local_ti);
    if (!scale_base) {
        out->k_scale0 = 0;
        out->v_scale0 = 0;
        out->k_scale1 = 0;
        out->v_scale1 = 0;
        out->kv_byte0 = 0;
        return;
    }

    const int scales_per_head = head_dim / 16;

    const uint8_t k_scale0 = scale_base[0];
    const uint8_t v_scale0 = scale_base[scales_per_head];

    // Also capture scale1 for di in [16, 31] to verify the
    // di/16 indexing in tests.
    const uint8_t k_scale1 = scales_per_head > 1 ? scale_base[1] : scale_base[0];
    const uint8_t v_scale1 = scales_per_head > 1 ? scale_base[scales_per_head + 1] : scale_base[scales_per_head];

    // Read the first packed FP4 byte from KV payload at di=0. We build a
    // Layout with a runtime head_dim via Fp4ProbeConfig so offsets match
    // the FP4 MXFP4 layout used in ProcessKV_v2.
    Fp4ProbeConfig<T, fp4_e2m1_t> config{head_dim, head_num, block_len};
    BlockLayout                   layout{config};
    block::Head<T, fp4_e2m1_t, BlockLayout> head{layout, layer_id, head_idx};
    int block_id, block_ti;
    head.get_block_coord(local_ti, block_id, block_ti);

    char* block = blocks[block_id];
    auto  k_ptr = head.k_data(block, block_ti);

    uint8_t kv_byte0 = reinterpret_cast<const uint8_t*>(k_ptr.ptr_)[0];

    out->k_scale0 = k_scale0;
    out->v_scale0 = v_scale0;
    out->k_scale1 = k_scale1;
    out->v_scale1 = v_scale1;
    out->kv_byte0 = kv_byte0;
}

void fp4_kv_probe(Fp4KvProbeResult* d_out,
                  char**            blocks,
                  char**            scale_blocks,
                  int               layer_id,
                  int               head_idx,
                  int               head_num,
                  int               block_len,
                  int               head_dim,
                  int               local_ti,
                  cudaStream_t      stream)
{
    using T           = half;
    using BlockLayout = block::Layout<Fp4ProbeConfig<T, fp4_e2m1_t>>;
    Fp4KvProbeKernel<T, BlockLayout><<<1, 1, 0, stream>>>(
        d_out, blocks, scale_blocks, layer_id, head_idx, head_num, block_len, head_dim, local_ti);
}

// Simple host-side harness to invoke the probe and copy the result back
// to host memory. This is intended for debug/testing only; callers are
// responsible for ensuring that `blocks` / `scale_blocks` correspond to
// a valid FP4 MXFP4 KV cache layout.
cudaError_t fp4_kv_probe_host(Fp4KvProbeResult& out,
                              char**            d_blocks,
                              char**            d_scale_blocks,
                              int               layer_id,
                              int               head_idx,
                              int               head_num,
                              int               block_len,
                              int               head_dim,
                              int               local_ti,
                              cudaStream_t      stream)
{
    Fp4KvProbeResult* d_out = nullptr;
    cudaError_t       err   = cudaMalloc(&d_out, sizeof(Fp4KvProbeResult));
    if (err != cudaSuccess) {
        return err;
    }

    fp4_kv_probe(d_out,
                 d_blocks,
                 d_scale_blocks,
                 layer_id,
                 head_idx,
                 head_num,
                 block_len,
                 head_dim,
                 local_ti,
                 stream);

    err = cudaMemcpyAsync(&out, d_out, sizeof(Fp4KvProbeResult), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }

    cudaFree(d_out);
    return err;
}

}  // namespace turbomind
