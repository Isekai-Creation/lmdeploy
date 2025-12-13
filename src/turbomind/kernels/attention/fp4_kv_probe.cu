// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/kernels/attention/fp4_kv_utils.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

struct Fp4KvProbeResult {
    uint8_t k_scale0;
    uint8_t v_scale0;
    uint8_t kv_byte0;
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
        out->kv_byte0 = 0;
        return;
    }

    const int scales_per_head = head_dim / 16;

    const uint8_t k_scale0 = scale_base[0];
    const uint8_t v_scale0 = scale_base[scales_per_head];

    // Read the first packed FP4 byte from KV payload at di=0.
    block::Head<T, fp4_e2m1_t, BlockLayout> head{BlockLayout{block::Config<T, fp4_e2m1_t, 0>{head_num, block_len}},
                                                 layer_id,
                                                 head_idx};
    int block_id, block_ti;
    head.get_block_coord(local_ti, block_id, block_ti);

    char* block = blocks[block_id];
    auto  k_ptr = head.k_data(block, block_ti);

    uint8_t kv_byte0 = reinterpret_cast<const uint8_t*>(k_ptr.ptr_)[0];

    out->k_scale0 = k_scale0;
    out->v_scale0 = v_scale0;
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
    using BlockLayout = block::Layout<block::Config<T, fp4_e2m1_t, 0>>;
    Fp4KvProbeKernel<T, BlockLayout><<<1, 1, 0, stream>>>(
        d_out, blocks, scale_blocks, layer_id, head_idx, head_num, block_len, head_dim, local_ti);
}

}  // namespace turbomind

