// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

/// Shared helper for FP4 MXFP4 KV cache scale layout.
///
/// Scales are stored in a dedicated scale pool with layout:
///   [layer][kv_head][token][ K_scales(0..head_dim/16),
///                            V_scales(0..head_dim/16) ]
///
/// Each "token" here is a timestep within a block of length `block_len`.
/// This helper returns the base address for the K_scales region for the
/// given (layer_id, head_idx, local_ti) triple, from which callers can
/// index into K and V scales by:
///   k_scale = base[block_idx]
///   v_scale = base[block_idx + head_dim/16]
///
/// The `scale_blocks_seq` pointer must already be offset by cu_block_num
/// for the current batch slot.
__device__ __forceinline__ uint8_t* get_fp4_mx_scale_base(char** scale_blocks_seq,
                                                          int    layer_id,
                                                          int    head_idx,
                                                          int    head_num,
                                                          int    block_len,
                                                          int    head_dim,
                                                          int    local_ti)
{
    if (!scale_blocks_seq) {
        return nullptr;
    }

    const int block_id = local_ti / block_len;
    const int block_ti = local_ti % block_len;

    char* scale_base = scale_blocks_seq[block_id];
    if (!scale_base) {
        return nullptr;
    }

    const int scales_per_head   = head_dim / 16;
    const int bytes_per_token   = 2 * scales_per_head;  // K scales + V scales
    const int head_size         = block_len * bytes_per_token;
    const int layer_size        = head_num * head_size;
    const size_t token_offset   = static_cast<size_t>(layer_id) * layer_size
                                + static_cast<size_t>(head_idx) * head_size
                                + static_cast<size_t>(block_ti) * bytes_per_token;

    return reinterpret_cast<uint8_t*>(scale_base + token_offset);
}

}  // namespace turbomind

