// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/kernels/attention/quantization.h"

#include <cstdint>

namespace turbomind {

/// MXFP4 helper: convert a packed FP4(E2M1) block to T and apply
/// a per-block power-of-two scale encoded as an 8-bit biased exponent.
///
/// The input `vi` holds E2M1 values (as produced by ProcessKV_v2).
/// The `scale_byte` is the biased exponent (exp + 127) for this 16‑value
/// block. We reconstruct:
///
///   scale = 2^(scale_byte - 127)
///   out   = ConvertKvCache<fp4_e2m1_t, T>::convert(vi) * scale
template<class T, int N>
__device__ inline auto decode_fp4mx_block(const Array<fp4_e2m1_t, N>& vi, uint8_t scale_byte) -> Array<T, N>
{
    auto  decoded = ConvertKvCache<fp4_e2m1_t, T>::convert(vi);
    int   exp     = static_cast<int>(scale_byte) - 127;
    float scale   = ldexpf(1.0f, exp);
    PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
        decoded[i] = static_cast<T>(static_cast<float>(decoded[i]) * scale);
    }
    return decoded;
}

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

// Decode-side convenience wrapper: derive the same scale base pointer
// from a BlockIterator-like cache iterator that exposes the scale pool
// pointer array and basic layout metadata. The iterator must provide:
//   - char** scale_blocks_seq()
//   - int    layer_id()
//   - int    head_idx()
//   - int    head_num()
//   - int    block_len()
template<class CacheIter>
__device__ __forceinline__ uint8_t* get_fp4_mx_scale_base_from_iter(const CacheIter& cache_iter,
                                                                    int              head_dim,
                                                                    int              local_ti)
{
    char** scale_blocks_seq = cache_iter.scale_blocks_seq();
    if (!scale_blocks_seq) {
        return nullptr;
    }
    return get_fp4_mx_scale_base(scale_blocks_seq,
                                 cache_iter.layer_id(),
                                 cache_iter.head_idx(),
                                 cache_iter.head_num(),
                                 cache_iter.block_len(),
                                 head_dim,
                                 local_ti);
}

/// Debug probe result for a single (layer, head, token) in MXFP4 KV cache.
///
/// This is used only by test / debug harnesses to verify that:
///   - scale bytes are written where we expect them, and
///   - payload packing is stable and non‑zero.
struct Fp4KvProbeResult {
    uint8_t k_scale0;
    uint8_t v_scale0;
    uint8_t k_scale1;
    uint8_t v_scale1;
    uint8_t kv_byte0;
};

/// Launch a tiny 1‑thread kernel that reads the first few scale bytes and one
/// packed FP4 payload byte for a given (layer, head, token).
///
/// The caller is responsible for allocating `d_out` on device memory and
/// copying it back to host after the kernel completes.
void fp4_kv_probe(Fp4KvProbeResult* d_out,
                  char**            blocks,
                  char**            scale_blocks,
                  int               layer_id,
                  int               head_idx,
                  int               head_num,
                  int               block_len,
                  int               head_dim,
                  int               local_ti,
                  cudaStream_t      stream);

/// Host-side helper that:
///   - allocates a temporary device buffer,
///   - launches `fp4_kv_probe`,
///   - copies the result back to `out`,
///   - and synchronizes the provided stream.
cudaError_t fp4_kv_probe_host(Fp4KvProbeResult& out,
                              char**            d_blocks,
                              char**            d_scale_blocks,
                              int               layer_id,
                              int               head_idx,
                              int               head_num,
                              int               block_len,
                              int               head_dim,
                              int               local_ti,
                              cudaStream_t      stream);

}  // namespace turbomind
