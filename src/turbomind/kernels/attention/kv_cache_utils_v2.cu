// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/attention/quantization.h"
#include "src/turbomind/kernels/attention/rotary_embedding.h"
#include "src/turbomind/kernels/attention/fp4_kv_utils.h"
#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/thread_map.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

using cutlass::FastDivmod;

// MXFP4 helper: encode a scaled float into FP4(E2M1) nibble (0..15).
// The mapping matches the E2M1 decode path in quantization.h
// (ConvertKvCache<fp4_e2m1_t, T>) and the reference implementation in
// floating_point.h.
__device__ inline uint8_t cvt_rn_sat_e2m1_f32(float x)
{
    // 0000  0.0
    // 0001  0.5
    // 0010  1.0
    // 0011  1.5
    // 0100  2.0
    // 0101  3.0
    // 0110  4.0
    // 0111  6.0

    float z = fabsf(x);
    //   0.25  0.75   1.25  1.75  2.5   3.5    5.0
    // 0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0
    // 0000  0001  0010  0011  0100  0101  0110  0111
    auto f = [](float z_val) {
        if (z_val <= .25f) {
            return 0;
        }
        else if (z_val < .75f) {
            return 1;  // 0.5
        }
        else if (z_val <= 1.25f) {
            return 2;  // 1.0
        }
        else if (z_val < 1.75f) {
            return 3;  // 1.5
        }
        else if (z_val <= 2.5f) {
            return 4;  // 2.0
        }
        else if (z_val < 3.5f) {
            return 5;  // 3.0
        }
        else if (z_val <= 5.f) {
            return 6;  // 4.0
        }
        else {
            return 7;  // 6.0
        }
    };

    const uint8_t mag  = static_cast<uint8_t>(f(z));
    const uint8_t sign = static_cast<uint8_t>((__float_as_uint(x) >> 31) << 3);
    return static_cast<uint8_t>(sign | mag);
}

template<class Tkv, int CTA_S, int HeadDim, int WarpCnt, class T, class BlockLayout>
__global__ void __launch_bounds__(128) ProcessKV_v2(char**          blocks,
                                                    char**          scale_blocks,
                                                    const T*        k,
                                                    const T*        v,
                                                    const T*        k_bias,
                                                    const T*        v_bias,
                                                    const int*      cu_q_len,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{

    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Vec = Array<T, kVecSize>;
    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_S;  // local offset into `input_length`
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int qi_beg = cu_q_len[batch_idx];
    const int qi_end = cu_q_len[batch_idx + 1];
    const int q_len  = qi_end - qi_beg;

    const int k_len       = cu_k_len[batch_idx + 1] - cu_k_len[batch_idx];
    const int history_len = k_len - q_len;

    if (qi_beg + token_idx >= qi_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Vec __align__(16) vec_K[ITER_S][ITER_C];
    Vec __align__(16) vec_V[ITER_S][ITER_C];

    Vec bias_V[ITER_C];
    Vec bias_K[ITER_C];

    if (k_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_K[c], &k_bias[head_idx * HeadDim + di]);
        }
    }
    if (v_bias) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            Ldg(bias_V[c], &v_bias[head_idx * HeadDim + di]);
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int     qi = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
            const int     di = offset.x + c * Map::kDeltaC;
            const int64_t index =
                (batch_idx * stride_b + qi_beg * stride_c + qi * stride_s + head_idx * stride_h) * HeadDim + di;
            if (qi < q_len) {
                Ldg(vec_K[s][c], &k[index]);
                Ldg(vec_V[s][c], &v[index]);
            }
        }
    }

    if (k_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_K[s][c] = vec_K[s][c] + bias_K[c];
            }
        }
    }
    if (v_bias) {
        using namespace ops;
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                vec_V[s][c] = vec_V[s][c] + bias_V[c];
            }
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = history_len + offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(vec_K[s][c], ti);
            }
        }
    }

    Array<Tkv, kVecSize> out_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> out_V[ITER_S][ITER_C];

    if constexpr (std::is_same_v<Tkv, fp4_e2m1_t>) {
        // MXFP4 path: FP4(E2M1) payload with per-16-element exponent
        // scales stored in a parallel scale pool. We compute block-wise
        // maxima over 16-dim groups, derive power-of-two exponents, and
        // then quantize each element into E2M1 before packing into
        // Array<fp4_e2m1_t, kVecSize>.

        static_assert(HeadDim % 16 == 0, "FP4 MXFP4 requires head_dim % 16 == 0");
        static_assert(kVecSize == 8, "FP4 MXFP4 assumes Vec covers 8 elements");

        constexpr int kScalesPerHead = HeadDim / 16;

        // Per-thread local maxima for K/V per token (ITER_S) and per
        // 16-element block along head_dim.
        float local_max_K[ITER_S][kScalesPerHead];
        float local_max_V[ITER_S][kScalesPerHead];

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int b = 0; b < kScalesPerHead; ++b) {
                local_max_K[s][b] = 0.f;
                local_max_V[s][b] = 0.f;
            }
        }

        // Accumulate per-block maxima over this CTA for the current
        // (batch, head) tile.
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di_base = offset.x + c * Map::kDeltaC;
                PRAGMA_UNROLL
                for (int i = 0; i < kVecSize; ++i) {
                    const int di = di_base + i;
                    if (di < HeadDim) {
                        const int   scale_idx = di / 16;
                        const float vK        = fabsf(static_cast<float>(vec_K[s][c][i]));
                        const float vV        = fabsf(static_cast<float>(vec_V[s][c][i]));
                        local_max_K[s][scale_idx] = fmaxf(local_max_K[s][scale_idx], vK);
                        local_max_V[s][scale_idx] = fmaxf(local_max_V[s][scale_idx], vV);
                    }
                }
            }
        }

        // Warp-level reduction for maxima.
        __shared__ float sh_warp_max_K[ITER_S][kScalesPerHead][WarpCnt];
        __shared__ float sh_warp_max_V[ITER_S][kScalesPerHead][WarpCnt];

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int b = 0; b < kScalesPerHead; ++b) {
                float vK = local_max_K[s][b];
                float vV = local_max_V[s][b];
                PRAGMA_UNROLL
                for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                    vK = fmaxf(vK, __shfl_xor_sync(uint32_t(-1), vK, mask));
                    vV = fmaxf(vV, __shfl_xor_sync(uint32_t(-1), vV, mask));
                }
                if (lane_id == 0) {
                    sh_warp_max_K[s][b][warp_id] = vK;
                    sh_warp_max_V[s][b][warp_id] = vV;
                }
            }
        }

        __shared__ float   sh_inv_scale_K[ITER_S][kScalesPerHead];
        __shared__ float   sh_inv_scale_V[ITER_S][kScalesPerHead];
        __shared__ uint8_t sh_scale_byte_K[ITER_S][kScalesPerHead];
        __shared__ uint8_t sh_scale_byte_V[ITER_S][kScalesPerHead];

        __syncthreads();

        // CTA-level reduction over warps + exponent/scale computation.
        if (warp_id == 0) {
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                PRAGMA_UNROLL
                for (int b = 0; b < kScalesPerHead; ++b) {
                    float maxK = 0.f;
                    float maxV = 0.f;
                    PRAGMA_UNROLL
                    for (int w = 0; w < WarpCnt; ++w) {
                        maxK = fmaxf(maxK, sh_warp_max_K[s][b][w]);
                        maxV = fmaxf(maxV, sh_warp_max_V[s][b][w]);
                    }

                    // Power-of-two exponent per 16-element block; when the
                    // block is all zeros, keep scale=1 and exponent byte
                    // as 127 (2^0) so decoded values stay at zero.
                    auto compute_scale = [](float max_val, float& inv_scale, uint8_t& scale_byte) {
                        int exp = 0;
                        if (max_val > 0.f) {
                            const float ratio = max_val / 6.0f;
                            const float lg2   = log2f(ratio);
                            exp               = static_cast<int>(ceilf(lg2));
                        }
                        // Clamp to a reasonable exponent range and convert
                        // to biased 8-bit form.
                        int biased = exp + 127;
                        if (biased < 0) {
                            biased = 0;
                        }
                        else if (biased > 255) {
                            biased = 255;
                        }
                        scale_byte = static_cast<uint8_t>(biased);
                        // inv_scale = 1 / 2^exp
                        inv_scale = ldexpf(1.0f, -exp);
                    };

                    compute_scale(maxK, sh_inv_scale_K[s][b], sh_scale_byte_K[s][b]);
                    compute_scale(maxV, sh_inv_scale_V[s][b], sh_scale_byte_V[s][b]);
                }
            }
        }

        __syncthreads();

        // Quantize K/V into FP4(E2M1) using the per-block inverse scales.
        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                const int di_base = offset.x + c * Map::kDeltaC;

                // Pack 8 FP4 codes into one 32-bit word for this Vec.
                uint32_t packed_K = 0;
                uint32_t packed_V = 0;

                PRAGMA_UNROLL
                for (int i = 0; i < kVecSize; ++i) {
                    const int di = di_base + i;
                    if (di < HeadDim) {
                        const int   scale_idx = di / 16;
                        const float inv_K     = sh_inv_scale_K[s][scale_idx];
                        const float inv_V     = sh_inv_scale_V[s][scale_idx];

                        const float scaled_K = static_cast<float>(vec_K[s][c][i]) * inv_K;
                        const float scaled_V = static_cast<float>(vec_V[s][c][i]) * inv_V;

                        const uint8_t code_K = cvt_rn_sat_e2m1_f32(scaled_K);
                        const uint8_t code_V = cvt_rn_sat_e2m1_f32(scaled_V);

                        packed_K |= static_cast<uint32_t>(code_K & 0xF) << (4 * i);
                        packed_V |= static_cast<uint32_t>(code_V & 0xF) << (4 * i);
                    }
                }

                reinterpret_cast<uint32_t&>(out_K[s][c]) = packed_K;
                reinterpret_cast<uint32_t&>(out_V[s][c]) = packed_V;
            }
        }

        // Write back packed FP4 data and per-block exponent bytes into the
        // data and scale pools. Data is written via block::Head and the
        // existing SubBytePtr machinery; scales are laid out as:
        //
        //   [layer][kv_head][token][ K_scales (head_dim/16),
        //                           V_scales (head_dim/16) ]
        int local_ti      = 0;
        int local_ti_rank = 0;

        blocks += cu_block_num[batch_idx];
        char** scale_blocks_seq = scale_blocks ? scale_blocks + cu_block_num[batch_idx] : nullptr;

        block::Head<T, Tkv, BlockLayout> block_head{block_layout, layer_id, head_idx};

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
            const int ti = history_len + qi;                         // timestep
            local_ti     = cp_size.divmod(local_ti_rank, ti);
            if (qi < q_len && local_ti_rank == cp_rank) {
                // Write packed FP4 payloads.
                block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T*, T*) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        const int di = offset.x + c * Map::kDeltaC;
                        Store(&k_cache[di], out_K[s][c]);
                        Store(&v_cache[di], out_V[s][c]);
                    }
                });

                // Write exponent bytes into the parallel scale pool once
                // per (token, head, layer). To avoid races, restrict to a
                // single writer thread in the CTA.
                if (scale_blocks_seq && warp_id == 0 && lane_id == 0) {
                    const int block_len = block_head.block_len();
                    const int head_num  = block_layout.config().head_num();

                    uint8_t* scale_ptr = get_fp4_mx_scale_base(
                        scale_blocks_seq, layer_id, head_idx, head_num, block_len, HeadDim, local_ti);

                    if (!scale_ptr) {
                        return;
                    }

                    // Layout: [K_scales[0..kScalesPerHead), V_scales[0..kScalesPerHead)].
                    PRAGMA_UNROLL
                    for (int b = 0; b < kScalesPerHead; ++b) {
                        scale_ptr[b] = sh_scale_byte_K[s][b];
                    }
                    PRAGMA_UNROLL
                    for (int b = 0; b < kScalesPerHead; ++b) {
                        scale_ptr[kScalesPerHead + b] = sh_scale_byte_V[s][b];
                    }
                }
            }
        }

        return;
    }
    else {
        // Int4/Int8/base KV cache path: per-tile affine quantization with
        // per-token (scale, zero) parameters written into the param region.
        Array<T, 2> param_K[ITER_S];
        Array<T, 2> param_V[ITER_S];

        if constexpr (!std::is_same_v<T, Tkv>) {
            warp_stats<Map::kWarpThreadC>(param_K, vec_K, bitsof<Tkv>);
            warp_stats<Map::kWarpThreadC>(param_V, vec_V, bitsof<Tkv>);
        }

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            ConvertKvCache<T, Tkv> conv_K{param_K[s][0], param_K[s][1]};
            ConvertKvCache<T, Tkv> conv_V{param_V[s][0], param_V[s][1]};
            PRAGMA_UNROLL
            for (int c = 0; c < ITER_C; ++c) {
                out_K[s][c] = conv_K(vec_K[s][c]);
                out_V[s][c] = conv_V(vec_V[s][c]);
            }
        }

        int local_ti, local_ti_rank;

        blocks += cu_block_num[batch_idx];
        (void)scale_blocks;

        block::Head<T, Tkv, BlockLayout> block_head{block_layout, layer_id, head_idx};

        PRAGMA_UNROLL
        for (int s = 0; s < ITER_S; ++s) {
            const int qi = offset.y + s * Map::kDeltaS + token_idx;  // local offset into `input_length`
            const int ti = history_len + qi;                         // timestep
            local_ti     = cp_size.divmod(local_ti_rank, ti);
            if (qi < q_len && local_ti_rank == cp_rank) {
                block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        int di = offset.x + c * Map::kDeltaC;
                        Store(&k_cache[di], out_K[s][c]);
                        Store(&v_cache[di], out_V[s][c]);
                    }
                    if constexpr (!std::is_same_v<T, Tkv>) {
                        if (offset.x == 0) {
                            StoreQuantParam<Tkv>(k_param, param_K[s]);
                            StoreQuantParam<Tkv>(v_param, param_V[s]);
                        }
                    }
                });
            }
        }

        return;
    }

}

template<class T>
void invokeProcessKV_v2(char**                 blocks,
                        char**                 scale_blocks,
                        const T*               k,
                        const T*               v,
                        const T*               k_bias,
                        const T*               v_bias,
                        const int*             cu_q_len,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        FastDivmod             cp_size,
                        int                    max_q_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        int                    arch,
                        cudaStream_t           stream)
{
    constexpr int WARPS = 4;
    constexpr int CTA_S = 64;

    int  block = WARPS * WARP_SIZE;
    dim3 grid((max_q_len + CTA_S - 1) / CTA_S, head_num, batch_size);

    auto invoke = [&](auto tkv, const auto dim) {
        using Tkv = decltype(tkv);

        constexpr int kHeadDim = dim;
        FT_CHECK(head_dim == kHeadDim);

        block::Layout block_layout{block::Config<T, Tkv, kHeadDim>{head_num, block_seq_len}};

        ProcessKV_v2<Tkv, CTA_S, kHeadDim, WARPS><<<grid, block, 0, stream>>>(blocks,
                                                                              scale_blocks,
                                                                              k,
                                                                              v,
                                                                              k_bias,
                                                                              v_bias,
                                                                              cu_q_len,
                                                                              cu_k_len,
                                                                              cu_block_num,
                                                                              rope_param,
                                                                              stride_b,
                                                                              stride_c,
                                                                              stride_h,
                                                                              stride_s,
                                                                              layer_id,
                                                                              cp_rank,
                                                                              cp_size,
                                                                              block_layout);
    };

    auto dispatch = [&](auto tkv) {
        if (head_dim == 64) {
            return invoke(tkv, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(tkv, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(tkv, std::integral_constant<int, 192>{});
        }
        FT_CHECK(0);
    };

    const KvCacheMode kv_mode = GetKvCacheMode(quant_policy, arch);

    if (kv_mode == KvCacheMode::kInt8) {
        dispatch(uint8_t{});
    }
    else if (kv_mode == KvCacheMode::kInt4) {
        dispatch(uint4_t{});
    }
    else if (kv_mode == KvCacheMode::kFp4Mx) {
#if defined(ENABLE_FP4)
        FT_CHECK_WITH_INFO(scale_blocks != nullptr,
                           "[kv_cache_v2][FP4] MXFP4 FP4 KV cache requires a scale pool; "
                           "scale_blocks must be non-null and initialized.");
        dispatch(fp4_e2m1_t{});
#else
        FT_CHECK_WITH_INFO(false,
                           "[kv_cache_v2][FP4] FP4 KV cache requested but ENABLE_FP4 is not defined; "
                           "rebuild TurboMind with -DENABLE_FP4 and CUDA 12.8+.");
#endif
    }
    else if (kv_mode == KvCacheMode::kFp4Nv) {
#if defined(ENABLE_FP4) && defined(ENABLE_FP8)
        FT_CHECK_WITH_INFO(scale_blocks != nullptr,
                           "[kv_cache_v2][FP8] NVFP4 FP4 KV cache requires a scale pool; "
                           "scale_blocks must be non-null and initialized.");
        dispatch(fp4_e2m1_t{});
#else
        FT_CHECK_WITH_INFO(false,
                           "[kv_cache_v2][FP8] NVFP4 KV cache requested but ENABLE_FP4/ENABLE_FP8 is not defined; "
                           "rebuild TurboMind with -DENABLE_FP4 -DENABLE_FP8 and CUDA 12.8+.");
#endif
    }
    else {
        dispatch(T{});
    }

    if (::turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_INVARIANTS_DEBUG")) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            TM_LOG_ERROR("[kv_cache_utils_v2][invariants] CUDA error %d (%s) after invokeProcessKV_v2 "
                         "(block_seq_len=%d, head_dim=%d, batch_size=%d)",
                         static_cast<int>(err),
                         cudaGetErrorString(err),
                         block_seq_len,
                         head_dim,
                         batch_size);
            std::abort();
        }
    }
}

#define INSTANTIATE_invokeProcessKV_v2(type)                                                                           \
    template void invokeProcessKV_v2(char**                 blocks,                                                    \
                                     char**                 scale_blocks,                                              \
                                     const type*            k,                                                         \
                                     const type*            v,                                                         \
                                     const type*            k_bias,                                                    \
                                     const type*            v_bias,                                                    \
                                     const int*             cu_q_len,                                                  \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_q_len,                                                 \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     int                    arch,                                                      \
                                     cudaStream_t           stream);

INSTANTIATE_invokeProcessKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeProcessKV_v2(nv_bfloat16);
#endif

template<int CTA_S, int HeadDim, int WarpCnt, class T, class Tkv, class BlockLayout>
__global__ void __launch_bounds__(128) flattenKV_v2(T*              k,
                                                    T*              v,
                                                    const Tkv**     blocks,
                                                    char**          scale_blocks,
                                                    const int*      cu_k_len,
                                                    const int*      cu_block_num,
                                                    RopeKernelParam rope_param,
                                                    int64_t         stride_b,
                                                    int64_t         stride_c,
                                                    int64_t         stride_h,
                                                    int64_t         stride_s,
                                                    int             layer_id,
                                                    int             cp_rank,
                                                    FastDivmod      cp_size,
                                                    BlockLayout     block_layout)
{
    constexpr int kVecSize = sizeof(uint4) / sizeof(T);

    using Map = RakedThreadMap<HeadDim, CTA_S, kVecSize, WarpCnt>;

    constexpr int ITER_C = Map::kIterC;
    constexpr int ITER_S = Map::kIterS;

    const int token_idx = blockIdx.x * CTA_S;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int ti_0   = cu_k_len[0];
    const int ti_beg = cu_k_len[batch_idx] - ti_0;
    const int ti_end = cu_k_len[batch_idx + 1] - ti_0;

    const int seq_len = ti_end - ti_beg;

    if (ti_beg + token_idx >= ti_end) {  // empty tile
        return;
    }

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int2 offset = Map::get_offset(warp_id, lane_id);

    Array<Tkv, kVecSize> __align__(16) vec_K[ITER_S][ITER_C];
    Array<Tkv, kVecSize> __align__(16) vec_V[ITER_S][ITER_C];

    Array<T, kVecSize> __align__(16) out_K[ITER_S][ITER_C];
    Array<T, kVecSize> __align__(16) out_V[ITER_S][ITER_C];

    blocks += cu_block_num[batch_idx];

    block::Head<T, Tkv, BlockLayout> block_head{block_layout, layer_id, head_idx};

    Array<T, 2> param_K[ITER_S];
    Array<T, 2> param_V[ITER_S];

    int local_ti, local_ti_rank;

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        const int si = offset.y + s * Map::kDeltaS + token_idx;
        local_ti     = cp_size.divmod(local_ti_rank, si);
        if (si < seq_len && local_ti_rank == cp_rank) {
            block_head.with((char**)blocks, local_ti, [&](auto k_cache, auto v_cache, T* k_param, T* v_param) {
                if constexpr (std::is_same_v<Tkv, fp4_e2m1_t>) {
                    // MXFP4 path: Dequantize FP4(E2M1) payload using per-16-element exponent scales.
                    static_assert(HeadDim % 16 == 0, "FP4 MXFP4 requires head_dim % 16 == 0");
                    static_assert(kVecSize == 8, "FP4 MXFP4 assumes Vec covers 8 elements");

                    constexpr int kScalesPerHead = HeadDim / 16;
                    
                    uint8_t* scale_ptr = get_fp4_mx_scale_base(
                        scale_blocks + cu_block_num[batch_idx], // scale_blocks is now directly accessible here
                        layer_id,
                        head_idx,
                        block_layout.config().head_num(),
                        block_layout.config().block_len(),
                        HeadDim,
                        local_ti
                    );

                    ConvertKvCache<fp4_e2m1_t, T> dequantizer{};
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        const int di_base = offset.x + c * Map::kDeltaC;
                        Ldg(vec_K[s][c], &k_cache[di_base]);
                        Ldg(vec_V[s][c], &v_cache[di_base]);

                        // Dequantize FP4 to T, then apply per-block scales
                        auto dequant_K = dequantizer.convert(vec_K[s][c]);
                        auto dequant_V = dequantizer.convert(vec_V[s][c]);

                        PRAGMA_UNROLL
                        for (int i = 0; i < kVecSize; ++i) {
                            const int di = di_base + i;
                            if (di < HeadDim) {
                                const int scale_idx = di / 16; // K and V scales are interleaved for a given token
                                const uint8_t exponent_u8_K = scale_ptr[scale_idx];
                                const float scale_val_K = __expf((float)(exponent_u8_K - 127) * 0.693147182f);

                                const uint8_t exponent_u8_V = scale_ptr[kScalesPerHead + scale_idx];
                                const float scale_val_V = __expf((float)(exponent_u8_V - 127) * 0.693147182f);
                                
                                out_K[s][c][i] = dequant_K[i] * static_cast<T>(scale_val_K);
                                out_V[s][c][i] = dequant_V[i] * static_cast<T>(scale_val_V);
                            }
                        }
                    }
                }
                else { // For Int4/Int8/Base KV
                    ConvertKvCache<Tkv, T> conv_K{param_K[s][0], param_K[s][1]};
                    ConvertKvCache<Tkv, T> conv_V{param_V[s][0], param_V[s][1]};
                    PRAGMA_UNROLL
                    for (int c = 0; c < ITER_C; ++c) {
                        int di = offset.x + c * Map::kDeltaC;
                        Ldg(vec_K[s][c], &k_cache[di]);
                        Ldg(vec_V[s][c], &v_cache[di]);
                        out_K[s][c] = conv_K(vec_K[s][c]);
                        out_V[s][c] = conv_V(vec_V[s][c]);
                    }
                    if constexpr (!std::is_same_v<T, Tkv>) {
                        Ldg(param_K[s], k_param);
                        Ldg(param_V[s], v_param);
                    }
                }
            });
        }
    }

    if (rope_param.type != RopeType::kNull) {
        FastRoPE rope(rope_param, batch_idx, std::integral_constant<int, kVecSize>{});
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int di = offset.x + c * Map::kDeltaC;
            rope.init(di);
            PRAGMA_UNROLL
            for (int s = 0; s < ITER_S; ++s) {
                const int ti = offset.y + s * Map::kDeltaS + token_idx;  // sequence local
                rope.apply(out_K[s][c], ti);
            }
        }
    }

    PRAGMA_UNROLL
    for (int s = 0; s < ITER_S; ++s) {
        PRAGMA_UNROLL
        for (int c = 0; c < ITER_C; ++c) {
            const int si = offset.y + s * Map::kDeltaS + token_idx;
            const int di = offset.x + c * Map::kDeltaC;
            local_ti     = cp_size.divmod(local_ti_rank, si);
            if (si < seq_len && local_ti_rank == cp_rank) {
                const int64_t index =
                    (batch_idx * stride_b + ti_beg * stride_c + local_ti * stride_s + head_idx * stride_h) * HeadDim
                    + di;
                Store(&k[index], out_K[s][c]);
                Store(&v[index], out_V[s][c]);
            }
        }
    }
}

template<class T>
void invokeFlattenKV_v2(T*                     k,
                        T*                     v,
                        char**                 blocks,
                        char**                 scale_blocks,
                        const int*             cu_k_len,
                        const int*             cu_block_num,
                        const RopeKernelParam& rope_param,
                        int64_t                stride_b,
                        int64_t                stride_c,
                        int64_t                stride_h,
                        int64_t                stride_s,
                        int                    block_seq_len,
                        int                    layer_id,
                        int                    cp_rank,
                        FastDivmod             cp_size,
                        int                    max_seq_len,
                        int                    head_num,
                        int                    head_dim,
                        int                    batch_size,
                        int                    quant_policy,
                        int                    arch,
                        cudaStream_t           stream)
{
    constexpr int kWarpCnt = 4;
    constexpr int CTA_S    = 64;

    constexpr int block = kWarpCnt * WARP_SIZE;
    const dim3    grid((max_seq_len + CTA_S - 1) / CTA_S, head_num, batch_size);

    auto invoke = [&](auto tkv, const auto dim) {
        using Tkv = decltype(tkv);

        constexpr int kHeadDim = dim;
        FT_CHECK(head_dim == kHeadDim);

        block::Layout block_layout{block::Config<T, Tkv, kHeadDim>{head_num, block_seq_len}};

        flattenKV_v2<CTA_S, kHeadDim, kWarpCnt><<<grid, block, 0, stream>>>(k,
                                                                            v,
                                                                            (const Tkv**)blocks,
                                                                            scale_blocks,
                                                                            cu_k_len,
                                                                            cu_block_num,
                                                                            rope_param,
                                                                            stride_b,
                                                                            stride_c,
                                                                            stride_h,
                                                                            stride_s,
                                                                            layer_id,
                                                                            cp_rank,
                                                                            cp_size,
                                                                            block_layout);
    };

    auto dispatch = [&](auto tkv) {
        if (head_dim == 64) {
            return invoke(tkv, std::integral_constant<int, 64>{});
        }
        else if (head_dim == 128) {
            return invoke(tkv, std::integral_constant<int, 128>{});
        }
        else if (head_dim == 192) {
            return invoke(tkv, std::integral_constant<int, 192>{});
        }
        FT_CHECK(0);
    };

    const KvCacheMode kv_mode = GetKvCacheMode(quant_policy, arch);

#if defined(ENABLE_FP4)
    if (kv_mode == KvCacheMode::kFp4Mx) {
        FT_CHECK_WITH_INFO(scale_blocks != nullptr,
                           "[kv_cache_v2][FP4] MXFP4 FP4 KV cache requires a scale pool; "
                           "scale_blocks must be non-null and initialized.");
        dispatch(fp4_e2m1_t{});
    }
    else if (kv_mode == KvCacheMode::kFp4Nv) {
        // NVFP4 path is kept gated for now. Fall back to unquantized KV.
        dispatch(T{});
    }
    else if (kv_mode == KvCacheMode::kInt8) {
        dispatch(uint8_t{});
    }
    else if (kv_mode == KvCacheMode::kInt4) {
        dispatch(uint4_t{});
    }
    else {
        dispatch(T{});
    }
#else
    if (kv_mode == KvCacheMode::kInt8) {
        dispatch(uint8_t{});
    }
    else if (kv_mode == KvCacheMode::kInt4) {
        dispatch(uint4_t{});
    }
    else {
        dispatch(T{});
    }
#endif
}

#define INSTANTIATE_invokeFlattenKV_v2(type)                                                                           \
    template void invokeFlattenKV_v2(type*                  k,                                                         \
                                     type*                  v,                                                         \
                                     char**                 blocks,                                                    \
                                     char**                 scale_blocks,                                              \
                                     const int*             cu_k_len,                                                  \
                                     const int*             cu_block_num,                                              \
                                     const RopeKernelParam& rope_param,                                                \
                                     int64_t                stride_b,                                                  \
                                     int64_t                stride_c,                                                  \
                                     int64_t                stride_h,                                                  \
                                     int64_t                stride_s,                                                  \
                                     int                    block_seq_len,                                             \
                                     int                    layer_id,                                                  \
                                     int                    cp_rank,                                                   \
                                     FastDivmod             cp_size,                                                   \
                                     int                    max_seq_len,                                               \
                                     int                    head_num,                                                  \
                                     int                    head_dim,                                                  \
                                     int                    batch_size,                                                \
                                     int                    quant_policy,                                              \
                                     int                    arch,                                                      \
                                     cudaStream_t           stream);

INSTANTIATE_invokeFlattenKV_v2(half);
#if ENABLE_BF16
INSTANTIATE_invokeFlattenKV_v2(nv_bfloat16);
#endif

size_t get_cache_block_size(DataType dtype,
                            DataType kvtype,
                            int      layer_num,
                            int      head_num,
                            int      head_dim,
                            int      block_seq_len)
{
    // Mirror the block::Layout math in block.h:
    //   token_data_size = q_bits * head_dim / 8
    //   token_param_size = t_bits * 2 / 8
    //   head_data_size  = block_len * token_data_size
    //   head_param_size = block_len * token_param_size
    //   layer_size      = head_num * 2 * head_data_size
    //                   + head_num * 2 * head_param_size
    //   block_size      = layer_size * layer_num

    auto bits_for = [](DataType dt) -> int {
        switch (dt) {
            case kFloat16:
            case kBfloat16:
                return 16;
            case kFloat32:
                return 32;
            case kUint8:
                return 8;
            case kUint4:
                return 4;
            case kUint2:
                return 2;
            default:
                return 8;
        }
    };

    const int q_bits = bits_for(kvtype);
    int       t_bits = 0;
    if (kvtype != dtype) {
        t_bits = bits_for(dtype);
    }

    const int token_data_bits   = q_bits * head_dim;
    const int token_param_bits  = t_bits * 2;
    const int token_data_bytes  = token_data_bits / 8;
    const int token_param_bytes = token_param_bits / 8;

    const size_t head_data_bytes  = static_cast<size_t>(block_seq_len) * token_data_bytes;
    const size_t head_param_bytes = static_cast<size_t>(block_seq_len) * token_param_bytes;

    const size_t layer_size =
        static_cast<size_t>(head_num) * 2 * head_data_bytes
        + static_cast<size_t>(head_num) * 2 * head_param_bytes;

    return layer_size * static_cast<size_t>(layer_num);
}

}  // namespace turbomind
