// Copyright (c) OpenMMLab. All rights reserved.

#include <type_traits>
#include <utility>

#include "decoding.h"
#include "decoding_config.h"
#include "src/turbomind/kernels/attention/arch.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class Kernel>
bool invokeDecoding(const typename Kernel::ParamType& params);

template<int... idxs>
using seq = std::integer_sequence<int, idxs...>;

template<class T, int is_kv_int8>
constexpr auto get_kv_type(std::integral_constant<int, is_kv_int8>)
{
    if constexpr (is_kv_int8) {
        return int8_t{};
    }
    else {
        return T{};
    }
}

template<class T>
void dispatchDecoding(const AttentionParams<T>& params)
{
    // Work on a local copy so we can annotate kv_quant_kind without
    // changing the caller's struct layout expectations.
    AttentionParams<T> p = params;

    const KvCacheMode kv_mode = GetKvCacheMode(p.quant_policy, p.arch);

    const bool is_kv_int8      = kv_mode == KvCacheMode::kInt8;
    const bool is_kv_int4      = kv_mode == KvCacheMode::kInt4;
    const bool is_kv_fp4       = kv_mode == KvCacheMode::kFp4Mx;  // MXFP4 only; NVFP4 falls back to base KV for now.
    const int  query_group_sz  = p.num_heads / p.num_kv_heads;

    if (is_kv_fp4) {
        // MXFP4 FP4 KV cache: FP4 payload + exponent-per-16 scales in a
        // separate scale pool. NVFP4 (kFp4Nv) continues to use base KV
        // until FP8(E4M3) scale semantics are implemented.
        p.kv_quant_kind = KvQuantKind::kFp4Mx;
    }
    else if (is_kv_int4 || is_kv_int8) {
        // Affine integer KV cache (int4/int8 scale+zero).
        p.kv_quant_kind = KvQuantKind::kAffineInt;
    }
    else {
        p.kv_quant_kind = KvQuantKind::kNone;
    }

    using namespace attention;

    /// TODO: we need better Qh dispatching, when #waves < 1, smaller Qh may outperform larger Qh due to better
    // concurrency
    auto dispatch_h = [&](auto arch, auto kv, const auto dim) -> bool {
        using Arch             = decltype(arch);
        using Tkv              = decltype(kv);
        constexpr int kHeadDim = dim;
        if (0) {}
        else if (query_group_sz > 8) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 9, kHeadDim>>(p);
        }
        else if (query_group_sz == 8) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 8, kHeadDim>>(p);
        }
        else if (query_group_sz == 7) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 7, kHeadDim>>(p);
        }
        else if (query_group_sz == 6) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 6, kHeadDim>>(p);
        }
        else if (query_group_sz == 5) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 5, kHeadDim>>(p);
        }
        else if (query_group_sz == 4) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 4, kHeadDim>>(p);
        }
        else if (query_group_sz == 3) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 3, kHeadDim>>(p);
        }
        else if (query_group_sz == 2) {
            return invokeDecoding<Decoding<Arch, T, Tkv, 2, kHeadDim>>(p);
        }
        else {
            return invokeDecoding<Decoding<Arch, T, Tkv, 1, kHeadDim>>(p);
        }
        return false;
    };

    auto dispatch_kv = [&](auto arch, const auto dim) -> bool {
        FT_CHECK(!(is_kv_int4 && is_kv_int8));
        if (is_kv_fp4) {
#if defined(ENABLE_FP4)
            // MXFP4 FP4 KV cache decode: currently supported on Hopper (SM89/90)
            // and uses SIMT mainloop specialization in DecodingConfig.
            FT_CHECK_WITH_INFO(p.arch == 90 || p.arch == 89, "[decoding][FP4] MXFP4 FP4 KV cache decode is only supported on SM90/SM89");
            FT_CHECK_WITH_INFO(p.block_iter_params.scale_block_ptrs != nullptr, "[decoding][FP4] MXFP4 FP4 KV cache requires scale_block_ptrs to be non-null");
            return dispatch_h(arch, fp4_e2m1_t{}, dim);
#else
            FT_CHECK_WITH_INFO(false, "[decoding][FP4] FP4 KV cache requested but ENABLE_FP4 is not defined");
#endif
        }
        else if (is_kv_int4) {
            return dispatch_h(arch, uint4_t{}, dim);
        }
        else if (is_kv_int8) {
            return dispatch_h(arch, uint8_t{}, dim);
        }
        else {
            return dispatch_h(arch, T{}, dim);
        }
        return false;
    };

    auto dispatch_head_dim = [&](auto arch) {
        if (p.size_per_head == 128) {
            return dispatch_kv(arch, std::integral_constant<int, 128>{});
        }
        else if (p.size_per_head == 64) {
            return dispatch_kv(arch, std::integral_constant<int, 64>{});
        }
        return false;
    };

    auto dispatch = [&]() {
        if (p.arch >= 80) {
            return dispatch_head_dim(arch::Sm80{});
        }

        if constexpr (!std::is_same_v<T, nv_bfloat16>) {
            if (p.arch == 75) {
                return dispatch_head_dim(arch::Sm75{});
            }
            else if (p.arch >= 70) {
                return dispatch_head_dim(arch::Sm70{});
            }
        }

        return false;
    };

    if (p.size_per_head == 192) {

        if (is_kv_int8) {
            invokeDecoding<Decoding<arch::Sm80, T, uint8_t, 1, 192>>(p);
        }
        else if (is_kv_int4) {
            FT_CHECK_WITH_INFO(!is_kv_int4, "not implemented");
            // invokeDecoding<Decoding<arch::Sm80, T, uint4_t, 1, 192>>(params);
        }
        else {
            invokeDecoding<Decoding<arch::Sm80, T, T, 1, 192>>(p);
        }
        return;
    }

    auto success = dispatch();

    FT_CHECK(success);
}

template void dispatchDecoding(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchDecoding(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
