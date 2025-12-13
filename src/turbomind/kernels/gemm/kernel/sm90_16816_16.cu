// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/arch/config_sm80_s16816.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/types.h"

namespace turbomind::gemm {

using namespace sm80_s16816;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

void Registry::sm90_16816_16()
{
    if constexpr (1) {
        // clang-format off
        using C = Config_F16_g<Sm90, half, kColMajor>;
        Add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 64,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        // clang-format on
    }

    if constexpr (1) {
        // clang-format off
        using C = Config_F16_g<Sm90, nv_bfloat16, kColMajor>;
        Add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 64,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        // clang-format on
    }

    // Dense BF16 row-major kernels for sm90+/sm120:
    // These cover non-quantized Eagle3 draft/FFN/LM-head GEMMs where
    // GemmDesc.pack_a/pack_b/pack_u/pack_v are all zero and layouts are
    // plain row-major [m,k] @ [k,n] -> [m,n]. We reuse the same tile
    // shapes as the grouped Config_F16_g BF16 kernels so the scheduler
    // can pick reasonable launch configs while providing a fused GMMA
    // path instead of falling back to cuBLAS for BF16 problems.
    if constexpr (1) {
        using C = Config_F16_dense<Sm90, nv_bfloat16, kColMajor>;
        // CTA tiles largely mirror the BF16 Config_F16_g set; this
        // gives good coverage for thin-M / wide-N shapes like
        // 1x34560x2880, 3x34560x2880, 4x34560x2880 that appear in
        // Eagle3 draft and LM-head paths.
        Add<C::Type<256, 128,  64, 4, 2, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  64, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 256,  32, 2, 4, 1, D, D, 3,   0 , 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type<128, 128,  32, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 96,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 64,  64,  64, 2, 2, 1, D, D, 5, true, 1, 1>>();
        Add<C::Type< 64,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 32, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16,  64, 128, 1, 2, 2, D, D, 3, true, 1, 1>>();
        Add<C::Type< 16, 128,  64, 1, 4, 1, D, D, 3, true, 1, 1>>();
    }
}

}  // namespace turbomind::gemm
