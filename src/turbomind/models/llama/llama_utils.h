// Copyright (c) OpenMMLab. All rights reserved.

#pragma once
#include "src/turbomind/utils/nvtx_utils.h"
#include <cuda_runtime.h>
#include <sstream>
#include <string>
#include <vector>

namespace turbomind {

enum QuantPolicy
{
    kNone = 0x00,
    // reserve 0x01 and 0x02 for backward compatibility
    kReserve1 = 0x01,
    kReserve2 = 0x02,
    // quantize cache kv (integer modes)
    kCacheKVInt8 = 0x08,
    kCacheKVInt4 = 0x04,
    // FP4 KV cache (FP4 E2M1 payload + per-16-element scales; MXFP4/NVFP4 family)
    kCacheKVFp4  = 0x10,
};

enum class KvCacheMode
{
    kNone,
    kInt4,
    kInt8,
    kFp4Mx,   // FP4 + exponent-style per-block scaling (MXFP4 fallback)
    kFp4Nv,   // FP4 + FP8(E4M3) per-block scales (NVFP4, Blackwell)
};

inline KvCacheMode GetKvCacheMode(int quant_policy, int sm_version)
{
    const bool has_fp4  = (quant_policy & kCacheKVFp4) != 0;
    const bool has_int8 = (quant_policy & kCacheKVInt8) != 0;
    const bool has_int4 = (quant_policy & kCacheKVInt4) != 0;

    // Prefer FP4 modes when explicitly requested.
    if (has_fp4) {
#if defined(ENABLE_FP4)
        // NVFP4 on known Blackwell-class SMs.
        // NOTE: keep this list in sync with CUDA / hardware docs.
        if (sm_version == 100 || sm_version == 101 || sm_version == 120 || sm_version == 121) {
            return KvCacheMode::kFp4Nv;
        }
        // MXFP4-style fallback on Hopper/SM90 if desired.
        if (sm_version == 90 || sm_version == 89) {
            return KvCacheMode::kFp4Mx;
        }
        // Unsupported SM for FP4 – fall through to integer modes.
#else
        // FP4 requested but not compiled in – fall through to integer modes.
#endif
    }

    if (has_int8) {
        return KvCacheMode::kInt8;
    }
    if (has_int4) {
        return KvCacheMode::kInt4;
    }
    return KvCacheMode::kNone;
}

enum CmpMode
{
    kCmpNone,
    kCmpRead,
    kCmpWrite,
};

extern CmpMode compare_mode;

template<typename T>
void Compare(T* ptr, size_t size, std::string key, CmpMode mode, cudaStream_t stream);

template<typename T>
void CheckNan(const T* ptr, size_t size, std::string key, cudaStream_t stream);

namespace detail {

template<typename T>
std::string to_string(T x)
{
    return std::to_string(x);
}

inline std::string to_string(std::string x)
{
    return x;
}

}  // namespace detail

template<typename... Args>
std::string Concat(std::string key, Args&&... args)
{
    std::vector<std::string> args_str{detail::to_string((Args &&) args)...};
    for (const auto& s : args_str) {
        key.append("_");
        key.append(s);
    }
    return key;
}

size_t curandStateGetSize();

bool isDebug();

struct NvtxScope {
    explicit NvtxScope(const std::string& name)
    {
        PUSH_RANGE(name.c_str());
    }

    ~NvtxScope()
    {
        POP_RANGE;
    }
};

int64_t& gSequenceIds(int batch_idx);

bool& isTuning();

}  // namespace turbomind
