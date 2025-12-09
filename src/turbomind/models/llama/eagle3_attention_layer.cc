// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/eagle3_attention_layer.h"

#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

Eagle3AttentionLayer::Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream):
    stream_{stream},
    device_prop_{prop}
{
    (void)device_prop_;
}

void Eagle3AttentionLayer::Forward(Eagle3AttentionParam& param)
{
    TM_LOG_DEBUG(__PRETTY_FUNCTION__);

    if (!param.input || !param.output || !param.weights || !param.weights->is_initialized) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid Eagle3AttentionParam; "
            "treating Eagle-3 attention as pass-through for this step.");
        if (param.input && param.output) {
            core::Copy(param.input, param.output);
        }
        return;
    }

    const int batch_size = param.input.shape(0);
    const int q_in_dim   = param.input.shape(1);
    const auto dtype     = param.input.dtype();

    if (batch_size <= 0 || q_in_dim <= 0) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] non-positive batch or hidden dim "
            "(batch=%d, q_in=%d); treating as pass-through.",
            batch_size,
            q_in_dim);
        core::Copy(param.input, param.output);
        return;
    }

    const auto& w = *param.weights;

    if (q_in_dim != w.q_in) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] input dim (%d) != q_in (%d); "
            "treating Eagle-3 attention as pass-through.",
            q_in_dim,
            w.q_in);
        core::Copy(param.input, param.output);
        return;
    }

    if (!w.q_proj || !w.o_proj || w.q_proj.ndim() != 2 || w.o_proj.ndim() != 2) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] missing or invalid q_proj/o_proj; "
            "treating Eagle-3 attention as pass-through.");
        core::Copy(param.input, param.output);
        return;
    }

    const int q_out_dim = w.q_out;
    if (q_out_dim <= 0 || w.o_proj.shape(0) != q_in_dim || w.o_proj.shape(1) != q_out_dim) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] Eagle-3 q/o geometry mismatch "
            "(q_out=%d, o_proj=[%d,%d], q_in=%d); treating as pass-through.",
            q_out_dim,
            w.o_proj.shape(0),
            w.o_proj.shape(1),
            q_in_dim);
        core::Copy(param.input, param.output);
        return;
    }

    if (w.q_proj.dtype() != dtype || w.o_proj.dtype() != dtype) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] dtype mismatch between input (%s) "
            "and Eagle3 q/o weights (q=%s, o=%s); treating as pass-through.",
            to_string(dtype),
            to_string(w.q_proj.dtype()),
            to_string(w.o_proj.dtype()));
        core::Copy(param.input, param.output);
        return;
    }

    // Ensure output buffer has the expected [B, q_in] layout and dtype.
    if (!param.output || param.output.ndim() != 2 || param.output.shape(0) != batch_size
        || param.output.shape(1) != q_in_dim || param.output.dtype() != dtype
        || param.output.device().type != param.input.device().type) {
        param.output = Tensor{{batch_size, q_in_dim}, dtype, param.input.device()};
    }

    // Temporary buffer for Q = X @ Wq^T of shape [B, q_out_dim].
    Tensor q{{batch_size, q_out_dim}, dtype, param.input.device()};

    auto do_gemm = [&](auto tag) {
        using T = decltype(tag);
        const T* x_ptr  = param.input.data<T>();
        const T* wq_ptr = w.q_proj.data<T>();
        const T* wo_ptr = w.o_proj.data<T>();
        T*       q_ptr  = q.data<T>();
        T*       y_ptr  = param.output.data<T>();

        if constexpr (std::is_same_v<T, half_t>) {
            launch_eagle3_matmul_rowmajor_half(x_ptr, wq_ptr, q_ptr, batch_size, q_in_dim, q_out_dim, stream_);
            sync_check_cuda_error();
            launch_eagle3_matmul_rowmajor_half(q_ptr, wo_ptr, y_ptr, batch_size, q_out_dim, q_in_dim, stream_);
            sync_check_cuda_error();
        }
#if ENABLE_BF16
        else if constexpr (std::is_same_v<T, bfloat16_t>) {
            launch_eagle3_matmul_rowmajor_bf16(x_ptr, wq_ptr, q_ptr, batch_size, q_in_dim, q_out_dim, stream_);
            sync_check_cuda_error();
            launch_eagle3_matmul_rowmajor_bf16(q_ptr, wo_ptr, y_ptr, batch_size, q_out_dim, q_in_dim, stream_);
            sync_check_cuda_error();
        }
#endif
#if ENABLE_FP32
        else if constexpr (std::is_same_v<T, float>) {
            launch_eagle3_matmul_rowmajor_float(x_ptr, wq_ptr, q_ptr, batch_size, q_in_dim, q_out_dim, stream_);
            sync_check_cuda_error();
            launch_eagle3_matmul_rowmajor_float(q_ptr, wo_ptr, y_ptr, batch_size, q_out_dim, q_in_dim, stream_);
            sync_check_cuda_error();
        }
#endif
        else {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] unsupported Eagle3Attention dtype=%s; "
                "treating as pass-through.",
                to_string(dtype));
            core::Copy(param.input, param.output);
        }
    };

    if (dtype == kFloat16) {
        do_gemm(half_t{});
    }
#if ENABLE_BF16
    else if (dtype == kBfloat16) {
        do_gemm(bfloat16_t{});
    }
#endif
#if ENABLE_FP32
    else if (dtype == kFloat32) {
        do_gemm(float{});
    }
    else
#endif
    {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] unsupported Eagle3Attention dtype=%s; "
            "treating as pass-through.",
            to_string(dtype));
        core::Copy(param.input, param.output);
    }
}

}  // namespace turbomind
