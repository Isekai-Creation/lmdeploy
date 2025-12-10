#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include <type_traits>

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

    if (!param.input || !param.weights || !param.weights->is_initialized) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid Eagle3AttentionParam; "
            "treating Eagle-3 attention as pass-through for this step.");
        if (param.input) {
            param.output = param.input;
        }
        return;
    }

    const auto& w = *param.weights;

    const int token_num = param.input.shape(0);
    const int q_in_dim  = param.input.shape(1);
    const auto dtype    = param.input.dtype();

    const int num_q_heads  = w.num_q_heads;
    const int num_kv_heads = w.num_kv_heads;
    const int head_dim     = w.head_dim;

    const int q_out_dim  = w.q_out;
    const int kv_out_dim = w.kv_out;

    if (token_num <= 0 || q_in_dim <= 0 || q_out_dim <= 0 || kv_out_dim <= 0 || num_q_heads <= 0
        || num_kv_heads <= 0 || head_dim <= 0) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid geometry (token=%d, q_in=%d, q_out=%d, kv_out=%d, "
            "q_heads=%d, kv_heads=%d, head_dim=%d); treating as pass-through.",
            token_num,
            q_in_dim,
            q_out_dim,
            kv_out_dim,
            num_q_heads,
            num_kv_heads,
            head_dim);
        param.output = param.input;
        return;
    }

    // Ensure output buffer matches the expected shape from the caller.
    if (!param.output || param.output.ndim() != 2 || param.output.shape(0) != token_num
        || param.output.shape(1) <= 0 || param.output.dtype() != dtype
        || param.output.device().type != param.input.device().type) {
        param.output = Tensor{{token_num, q_in_dim}, dtype, param.input.device()};
    }

    // Fallback when q_in does not line up with weights.
    if (q_in_dim != w.q_in || w.q_proj.dtype() != dtype || w.k_proj.dtype() != dtype || w.v_proj.dtype() != dtype) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] q_in/weight dtype mismatch (q_in=%d vs weight=%d, dtype=%s/%s); "
            "pass-through.",
            q_in_dim,
            w.q_in,
            to_string(dtype),
            to_string(w.q_proj.dtype()));
        param.output = param.input;
        return;
    }

    const int q_len = param.q_len > 0 ? param.q_len : 1;
    const int slot_count =
        (q_len > 0 && token_num % q_len == 0) ? (token_num / q_len) : (param.batch_size > 0 ? param.batch_size : 1);
    const int batch_size = slot_count;
    const int kv_len     = param.kv_len > 0 ? param.kv_len : q_len;
    const int group_size = std::max(1, num_q_heads / num_kv_heads);
    const auto* runtime_offsets = param.runtime_offsets;
    const auto* tree_offsets    = param.tree_offsets;
    auto offsets_invalid = [&](const Tensor* t) -> bool {
        return t && (!(*t) || t->dtype() != kInt32 || t->ndim() != 1 || t->shape(0) < slot_count + 1);
    };
    if (offsets_invalid(runtime_offsets)) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] runtime_offsets invalid (len=%d, expect=%d); ignoring offsets.",
            runtime_offsets ? runtime_offsets->shape(0) : -1,
            slot_count + 1);
        runtime_offsets = nullptr;
    }
    if (offsets_invalid(tree_offsets)) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] tree_offsets invalid (len=%d, expect=%d); ignoring offsets.",
            tree_offsets ? tree_offsets->shape(0) : -1,
            slot_count + 1);
        tree_offsets = nullptr;
    }

    // Temporary buffers for Q/K/V and context.
    Tensor q{{token_num, q_out_dim}, dtype, param.input.device()};
    Tensor k{{token_num, kv_out_dim}, dtype, param.input.device()};
    Tensor v{{token_num, kv_out_dim}, dtype, param.input.device()};
    Tensor ctx{{token_num, q_out_dim}, dtype, param.input.device()};

    auto do_forward = [&](auto tag) {
        using T = decltype(tag);
        const T* x_ptr  = param.input.data<T>();
        const T* wq_ptr = w.q_proj.data<T>();
        const T* wk_ptr = w.k_proj.data<T>();
        const T* wv_ptr = w.v_proj.data<T>();
        auto matmul = [&](const T* A, const T* B, T* C, int M, int K, int N) {
            ft::launch_eagle3_matmul_rowmajor_dispatch(A, B, C, M, K, N, stream_);
        };

        // Q/K/V projections.
        matmul(x_ptr, wq_ptr, q.data<T>(), token_num, q_in_dim, q_out_dim);
        matmul(x_ptr, wk_ptr, k.data<T>(), token_num, q_in_dim, kv_out_dim);
        matmul(x_ptr, wv_ptr, v.data<T>(), token_num, q_in_dim, kv_out_dim);
        sync_check_cuda_error();

        // Optional RoPE on Q/K.
        ft::launch_apply_rope_kernel<T>(q.data<T>(),
                                        k.data<T>(),
                                        token_num,
                                        num_q_heads,
                                        num_kv_heads,
                                        head_dim,
                                        q_len,
                                        param.past_kv_len,
                                        param.position_ids,
                                        param.rope_base,
                                        param.rope_scale,
                                        stream_);
        sync_check_cuda_error();

        // SDPA (naive) over kv_len tokens.
        ft::launch_sdpa_kernel<T>(q.data<T>(),
                                  k.data<T>(),
                                  v.data<T>(),
                                  ctx.data<T>(),
                                  token_num,
                                  batch_size,
                                  q_len,
                                  kv_len,
                                  num_q_heads,
                                  num_kv_heads,
                                  head_dim,
                                  param.past_kv_len,
                                  param.position_ids,
                                  param.packed_mask,
                                  param.packed_mask_stride,
                                  runtime_offsets,
                                  tree_offsets,
                                  param.kv_lens_runtime,
                                  param.successor_offsets,
                                  param.successor_counts,
                                  stream_);
        sync_check_cuda_error();

        // Optional debug capture of QKV/attn_out.
        if (param.debug_qkv) {
            *param.debug_qkv = Tensor{{token_num, q_out_dim + 2 * kv_out_dim}, q.dtype(), q.device()};
            // Layout: [Q | K | V] concatenated along the last dim.
            Tensor& dbg = *param.debug_qkv;
            check_cuda_error(cudaMemcpyAsync(
                dbg.raw_data(), q.raw_data(), q.byte_size(), cudaMemcpyDeviceToDevice, stream_));
            check_cuda_error(cudaMemcpyAsync(static_cast<char*>(dbg.raw_data()) + q.byte_size(),
                                             k.raw_data(),
                                             k.byte_size(),
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
            check_cuda_error(cudaMemcpyAsync(static_cast<char*>(dbg.raw_data()) + q.byte_size() + k.byte_size(),
                                             v.raw_data(),
                                             v.byte_size(),
                                             cudaMemcpyDeviceToDevice,
                                             stream_));
        }
        if (param.debug_attn_out) {
            *param.debug_attn_out = ctx;
        }

        // Final Wo projection. Prefer fused LLaMA-style Wo when present;
        // fall back to native o_proj otherwise.
        const Tensor* wo = param.attn_weights ? &param.attn_weights->output.weight : nullptr;
        if (!wo || !(*wo)) {
            wo = &w.o_proj;
        }

        if (!wo || !(*wo) || wo->ndim() != 2) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] missing Wo; returning context without projection.");
            param.output = ctx;
            return;
        }

        const int wo_in  = wo->shape(0);
        const int wo_out = wo->shape(1);
        if (wo_in != q_out_dim) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] Wo shape mismatch (Wo=[%d,%d], q_out=%d); "
                "returning context.",
                wo_in,
                wo_out,
                q_out_dim);
            param.output = ctx;
            return;
        }
        if (wo->dtype() != ctx.dtype()) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] Wo dtype mismatch (Wo=%s, ctx=%s); returning context.",
                to_string(wo->dtype()),
                to_string(ctx.dtype()));
            param.output = ctx;
            return;
        }

        if (!param.output || param.output.ndim() != 2 || param.output.shape(0) != token_num
            || param.output.shape(1) != wo_out || param.output.dtype() != ctx.dtype()
            || param.output.device().type != ctx.device().type) {
            param.output = Tensor{{token_num, wo_out}, ctx.dtype(), ctx.device()};
        }

        ft::launch_eagle3_matmul_rowmajor_dispatch(ctx.data<T>(), wo->data<T>(), param.output.data<T>(), token_num, wo_in, wo_out, stream_);
        sync_check_cuda_error();
    };

    if (dtype == kFloat16) {
        do_forward(half_t{});
    }
#if ENABLE_BF16
    else if (dtype == kBfloat16) {
        do_forward(bfloat16_t{});
    }
#endif
#if ENABLE_FP32
    else if (dtype == kFloat32) {
        do_forward(float{});
    }
    else
#endif
    {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] unsupported Eagle3Attention dtype=%s; treating as pass-through.",
            to_string(dtype));
        param.output = param.input;
    }
}

}  // namespace turbomind
