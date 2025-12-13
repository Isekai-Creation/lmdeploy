#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/kernels/attention/attention_params.h"

#include <type_traits>
#include <cstdlib>
#include <cstring>

namespace turbomind {

Eagle3AttentionLayer::Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream):
    stream_{stream},
    device_prop_{prop}
{
    (void)device_prop_;
}


void Eagle3AttentionLayer::Forward(Eagle3AttentionParam& param)
{
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

    static bool logged_active_path = false;

    auto is_fmha_enabled = []() {
        static int cached = -1;
        if (cached >= 0) {
            return cached == 1;
        }
        const char* env = std::getenv("TM_ENABLE_EAGLE3_FMHA");
        cached          = (env && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
        return cached == 1;
    };

    const bool enable_fmha = is_fmha_enabled();

    auto do_forward = [&](auto tag) {
        using T = decltype(tag);
        const T* x_ptr  = param.input.data<T>();
        const T* wq_ptr = w.q_proj.data<T>();
        const T* wk_ptr = w.k_proj.data<T>();
        const T* wv_ptr = w.v_proj.data<T>();
        auto matmul = [&](const T* A, const T* B, T* C, int M, int K, int N) {
            ft::launch_eagle3_matmul_rowmajor_dispatch(A, B, C, M, K, N, stream_);
        };

        if (!logged_active_path) {
            TM_LOG_INFO(
                "[EAGLE3][Attention][stage] begin native Eagle3 attention "
                "(dtype=%s, token_num=%d, q_in=%d, q_out=%d, kv_out=%d, q_heads=%d, kv_heads=%d, head_dim=%d, "
                "q_len=%d, kv_len=%d, group_size=%d)",
                to_string(dtype),
                token_num,
                q_in_dim,
                q_out_dim,
                kv_out_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                q_len,
                kv_len,
                group_size);
            logged_active_path = true;
        }

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

        // Optional FMHA wiring: when enabled via TM_ENABLE_EAGLE3_FMHA, pack
        // Q/K/V into a unified [token_num, q_heads + 2 * kv_heads, head_dim]
        // layout and construct an AttentionParams<T> instance. For now, this
        // is a wiring layer only; we still compute attention via the existing
        // Eagle3 SDPA kernel so behaviour remains unchanged.
        if (enable_fmha) {
            const int local_q_heads      = num_q_heads;
            const int local_kv_heads     = num_kv_heads;
            const int local_qkv_head_num = local_q_heads + 2 * local_kv_heads;

            Tensor qkv{{token_num, local_qkv_head_num, head_dim}, q.dtype(), q.device()};

            const size_t elem_bytes    = sizeof(T);
            const size_t q_row_bytes   = static_cast<size_t>(q_out_dim) * elem_bytes;
            const size_t kv_row_bytes  = static_cast<size_t>(kv_out_dim) * elem_bytes;
            const size_t qkv_row_bytes = static_cast<size_t>(local_qkv_head_num * head_dim) * elem_bytes;

            char*       qkv_base = static_cast<char*>(qkv.raw_data());
            const char* q_base   = static_cast<const char*>(q.raw_data());
            const char* k_base   = static_cast<const char*>(k.raw_data());
            const char* v_base   = static_cast<const char*>(v.raw_data());

            for (int t = 0; t < token_num; ++t) {
                char*       dst_row = qkv_base + static_cast<size_t>(t) * qkv_row_bytes;
                const char* src_q   = q_base + static_cast<size_t>(t) * q_row_bytes;
                const char* src_k   = k_base + static_cast<size_t>(t) * kv_row_bytes;
                const char* src_v   = v_base + static_cast<size_t>(t) * kv_row_bytes;

                check_cuda_error(
                    cudaMemcpyAsync(dst_row, src_q, q_row_bytes, cudaMemcpyDeviceToDevice, stream_));
                check_cuda_error(cudaMemcpyAsync(
                    dst_row + q_row_bytes, src_k, kv_row_bytes, cudaMemcpyDeviceToDevice, stream_));
                check_cuda_error(cudaMemcpyAsync(dst_row + q_row_bytes + kv_row_bytes,
                                                 src_v,
                                                 kv_row_bytes,
                                                 cudaMemcpyDeviceToDevice,
                                                 stream_));
            }

            AttentionParams<T> fmha{};
            fmha.out     = ctx.data<T>();
            fmha.q       = qkv.data<T>();
            fmha.k       = fmha.q + local_q_heads * head_dim;
            fmha.v       = fmha.k + local_kv_heads * head_dim;
            fmha.stride  = static_cast<int64_t>(local_qkv_head_num * head_dim);
            fmha.token_num     = token_num;
            fmha.batch_size    = batch_size;
            fmha.max_q_len     = q_len;
            fmha.max_k_len     = kv_len;
            fmha.num_heads     = local_q_heads;
            fmha.num_kv_heads  = local_kv_heads;
            fmha.size_per_head = head_dim;

            if (param.packed_mask && param.packed_mask_stride > 0) {
                fmha.spec_decoding_packed_mask        = param.packed_mask->data<int32_t>();
                fmha.spec_decoding_packed_mask_stride = param.packed_mask_stride;
            }

            fmha.spec_runtime_offsets    = runtime_offsets ? runtime_offsets->data<int32_t>() : nullptr;
            fmha.spec_tree_offsets       = tree_offsets ? tree_offsets->data<int32_t>() : nullptr;
            fmha.spec_successor_offsets  = param.successor_offsets ? param.successor_offsets->data<int32_t>() : nullptr;
            fmha.spec_successor_counts   = param.successor_counts ? param.successor_counts->data<int32_t>() : nullptr;
            fmha.spec_tree_batch_size    = batch_size;

            (void)fmha;
        }

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
            // Fall back to native Eagle3 o_proj when LLaMA-style Wo is
            // not provided. o_proj is stored in [draft_hidden, q_out]
            // row-major layout, so we treat its rows as output_dim and
            // its columns as input_dim.
            wo = &w.o_proj;
        }

        if (!wo || !(*wo) || wo->ndim() != 2) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] missing Wo; returning context without projection.");
            param.output = ctx;
            return;
        }

        // Interpret Wo in standard row-major [output_dim, input_dim]
        // form so that MatmulRowMajorKernel sees B as [N,K] with
        // K = q_out_dim and N = draft_hidden (or base hidden) width.
        const int wo_out = wo->shape(0);  // output dim (e.g. draft_hidden)
        const int wo_in  = wo->shape(1);  // input dim  (q_out_dim)

        if (wo_in != q_out_dim) {
            TM_LOG_WARNING(
                "[EAGLE3][Attention][fallback] Wo shape mismatch (Wo=[%d,%d], expected input=%d); "
                "returning context without projection.",
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

        // C[M,N] = A[M,K] @ B[N,K]^T with:
        //   A = ctx         [token_num, q_out_dim]
        //   B = Wo          [wo_out, q_out_dim]
        //   C = output      [token_num, wo_out]
        ft::launch_eagle3_matmul_rowmajor_dispatch(
            ctx.data<T>(), wo->data<T>(), param.output.data<T>(), token_num, wo_in, wo_out, stream_);
        TM_LOG_INFO(
            "[EAGLE3][Attention][stage] completed native Eagle3 attention "
            "(token_num=%d, q_out=%d, wo_out=%d)",
            token_num,
            q_out_dim,
            wo_out);
        sync_check_cuda_error();
    };

    // Prefer native BF16 path when available, then FP16, then FP32.
    if (dtype == kBfloat16) {
        do_forward(bfloat16_t{});
    }
    else if (dtype == kFloat16) {
        do_forward(half_t{});
    }
#if ENABLE_FP32
    else if (dtype == kFloat32) {
        do_forward(float{});
    }
#endif
    else {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] unsupported Eagle3Attention dtype=%s "
            "(code=%d, kBfloat16=%d, kFloat16=%d, kFloat32=%d); treating as pass-through.",
            to_string(dtype),
            static_cast<int>(dtype),
            static_cast<int>(kBfloat16),
            static_cast<int>(kFloat16),
            static_cast<int>(kFloat32));
        param.output = param.input;
    }
}

}  // namespace turbomind
