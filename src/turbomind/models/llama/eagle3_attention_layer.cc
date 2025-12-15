#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/kernels/attention/attention_params.h"

#include <type_traits>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

namespace turbomind {

namespace ft {
// Helper function to get the address of the device-side tile statistics.
cudaError_t GetEagle3FmhaTileStats(void** dev_ptr);
}  // namespace ft


namespace {

enum Eagle3FmhaTileStatIndex : int {
    kEagle3FmhaTilesTotal      = 0,
    kEagle3FmhaTilesSpanEmpty  = 1,
    kEagle3FmhaTilesMaskEmpty  = 2,
    kEagle3FmhaTilesExecuted   = 3,
    kEagle3FmhaTilesCount      = 4,
};

inline bool is_env_enabled(const char* name)
{
    const char* v = std::getenv(name);
    if (!v || !v[0]) {
        return false;
    }
    return !(v[0] == '0' && v[1] == '\0');
}

}  // namespace

Eagle3AttentionLayer::Eagle3AttentionLayer(const cudaDeviceProp* prop, cudaStream_t stream):
    stream_{stream},
    device_prop_{prop}
{
    (void)device_prop_;
}


void Eagle3AttentionLayer::Forward(Eagle3AttentionParam& param)
{
    if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
        if (fmha_in_flight_) {
            TM_LOG_WARNING("[EAGLE3][Attention] Concurrent Forward() detected on same layer instance");
            std::abort();
        }
        fmha_in_flight_ = true;
    }

    if (!param.input || !param.weights || !param.weights->is_initialized) {
        TM_LOG_WARNING(
            "[EAGLE3][Attention][fallback] invalid Eagle3AttentionParam; "
            "treating Eagle-3 attention as pass-through for this step.");
        if (param.input) {
            param.output = param.input;
        }
        if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
            fmha_in_flight_ = false;
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
        if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
            fmha_in_flight_ = false;
        }
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
        if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
            fmha_in_flight_ = false;
        }
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
        cached = is_env_enabled("TM_ENABLE_EAGLE3_FMHA") ? 1 : 0;
        return cached == 1;
    };

    auto is_fmha_ab_debug_enabled = []() {
        static int cached = -1;
        if (cached >= 0) {
            return cached == 1;
        }
        cached = is_env_enabled("TM_EAGLE3_FMHA_AB") ? 1 : 0;
        return cached == 1;
    };

    const bool enable_fmha       = is_fmha_enabled();
    const bool enable_fmha_ab    = is_fmha_ab_debug_enabled();
    const bool perf_mode_enabled = turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_PERF_MODE");

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
        // layout, build per-token KV spans on GPU, and route attention
        // through a tree-aware FMHA-style kernel that consumes those spans.
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

            // Build per-token KV spans on device using the same runtime/tree/
            // successor/kv_lens semantics as the current SDPA kernels. This
            // gives FMHA a cheap way to know which KV window each token
            // should attend to.
            core::Tensor kv_start_t{{token_num}, kInt32, param.input.device()};
            core::Tensor kv_len_t{{token_num}, kInt32, param.input.device()};

            const int32_t* runtime_ptr =
                (runtime_offsets && runtime_offsets->dtype() == kInt32) ? runtime_offsets->data<int32_t>() : nullptr;
            const int32_t* tree_ptr =
                (tree_offsets && tree_offsets->dtype() == kInt32) ? tree_offsets->data<int32_t>() : nullptr;
            const int32_t* kv_runtime_ptr = (param.kv_lens_runtime && param.kv_lens_runtime->dtype() == kInt32)
                                                ? param.kv_lens_runtime->data<int32_t>()
                                                : nullptr;
            const int32_t* succ_off_ptr = (param.successor_offsets && param.successor_offsets->dtype() == kInt32)
                                              ? param.successor_offsets->data<int32_t>()
                                              : nullptr;
            const int32_t* succ_cnt_ptr = (param.successor_counts && param.successor_counts->dtype() == kInt32)
                                              ? param.successor_counts->data<int32_t>()
                                              : nullptr;

            ft::launch_build_eagle3_kv_spans(token_num,
                                             batch_size,
                                             q_len,
                                             kv_len,
                                             runtime_ptr,
                                             tree_ptr,
                                             kv_runtime_ptr,
                                             succ_off_ptr,
                                             succ_cnt_ptr,
                                             kv_start_t.data<int32_t>(),
                                             kv_len_t.data<int32_t>(),
                                             stream_);

            // Optional KV-span integrity check for debugging: when
            // LMDEPLOY_EAGLE_SPAN_DEBUG is enabled, verify on host that
            // per-token spans lie within [0, kv_len] for this step and
            // (when packed masks are in use) within the mask capacity
            // implied by packed_mask_stride. Treat any violation as a
            // hard failure under this debug gate so that spec runs never
            // silently proceed with invalid spans.
            if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_SPAN_DEBUG")) {
                std::vector<int32_t> h_kv_start(token_num, 0);
                std::vector<int32_t> h_kv_len(token_num, 0);

                check_cuda_error(cudaMemcpyAsync(h_kv_start.data(),
                                                 kv_start_t.data<int32_t>(),
                                                 static_cast<size_t>(token_num) * sizeof(int32_t),
                                                 cudaMemcpyDeviceToHost,
                                                 stream_));
                check_cuda_error(cudaMemcpyAsync(h_kv_len.data(),
                                                 kv_len_t.data<int32_t>(),
                                                 static_cast<size_t>(token_num) * sizeof(int32_t),
                                                 cudaMemcpyDeviceToHost,
                                                 stream_));
                check_cuda_error(cudaStreamSynchronize(stream_));

                const int packed_stride = param.packed_mask_stride;
                const int packed_cap    = packed_stride > 0 ? packed_stride * 32 : kv_len;

                for (int idx = 0; idx < token_num; ++idx) {
                    const int start = h_kv_start[idx];
                    const int len   = h_kv_len[idx];
                    const int end   = start + len;
                    if (start < 0 || len < 0 || end > kv_len || end > packed_cap) {
                        TM_LOG_WARNING(
                            "[EAGLE3][Attention][kv-span] invalid span at token=%d batch=%d "
                            "(start=%d len=%d end=%d kv_len=%d packed_stride=%d packed_cap=%d)",
                            idx,
                            batch_size,
                            start,
                            len,
                            end,
                            kv_len,
                            packed_stride,
                            packed_cap);
                        // Under span debug, fail fast so that any
                        // invalid span is fixed before further perf
                        // tuning or long benchmarks.
                        std::abort();
                    }
                }

                // Optional packed-mask coverage invariant: when a packed
                // mask is present and the KV span length is non-zero, we
                // expect at least one allowed position in the span. If
                // the mask is all-zero over [start, start+len), we are
                // doing useless work or have inconsistent metadata.
                if (param.packed_mask && param.packed_mask_stride > 0) {
                    const int32_t* d_mask = param.packed_mask->data<int32_t>();
                    const int      stride = param.packed_mask_stride;
                    std::vector<int32_t> h_mask(static_cast<size_t>(token_num) * stride, 0);

                    check_cuda_error(cudaMemcpyAsync(h_mask.data(),
                                                     d_mask,
                                                     static_cast<size_t>(token_num * stride) * sizeof(int32_t),
                                                     cudaMemcpyDeviceToHost,
                                                     stream_));
                    check_cuda_error(cudaStreamSynchronize(stream_));

                    for (int idx = 0; idx < token_num; ++idx) {
                        const int start = h_kv_start[idx];
                        const int len   = h_kv_len[idx];
                        if (len <= 0) {
                            continue;
                        }
                        const int end = start + len;
                        if (start < 0 || end > kv_len) {
                            continue;
                        }

                        const int word_start = start / 32;
                        const int word_end   = (end + 31) / 32;
                        const int row_off    = idx * stride;

                        bool any = false;
                        for (int w = word_start; w < word_end && !any; ++w) {
                            const int32_t mask_val = h_mask[row_off + w];
                            if (mask_val == 0) {
                                continue;
                            }
                            const int bit_lo = (w == word_start) ? (start & 31) : 0;
                            const int bit_hi = (w == word_end - 1) ? ((end - 1) & 31) : 31;
                            const int bit_count = bit_hi - bit_lo + 1;
                            const uint32_t mask = (bit_count >= 32)
                                                      ? 0xFFFFFFFFu
                                                      : ((static_cast<uint32_t>(1u) << bit_count) - 1u) << bit_lo;
                            if ((static_cast<uint32_t>(mask_val) & mask) != 0u) {
                                any = true;
                            }
                        }

                        if (!any) {
                            TM_LOG_WARNING(
                                "[EAGLE3][Attention][kv-span] span has no allowed positions under packed mask "
                                "(token=%d start=%d len=%d kv_len=%d packed_stride=%d)",
                                idx,
                                start,
                                len,
                                kv_len,
                                stride);
                            std::abort();
                        }
                    }
                }
            }

            // Populate a unified AttentionParams view for this Eagle3 FMHA
            // launch so that we stay aligned with the rest of the TurboMind
            // attention stack. Today this is only used for debug /
            // introspection – the actual kernels below take raw pointers –
            // but it keeps all of the tree / span / mask wiring in one
            // struct that can be surfaced to generic helpers later.
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
            fmha.spec_successor_offsets  =
                param.successor_offsets ? param.successor_offsets->data<int32_t>() : nullptr;
            fmha.spec_successor_counts = param.successor_counts ? param.successor_counts->data<int32_t>() : nullptr;
            fmha.spec_tree_batch_size  = batch_size;

            fmha.spec_kv_start_per_token = kv_start_t.data<int32_t>();
            fmha.spec_kv_len_per_token   = kv_len_t.data<int32_t>();

            (void)fmha;

            // Tree-aware FMHA-style SDPA that trusts the GPU span builder
            // for runtime/tree/successor clamping. This is gated by
            // TM_ENABLE_EAGLE3_FMHA so we can A/B it against the original
            // SDPA path.
            //
            // For the tiled multi-CTA kernel we choose a small number of
            // KV tiles per (token, head) based on kv_len and a desired
            // kv_tile size, and reuse layer-owned scratch buffers for
            // partial m/l/o across steps. Additional knobs allow tuning
            // block size and heuristic grouping parameters.
            int kv_tile = 128;
            if (const char* env = std::getenv("TM_EAGLE3_FMHA_KV_TILE")) {
                int v = std::atoi(env);
                if (v > 0) {
                    kv_tile = v;
                }
            }
            else {
                // Heuristic default for sm120: use smaller KV tiles for
                // large-context batch runs so that multi-CTA work is
                // better balanced, and a slightly larger tile for small
                // batches to reduce scheduling overhead.
                if (batch_size >= 4 && kv_len >= 8192) {
                    kv_tile = 64;
                }
                else {
                    kv_tile = 128;
                }
            }
            int max_tiles_cap = 8;
            if (const char* env = std::getenv("TM_EAGLE3_FMHA_MAX_TILES")) {
                int v = std::atoi(env);
                if (v > 0) {
                    max_tiles_cap = v;
                }
            }

            int fmha_block = 256;
            if (const char* env = std::getenv("TM_EAGLE3_FMHA_BLOCK")) {
                int v = std::atoi(env);
                if (v >= 32 && v <= 256 && (v % 32) == 0) {
                    fmha_block = v;
                }
            }

            int heads_per_cta = 1;
            if (const char* env = std::getenv("TM_EAGLE3_FMHA_HEADS_PER_CTA")) {
                int v = std::atoi(env);
                if (v > 0) {
                    heads_per_cta = v;
                }
            }

            int qtokens_per_cta = 0;
            if (const char* env = std::getenv("TM_EAGLE3_FMHA_QTOKENS_PER_CTA")) {
                int v = std::atoi(env);
                if (v > 0) {
                    qtokens_per_cta = v;
                }
            }

            const int max_tiles_est = kv_tile > 0 ? (kv_len + kv_tile - 1) / kv_tile : 1;
            int       fmha_tiles    = std::max(1, std::min(max_tiles_est, max_tiles_cap));

            if (heads_per_cta > 1) {
                fmha_tiles = std::max(1, fmha_tiles / heads_per_cta);
            }
            if (qtokens_per_cta > 0 && q_len > 0 && batch_size > 0) {
                const int q_tokens       = batch_size * q_len;
                const int tiles_for_qtok = std::max(1, q_tokens / qtokens_per_cta);
                fmha_tiles               = std::max(1, std::min(fmha_tiles, tiles_for_qtok));
            }

            const int total_fmha = token_num * num_q_heads;
            static bool logged_fmha_cfg = false;
            if (!logged_fmha_cfg && perf_mode_enabled) {
                TM_LOG_INFO(
                    "[EAGLE3][FMHA_CFG] batch=%d kv_len=%d kv_tile=%d max_tiles_cap=%d fmha_tiles=%d "
                    "block=%d heads_per_cta=%d qtokens_per_cta=%d",
                    batch_size,
                    kv_len,
                    kv_tile,
                    max_tiles_cap,
                    fmha_tiles,
                    fmha_block,
                    heads_per_cta,
                    qtokens_per_cta);
                logged_fmha_cfg = true;
            }
            if (total_fmha > 0) {
                const size_t partial_count   = static_cast<size_t>(total_fmha) * fmha_tiles;
                const size_t partial_m_bytes = partial_count * sizeof(float);
                const size_t partial_l_bytes = partial_count * sizeof(float);
                const size_t partial_o_bytes = partial_count * static_cast<size_t>(head_dim) * sizeof(float);

                if (!fmha_partial_m_ || fmha_partial_m_.byte_size() < partial_m_bytes) {
                    fmha_partial_m_ = core::Tensor{{static_cast<int>(partial_count)}, kFloat32, param.input.device()};
                }
                if (!fmha_partial_l_ || fmha_partial_l_.byte_size() < partial_l_bytes) {
                    fmha_partial_l_ = core::Tensor{{static_cast<int>(partial_count)}, kFloat32, param.input.device()};
                }
                if (!fmha_partial_o_ || fmha_partial_o_.byte_size() < partial_o_bytes) {
                    fmha_partial_o_ = core::Tensor{
                        {static_cast<int>(partial_count * static_cast<size_t>(head_dim))}, kFloat32, param.input.device()};
                }
            }

            // Optional FMHA tile statistics: when TM_EAGLE3_FMHA_TILE_STATS is
            // enabled, reset device-side counters once and emit a single
            // summary line per run. This is explicitly opt-in so PERF_MODE
            // remains free of D2H copies by default.
            const bool fmha_tile_stats = is_env_enabled("TM_EAGLE3_FMHA_TILE_STATS");
            if (fmha_tile_stats) {
                static bool tile_stats_initialized = false;
                if (!tile_stats_initialized) {
                    tile_stats_initialized = true;
                    void*  dev_ptr  = nullptr;
                    size_t dev_size = 0;
                    check_cuda_error(ft::GetEagle3FmhaTileStats(&dev_ptr));
                    dev_size = static_cast<size_t>(4) * sizeof(unsigned long long);
                    check_cuda_error(cudaMemsetAsync(dev_ptr, 0, dev_size, stream_));
                    check_cuda_error(cudaStreamSynchronize(stream_));
                }
            }

            if constexpr (std::is_same_v<T, half_t>) {
                ft::launch_eagle3_fmha_kernel_fp16(q.data<T>(),
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
                                                   kv_start_t.data<int32_t>(),
                                                   kv_len_t.data<int32_t>(),
                                                   fmha_partial_o_.data<float>(),
                                                   fmha_partial_m_.data<float>(),
                                                   fmha_partial_l_.data<float>(),
                                                   fmha_tiles,
                                                   stream_);
            }
#if ENABLE_BF16
            else if constexpr (std::is_same_v<T, bfloat16_t>) {
                ft::launch_eagle3_fmha_kernel_bf16(q.data<T>(),
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
                                                   kv_start_t.data<int32_t>(),
                                                   kv_len_t.data<int32_t>(),
                                                   fmha_partial_o_.data<float>(),
                                                   fmha_partial_m_.data<float>(),
                                                   fmha_partial_l_.data<float>(),
                                                   fmha_tiles,
                                                   stream_);
            }
#endif
#if ENABLE_FP32
            else if constexpr (std::is_same_v<T, float>) {
                ft::launch_eagle3_fmha_kernel_fp32(q.data<T>(),
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
                                                   kv_start_t.data<int32_t>(),
                                                   kv_len_t.data<int32_t>(),
                                                   fmha_partial_o_.data<float>(),
                                                   fmha_partial_m_.data<float>(),
                                                   fmha_partial_l_.data<float>(),
                                                   fmha_tiles,
                                                   stream_);
            }
#endif
            sync_check_cuda_error();

            if (fmha_tile_stats) {
                void*  dev_ptr  = nullptr;
                size_t dev_size = 0;
                check_cuda_error(ft::GetEagle3FmhaTileStats(&dev_ptr));
                dev_size = static_cast<size_t>(kEagle3FmhaTilesCount) * sizeof(unsigned long long);
                unsigned long long h_stats[kEagle3FmhaTilesCount] = {};
                check_cuda_error(cudaMemcpyAsync(
                    h_stats, dev_ptr, dev_size, cudaMemcpyDeviceToHost, stream_));
                check_cuda_error(cudaStreamSynchronize(stream_));
                TM_LOG_INFO(
                    "[EAGLE3][FMHA_TILE_STATS] total=%llu span_empty=%llu mask_empty=%llu executed=%llu",
                    static_cast<unsigned long long>(h_stats[kEagle3FmhaTilesTotal]),
                    static_cast<unsigned long long>(h_stats[kEagle3FmhaTilesSpanEmpty]),
                    static_cast<unsigned long long>(h_stats[kEagle3FmhaTilesMaskEmpty]),
                    static_cast<unsigned long long>(h_stats[kEagle3FmhaTilesExecuted]));
            }

            // Optional A/B correctness harness: when TM_EAGLE3_FMHA_AB is
            // enabled and we are in a reasonably small regime, run the
            // original SDPA path in parallel and log the maximum absolute
            // difference between SDPA and FMHA contexts. This is for
            // debugging only and is not meant for production perf runs.
            if (enable_fmha_ab && token_num <= 32 && kv_len <= 512) {
                Tensor ctx_ref{{token_num, q_out_dim}, dtype, param.input.device()};
                ft::launch_sdpa_kernel<T>(q.data<T>(),
                                          k.data<T>(),
                                          v.data<T>(),
                                          ctx_ref.data<T>(),
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

                const size_t elem_count = static_cast<size_t>(token_num) * q_out_dim;
                std::vector<float> h_fmha(elem_count);
                std::vector<float> h_ref(elem_count);

                if constexpr (std::is_same_v<T, half_t>) {
                    std::vector<half_t> tmp_fmha(elem_count);
                    std::vector<half_t> tmp_ref(elem_count);
                    check_cuda_error(cudaMemcpyAsync(
                        tmp_fmha.data(), ctx.data<half_t>(), elem_count * sizeof(half_t), cudaMemcpyDeviceToHost, stream_));
                    check_cuda_error(cudaMemcpyAsync(
                        tmp_ref.data(), ctx_ref.data<half_t>(), elem_count * sizeof(half_t), cudaMemcpyDeviceToHost, stream_));
                    check_cuda_error(cudaStreamSynchronize(stream_));
                    for (size_t i = 0; i < elem_count; ++i) {
                        h_fmha[i] = __half2float(tmp_fmha[i]);
                        h_ref[i]  = __half2float(tmp_ref[i]);
                    }
                }
#if ENABLE_BF16
                else if constexpr (std::is_same_v<T, bfloat16_t>) {
                    std::vector<bfloat16_t> tmp_fmha(elem_count);
                    std::vector<bfloat16_t> tmp_ref(elem_count);
                    check_cuda_error(cudaMemcpyAsync(tmp_fmha.data(),
                                                     ctx.data<bfloat16_t>(),
                                                     elem_count * sizeof(bfloat16_t),
                                                     cudaMemcpyDeviceToHost,
                                                     stream_));
                    check_cuda_error(cudaMemcpyAsync(tmp_ref.data(),
                                                     ctx_ref.data<bfloat16_t>(),
                                                     elem_count * sizeof(bfloat16_t),
                                                     cudaMemcpyDeviceToHost,
                                                     stream_));
                    check_cuda_error(cudaStreamSynchronize(stream_));
                    for (size_t i = 0; i < elem_count; ++i) {
                        h_fmha[i] = __bfloat162float(tmp_fmha[i]);
                        h_ref[i]  = __bfloat162float(tmp_ref[i]);
                    }
                }
#endif
#if ENABLE_FP32
                else if constexpr (std::is_same_v<T, float>) {
                    check_cuda_error(cudaMemcpyAsync(
                        h_fmha.data(), ctx.data<float>(), elem_count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                    check_cuda_error(cudaMemcpyAsync(
                        h_ref.data(), ctx_ref.data<float>(), elem_count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
                    check_cuda_error(cudaStreamSynchronize(stream_));
                }
#endif
                else {
                    h_fmha.clear();
                    h_ref.clear();
                }

                if (!h_fmha.empty() && h_fmha.size() == h_ref.size()) {
                    float max_abs = 0.f;
                    for (size_t i = 0; i < h_fmha.size(); ++i) {
                        const float diff = std::fabs(h_fmha[i] - h_ref[i]);
                        if (diff > max_abs) {
                            max_abs = diff;
                        }
                    }
                    TM_LOG_INFO(
                        "[EAGLE3][Attention][fmha_ab] token_num=%d kv_len=%d max_abs_diff=%.6f",
                        token_num,
                        kv_len,
                        max_abs);
                }
            }
        }

        // SDPA (naive) over kv_len tokens when FMHA path is disabled.
        if (!enable_fmha) {
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
        }

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

        if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
            fmha_in_flight_ = false;
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
            if (turbomind::isEnvVarEnabled("LMDEPLOY_EAGLE_FMHA_CONCURRENCY_DEBUG")) {
                fmha_in_flight_ = false;
            }
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
