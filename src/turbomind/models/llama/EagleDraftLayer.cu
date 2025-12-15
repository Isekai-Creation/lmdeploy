#include "src/turbomind/models/llama/EagleDraftLayer.h"

#include <algorithm>

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/eagle3_attention_layer.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/models/llama/llama_utils.h"

namespace {

template<typename T>
__global__ void copy_and_pad_rows_kernel(const T* __restrict__ src,
                                         int                   src_ld,
                                         T* __restrict__       dst,
                                         int                   dst_ld,
                                         int                   batch_size,
                                         int                   src_cols,
                                         int                   dst_cols,
                                         int                   copy_cols)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (row >= batch_size || col >= dst_cols) {
        return;
    }

    const T* src_row = src + static_cast<size_t>(row) * src_ld;
    T*       dst_row = dst + static_cast<size_t>(row) * dst_ld;

    if (col < copy_cols && col < src_cols) {
        dst_row[col] = src_row[col];
    }
    else {
        dst_row[col] = T(0);
    }
}

template<typename T>
void launch_copy_and_pad_rows(const turbomind::Tensor& src,
                              turbomind::Tensor&       dst,
                              int                      copy_cols,
                              cudaStream_t             stream)
{
    const int batch_size = static_cast<int>(src.shape(0));
    const int src_cols   = static_cast<int>(src.shape(1));
    const int dst_cols   = static_cast<int>(dst.shape(1));
    const int src_ld     = static_cast<int>(src.stride(0));
    const int dst_ld     = static_cast<int>(dst.stride(0));

    dim3 block(128);
    dim3 grid((dst_cols + block.x - 1) / block.x, batch_size);
    copy_and_pad_rows_kernel<T><<<grid, block, 0, stream>>>(src.data<T>(),
                                                            src_ld,
                                                            dst.data<T>(),
                                                            dst_ld,
                                                            batch_size,
                                                            src_cols,
                                                            dst_cols,
                                                            copy_cols);
}

template<typename T>
__global__ void pack_two_halves_kernel(const T* __restrict__ first,
                                       int                   first_ld,
                                       const T* __restrict__ second,
                                       int                   second_ld,
                                       T* __restrict__       dst,
                                       int                   dst_ld,
                                       int                   batch_size,
                                       int                   half_cols,
                                       bool                  have_second)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;

    if (row >= batch_size || col >= 2 * half_cols) {
        return;
    }

    T*       dst_row = dst + static_cast<size_t>(row) * dst_ld;
    const T* first_row =
        first ? first + static_cast<size_t>(row) * first_ld : nullptr;
    const T* second_row =
        second ? second + static_cast<size_t>(row) * second_ld : nullptr;

    if (col < half_cols) {
        if (first_row && col < first_ld) {
            dst_row[col] = first_row[col];
        }
        else {
            dst_row[col] = T(0);
        }
    }
    else {
        const int second_col = col - half_cols;
        if (have_second && second_row && second_col < second_ld) {
            dst_row[col] = second_row[second_col];
        }
        else {
            dst_row[col] = T(0);
        }
    }
}

template<typename T>
void launch_pack_two_halves(const turbomind::Tensor& first,
                            const turbomind::Tensor* second,
                            turbomind::Tensor&       dst,
                            int                      half_cols,
                            bool                     have_second,
                            cudaStream_t             stream)
{
    const int batch_size = static_cast<int>(dst.shape(0));
    const int dst_ld     = static_cast<int>(dst.stride(0));
    const int first_ld   = first ? static_cast<int>(first.stride(0)) : 0;
    const int second_ld  = (second && *second) ? static_cast<int>((*second).stride(0)) : 0;

    dim3 block(128);
    dim3 grid((2 * half_cols + block.x - 1) / block.x, batch_size);
    pack_two_halves_kernel<T><<<grid, block, 0, stream>>>(first ? first.data<T>() : nullptr,
                                                          first_ld,
                                                          (second && *second) ? second->data<T>() : nullptr,
                                                          second_ld,
                                                          dst.data<T>(),
                                                          dst_ld,
                                                          batch_size,
                                                          half_cols,
                                                          have_second);
}

}  // anonymous namespace

namespace turbomind {

Eagle3DraftLayer::Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight,
                                   UnifiedAttentionLayer*        attn_layer,
                                   Eagle3AttentionLayer*         eagle3_attn_layer,
                                   LlamaFfnLayer*                ffn_layer,
                                   float                         rmsnorm_eps):
    weight_{weight},
    ffn_layer_{ffn_layer},
    rmsnorm_eps_{rmsnorm_eps},
    debug_fc_out_{},
    debug_attn_out_{},
    debug_ffn_out_{},
    debug_pre_head_hidden_{},
    attn_layer_{attn_layer},
    eagle3_attn_layer_{eagle3_attn_layer},
    head_num_{0},
    kv_head_num_{0},
    size_per_head_{0},
    attn_geom_ok_{true}
{
    if (weight_) {
        head_num_         = weight_->head_num;
        kv_head_num_      = weight_->kv_head_num;
        size_per_head_    = weight_->size_per_head;
        draft_hidden_dim_ = weight_->draft_hidden_dim;
        base_hidden_dim_  = weight_->base_hidden_dim;
    }
}

bool Eagle3DraftLayer::is_qkv_compatible_() const
{
    // Require standard LLaMA attention geometry so UnifiedAttentionLayer
    // can consume the weights without reinterpretation.
    const Tensor& qkv_w = weight_->attn.qkv.weight;
    const Tensor& wo_w  = weight_->attn.output.weight;

    if (!qkv_w || !wo_w || qkv_w.ndim() != 2 || wo_w.ndim() != 2) {
        return false;
    }
    const int qkv_in  = qkv_w.shape(0);
    const int qkv_out = qkv_w.shape(1);
    const int wo_in   = wo_w.shape(0);
    const int wo_out  = wo_w.shape(1);
    // Expect qkv: [hidden, 3 * hidden] and wo: [hidden, hidden].
    if (qkv_out != 3 * qkv_in) {
        return false;
    }
    if (wo_in != qkv_in || wo_out != qkv_in) {
        return false;
    }
    return true;
}



void Eagle3DraftLayer::Forward(const Tensor& input_hidden,
                               const Tensor& captured_hidden,
                               const Tensor& input_ids,
                               const Tensor& embed_tokens_weights,
                               const Tensor& position_ids,
                               const Tensor& packed_mask,
                               const Tensor& tree_offsets,
                               const Tensor& runtime_offsets,
                               const Tensor& kv_lens_runtime,
                               const Tensor& successor_offsets,
                               const Tensor& successor_counts,
                               int           q_len,
                               int           kv_len,
                               int           past_kv_len,
                               Tensor&       output_hidden,
                               cudaStream_t  stream)
{
    // Tree/run-time metadata is threaded through to enable full TRT-style
    // masking/branching once the real attention backend is wired. For now
    // they are unused in the shallow path.
    if (!input_hidden) {
        return;
    }

    if (!weight_) {
        TM_LOG_WARNING(
            "[EAGLE][Eagle3DraftLayer][fallback] draft layer weights unavailable; "
            "treating Eagle3 draft as pass-through.");
        output_hidden = input_hidden;
        return;
    }
    const bool debug_enabled = isEagleDebugEnabled();

    debug_fc_out_          = Tensor{};
    debug_attn_out_        = Tensor{};
    debug_ffn_out_         = Tensor{};
    debug_pre_head_hidden_ = Tensor{};
    debug_qkv_             = Tensor{};
    attn_geom_ok_          = true;

    const int batch_size   = input_hidden.shape(0);
    const int hidden_dim   = input_hidden.shape(1);          // incoming hidden (base)
    const int draft_dim    = draft_hidden_dim_ > 0 ? draft_hidden_dim_ : hidden_dim;
    const int wo_out_dim   = weight_->attn.output.weight ? weight_->attn.output.weight.shape(1) : base_hidden_dim_;

    if (batch_size <= 0 || hidden_dim <= 0) {
        output_hidden = input_hidden;
        return;
    }

    const auto dtype = input_hidden.dtype();

    if (debug_enabled) {
        TM_LOG_INFO(
            "[EAGLE3][Draft][stage] begin Eagle3DraftLayer::Forward "
            "(batch=%d, hidden_dim=%d, draft_dim=%d, base_hidden_dim=%d, q_len=%d, kv_len=%d, past_kv_len=%d)",
            batch_size,
            hidden_dim,
            draft_dim,
            base_hidden_dim_,
            q_len,
            kv_len,
            past_kv_len);
    }

    // New code for embedding lookup and norm
    Tensor embed_input_base_dim{{batch_size, base_hidden_dim_}, dtype, input_hidden.device()}; // Use base_hidden_dim_ here
    Tensor embed_norm{{batch_size, draft_dim}, dtype, input_hidden.device()}; // Normed to draft_dim
    bool have_embed = false;

    if (input_ids && embed_tokens_weights && input_ids.size() == batch_size && embed_tokens_weights.ndim() == 2 && embed_tokens_weights.shape(1) == base_hidden_dim_) {
        core::Buffer raw_ids(const_cast<void*>(input_ids.raw_data()), batch_size, kInt32, input_hidden.device());
        core::Buffer_<int> token_ids(raw_ids);
        core::Ref<Tensor>  embed_out(embed_input_base_dim); // Lookup into base_hidden_dim_ size
        invokeEmbeddingLookup(embed_out, token_ids, embed_tokens_weights, stream);

        // Convert from base_hidden_dim_ to draft_dim before RMSNorm
        if (base_hidden_dim_ != draft_dim) {
            Tensor temp_embed_for_norm{{batch_size, draft_dim}, dtype, input_hidden.device()};
            const int copy_cols = std::min(base_hidden_dim_, draft_dim);
            switch (dtype) {
            case kFloat16:
                launch_copy_and_pad_rows<half_t>(embed_input_base_dim, temp_embed_for_norm, copy_cols, stream);
                break;
#if ENABLE_BF16
            case kBfloat16:
                launch_copy_and_pad_rows<bfloat16_t>(embed_input_base_dim, temp_embed_for_norm, copy_cols, stream);
                break;
#endif
            case kFloat32:
                launch_copy_and_pad_rows<float>(embed_input_base_dim, temp_embed_for_norm, copy_cols, stream);
                break;
            default:
                TM_LOG_WARNING("[EAGLE3][Draft] unsupported dtype for embedding resize; falling back to memcpy path.");
                {
                    check_cuda_error(cudaMemsetAsync(
                        temp_embed_for_norm.raw_data(), 0, temp_embed_for_norm.byte_size(), stream));
                    const size_t elem_bytes = byte_size(dtype, 1);
                    const size_t src_stride =
                        static_cast<size_t>(embed_input_base_dim.stride(0)) * elem_bytes;
                    const size_t dst_stride =
                        static_cast<size_t>(temp_embed_for_norm.stride(0)) * elem_bytes;
                    const size_t copy_bytes = static_cast<size_t>(copy_cols) * elem_bytes;
                    for (int b = 0; b < batch_size; ++b) {
                        const char* src_row =
                            static_cast<const char*>(embed_input_base_dim.raw_data())
                            + static_cast<size_t>(b) * src_stride;
                        char* dst_row = static_cast<char*>(temp_embed_for_norm.raw_data())
                            + static_cast<size_t>(b) * dst_stride;
                        check_cuda_error(cudaMemcpyAsync(
                            dst_row, src_row, copy_bytes, cudaMemcpyDeviceToDevice, stream));
                    }
                }
                break;
            }
            invokeRMSNorm(embed_norm, temp_embed_for_norm, weight_->input_norm, rmsnorm_eps_, stream);
        } else {
            invokeRMSNorm(embed_norm, embed_input_base_dim, weight_->input_norm, rmsnorm_eps_, stream);
        }
        have_embed = true;
    } else {
        if (input_ids && !embed_tokens_weights) {
            TM_LOG_WARNING("[EAGLE3][Draft] input_ids present but embed_tokens_weights missing; cannot use embedding for 2H QKV input.");
        } else if (!input_ids && embed_tokens_weights) {
            TM_LOG_WARNING("[EAGLE3][Draft] embed_tokens_weights present but input_ids missing; cannot use embedding for 2H QKV input.");
        } else {
             TM_LOG_WARNING("[EAGLE3][Draft] input_ids and embed_tokens_weights missing; cannot use embedding for 2H QKV input.");
        }
    }
    // End new code

    auto norm_mismatch = [&](const Tensor& w, const char* name) -> bool {
        if (!w) {
            return false;
        }
        if (w.dtype() != dtype || w.ndim() != 1 || w.shape(0) != draft_dim) {
            TM_LOG_WARNING(
                "[EAGLE][Eagle3DraftLayer][fallback] %s norm mismatch: "
                "weight_dtype=%d hidden_dtype=%d weight_dim=%d hidden_dim=%d; treating draft as pass-through.",
                name,
                static_cast<int>(w.dtype()),
                static_cast<int>(dtype),
                w.shape(0),
                draft_dim);
            return true;
        }
        return false;
    };

    if (norm_mismatch(weight_->input_norm, "input")
        || norm_mismatch(weight_->post_attn_norm, "post_attn")
        || norm_mismatch(weight_->output_norm, "output")) {
        output_hidden = input_hidden;
        return;
    }

    // 1) Input RMSNorm before attention.
    // Reduce/expand incoming hidden to draft_dim if needed.
    Tensor draft_view = input_hidden;
    if (hidden_dim != draft_dim) {
        // If input is wider than draft_dim, truncate; if narrower, pad zeros.
        draft_view = Tensor{{batch_size, draft_dim}, dtype, input_hidden.device()};
        const int copy_cols = std::min(hidden_dim, draft_dim);
        switch (dtype) {
        case kFloat16:
            launch_copy_and_pad_rows<half_t>(input_hidden, draft_view, copy_cols, stream);
            break;
#if ENABLE_BF16
        case kBfloat16:
            launch_copy_and_pad_rows<bfloat16_t>(input_hidden, draft_view, copy_cols, stream);
            break;
#endif
        case kFloat32:
            launch_copy_and_pad_rows<float>(input_hidden, draft_view, copy_cols, stream);
            break;
        default:
            TM_LOG_WARNING("[EAGLE3][Draft] unsupported dtype for draft_view resize; falling back to memcpy path.");
            {
                check_cuda_error(
                    cudaMemsetAsync(draft_view.raw_data(), 0, draft_view.byte_size(), stream));
                const size_t elem_bytes = byte_size(dtype, 1);
                const size_t src_stride =
                    static_cast<size_t>(input_hidden.stride(0)) * elem_bytes;
                const size_t dst_stride =
                    static_cast<size_t>(draft_view.stride(0)) * elem_bytes;
                const size_t copy_bytes =
                    static_cast<size_t>(copy_cols) * elem_bytes;
                for (int b = 0; b < batch_size; ++b) {
                    char* dst =
                        static_cast<char*>(draft_view.raw_data())
                        + static_cast<size_t>(b) * dst_stride;
                    const char* src =
                        static_cast<const char*>(input_hidden.raw_data())
                        + static_cast<size_t>(b) * src_stride;
                    check_cuda_error(cudaMemcpyAsync(
                        dst, src, copy_bytes, cudaMemcpyDeviceToDevice, stream));
                }
            }
            break;
        }
    }

    Tensor hidden_norm{{batch_size, draft_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(hidden_norm, draft_view, weight_->input_norm, rmsnorm_eps_, stream);

    // Optional FC branch for 2H packing (embedding_norm + FC_norm).
    Tensor fc_norm;
    bool   have_fc = false;
    if (ffn_layer_ && captured_hidden && captured_hidden.ndim() == 2 && captured_hidden.shape(0) == batch_size
        && captured_hidden.dtype() == dtype && weight_->fc_weight) {
        LlamaLinear&     linear_fc = ffn_layer_->linear();
        LlamaDenseWeight fc_w;

        // Construct a minimal dense weight wrapper around the Eagle3 FC
        // tensor so that LlamaLinear can run a standard row-major GEMM
        // without relying on any quantization metadata. Geometry:
        //   captured_hidden: [batch_size, eagle_fc_in_dim]
        //   fc_weight:       [eagle_fc_in_dim, draft_dim]
        const Tensor& fc_tensor = weight_->fc_weight;
        const int     fc_in     = fc_tensor.ndim() == 2 ? fc_tensor.shape(0) : 0;
        const int     fc_out    = fc_tensor.ndim() == 2 ? fc_tensor.shape(1) : 0;

        if (fc_in > 0 && fc_out == draft_dim && captured_hidden.shape(1) == fc_in) {
            fc_w.data_type   = dtype;
            fc_w.input_type  = dtype;
            fc_w.weight_type = dtype;
            fc_w.input_dim   = fc_in;
            fc_w.output_dim  = fc_out;
            fc_w.group_size  = 1;

            fc_w.weight = fc_tensor.borrow();
            fc_w.bias   = {};

            fc_w.weight_quant = {};
            fc_w.input_quant  = {};
            fc_w.weight_quant.type = gemm::QuantType::kNone;
            fc_w.input_quant.type  = gemm::QuantType::kNone;

            fc_w.epilogue = Epilogue::kNone;

            fc_w.k_desc       = {};
            fc_w.k_desc.type  = dtype;
            fc_w.k_desc.order = gemm::kRowMajor;
            fc_w.k_desc.rows  = fc_in;
            fc_w.k_desc.cols  = fc_out;
            fc_w.k_desc.ld    = fc_out;
            fc_w.k_desc.pack  = 0;
            fc_w.k_desc.num   = 0;
            fc_w.k_desc.offsets = nullptr;
            fc_w.k_desc.idxs    = nullptr;

            fc_w.q_desc = {};

            if (isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG")) {
                logEagleGemmShape("EAGLE3_FC",
                                  batch_size,
                                  fc_in,
                                  fc_out,
                                  static_cast<int>(dtype),
                                  "row_major");
            }

            EagleGemmTagGuard gemm_guard("EAGLE3_FC");
            Tensor            fc_out = linear_fc.Forward(captured_hidden, fc_w);
            fc_norm       = Tensor{{batch_size, draft_dim}, dtype, input_hidden.device()};
            invokeRMSNorm(fc_norm, fc_out, weight_->input_norm, rmsnorm_eps_, stream);
            have_fc = true;
        }
        else {
            TM_LOG_WARNING(
                "[EAGLE3][Draft][fallback] fc_weight geometry (%d,%d) or captured_hidden width (%d) "
                "incompatible with draft_dim=%d; skipping FC branch.",
                fc_in,
                fc_out,
                captured_hidden.shape(1),
                draft_dim);
        }
    }
    else if (!ffn_layer_ && captured_hidden && weight_->fc_weight) {
        static bool logged_missing_ffn_fc = false;
        if (!logged_missing_ffn_fc) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft][fallback] FC capture available but ffn_layer_ is null; "
                "skipping FC branch and using attention-only draft.");
            logged_missing_ffn_fc = true;
        }
    }

    // Build 2H QKV input for Eagle-3 attention. TRT packs
    // [embedding_norm, fc_norm] so default to that layout when shapes match.
    int qkv_in_dim = weight_->attn.qkv.weight ? weight_->attn.qkv.weight.shape(0) : 2 * draft_dim;
    if (weight_->eagle3_attn.is_initialized && weight_->eagle3_attn.q_in > 0) {
        qkv_in_dim = weight_->eagle3_attn.q_in;
    }
    const bool expect_two_h  = (qkv_in_dim == 2 * draft_dim);
    Tensor     qkv_input{{batch_size, qkv_in_dim}, dtype, input_hidden.device()};
    check_cuda_error(cudaMemsetAsync(qkv_input.raw_data(), 0, qkv_input.byte_size(), stream));

    if (!expect_two_h) {
        TM_LOG_ERROR(
            "[EAGLE3][Draft] QKV input dimension (%d) is not 2 * draft_hidden (%d); "
            "disabling Eagle3 attention for this layer.",
            qkv_in_dim,
            2 * draft_dim);
        attn_geom_ok_ = false;
        output_hidden = input_hidden; // Fallback to pass-through
        return;
    }

    // Strict TRT packing: first half embedding_norm (or hidden_norm), second half FC_norm.
    static bool logged_missing_fc = false;
    const turbomind::Tensor* second_half = nullptr;
    if (have_fc) {
        second_half = &fc_norm;
    }
    else {
        if (!logged_missing_fc) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft] FC capture missing; filling 2H QKV input second half with zeros.");
            logged_missing_fc = true;
        }
    }

    switch (dtype) {
    case kFloat16:
        launch_pack_two_halves<half_t>(
            embed_norm, second_half, qkv_input, draft_dim, have_fc, stream);
        break;
#if ENABLE_BF16
    case kBfloat16:
        launch_pack_two_halves<bfloat16_t>(
            embed_norm, second_half, qkv_input, draft_dim, have_fc, stream);
        break;
#endif
    case kFloat32:
        launch_pack_two_halves<float>(
            embed_norm, second_half, qkv_input, draft_dim, have_fc, stream);
        break;
    default:
        TM_LOG_WARNING(
            "[EAGLE3][Draft] unsupported dtype for QKV packing; falling back to memcpy path.");
        {
            const size_t elem_bytes = byte_size(dtype, 1);
            const size_t src_row_bytes =
                static_cast<size_t>(embed_norm.stride(0)) * elem_bytes;
            const size_t dst_row_bytes =
                static_cast<size_t>(qkv_input.stride(0)) * elem_bytes;
            const char* src_base =
                static_cast<const char*>(embed_norm.raw_data());
            char* dst_base = static_cast<char*>(qkv_input.raw_data());
            const size_t slice_bytes =
                static_cast<size_t>(draft_dim) * elem_bytes;
            for (int b = 0; b < batch_size; ++b) {
                const char* src_row =
                    src_base + static_cast<size_t>(b) * src_row_bytes;
                char* dst_row =
                    dst_base + static_cast<size_t>(b) * dst_row_bytes;
                check_cuda_error(cudaMemcpyAsync(
                    dst_row, src_row, slice_bytes, cudaMemcpyDeviceToDevice, stream));
            }
        }
        break;
    }

    if (debug_enabled) {
        debug_fc_out_ = have_fc ? fc_norm : hidden_norm;
    }

    // 2) Attention: prefer the native Eagle3 fused-QKV path when geometry matches;
    //    otherwise fall back to UnifiedAttentionLayer or shallow pass-through.
    Tensor attn_out{{batch_size, wo_out_dim}, dtype, input_hidden.device()};

    const bool has_attn_layer   = (attn_layer_ != nullptr);
    const bool has_eagle3_layer = (eagle3_attn_layer_ != nullptr && weight_->eagle3_attn.is_initialized);
    const bool native_eagle3_ok = has_eagle3_layer && weight_->eagle3_attn.is_initialized
        && qkv_input.shape(1) == weight_->eagle3_attn.q_in && weight_->eagle3_attn.q_out > 0
        && weight_->eagle3_attn.kv_out > 0 && attn_geom_ok_;

    static bool logged_invalid_geom = false;

    if (native_eagle3_ok) {
        if (debug_enabled) {
            TM_LOG_INFO(
                "[EAGLE3][Draft][stage] running Eagle3AttentionLayer::Forward "
                "(q_in=%d, q_out=%d, kv_out=%d, head_num=%d, kv_head_num=%d)",
                weight_->eagle3_attn.q_in,
                weight_->eagle3_attn.q_out,
                weight_->eagle3_attn.kv_out,
                weight_->eagle3_attn.num_q_heads,
                weight_->eagle3_attn.num_kv_heads);
        }
        {
            NvtxScope scope("EAGLE_DRAFT_ATTENTION_FMHA");
            Eagle3AttentionParam ep{};
            ep.input              = qkv_input;
            ep.output             = attn_out;
            ep.attn_weights       = &weight_->attn;
            ep.weights            = &weight_->eagle3_attn;
            ep.layer_id           = 0;
            ep.past_kv_len        = past_kv_len;
            ep.packed_mask        = packed_mask ? &packed_mask : nullptr;
            ep.packed_mask_stride = (packed_mask && packed_mask.ndim() >= 2) ? packed_mask.shape(1) : 0;
            ep.position_ids       = position_ids ? &position_ids : nullptr;
            ep.tree_offsets       = tree_offsets ? &tree_offsets : nullptr;
            ep.runtime_offsets    = runtime_offsets ? &runtime_offsets : nullptr;
            ep.kv_lens_runtime    = kv_lens_runtime ? &kv_lens_runtime : nullptr;
            ep.successor_offsets  = successor_offsets ? &successor_offsets : nullptr;
            ep.successor_counts   = successor_counts ? &successor_counts : nullptr;
            ep.batch_size         = batch_size;
            ep.q_len              = q_len > 0 ? q_len : 1;
            ep.kv_len             = kv_len > 0 ? kv_len : ep.q_len;
            ep.rope_base          = weight_->eagle3_attn.rope_base > 0.f ? weight_->eagle3_attn.rope_base : 10000.f;
            ep.rope_scale         = weight_->eagle3_attn.rope_scale > 0.f ? weight_->eagle3_attn.rope_scale : 1.f;
            ep.debug_qkv          = debug_enabled ? &debug_qkv_ : nullptr;
            ep.debug_attn_out     = debug_enabled ? &debug_attn_out_ : nullptr;
            eagle3_attn_layer_->Forward(ep);
            attn_out = ep.output;
            sync_check_cuda_error();
        }
    }
    else {
        if (!logged_invalid_geom) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft] invalid Eagle3 geometry (q_in=%d, expected=%d); skipping Eagle3 attention.",
                qkv_input.shape(1),
                weight_->eagle3_attn.q_in);
            logged_invalid_geom = true;
        }
        attn_out = hidden_norm;
    }

    if (debug_enabled) {
        debug_attn_out_ = attn_out;
    }

    // 3) FFN path: Ensure attn_out is converted to draft_dim before FFN.
    if (!ffn_layer_) { // Still need to check if FFN is available
        output_hidden = attn_out;
        if (debug_enabled) {
            debug_ffn_out_         = Tensor{};
            debug_pre_head_hidden_ = output_hidden;
        }
        return;
    }

    Tensor ffn_input;

    // Convert attn_out to ffn_input (draft_dim = draft_hidden_dim_). Prefer
    // the actual attn_out width when available, falling back to the
    // configured base_hidden_dim_ only as a sanity default.
    const int current_attn_out_dim =
        (attn_out.ndim() == 2 && attn_out.shape(1) > 0) ? attn_out.shape(1) : base_hidden_dim_;
    const int target_ffn_in_dim = draft_dim;  // Use authoritative draft_hidden_dim

    if (current_attn_out_dim != target_ffn_in_dim) {
        ffn_input = Tensor{{batch_size, draft_dim}, dtype, input_hidden.device()};
        const int copy_cols = std::min(current_attn_out_dim, target_ffn_in_dim);
        switch (dtype) {
        case kFloat16:
            launch_copy_and_pad_rows<half_t>(attn_out, ffn_input, copy_cols, stream);
            break;
#if ENABLE_BF16
        case kBfloat16:
            launch_copy_and_pad_rows<bfloat16_t>(attn_out, ffn_input, copy_cols, stream);
            break;
#endif
        case kFloat32:
            launch_copy_and_pad_rows<float>(attn_out, ffn_input, copy_cols, stream);
            break;
        default:
            TM_LOG_WARNING("[EAGLE3][Draft] unsupported dtype for FFN input resize; falling back to memcpy path.");
            {
                check_cuda_error(
                    cudaMemsetAsync(ffn_input.raw_data(), 0, ffn_input.byte_size(), stream));
                const size_t elem_bytes = byte_size(dtype, 1);
                const size_t src_stride =
                    static_cast<size_t>(attn_out.stride(0)) * elem_bytes;
                const size_t dst_stride =
                    static_cast<size_t>(ffn_input.stride(0)) * elem_bytes;
                const size_t copy_bytes =
                    static_cast<size_t>(copy_cols) * elem_bytes;
                for (int b = 0; b < batch_size; ++b) {
                    const char* src_row =
                        static_cast<const char*>(attn_out.raw_data())
                        + static_cast<size_t>(b) * src_stride;
                    char* dst_row =
                        static_cast<char*>(ffn_input.raw_data())
                        + static_cast<size_t>(b) * dst_stride;
                    check_cuda_error(cudaMemcpyAsync(
                        dst_row, src_row, copy_bytes, cudaMemcpyDeviceToDevice, stream));
                }
            }
            break;
        }
    }
    else {
        // When attention output already matches the draft dimension, just
        // reuse the buffer directly to avoid an unnecessary device-to-device
        // memcpy and potential CUDA argument issues.
        ffn_input = attn_out;
    }

    {
        NvtxScope scope("EAGLE_DRAFT_FFN");
        invokeRMSNorm(ffn_input, ffn_input, weight_->post_attn_norm, rmsnorm_eps_, stream);

    // Ensure output buffer is allocated with the correct shape.
    if (!output_hidden || output_hidden.ndim() != 2 || output_hidden.shape(0) != batch_size
        || output_hidden.shape(1) != draft_dim || output_hidden.dtype() != dtype
        || output_hidden.device().type != input_hidden.device().type) {
        output_hidden = Tensor{{batch_size, draft_dim}, dtype, input_hidden.device()};
    }

    // Gated MLP using the prepared Eagle3 FFN weights.
        LlamaFfnLayer::ForwardParam ffn_param{};
        ffn_param.input    = ffn_input;
        ffn_param.output   = output_hidden;
        ffn_param.weights  = &weight_->ffn;
        ffn_param.layer_id = 0;
        ffn_layer_->forward(ffn_param);
        sync_check_cuda_error();
    }
    if (debug_enabled) {
        debug_ffn_out_ = output_hidden;
    }

    // Residual + output RMSNorm in draft space.
    invokeResidualBiasRMSNorm(
        /*hidden_states=*/output_hidden.raw_data(),
        /*residual=*/const_cast<void*>(draft_view.raw_data()),
        /*weights=*/weight_->output_norm.raw_data(),
        /*bias=*/nullptr,
        dtype,
        draft_dim,
        batch_size,
        rmsnorm_eps_,
        stream);

    if (debug_enabled) {
        debug_pre_head_hidden_ = output_hidden;
        TM_LOG_INFO("[EAGLE3][Draft][stage] completed Eagle3DraftLayer::Forward");
    }
}

}  // namespace turbomind
