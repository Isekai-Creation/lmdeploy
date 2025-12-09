// Copyright (c) OpenMMLab. All rights reserved.

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

namespace {

template<typename T>
__global__ void expand_kv_to_q_kernel(const T* __restrict__ kv,
                                      T* __restrict__ q,
                                      int batch,
                                      int kv_heads,
                                      int head_dim,
                                      int group_size)
{
    // kv layout: [batch, kv_heads, head_dim]
    // output q layout: [batch, kv_heads * group_size, head_dim]
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * kv_heads * head_dim;
    if (idx >= total) {
        return;
    }
    int d      = idx % head_dim;
    int kv_idx = (idx / head_dim) % kv_heads;
    int b      = idx / (head_dim * kv_heads);

    const T* src = kv + ((b * kv_heads + kv_idx) * head_dim + d);
    T*       dst = q + ((b * kv_heads * group_size + kv_idx * group_size) * head_dim + d);
    for (int g = 0; g < group_size; ++g) {
        dst[g * head_dim] = *src;
    }
}

template<typename T>
void expand_kv_to_q(const Tensor& kv, Tensor& q_expanded, int kv_heads, int head_dim, int group_size, cudaStream_t stream)
{
    const int batch = kv.shape(0);
    const int elems = batch * kv_heads * head_dim;
    const int threads = 256;
    const int blocks  = (elems + threads - 1) / threads;
    expand_kv_to_q_kernel<<<blocks, threads, 0, stream>>>(
        kv.data<T>(), q_expanded.data<T>(), batch, kv_heads, head_dim, group_size);
}

}  // namespace

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
    size_per_head_{0}
{
    if (weight_) {
        head_num_      = weight_->attn.head_num;
        kv_head_num_   = weight_->attn.kv_head_num;
        size_per_head_ = weight_->attn.size_per_head;
        draft_hidden_dim_ = weight_->attn.qkv.weight ? weight_->attn.qkv.weight.shape(0) / 2 : 0;
        base_hidden_dim_  = weight_->attn.output.weight ? weight_->attn.output.weight.shape(1) : 0;
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

// Validate QKV/Wo shapes instead of head_num * size_per_head gating.
bool Eagle3DraftLayer::attn_geom_ok_(int hidden_dim) const
{
    const Tensor& qkv_w = weight_->attn.qkv.weight;
    const Tensor& wo_w  = weight_->attn.output.weight;
    if (!qkv_w || !wo_w || qkv_w.ndim() != 2 || wo_w.ndim() != 2) {
        return false;
    }
    if (qkv_w.shape(1) <= 0 || wo_w.shape(0) <= 0 || wo_w.shape(1) <= 0) {
        return false;
    }
    const int q_dim   = wo_w.shape(0);
    const int kv_span = qkv_w.shape(1) - q_dim;
    if (kv_span <= 0 || (kv_span % 2) != 0) {
        return false;
    }
    // Wo output width must match the base hidden (residual) width when provided.
    if (base_hidden_dim_ > 0 && wo_w.shape(1) != base_hidden_dim_) {
        return false;
    }
    // Require QKV input width to be either draft_dim or 2 * draft_dim.
    const int draft_dim = draft_hidden_dim_ > 0 ? draft_hidden_dim_ : hidden_dim;
    if (qkv_w.shape(0) != draft_dim && qkv_w.shape(0) != 2 * draft_dim) {
        return false;
    }
    return true;
}

void Eagle3DraftLayer::Forward(const Tensor& input_hidden, Tensor& output_hidden, cudaStream_t stream)
{
    Tensor empty;
    Forward(input_hidden,
            empty,
            Tensor{},
            Tensor{},
            Tensor{},
            Tensor{},
            /*q_len=*/1,
            /*kv_len=*/1,
            /*past_kv_len=*/0,
            output_hidden,
            stream);
}

void Eagle3DraftLayer::Forward(const Tensor& input_hidden,
                               const Tensor& captured_hidden,
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

    const int batch_size   = input_hidden.shape(0);
    const int hidden_dim   = input_hidden.shape(1);          // incoming hidden (base)
    const int draft_dim    = draft_hidden_dim_ > 0 ? draft_hidden_dim_ : hidden_dim;
    const int wo_out_dim   = weight_->attn.output.weight ? weight_->attn.output.weight.shape(1) : base_hidden_dim_;

    if (batch_size <= 0 || hidden_dim <= 0) {
        output_hidden = input_hidden;
        return;
    }

    const auto dtype = input_hidden.dtype();

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
        const size_t elem_bytes = byte_size(dtype, 1);
        const size_t copy_bytes = static_cast<size_t>(std::min(hidden_dim, draft_dim)) * elem_bytes;
        const size_t src_stride = static_cast<size_t>(input_hidden.stride(0)) * elem_bytes;
        const size_t dst_stride = static_cast<size_t>(draft_view.stride(0)) * elem_bytes;
        check_cuda_error(cudaMemsetAsync(draft_view.raw_data(), 0, draft_view.byte_size(), stream));
        for (int b = 0; b < batch_size; ++b) {
            char*       dst = static_cast<char*>(draft_view.raw_data()) + static_cast<size_t>(b) * dst_stride;
            const char* src = static_cast<const char*>(input_hidden.raw_data()) + static_cast<size_t>(b) * src_stride;
            check_cuda_error(cudaMemcpyAsync(dst, src, copy_bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }

    Tensor hidden_norm{{batch_size, draft_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(hidden_norm, draft_view, weight_->input_norm, rmsnorm_eps_, stream);

    // Optional FC branch for 2H packing (embedding_norm + FC_norm).
    Tensor fc_norm;
    bool   have_fc = false;
    if (captured_hidden && captured_hidden.ndim() == 2 && captured_hidden.shape(0) == batch_size
        && captured_hidden.dtype() == dtype && weight_->fc_weight) {
        LlamaLinear& linear_fc = ffn_layer_->linear();
        Tensor       fc_out{{batch_size, draft_dim}, dtype, input_hidden.device()};
        linear_fc.Forward(captured_hidden, weight_->fc_weight, fc_out);
        fc_norm = Tensor{{batch_size, draft_dim}, dtype, input_hidden.device()};
        invokeRMSNorm(fc_norm, fc_out, weight_->input_norm, rmsnorm_eps_, stream);
        have_fc = true;
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

    const size_t elem_bytes    = byte_size(dtype, 1);
    const size_t src_row_bytes = static_cast<size_t>(hidden_norm.stride(0)) * elem_bytes;
    const size_t dst_row_bytes = static_cast<size_t>(qkv_input.stride(0)) * elem_bytes;

    char*       dst_base = static_cast<char*>(qkv_input.raw_data());
    const char* src_base = static_cast<const char*>(hidden_norm.raw_data());
    const size_t fc_row_bytes = have_fc ? static_cast<size_t>(fc_norm.stride(0)) * elem_bytes : src_row_bytes;
    const char* fc_base       = have_fc ? static_cast<const char*>(fc_norm.raw_data()) : src_base;

    if (expect_two_h) {
        // Strict TRT packing: first half embedding_norm, second half FC_norm.
        const size_t slice_bytes = static_cast<size_t>(draft_dim) * elem_bytes;
        for (int b = 0; b < batch_size; ++b) {
            char*       dst_row = dst_base + static_cast<size_t>(b) * dst_row_bytes;
            const char* src_row = src_base + static_cast<size_t>(b) * src_row_bytes;
            check_cuda_error(
                cudaMemcpyAsync(dst_row, src_row, slice_bytes, cudaMemcpyDeviceToDevice, stream));

            char* second_dst = dst_row + slice_bytes;
            if (have_fc) {
                const char* second_src = fc_base + static_cast<size_t>(b) * fc_row_bytes;
                check_cuda_error(cudaMemcpyAsync(
                    second_dst, second_src, slice_bytes, cudaMemcpyDeviceToDevice, stream));
            }
            else {
                // No FC capture available; leave the second half zero but log once.
                static bool logged_missing_fc = false;
                if (!logged_missing_fc) {
                    TM_LOG_WARNING(
                        "[EAGLE3][Draft] FC capture missing; filling 2H QKV input second half with zeros.");
                    logged_missing_fc = true;
                }
            }
        }
    }
    else {
        // Fallback packing: copy as much as fits from hidden_norm then FC/hidden_norm.
        const size_t copy_bytes = static_cast<size_t>(std::min(draft_dim, qkv_in_dim)) * elem_bytes;
        for (int b = 0; b < batch_size; ++b) {
            char*       dst_row = dst_base + static_cast<size_t>(b) * dst_row_bytes;
            const char* src_row = src_base + static_cast<size_t>(b) * src_row_bytes;
            check_cuda_error(cudaMemcpyAsync(dst_row, src_row, copy_bytes, cudaMemcpyDeviceToDevice, stream));

            if (qkv_in_dim > draft_dim) {
                const size_t second_bytes = static_cast<size_t>(std::min(draft_dim, qkv_in_dim - draft_dim)) * elem_bytes;
                const char*  second_src   = fc_base + static_cast<size_t>(b) * fc_row_bytes;
                check_cuda_error(cudaMemcpyAsync(
                    dst_row + static_cast<size_t>(draft_dim) * elem_bytes,
                    second_src,
                    second_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
        }
    }

    if (debug_enabled) {
        debug_fc_out_ = have_fc ? fc_norm : hidden_norm;
    }

    // 2) Attention: prefer the native Eagle3 fused-QKV path when geometry matches;
    //    otherwise fall back to UnifiedAttentionLayer or shallow pass-through.
    Tensor attn_out{{batch_size, wo_out_dim}, dtype, input_hidden.device()};

    const bool has_attn_layer   = (attn_layer_ != nullptr);
    const bool has_eagle3_layer = (eagle3_attn_layer_ != nullptr && weight_->eagle3_attn.is_initialized);
    const bool can_use_unified  = has_attn_layer && attn_geom_ok_(draft_dim) && weight_->attn.qkv.weight
                                  && weight_->attn.output.weight
                                  && weight_->attn.qkv.weight.shape(0) == draft_dim
                                  && weight_->attn.qkv.weight.shape(1) == 3 * draft_dim
                                  && weight_->attn.output.weight.shape(0) == draft_dim
                                  && weight_->attn.output.weight.shape(1) == draft_dim;
    const bool native_eagle3_ok = has_eagle3_layer && weight_->eagle3_attn.is_initialized
        && qkv_input.shape(1) == weight_->eagle3_attn.q_in && weight_->eagle3_attn.q_out > 0
        && weight_->eagle3_attn.kv_out > 0;

    if (native_eagle3_ok) {
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
        ep.debug_qkv          = debug_enabled ? &debug_qkv_ : nullptr;
        ep.debug_attn_out     = debug_enabled ? &debug_attn_out_ : nullptr;
        eagle3_attn_layer_->Forward(ep);
        attn_out = ep.output;
        sync_check_cuda_error();
    }
    else if (can_use_unified) {
        UnifiedAttentionLayer::ForwardParam param{};
        param.input    = qkv_input;
        if (!attn_out || attn_out.shape(0) != batch_size || attn_out.shape(1) != wo_out_dim
            || attn_out.dtype() != dtype || attn_out.device().type != input_hidden.device().type) {
            attn_out = Tensor{{batch_size, wo_out_dim}, dtype, input_hidden.device()};
        }
        param.output   = attn_out;
        param.weights  = &weight_->attn;
        param.layer_id = 0;
        param.q_len    = q_len > 0 ? q_len : 1;
        param.kv_len   = kv_len > 0 ? kv_len : param.q_len;

        attn_layer_->Forward(param);
        sync_check_cuda_error();
    }
    else if (has_eagle3_layer && attn_geom_ok_(draft_dim)) {
        Eagle3AttentionParam ep{};
        ep.input        = qkv_input;
        ep.output       = attn_out;
        ep.attn_weights = &weight_->attn;
        ep.weights      = &weight_->eagle3_attn;
        ep.batch_size   = batch_size;
        ep.layer_id     = 0;
        ep.q_len        = q_len > 0 ? q_len : 1;
        ep.kv_len       = kv_len > 0 ? kv_len : ep.q_len;
        ep.past_kv_len  = past_kv_len;
        ep.rope_base    = weight_->eagle3_attn.rope_base > 0.f ? weight_->eagle3_attn.rope_base : 10000.f;
        ep.rope_scale   = weight_->eagle3_attn.rope_scale > 0.f ? weight_->eagle3_attn.rope_scale : 1.f;
        ep.tree_offsets       = tree_offsets ? &tree_offsets : nullptr;
        ep.runtime_offsets    = runtime_offsets ? &runtime_offsets : nullptr;
        ep.kv_lens_runtime    = kv_lens_runtime ? &kv_lens_runtime : nullptr;
        ep.successor_offsets = successor_offsets ? &successor_offsets : nullptr;
        ep.successor_counts  = successor_counts ? &successor_counts : nullptr;
        if (position_ids) {
            ep.position_ids = &position_ids;
        }
        if (packed_mask) {
            ep.packed_mask = &packed_mask;
            ep.packed_mask_stride = (packed_mask.ndim() >= 2) ? packed_mask.shape(1) : 0;
        }
        if (debug_enabled) {
            ep.debug_qkv      = &debug_qkv_;
            ep.debug_attn_out = &debug_attn_out_;
        }
        eagle3_attn_layer_->Forward(ep);
        sync_check_cuda_error();
    }
    else {
        if (has_attn_layer && !can_use_unified) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft] invalid attention geometry (head_num=%d, size_per_head=%d, hidden_dim=%d); "
                "falling back to pass-through until real Eagle3 attention is wired.",
                head_num_,
                size_per_head_,
                draft_dim);
        }
        else if (debug_enabled) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft] shallow pass-through attention path active; replace once Eagle3 kernels land.");
        }
        attn_out = hidden_norm;
    }

    if (debug_enabled) {
        debug_attn_out_ = attn_out;
    }

    // 3) FFN path: only valid when Wo_out matches draft_dim. Otherwise, return attn_out.
    if (wo_out_dim != draft_dim || !ffn_layer_) {
        output_hidden = attn_out;
        if (debug_enabled) {
            debug_ffn_out_         = Tensor{};
            debug_pre_head_hidden_ = output_hidden;
        }
        return;
    }

    Tensor ffn_input{{batch_size, draft_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(ffn_input, attn_out, weight_->post_attn_norm, rmsnorm_eps_, stream);

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
    }
}

}  // namespace turbomind
