// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/EagleDraftLayer.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/eagle_debug.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

Eagle3DraftLayer::Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight,
                                   UnifiedAttentionLayer*        attn_layer,
                                   LlamaFfnLayer*                ffn_layer,
                                   float                         rmsnorm_eps):
    weight_{weight},
    attn_layer_{attn_layer},
    ffn_layer_{ffn_layer},
    rmsnorm_eps_{rmsnorm_eps},
    debug_fc_out_{},
    debug_attn_out_{},
    debug_ffn_out_{},
    debug_pre_head_hidden_{}
{
}

void Eagle3DraftLayer::Forward(const Tensor& input_hidden, Tensor& output_hidden, cudaStream_t stream)
{
    if (!input_hidden) {
        return;
    }

    if (!weight_ || !ffn_layer_) {
        TM_LOG_WARNING(
            "[EAGLE][Eagle3DraftLayer][fallback] draft layer weights or FFN layer unavailable; "
            "treating Eagle3 draft as pass-through.");
        output_hidden = input_hidden;
        return;
    }

    const bool debug_enabled = isEagleDebugEnabled();

    debug_fc_out_          = Tensor{};
    debug_attn_out_        = Tensor{};
    debug_ffn_out_         = Tensor{};
    debug_pre_head_hidden_ = Tensor{};

    const int batch_size = input_hidden.shape(0);
    const int hidden_dim = input_hidden.shape(1);

    if (batch_size <= 0 || hidden_dim <= 0) {
        output_hidden = input_hidden;
        return;
    }

    const auto dtype = input_hidden.dtype();

    auto norm_mismatch = [&](const Tensor& w, const char* name) -> bool {
        if (!w) {
            return false;
        }
        if (w.dtype() != dtype || w.ndim() != 1 || w.shape(0) != hidden_dim) {
            TM_LOG_WARNING(
                "[EAGLE][Eagle3DraftLayer][fallback] %s norm mismatch: "
                "weight_dtype=%d hidden_dtype=%d weight_dim=%d hidden_dim=%d; treating draft as pass-through.",
                name,
                static_cast<int>(w.dtype()),
                static_cast<int>(dtype),
                w.shape(0),
                hidden_dim);
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
    Tensor hidden_norm{{batch_size, hidden_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(hidden_norm, input_hidden, weight_->input_norm, rmsnorm_eps_, stream);

    if (debug_enabled) {
        debug_fc_out_ = hidden_norm;
    }

    // 2) Attention: prefer real UnifiedAttentionLayer if available and geometry is sane,
    //    otherwise fall back to the shallow QKV+V→Wo path.
    Tensor attn_out{{batch_size, hidden_dim}, dtype, input_hidden.device()};

    // For now we always use the shallow QKV path that mirrors the
    // existing Eagle3 attention approximation. Once UnifiedAttentionLayer
    // is fully wired for the draft layer we can switch over to the real
    // attention backend here.
    LlamaLinear& linear = ffn_layer_->linear();

    const Tensor& qkv_w = weight_->attn.qkv.weight;
    const Tensor& wo_w  = weight_->attn.output.weight;

    if (!qkv_w || !wo_w || qkv_w.ndim() != 2 || wo_w.ndim() != 2) {
        TM_LOG_WARNING(
            "[EAGLE][Eagle3DraftLayer][fallback] invalid QKV/WO tensors in Forward; "
            "treating Eagle3 draft as pass-through.");
        output_hidden = input_hidden;
        return;
    }

    const int qkv_out_dim = qkv_w.shape(1);
    const int q_dim       = wo_w.shape(0);
    const int kv_span     = qkv_out_dim - q_dim;

    if (qkv_out_dim <= 0 || q_dim <= 0 || kv_span <= 0 || (kv_span % 2) != 0 || wo_w.shape(1) != hidden_dim) {
        TM_LOG_WARNING(
            "[EAGLE][Eagle3DraftLayer][fallback] invalid QKV/WO geometry in Forward "
            "(qkv=[%d,%d], wo=[%d,%d], hidden_dim=%d); treating draft as pass-through.",
            qkv_w.shape(0),
            qkv_w.shape(1),
            wo_w.shape(0),
            wo_w.shape(1),
            hidden_dim);
        output_hidden = input_hidden;
        return;
    }

    const int kv_dim = kv_span / 2;
    if (kv_dim != q_dim) {
        TM_LOG_WARNING(
            "[EAGLE][Eagle3DraftLayer][fallback] unsupported Eagle3 QKV layout in Forward "
            "(q_dim=%d, kv_dim=%d); treating draft as pass-through.",
            q_dim,
            kv_dim);
        output_hidden = input_hidden;
        return;
    }

    Tensor qkv{{batch_size, qkv_out_dim}, dtype, input_hidden.device()};
    linear.Forward(hidden_norm, weight_->attn.qkv, qkv);
    sync_check_cuda_error();

    // QKV layout is [Q, K, V] with sizes [q_dim, kv_dim, kv_dim]. For a
    // single-position attention block, the pre-Wo attention output is
    // equivalent to V, so we select the final kv_dim slice here.
    const int v_offset = q_dim + kv_dim;
    Tensor     value   = qkv.slice({0, v_offset}, {batch_size, kv_dim});

    linear.Forward(value, weight_->attn.output, attn_out);
    sync_check_cuda_error();

    if (debug_enabled) {
        debug_attn_out_ = attn_out;
    }

    // 3) FFN path: post-attention norm, gated MLP, and final residual norm.
    Tensor ffn_input{{batch_size, hidden_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(ffn_input, attn_out, weight_->post_attn_norm, rmsnorm_eps_, stream);

    // Ensure output buffer is allocated with the correct shape.
    if (!output_hidden || output_hidden.ndim() != 2 || output_hidden.shape(0) != batch_size
        || output_hidden.shape(1) != hidden_dim || output_hidden.dtype() != dtype
        || output_hidden.device().type != input_hidden.device().type) {
        output_hidden = Tensor{{batch_size, hidden_dim}, dtype, input_hidden.device()};
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

    // Residual + output RMSNorm:
    //   y = RMSNorm(attn_out + FFN(norm(attn_out)), output_norm)
    // Here `output_hidden` holds FFN(norm(attn_out)) and `input_hidden` is
    // the residual from attention.
    void* residual_ptr = const_cast<void*>(input_hidden.raw_data());
    invokeResidualBiasRMSNorm(
        /*hidden_states=*/output_hidden.raw_data(),
        /*residual=*/residual_ptr,
        /*weights=*/weight_->output_norm.raw_data(),
        /*bias=*/nullptr,
        dtype,
        hidden_dim,
        batch_size,
        rmsnorm_eps_,
        stream);

    if (debug_enabled) {
        debug_pre_head_hidden_ = output_hidden;
    }
}

}  // namespace turbomind
