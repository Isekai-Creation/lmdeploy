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
    debug_pre_head_hidden_{},
    head_num_{0},
    kv_head_num_{0},
    size_per_head_{0}
{
    if (weight_) {
        head_num_      = weight_->attn.head_num;
        kv_head_num_   = weight_->attn.kv_head_num;
        size_per_head_ = weight_->attn.size_per_head;
    }
}

void Eagle3DraftLayer::Forward(const Tensor& input_hidden,
                               Tensor&       output_hidden,
                               cudaStream_t  stream)
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

    // 2) Attention: prefer real UnifiedAttentionLayer if available and geometry sane,
    //    otherwise fall back to shallow QKV+V→Wo path.
    Tensor attn_out{{batch_size, hidden_dim}, dtype, input_hidden.device()};

    const bool has_attn_layer = (attn_layer_ != nullptr);

    auto attn_geom_ok = [&]() -> bool {
        if (head_num_ <= 0 || size_per_head_ <= 0) {
            return false;
        }
        // Standard MHA: hidden_dim = head_num * size_per_head
        if (hidden_dim != head_num_ * size_per_head_) {
            return false;
        }
        return true;
    };

    bool used_unified_attention = false;

    if (has_attn_layer && attn_geom_ok()) {
        UnifiedAttentionLayer::ForwardParam param{};
        param.input    = hidden_norm;
        param.output   = attn_out;
        param.weights  = &weight_->attn;
        param.layer_id = 0;  // no layer-specific KV for draft v1

        attn_layer_->Forward(param);
        sync_check_cuda_error();

        used_unified_attention = true;
    }
    else {
        if (has_attn_layer && !attn_geom_ok()) {
            TM_LOG_WARNING(
                "[EAGLE3][Draft] invalid attention geometry (head_num=%d, size_per_head=%d, hidden_dim=%d); "
                "falling back to shallow QKV path.",
                head_num_,
                size_per_head_,
                hidden_dim);
        }

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

        const int v_offset = q_dim + kv_dim;
        Tensor     value   = qkv.slice({0, v_offset}, {batch_size, kv_dim});

        linear.Forward(value, weight_->attn.output, attn_out);
        sync_check_cuda_error();
    }

    if (debug_enabled) {
        debug_attn_out_ = attn_out;
    }

    // 3) FFN path: post-attention norm, gated MLP, and final residual norm.
    Tensor ffn_input{{batch_size, hidden_dim}, dtype, input_hidden.device()};
    invokeRMSNorm(ffn_input, attn_out, weight_->post_attn_norm, rmsnorm_eps_, stream);

    if (!output_hidden || output_hidden.ndim() != 2 || output_hidden.shape(0) != batch_size
        || output_hidden.shape(1) != hidden_dim || output_hidden.dtype() != dtype
        || output_hidden.device().type != input_hidden.device().type) {
        output_hidden = Tensor{{batch_size, hidden_dim}, dtype, input_hidden.device()};
    }

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
        if (used_unified_attention) {
            TM_LOG_DEBUG("[EAGLE3][Draft] UnifiedAttentionLayer used for Eagle3 draft attention.");
        }
    }
}

}  // namespace turbomind
