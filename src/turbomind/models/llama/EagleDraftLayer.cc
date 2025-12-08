// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/EagleDraftLayer.h"

#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/logger.h"

namespace turbomind {

Eagle3DraftLayer::Eagle3DraftLayer(const Eagle3DraftLayerWeight* weight, LlamaFfnLayer* ffn_layer, float rmsnorm_eps):
    weight_{weight},
    ffn_layer_{ffn_layer},
    rmsnorm_eps_{rmsnorm_eps}
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

    debug_attn_out_         = Tensor{};
    debug_ffn_out_          = Tensor{};
    debug_pre_head_hidden_  = Tensor{};

    const int batch_size = input_hidden.shape(0);
    const int hidden_dim = input_hidden.shape(1);

    if (batch_size <= 0 || hidden_dim <= 0) {
        output_hidden = input_hidden;
        return;
    }

    auto geom_mismatch = [&]() -> bool {
        // Require basic availability and 2D shape for attention and FFN
        // weights. We keep the checks intentionally light here and rely
        // on EagleModule::load for stricter geometry validation.
        if (!weight_->attn.qkv.weight || !weight_->attn.output.weight) {
            return true;
        }
        if (weight_->attn.qkv.weight.ndim() != 2 || weight_->attn.output.weight.ndim() != 2) {
            return true;
        }

        const int wo_out = weight_->attn.output.weight.shape(1);
        if (wo_out != hidden_dim) {
            return true;
        }

        return false;
    };

    if (geom_mismatch()) {
        TM_LOG_WARNING(
            "[EAGLE3][Draft] invalid geometry; skipping Eagle3DraftLayer::Forward");
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
                "[EAGLE][Eagle3DraftLayer][fallback] %s norm mismatch: weight_dtype=%d hidden_dtype=%d "
                "weight_dim=%d hidden_dim=%d; treating draft as pass-through.",
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

    // 2) Single-step draft "attention" using the fused QKV / Wo weights
    // from Eagle3. The current implementation still follows the
    // simplified EagleModule shallow path (no explicit softmax or KV
    // cache) but guards QKV / Wo geometry so any unexpected layout
    // falls back to pass-through.
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

    if (qkv_out_dim <= 0 || q_dim <= 0 || wo_w.shape(1) != hidden_dim) {
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

    Tensor qkv{{batch_size, qkv_out_dim}, dtype, input_hidden.device()};
    linear.Forward(hidden_norm, weight_->attn.qkv, qkv);
    sync_check_cuda_error();

    Tensor value = qkv.slice({0, 0}, {batch_size, q_dim});

    Tensor attn_out{{batch_size, hidden_dim}, dtype, input_hidden.device()};
    linear.Forward(value, weight_->attn.output, attn_out);
    sync_check_cuda_error();
    debug_attn_out_ = attn_out;

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
    debug_ffn_out_ = output_hidden;

    // Residual + output RMSNorm:
    //   y = RMSNorm(attn_out + FFN(norm(attn_out)), output_norm)
    // Here `output_hidden` holds FFN(norm(attn_out)) and input_hidden is
    // the residual from attention.
    invokeResidualBiasRMSNorm(
        /*hidden_states=*/output_hidden.raw_data(),
        /*residual=*/input_hidden.raw_data(),
        /*weights=*/weight_->output_norm.raw_data(),
        /*bias=*/nullptr,
        dtype,
        hidden_dim,
        batch_size,
        rmsnorm_eps_,
        stream);
    debug_pre_head_hidden_ = output_hidden;
}

}  // namespace turbomind
