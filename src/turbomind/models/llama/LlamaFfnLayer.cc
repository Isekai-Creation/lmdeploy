/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.h

#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/eagle_debug.h"

namespace turbomind {

void LlamaFfnLayer::forward(ForwardParam param)
{
    NvtxScope scope("ffn");

    const auto& mlp = *param.weights;

    const bool is_eagle3_ffn =
        (mlp.debug_name != nullptr) && std::strncmp(mlp.debug_name, "EAGLE3", 6) == 0;
    const bool log_gemm_shapes =
        is_eagle3_ffn && isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG");
    const int dtype_int = static_cast<int>(mlp.output.data_type);

    const int token_num  = param.input.shape(0);
    const int inter_size = mlp.inter_size;
    const int layer_id   = param.layer_id;

    const auto stream = core::Context::stream().handle();

    Tensor gating;
    Tensor inter;

    if (mlp.fused_gating_intermediate.weight) {
        if (log_gemm_shapes) {
            const int m = token_num;
            const int k = mlp.fused_gating_intermediate.input_dim;
            const int n = mlp.fused_gating_intermediate.output_dim;
            logEagleGemmShape("EAGLE3_FFN_FUSED_GATE_UP", m, k, n, dtype_int, "row_major");
        }
        EagleGemmTagGuard tag_guard(is_eagle3_ffn ? "EAGLE3_FFN" : nullptr);
        auto              mix = linear_.Forward(param.input, mlp.fused_gating_intermediate);
        sync_check_cuda_error();

        gating = mix.slice({0, 0}, {(int)token_num, inter_size});
        if (!mlp.is_fused_silu) {
            inter = mix.slice({0, inter_size}, {(ssize_t)token_num, inter_size});
        }
    }
    else {
        if (log_gemm_shapes && mlp.gating) {
            logEagleGemmShape("EAGLE3_FFN_GATE",
                              token_num,
                              mlp.gating.input_dim,
                              mlp.gating.output_dim,
                              dtype_int,
                              "row_major");
        }
        {
            EagleGemmTagGuard tag_guard(is_eagle3_ffn ? "EAGLE3_FFN" : nullptr);
            gating = linear_.Forward(param.input, mlp.gating);
        }
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(gating, Concat("w1", layer_id), 3);

        if (log_gemm_shapes && mlp.intermediate) {
            logEagleGemmShape("EAGLE3_FFN_UP",
                              token_num,
                              mlp.intermediate.input_dim,
                              mlp.intermediate.output_dim,
                              dtype_int,
                              "row_major");
        }
        {
            EagleGemmTagGuard tag_guard(is_eagle3_ffn ? "EAGLE3_FFN" : nullptr);
            inter = linear_.Forward(param.input, mlp.intermediate);
        }
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(inter, Concat("w3", layer_id), 3);
    }

    if (!mlp.is_fused_silu) {
        // gate' = silu(gate) * up
        Activation(gating, inter, mlp.act_type, stream);
        sync_check_cuda_error();
        TM_DEBUG_TENSOR(gating, Concat("act", layer_id), 3);
    }

    {  // w2(x)
        NvtxScope scope("w2");
        if (log_gemm_shapes && mlp.output) {
            logEagleGemmShape("EAGLE3_FFN_DOWN",
                              token_num,
                              mlp.output.input_dim,
                              mlp.output.output_dim,
                              dtype_int,
                              "row_major");
        }
        EagleGemmTagGuard tag_guard(is_eagle3_ffn ? "EAGLE3_FFN" : nullptr);
        linear_.Forward(gating, mlp.output, param.output);
        sync_check_cuda_error();
    }
}

}  // namespace turbomind
