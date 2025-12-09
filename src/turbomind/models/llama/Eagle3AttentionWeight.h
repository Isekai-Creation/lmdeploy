// Copyright (c) OpenMMLab. All rights reserved.
// Lightweight Eagle-3 attention weight holder. This mirrors the
// Eagle-3 midlayer.self_attn.{q,k,v,o}_proj geometry from HF/TRT
// instead of the standard LLaMA fused QKV layout.

#pragma once

#include "src/turbomind/core/core.h"

namespace turbomind {

struct Eagle3AttentionWeight {
    // Pre-FC is handled separately (fc.weight), so this struct only
    // owns the midlayer.self_attn.{q,k,v,o}_proj weights.
    //
    // Expected Eagle-3 GPT-OSS shapes (for reference):
    //   q_proj: [q_out, q_in]   = [4096, 2880]
    //   k_proj: [kv_out, q_in]  = [ 512, 2880]
    //   v_proj: [kv_out, q_in]  = [ 512, 2880]
    //   o_proj: [q_in, q_out]   = [2880, 4096]

    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;

    int q_out{0};       // total Q width (e.g. 4096)
    int kv_out{0};      // total KV width (e.g. 512)
    int q_in{0};        // input hidden size (e.g. 2880)

    int num_q_heads{0};
    int num_kv_heads{0};
    int head_dim{0};
    float rope_base{10000.f};
    float rope_scale{1.f};

    bool is_initialized{false};

    Eagle3AttentionWeight() = default;

    void init_from_tensors(const Tensor& q,
                           const Tensor& k,
                           const Tensor& v,
                           const Tensor& o)
    {
        q_proj = q;
        k_proj = k;
        v_proj = v;
        o_proj = o;

        q_out  = q_proj ? q_proj.shape(0) : 0;
        q_in   = q_proj && q_proj.ndim() == 2 ? q_proj.shape(1) : 0;
        kv_out = k_proj ? k_proj.shape(0) : 0;

        // num_q_heads / num_kv_heads / head_dim should be derived from
        // Eagle-3 config (TensorRT-LLM / HF). For now they remain zero
        // and are filled once we port the exact head layout.

        is_initialized = q_proj && k_proj && v_proj && o_proj && q_out > 0 && q_in > 0 && kv_out > 0;
    }
};

}  // namespace turbomind
