// Copyright (c) OpenMMLab. All rights reserved.
//
// Minimal SpecPV-style partial KV cache scaffolding for TurboMind.
//
// This first iteration focuses on owning a per-layer KV buffer with the
// expected sink/retrieval/window/buffer capacity and exposing a small
// API surface that LlamaV2 can use for gating and basic bookkeeping.
// The more advanced summary/retrieval logic from SpecPV will be wired
// on top of this container in subsequent steps.

#include "src/turbomind/models/llama/specpv_kv_cache.h"

namespace turbomind {

PartialKVCache::PartialKVCache(const SpecPVCacheConfig& cfg,
                               int                      max_batch_size,
                               int                      num_layers,
                               int                      num_kv_heads,
                               int                      head_dim,
                               DataType                 kv_dtype):
    cfg_{cfg},
    max_batch_size_{max_batch_size},
    num_layers_{num_layers},
    num_kv_heads_{num_kv_heads},
    head_dim_{head_dim},
    kv_dtype_{kv_dtype}
{
    const int budget = cfg_.total_budget();
    if (budget <= 0 || max_batch_size_ <= 0 || num_layers_ <= 0 || num_kv_heads_ <= 0 || head_dim_ <= 0) {
        return;
    }

    // Allocate a contiguous KV buffer per layer with shape:
    //   [max_batch_size, num_kv_heads, total_budget, head_dim]
    // The cache is logically split into sink / retrieval / window /
    // buffer segments via SpecPVCacheConfig; we defer explicit slice
    // helpers until the corresponding kernels are in place.
    key_cache_.reserve(num_layers_);
    value_cache_.reserve(num_layers_);
    verified_lens_.assign(num_layers_, 0);

    for (int layer = 0; layer < num_layers_; ++layer) {
        Tensor k{{max_batch_size_, num_kv_heads_, budget, head_dim_}, kv_dtype_, kDEVICE};
        Tensor v{{max_batch_size_, num_kv_heads_, budget, head_dim_}, kv_dtype_, kDEVICE};
        key_cache_.push_back(std::move(k));
        value_cache_.push_back(std::move(v));
    }

    enabled_ = true;
}

int PartialKVCache::get_seq_length(int layer_idx) const noexcept
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(verified_lens_.size())) {
        return 0;
    }
    return verified_lens_[layer_idx];
}

void PartialKVCache::set_verified_length(int layer_idx, int len) noexcept
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(verified_lens_.size())) {
        return;
    }
    verified_lens_[layer_idx] = len;
    recompute_global_verified_len();
}

void PartialKVCache::reset() noexcept
{
    std::fill(verified_lens_.begin(), verified_lens_.end(), 0);
    global_verified_len_ = 0;
}

void PartialKVCache::recompute_global_verified_len() noexcept
{
    int v = 0;
    for (int len : verified_lens_) {
        v = std::max(v, len);
    }
    global_verified_len_ = v;
}

}  // namespace turbomind
