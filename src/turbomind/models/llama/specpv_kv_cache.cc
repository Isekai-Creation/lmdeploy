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

class PartialKVCache {
public:
    PartialKVCache() = default;

    PartialKVCache(const SpecPVCacheConfig& cfg,
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

    bool is_enabled() const noexcept
    {
        return enabled_;
    }

    int max_batch_size() const noexcept
    {
        return max_batch_size_;
    }

    int num_layers() const noexcept
    {
        return num_layers_;
    }

    int num_kv_heads() const noexcept
    {
        return num_kv_heads_;
    }

    int head_dim() const noexcept
    {
        return head_dim_;
    }

    const SpecPVCacheConfig& config() const noexcept
    {
        return cfg_;
    }

    // Return the number of verified tokens currently tracked for the
    // given layer. For now this is a simple per-layer scalar; more
    // fine-grained accounting can be added later.
    int get_seq_length(int layer_idx = 0) const noexcept
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(verified_lens_.size())) {
            return 0;
        }
        return verified_lens_[layer_idx];
    }

    void set_verified_length(int layer_idx, int len) noexcept
    {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(verified_lens_.size())) {
            return;
        }
        verified_lens_[layer_idx] = len;
        recompute_global_verified_len();
    }

    int global_verified_len() const noexcept
    {
        return global_verified_len_;
    }

    void reset() noexcept
    {
        std::fill(verified_lens_.begin(), verified_lens_.end(), 0);
        global_verified_len_ = 0;
    }

private:
    void recompute_global_verified_len() noexcept
    {
        int v = 0;
        for (int len : verified_lens_) {
            v = std::max(v, len);
        }
        global_verified_len_ = v;
    }

private:
    SpecPVCacheConfig cfg_{};
    int               max_batch_size_{0};
    int               num_layers_{0};
    int               num_kv_heads_{0};
    int               head_dim_{0};
    DataType          kv_dtype_{kFloat32};

    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;

    std::vector<int> verified_lens_;
    int              global_verified_len_{0};
    bool             enabled_{false};
};

}  // namespace turbomind

