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
    key_summary_max_.reserve(num_layers_);
    key_summary_min_.reserve(num_layers_);
    verified_lens_.assign(num_layers_, 0);
    summary_block_count_.assign(num_layers_, 0);

    // A conservative upper bound on the number of summary blocks we may
    // need, based on the partial KV budget. This keeps the initial
    // implementation simple; future iterations can tighten it using the
    // engine session length.
    max_seq_len_        = cfg_.total_budget();
    max_summary_blocks_ = cfg_.block_size > 0 ? (max_seq_len_ + cfg_.block_size - 1) / cfg_.block_size : 0;

    for (int layer = 0; layer < num_layers_; ++layer) {
        Tensor k{{max_batch_size_, num_kv_heads_, budget, head_dim_}, kv_dtype_, kDEVICE};
        Tensor v{{max_batch_size_, num_kv_heads_, budget, head_dim_}, kv_dtype_, kDEVICE};
        key_cache_.push_back(std::move(k));
        value_cache_.push_back(std::move(v));

        if (max_summary_blocks_ > 0) {
            Tensor smax{{max_batch_size_, num_kv_heads_, max_summary_blocks_, head_dim_}, kv_dtype_, kCPU};
            Tensor smin{{max_batch_size_, num_kv_heads_, max_summary_blocks_, head_dim_}, kv_dtype_, kCPU};
            key_summary_max_.push_back(std::move(smax));
            key_summary_min_.push_back(std::move(smin));
        }
        else {
            key_summary_max_.push_back(Tensor{});
            key_summary_min_.push_back(Tensor{});
        }
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
    std::fill(summary_block_count_.begin(), summary_block_count_.end(), 0);
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

Tensor PartialKVCache::sink(int layer_idx)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_cache_.size())) {
        return {};
    }
    const int sink_tokens = cfg_.sink_size();
    if (sink_tokens <= 0) {
        return {};
    }
    Tensor& base = key_cache_[layer_idx];
    std::vector<ssize_t> base_idx{0, 0, 0, 0};
    std::vector<ssize_t> shape{base.shape(0), base.shape(1), sink_tokens, base.shape(3)};
    return base.slice(base_idx, shape);
}

Tensor PartialKVCache::retrieval(int layer_idx)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_cache_.size())) {
        return {};
    }
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    if (retrieval_tokens <= 0) {
        return {};
    }
    Tensor& base = key_cache_[layer_idx];
    std::vector<ssize_t> base_idx{0, 0, sink_tokens, 0};
    std::vector<ssize_t> shape{base.shape(0), base.shape(1), retrieval_tokens, base.shape(3)};
    return base.slice(base_idx, shape);
}

Tensor PartialKVCache::window(int layer_idx)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_cache_.size())) {
        return {};
    }
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    if (window_tokens <= 0) {
        return {};
    }
    Tensor& base = key_cache_[layer_idx];
    std::vector<ssize_t> base_idx{0, 0, sink_tokens + retrieval_tokens, 0};
    std::vector<ssize_t> shape{base.shape(0), base.shape(1), window_tokens, base.shape(3)};
    return base.slice(base_idx, shape);
}

Tensor PartialKVCache::buffer(int layer_idx)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_cache_.size())) {
        return {};
    }
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    const int buffer_tokens    = cfg_.buffer_size();
    if (buffer_tokens <= 0) {
        return {};
    }
    const int start = sink_tokens + retrieval_tokens + window_tokens;

    Tensor& base = key_cache_[layer_idx];
    std::vector<ssize_t> base_idx{0, 0, start, 0};
    std::vector<ssize_t> shape{base.shape(0), base.shape(1), buffer_tokens, base.shape(3)};
    return base.slice(base_idx, shape);
}

void PartialKVCache::summary_key_states(int layer_idx, const Tensor& key_states, int seq_len)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_summary_max_.size())) {
        return;
    }
    if (!key_summary_max_[layer_idx] || !key_summary_min_[layer_idx] || cfg_.block_size <= 0) {
        return;
    }
    const int sink_tokens = cfg_.sink_size();
    if (seq_len <= sink_tokens) {
        return;
    }

    const int total_tokens    = std::min(seq_len, max_seq_len_);
    const int existing_blocks = summary_block_count_[layer_idx];
    const int new_tokens      = std::max(0, total_tokens - sink_tokens);
    const int expected_blocks = std::max(0, new_tokens / cfg_.block_size);

    if (expected_blocks <= existing_blocks || expected_blocks > max_summary_blocks_) {
        return;
    }

    // For now, just advance the block count to the expected value. A later
    // revision will materialize true Kmax/Kmin summaries using either host
    // loops or dedicated CUDA kernels.
    summary_block_count_[layer_idx] = expected_blocks;
}

void PartialKVCache::refresh_retrieval(int         layer_idx,
                                       const Tensor& /*query_states*/,
                                       const Tensor& /*key_states*/,
                                       const Tensor& /*value_states*/,
                                       int           /*seq_len*/)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(summary_block_count_.size())) {
        return;
    }

    // Placeholder implementation: retrieval/window content is not yet
    // updated from full KV. Behaviourally this keeps the partial KV view
    // as an empty prefix, so SpecPV does not alter attention semantics
    // until the full retrieval kernels are implemented.
}

std::pair<Tensor, Tensor> PartialKVCache::update(int layer_idx,
                                                 const Tensor& new_keys,
                                                 const Tensor& new_values)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_cache_.size())) {
        return {Tensor{}, Tensor{}};
    }

    const int buffer_tokens = cfg_.buffer_size();
    if (buffer_tokens <= 0) {
        return {Tensor{}, Tensor{}};
    }

    const int cur_len   = verified_lens_[layer_idx];
    const int new_len   = new_keys.shape(2);  // tokens dimension
    const int next_len  = cur_len + new_len;
    const int max_tokens = buffer_tokens;

    if (new_len <= 0 || cur_len >= max_tokens) {
        return {Tensor{}, Tensor{}};
    }

    // Clamp to available buffer capacity; no data movement yet.
    const int committed = std::min(new_len, max_tokens - cur_len);
    verified_lens_[layer_idx] = cur_len + committed;
    recompute_global_verified_len();

    // Active KV view currently covers the entire allocated budget; a more
    // precise view (up to sink+retrieval+window+verified_len) can be wired
    // once SpecPV is fully integrated into attention.
    Tensor k_active = key_cache_[layer_idx];
    Tensor v_active = value_cache_[layer_idx];
    return {k_active, v_active};
}

void PartialKVCache::reset_buffer()
{
    std::fill(verified_lens_.begin(), verified_lens_.end(), 0);
    recompute_global_verified_len();
}

}  // namespace turbomind
