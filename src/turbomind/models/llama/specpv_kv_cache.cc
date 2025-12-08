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

#include <limits>

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

Tensor PartialKVCache::slice_tokens(std::vector<Tensor>& cache, int layer_idx, int token_start, int token_count)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(cache.size())) {
        return {};
    }
    if (token_count <= 0) {
        return {};
    }

    Tensor& base = cache[layer_idx];
    if (!base) {
        return {};
    }

    const int L = static_cast<int>(base.shape(2));
    if (token_start < 0 || token_start >= L) {
        return {};
    }

    const int clamped_count = std::min(token_count, L - token_start);
    if (clamped_count <= 0) {
        return {};
    }

    std::vector<ssize_t> base_idx{0, 0, token_start, 0};
    std::vector<ssize_t> shape{base.shape(0), base.shape(1), clamped_count, base.shape(3)};
    return base.slice(base_idx, shape);
}

Tensor PartialKVCache::sink(int layer_idx)
{
    const int sink_tokens = cfg_.sink_size();
    return slice_tokens(key_cache_, layer_idx, 0, sink_tokens);
}

Tensor PartialKVCache::retrieval(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    return slice_tokens(key_cache_, layer_idx, sink_tokens, retrieval_tokens);
}

Tensor PartialKVCache::window(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    return slice_tokens(key_cache_, layer_idx, sink_tokens + retrieval_tokens, window_tokens);
}

Tensor PartialKVCache::buffer(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    const int buffer_tokens    = cfg_.buffer_size();
    const int start            = sink_tokens + retrieval_tokens + window_tokens;
    return slice_tokens(key_cache_, layer_idx, start, buffer_tokens);
}

Tensor PartialKVCache::sink_v(int layer_idx)
{
    const int sink_tokens = cfg_.sink_size();
    return slice_tokens(value_cache_, layer_idx, 0, sink_tokens);
}

Tensor PartialKVCache::retrieval_v(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    return slice_tokens(value_cache_, layer_idx, sink_tokens, retrieval_tokens);
}

Tensor PartialKVCache::window_v(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    return slice_tokens(value_cache_, layer_idx, sink_tokens + retrieval_tokens, window_tokens);
}

Tensor PartialKVCache::buffer_v(int layer_idx)
{
    const int sink_tokens      = cfg_.sink_size();
    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = cfg_.window_size();
    const int buffer_tokens    = cfg_.buffer_size();
    const int start            = sink_tokens + retrieval_tokens + window_tokens;
    return slice_tokens(value_cache_, layer_idx, start, buffer_tokens);
}

void PartialKVCache::summary_key_states(int layer_idx, const Tensor& key_states, int seq_len)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(key_summary_max_.size())) {
        return;
    }
    if (!key_summary_max_[layer_idx] || !key_summary_min_[layer_idx] || cfg_.block_size <= 0) {
        return;
    }
    if (kv_dtype_ != kFloat32 || key_states.dtype() != kFloat32) {
        // Only float32 summaries are supported in the initial implementation.
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

    const int B = static_cast<int>(key_states.shape(0));
    const int H = static_cast<int>(key_states.shape(1));
    const int L = static_cast<int>(key_states.shape(2));
    const int D = static_cast<int>(key_states.shape(3));

    if (L <= sink_tokens || D != head_dim_) {
        return;
    }

    const int max_B = std::min(B, max_batch_size_);
    const int max_H = std::min(H, num_kv_heads_);

    // Copy the full [B,H,L,D] keys to host once; summaries are small compared
    // to the main KV so this is acceptable for a first implementation.
    Tensor host_keys{{B, H, L, D}, kv_dtype_, kCPU};
    core::Copy(key_states, host_keys);

    float* k_host = host_keys.data<float>();
    float* smax   = key_summary_max_[layer_idx].data<float>();
    float* smin   = key_summary_min_[layer_idx].data<float>();

    const int summary_blocks = max_summary_blocks_;

    for (int b = existing_blocks; b < expected_blocks; ++b) {
        const int start = sink_tokens + b * cfg_.block_size;
        const int end   = std::min(start + cfg_.block_size, total_tokens);
        const int len   = std::max(0, end - start);
        if (len <= 0) {
            continue;
        }

        for (int bb = 0; bb < max_B; ++bb) {
            for (int hh = 0; hh < max_H; ++hh) {
                for (int d = 0; d < D; ++d) {
                    float vmax = -std::numeric_limits<float>::infinity();
                    float vmin = std::numeric_limits<float>::infinity();

                    for (int t = 0; t < len; ++t) {
                        const int token_idx = start + t;
                        if (token_idx >= L) {
                            break;
                        }

                        const ssize_t idx =
                            (((static_cast<ssize_t>(bb) * H + hh) * L + token_idx) * D) + d;
                        const float v = k_host[idx];
                        vmax          = std::max(vmax, v);
                        vmin          = std::min(vmin, v);
                    }

                    const ssize_t sum_idx =
                        (((static_cast<ssize_t>(bb) * num_kv_heads_ + hh) * summary_blocks + b) * D) + d;
                    smax[sum_idx] = vmax;
                    smin[sum_idx] = vmin;
                }
            }
        }
    }

    summary_block_count_[layer_idx] = expected_blocks;
}

void PartialKVCache::refresh_retrieval(int         layer_idx,
                                       const Tensor& query_states,
                                       const Tensor& key_states,
                                       const Tensor& value_states,
                                       int           seq_len)
{
    if (layer_idx < 0 || layer_idx >= static_cast<int>(summary_block_count_.size())) {
        return;
    }
    if (kv_dtype_ != kFloat32 || key_states.dtype() != kFloat32 || value_states.dtype() != kFloat32
        || query_states.dtype() != kFloat32) {
        // Float32-only implementation for now.
        return;
    }

    const int blocks = summary_block_count_[layer_idx];
    if (blocks <= 0 || cfg_.block_size <= 0) {
        return;
    }

    const int B = static_cast<int>(key_states.shape(0));
    const int H = static_cast<int>(key_states.shape(1));
    const int L = std::min(static_cast<int>(key_states.shape(2)), max_seq_len_);
    const int D = static_cast<int>(key_states.shape(3));

    if (D != head_dim_ || L <= 0) {
        return;
    }

    const int max_B = std::min(B, max_batch_size_);
    const int max_H = std::min(H, num_kv_heads_);

    const int Q = static_cast<int>(query_states.shape(2));
    if (Q <= 0 || query_states.shape(0) != B || query_states.shape(1) != H
        || query_states.shape(3) != D) {
        return;
    }

    const int R_blocks = std::min(cfg_.n_retrieval_blocks, blocks);
    if (R_blocks <= 0) {
        return;
    }

    // Host copies for queries and full K/V.
    Tensor host_q{{B, H, Q, D}, kv_dtype_, kCPU};
    Tensor host_k{{B, H, L, D}, kv_dtype_, kCPU};
    Tensor host_v{{B, H, L, D}, kv_dtype_, kCPU};

    core::Copy(query_states, host_q);
    core::Copy(key_states, host_k);
    core::Copy(value_states, host_v);

    float* q_host   = host_q.data<float>();
    float* k_host   = host_k.data<float>();
    float* v_host   = host_v.data<float>();
    float* smax_buf = key_summary_max_[layer_idx].data<float>();
    float* smin_buf = key_summary_min_[layer_idx].data<float>();

    const int summary_blocks = max_summary_blocks_;

    // Per-[B,H] scores over blocks.
    std::vector<float> scores(static_cast<size_t>(blocks));
    std::vector<int>   top_indices(static_cast<size_t>(blocks));

    const int retrieval_tokens = cfg_.retrieval_size();
    const int window_tokens    = std::min(cfg_.window_size(), L);

    Tensor retrieval_k_t = retrieval(layer_idx);
    Tensor retrieval_v_t = retrieval_v(layer_idx);
    Tensor window_k_t    = window(layer_idx);
    Tensor window_v_t    = window_v(layer_idx);

    if (!retrieval_k_t || !retrieval_v_t || !window_k_t || !window_v_t) {
        return;
    }

    Tensor host_retrieval_k{retrieval_k_t.layout(), retrieval_k_t.dtype(), kCPU};
    Tensor host_retrieval_v{retrieval_v_t.layout(), retrieval_v_t.dtype(), kCPU};
    Tensor host_window_k{window_k_t.layout(), window_k_t.dtype(), kCPU};
    Tensor host_window_v{window_v_t.layout(), window_v_t.dtype(), kCPU};

    float* retr_k_host = host_retrieval_k.data<float>();
    float* retr_v_host = host_retrieval_v.data<float>();
    float* win_k_host  = host_window_k.data<float>();
    float* win_v_host  = host_window_v.data<float>();

    for (int bb = 0; bb < max_B; ++bb) {
        for (int hh = 0; hh < max_H; ++hh) {
            // Compute scores per block.
            for (int blk = 0; blk < blocks; ++blk) {
                float best = -std::numeric_limits<float>::infinity();

                for (int qq = 0; qq < Q; ++qq) {
                    float dot_max = 0.f;
                    float dot_min = 0.f;

                    for (int d = 0; d < D; ++d) {
                        const ssize_t q_idx =
                            (((static_cast<ssize_t>(bb) * H + hh) * Q + qq) * D) + d;
                        const ssize_t s_idx =
                            (((static_cast<ssize_t>(bb) * num_kv_heads_ + hh) * summary_blocks
                              + blk)
                             * D)
                            + d;
                        const float qv = q_host[q_idx];
                        dot_max += qv * smax_buf[s_idx];
                        dot_min += qv * smin_buf[s_idx];
                    }

                    const float s = std::max(dot_max, dot_min);
                    if (s > best) {
                        best = s;
                    }
                }

                scores[blk]    = best;
                top_indices[blk] = blk;
            }

            // Select top R_blocks by score (simple partial sort).
            const int K = R_blocks;
            std::partial_sort(top_indices.begin(),
                              top_indices.begin() + K,
                              top_indices.begin() + blocks,
                              [&](int a, int b) { return scores[a] > scores[b]; });

            // Gather retrieval tokens: pack selected blocks sequentially.
            const int tokens_per_block = cfg_.block_size;
            const int max_retr_tokens  = retrieval_tokens;

            for (int r = 0; r < K; ++r) {
                const int blk     = top_indices[r];
                const int srcbase = blk * tokens_per_block;

                for (int t = 0; t < tokens_per_block; ++t) {
                    const int src_token = srcbase + t;
                    if (src_token >= L) {
                        break;
                    }
                    const int dst_token = r * tokens_per_block + t;
                    if (dst_token >= max_retr_tokens) {
                        break;
                    }

                    for (int d = 0; d < D; ++d) {
                        const ssize_t src_idx =
                            (((static_cast<ssize_t>(bb) * H + hh) * L + src_token) * D) + d;
                        const ssize_t dst_idx =
                            (((static_cast<ssize_t>(bb) * num_kv_heads_ + hh) * max_retr_tokens
                              + dst_token)
                             * D)
                            + d;
                        retr_k_host[dst_idx] = k_host[src_idx];
                        retr_v_host[dst_idx] = v_host[src_idx];
                    }
                }
            }

            // Gather window tokens: last window_tokens from full KV.
            if (window_tokens > 0) {
                const int win_start = L - window_tokens;
                for (int t = 0; t < window_tokens; ++t) {
                    const int src_token = win_start + t;
                    const int dst_token = t;
                    for (int d = 0; d < D; ++d) {
                        const ssize_t src_idx =
                            (((static_cast<ssize_t>(bb) * H + hh) * L + src_token) * D) + d;
                        const ssize_t dst_idx =
                            (((static_cast<ssize_t>(bb) * num_kv_heads_ + hh) * window_tokens
                              + dst_token)
                             * D)
                            + d;
                        win_k_host[dst_idx] = k_host[src_idx];
                        win_v_host[dst_idx] = v_host[src_idx];
                    }
                }
            }
        }
    }

    // Copy retrieval/window back to device for both K and V.
    core::Copy(host_retrieval_k, retrieval_k_t);
    core::Copy(host_retrieval_v, retrieval_v_t);
    core::Copy(host_window_k, window_k_t);
    core::Copy(host_window_v, window_v_t);
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

    const int B = static_cast<int>(new_keys.shape(0));
    const int H = static_cast<int>(new_keys.shape(1));
    const int D = static_cast<int>(new_keys.shape(3));

    if (D != head_dim_) {
        return {Tensor{}, Tensor{}};
    }

    const int max_B = std::min(B, max_batch_size_);
    const int max_H = std::min(H, num_kv_heads_);

    const int committed = std::min(new_len, max_tokens - cur_len);
    if (committed <= 0) {
        return {Tensor{}, Tensor{}};
    }

    // Write into the buffer segment for this layer.
    Tensor buf_k = buffer(layer_idx);
    Tensor buf_v = buffer(layer_idx);
    if (!buf_k || !buf_v) {
        return {Tensor{}, Tensor{}};
    }

    std::vector<ssize_t> base_idx{0, 0, cur_len, 0};
    std::vector<ssize_t> shape{max_B, max_H, committed, D};

    Tensor dst_k = buf_k.slice(base_idx, shape);
    Tensor dst_v = buf_v.slice(base_idx, shape);

    Tensor src_k = new_keys.slice({0, 0, 0, 0}, shape);
    Tensor src_v = new_values.slice({0, 0, 0, 0}, shape);

    core::Copy(src_k, dst_k);
    core::Copy(src_v, dst_v);

    verified_lens_[layer_idx] = cur_len + committed;
    recompute_global_verified_len();

    // Active KV view covers sink+retrieval+window+verified buffer tokens.
    const int prefix_tokens = cfg_.sink_size() + cfg_.retrieval_size() + cfg_.window_size();
    const int active_tokens = std::min(prefix_tokens + verified_lens_[layer_idx],
                                       cfg_.total_budget());

    Tensor& base_k = key_cache_[layer_idx];
    Tensor& base_v = value_cache_[layer_idx];

    std::vector<ssize_t> base0{0, 0, 0, 0};
    std::vector<ssize_t> shape_active{max_B, max_H, active_tokens, D};

    Tensor k_active = base_k.slice(base0, shape_active);
    Tensor v_active = base_v.slice(base0, shape_active);
    return {k_active, v_active};
}

void PartialKVCache::reset_buffer()
{
    for (int layer = 0; layer < num_layers_; ++layer) {
        Tensor buf_k = buffer(layer);
        Tensor buf_v = buffer_v(layer);
        if (buf_k) {
            core::Clear(buf_k);
        }
        if (buf_v) {
            core::Clear(buf_v);
        }
        verified_lens_[layer] = 0;
    }
    recompute_global_verified_len();
}

}  // namespace turbomind
