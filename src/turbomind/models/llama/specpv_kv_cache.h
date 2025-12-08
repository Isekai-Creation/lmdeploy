// Copyright (c) OpenMMLab. All rights reserved.
//
// SpecPV-style partial KV cache configuration and forward declarations.
//
// This header intentionally keeps the core configuration struct small and
// self-contained. The full PartialKVCache implementation lives in the
// corresponding specpv_kv_cache.cc file.

#pragma once

#include <algorithm>
#include <vector>

#include "src/turbomind/core/core.h"

namespace turbomind {

struct SpecPVCacheConfig {
    int block_size{0};
    int n_sink_blocks{0};
    int n_retrieval_blocks{0};
    int n_window_blocks{0};
    int n_spec_tokens_buf{0};

    int sink_size() const
    {
        return n_sink_blocks * block_size;
    }

    int retrieval_size() const
    {
        return n_retrieval_blocks * block_size;
    }

    int window_size() const
    {
        return n_window_blocks * block_size;
    }

    int buffer_size() const
    {
        return n_spec_tokens_buf;
    }

    int total_budget() const
    {
        return sink_size() + retrieval_size() + window_size() + buffer_size();
    }
};

class PartialKVCache {
public:
    PartialKVCache() = default;

    PartialKVCache(const SpecPVCacheConfig& cfg,
                   int                      max_batch_size,
                   int                      num_layers,
                   int                      num_kv_heads,
                   int                      head_dim,
                   DataType                 kv_dtype);

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
    int get_seq_length(int layer_idx = 0) const noexcept;

    void set_verified_length(int layer_idx, int len) noexcept;

    int global_verified_len() const noexcept
    {
        return global_verified_len_;
    }

    void reset() noexcept;

    // Segment views over the per-layer KV buffers. These are lightweight
    // Tensor slices; they do not allocate or copy.
    Tensor sink(int layer_idx);     // K view
    Tensor retrieval(int layer_idx);
    Tensor window(int layer_idx);
    Tensor buffer(int layer_idx);

    Tensor sink_v(int layer_idx);   // V view
    Tensor retrieval_v(int layer_idx);
    Tensor window_v(int layer_idx);
    Tensor buffer_v(int layer_idx);

    // Summary and retrieval helpers modelled after SpecPV. The initial
    // implementation focuses on maintaining block counts and placeholders
    // for future CUDA kernels; they do not yet drive any change in the
    // attention layout.
    void summary_key_states(int layer_idx, const Tensor& key_states, int seq_len);

    void refresh_retrieval(int         layer_idx,
                           const Tensor& query_states,
                           const Tensor& key_states,
                           const Tensor& value_states,
                           int           seq_len);

    // Append newly verified tokens into the speculative buffer slice and
    // return the active KV views [sink+retrieval+window+buffer] for this
    // layer. The current implementation only updates verified lengths; it
    // does not yet move real KV data.
    std::pair<Tensor, Tensor> update(int layer_idx, const Tensor& new_keys, const Tensor& new_values);

    void reset_buffer();

private:
    void recompute_global_verified_len() noexcept;

    Tensor slice_tokens(std::vector<Tensor>& cache, int layer_idx, int token_start, int token_count);

private:
    SpecPVCacheConfig cfg_{};
    int               max_batch_size_{0};
    int               num_layers_{0};
    int               num_kv_heads_{0};
    int               head_dim_{0};
    DataType          kv_dtype_{kFloat32};

    int max_seq_len_{0};
    int max_summary_blocks_{0};

    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;

    std::vector<Tensor> key_summary_max_;
    std::vector<Tensor> key_summary_min_;
    std::vector<int>    summary_block_count_;

    std::vector<int> verified_lens_;
    int              global_verified_len_{0};
    bool             enabled_{false};
};

}  // namespace turbomind
