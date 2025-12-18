#pragma once

#include "src/turbomind/engine/scheduler_config.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include <string>

namespace turbomind {

struct DriftEngineConfig {
    // Model / parallelism
    std::string dtype{"bf16"};
    int         tp{1};
    int         pp{1};
    int         session_len{8192};
    int         max_batch_size{256};
    int         quant_policy{0};

    SchedulerConfig scheduler_config{};
    ModelLayout     model_layout{};   // canonical layout for the model
    KVLayout        kv_layout{};      // derived from model_layout and kv dtype
    size_t          kv_capacity_bytes{0}; // optional; when 0 expect external KV manager

    // Speculative decoding scaffolding (no effect until wired)
    bool        enable_speculative_decoding{false};
    std::string spec_method{"none"};
    int         spec_max_draft_tokens{0};

    // SpecPV partial-KV hints (mirrors EngineParam::specpv_*)
    bool enable_specpv{false};
    int  specpv_block_size{16};
    int  specpv_n_sink_blocks{2};
    int  specpv_n_retrieval_blocks{256};
    int  specpv_n_window_blocks{8};
    int  specpv_n_spec_tokens_buf{128};
    int  specpv_partial_threshold{4096};
    int  specpv_full_refresh_steps{32};

    // Suffix decoding knobs (STUFFIX)
    bool  enable_suffix_decoding{false};
    int   suffix_cache_max_depth{64};
    int   suffix_cache_max_requests{-1};
    float suffix_max_spec_factor{1.0f};
    float suffix_max_spec_offset{0.0f};
    float suffix_min_token_prob{0.1f};

    bool prefer_high_throughput{true};
    int  target_latency_ms_p50{50};
    int         target_latency_ms_p95{200};
    int         max_queued_requests{4096};
    bool        abort_on_oom{true};
    std::string log_level{"INFO"};
};

}  // namespace turbomind
