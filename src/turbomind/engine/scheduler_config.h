#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

namespace turbomind {

class DriftEngineConfig; // Forward declaration

struct SchedulerConfig {
    // Per-step budgets
    int max_num_batched_tokens{2048};
    int max_num_seqs{128};

    // Chunked prefill controls
    bool enable_chunked_prefill{true};
    int  max_num_partial_prefills{1};
    int  max_long_partial_prefills{1};
    int  long_prefill_token_threshold{4096};

    // Policy knobs
    bool prefer_decode_over_prefill{true};
    enum class SchedulePolicy : uint8_t { kFcfs, kSmallFirst };
    SchedulePolicy schedule_policy{SchedulePolicy::kFcfs};
    
    // Step D: Speculative decoding configuration
    bool enable_speculative_decoding{false};
    std::string spec_method{"none"};  // "eagle", "eagle3", "ngram", etc.
    int max_draft_tokens_per_seq{4};  // Max draft tokens to generate per sequence

    // Prefix/KV hints and latency targets
    bool enable_prefix_caching{false};
    int  target_latency_ms_p50{50};
    int  target_latency_ms_p95{200};

    int max_num_batched_tokens_per_seq() const {
        if (max_num_seqs <= 0) {
            return max_num_batched_tokens;
        }
        return std::max(1, max_num_batched_tokens / max_num_seqs);
    }

    static SchedulerConfig from_engine_config(const DriftEngineConfig& engine_config);
};

}  // namespace turbomind

