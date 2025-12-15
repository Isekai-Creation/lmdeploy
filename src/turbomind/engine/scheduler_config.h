#pragma once

namespace turbomind {

struct SchedulerConfig {
    int  max_num_batched_tokens{2048};
    int  max_num_seqs{128};
    bool enable_chunked_prefill{true};
    int  max_num_partial_prefills{1};
    int  long_prefill_token_threshold{0};
    bool prefer_decode_over_prefill{true};

    int target_latency_ms_p50{50}; // New member
    int target_latency_ms_p95{200}; // New member

    bool enable_prefix_caching{false}; // New member

    int max_num_batched_tokens_per_seq() const {
        return max_num_batched_tokens / max_num_seqs; // Simplified for now
    }
};

}  // namespace turbomind