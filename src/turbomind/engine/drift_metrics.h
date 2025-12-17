
#pragma once

namespace turbomind {

struct DriftMetrics {
    double ema_tokens_per_second{0.0};
    double ema_p50_latency_ms{0.0};
    double ema_p95_latency_ms{0.0};

    size_t step_prefill_tokens{0};
    size_t step_decode_tokens{0};

    size_t queued_prefill{0};
    size_t queued_decode{0};
    size_t active_requests{0};

    size_t kv_total_pages{0};
    size_t kv_used_pages{0};
    size_t kv_free_pages{0};
    size_t kv_blocked{0};
    size_t kv_rejected{0};

    size_t prefix_hits{0};
    size_t prefix_misses{0};
    size_t prefix_evictions{0};
    size_t prefix_bytes_evicted{0};
};

} // namespace turbomind
