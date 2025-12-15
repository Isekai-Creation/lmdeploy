
#pragma once

namespace turbomind {

struct DriftMetrics {
    double ema_tokens_per_second;
    double ema_p50_latency_ms;
    double ema_p95_latency_ms;

    // TODO: Add other relevant metrics like KV cache hit rate, eviction rate, etc.
};

} // namespace turbomind
