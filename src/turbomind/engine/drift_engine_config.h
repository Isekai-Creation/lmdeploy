
#pragma once

#include "src/turbomind/engine/scheduler_config.h"
#include "src/turbomind/core/kv_cache_manager.h" // For KVLayout
#include "src/turbomind/models/common/model_layout.h"

namespace turbomind {

struct DriftEngineConfig {
    SchedulerConfig scheduler;
    KVLayout        kv_layout; // From KVCacheManager
    ModelLayout     model_layout; // From KVCacheManager

    bool prefer_high_throughput{true};
    int  target_latency_ms_p50{50};
    int  target_latency_ms_p95{200};
    int  max_queued_requests{4096};
    bool abort_on_oom{true};
};

} // namespace turbomind
