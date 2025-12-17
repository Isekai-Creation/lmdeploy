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

    SchedulerConfig scheduler_config{};
    ModelLayout     model_layout{};   // canonical layout for the model
    KVLayout        kv_layout{};      // derived from model_layout and kv dtype
    size_t          kv_capacity_bytes{0}; // optional; when 0 expect external KV manager

    bool prefer_high_throughput{true};
    int  target_latency_ms_p50{50};
    int         target_latency_ms_p95{200};
    int         max_queued_requests{4096};
    bool        abort_on_oom{true};
    std::string log_level{"INFO"};
};

}  // namespace turbomind
