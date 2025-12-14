
#pragma once

#include "src/turbomind/engine/drift_engine_config.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/models/llama/LlamaBatch.h" // Add include for LlamaBatch

#include <memory>
#include <atomic>
#include <thread> // For worker_loop

namespace turbomind {

class LlamaBatch;

class DriftEngine {
public:
    DriftEngine(const DriftEngineConfig& cfg,
                std::shared_ptr<Gateway> gateway,
                std::shared_ptr<KVCacheManager> kv_mgr,
                std::shared_ptr<PrefixCache> prefix_cache,
                std::shared_ptr<LlamaBatch> llama_batch); // Add llama_batch

    // Main loop entry for a worker rank (run in a dedicated thread).
    void run(int rank);

    // Shutdown coordination
    void shutdown();

private:
    DriftEngineConfig             cfg_;
    std::shared_ptr<Gateway>     gateway_;
    std::shared_ptr<KVCacheManager> kv_mgr_;
    std::shared_ptr<PrefixCache> prefix_cache_;
    EngineScheduler              scheduler_;
    CapacityScheduler            capacity_sched_;
    std::atomic<bool>           abort_{false};

    std::shared_ptr<LlamaBatch> llama_batch_; // Changed from unique_ptr to shared_ptr

    void worker_loop(int rank);
};

} // namespace turbomind
