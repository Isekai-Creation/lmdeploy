#pragma once

#include "src/turbomind/engine/drift_engine_config.h"
#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/drift_metrics.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include <memory>
#include <atomic>
#include <functional>

namespace turbomind {

struct PrefillChunk;
class LlamaBatch;

class DriftEngine {
public:
    DriftEngine(const DriftEngineConfig& cfg,
                std::shared_ptr<Gateway> gateway,
                std::shared_ptr<KVCacheManager> kv_mgr,
                std::shared_ptr<PrefixCache> prefix_cache);

    // Main loop entry for a worker rank (run in a dedicated thread).
    void run(int rank);

    // Shutdown coordination
    void shutdown();

    // Snapshot current engine metrics (queues, KV/prefix stats)
    DriftMetrics metrics() const;

    // Hook to plug in a concrete model executor (e.g., LlamaBatch). Optional
    // shutdown callback lets the engine ask the executor to stop on abort.
    void set_batch_executor(std::function<void(const std::vector<PrefillChunk>&,
                                               const std::vector<std::shared_ptr<Request>>&)> executor,
                            std::function<void()> shutdown_cb = {});

    // Convenience helper: bind a LlamaBatch executor/shutdown pair.
    void bind_llama_batch(LlamaBatch* batch);

    // Step D: Configure speculative decoding parameters
    void configure_speculative_decoding(bool enable, const std::string& method, int max_draft_tokens = 4);

private:
    DriftEngineConfig                  cfg_;
    std::shared_ptr<Gateway>           gateway_;
    std::shared_ptr<KVCacheManager>    kv_mgr_;
    std::shared_ptr<PrefixCache>       prefix_cache_;
    std::unique_ptr<CapacityScheduler> capacity_sched_; // Constructed internally
    std::unique_ptr<EngineScheduler>   scheduler_; // Constructed internally
     std::atomic<bool>                  abort_{false};
     std::function<void(const std::vector<PrefillChunk>&,
                        const std::vector<std::shared_ptr<Request>>&)> batch_executor_;
     std::function<void()> batch_shutdown_;
     std::atomic<bool>                  progress_ready_logged_{false};
 
     // Optional bound LlamaBatch executor for DriftEngine TP=1 path.
     LlamaBatch* llama_batch_{nullptr};


    KVLayout derive_kv_layout(const ModelLayout& model_layout, const KVLayout& provided) const;
    void     worker_loop(int rank); // Internal worker loop
};

}  // namespace turbomind
