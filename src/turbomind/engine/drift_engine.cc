
#include "src/turbomind/engine/drift_engine.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/models/llama/LlamaBatch.h" // For LlamaBatch
#include "src/turbomind/models/llama/llama_utils.h"

namespace turbomind {

DriftEngine::DriftEngine(const DriftEngineConfig&   cfg,
                         std::shared_ptr<Gateway>   gateway,
                         std::shared_ptr<KVCacheManager> kv_mgr,
                         std::shared_ptr<PrefixCache>    prefix_cache,
                         std::shared_ptr<LlamaBatch>     llama_batch):
    cfg_(cfg),
    gateway_(std::move(gateway)),
    kv_mgr_(std::move(kv_mgr)),
    prefix_cache_(std::move(prefix_cache)),
    capacity_sched_(kv_mgr_.get(), prefix_cache_.get()),
    scheduler_(cfg.scheduler, kv_mgr_.get(), cfg.model_layout, prefix_cache_.get(), &capacity_sched_),
    llama_batch_(std::move(llama_batch))
{
    TM_LOG_INFO("[DriftEngine] Initialized.");
}

void DriftEngine::run(int rank) {
    TM_LOG_INFO("[DriftEngine] Starting worker_loop for rank %d.", rank);
    // Use std::async for thread management instead of directly creating std::thread,
    // as std::thread can be tricky with class members.
    // For now, let's just directly call the worker_loop. The actual threading will be done by the top-level orchestrator.
    worker_loop(rank);
}

void DriftEngine::shutdown() {
    TM_LOG_INFO("[DriftEngine] Shutting down.");
    abort_.store(true, std::memory_order_release);
    // If worker_loop was in a separate thread, join it here.
}

void DriftEngine::worker_loop(int rank)
{
    TM_LOG_INFO("[DriftEngine] Worker loop started for rank %d.", rank);

    GenerationState g{}; // Manages generation state across steps

    while (!abort_.load(std::memory_order_acquire)) {
        std::vector<std::shared_ptr<Request>> infer_reqs_local;
        std::vector<std::shared_ptr<Request>> kill_reqs_local;
        bool abort_flag = false;

        {
            NvtxScope _("pop");
            const int free_slot_count = cfg_.scheduler.max_num_seqs - (scheduler_.get_active_requests_count() + scheduler_.get_queued_requests_count());
            bool is_empty = scheduler_.empty();
            gateway_->pop(infer_reqs_local, kill_reqs_local, free_slot_count, is_empty, abort_flag, rank);
        }

        if (abort_flag) {
            TM_LOG_INFO("[DriftEngine] Abort signal received, exiting worker loop.");
            break;
        }

        NvtxScope scope("mainloop");

        scheduler_.on_new_requests(infer_reqs_local, kill_reqs_local);

        std::vector<PrefillChunk> prefill_batch;
        std::vector<std::shared_ptr<Request>> decode_batch;
        scheduler_.schedule_step(prefill_batch, decode_batch);

        if (prefill_batch.empty() && decode_batch.empty() && scheduler_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // NOTE: DriftEngine scheduling loop is still under active
        // development. The current TurboMind LlamaBatch path manages
        // its own internal batching and KV via SequenceManager and
        // does not yet expose a stable execute_batches API. Until the
        // integration is complete, we intentionally avoid issuing
        // per-step work here to keep the legacy TurboMind/EAGLE3 path
        // unchanged.
    }

    TM_LOG_INFO("[DriftEngine] Worker loop for rank %d stopped.", rank);
}

} // namespace turbomind
