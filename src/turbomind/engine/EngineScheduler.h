
#pragma once

#include "src/turbomind/engine/scheduler_config.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/engine/drift_metrics.h"
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace turbomind {

class CapacityScheduler;

enum class SequencePhase : uint8_t { kPrefill, kDecode, kFinished };

struct SequenceState {
    uint64_t     seq_id;
    int          prompt_len;      // total prompt tokens
    int          prefilled_len;   // how many prompt tokens already processed
    int          generated_len;   // how many new tokens generated
    int          max_new_tokens;
    SequencePhase phase;
    int          prefill_chunk_size; // Added for chunked prefill
};

struct PrefillChunk {
    std::shared_ptr<Request> req;
    int                      start_pos;  // inclusive, in prompt tokens
    int                      len;        // number of tokens in this chunk
};

class EngineScheduler {
public:
    EngineScheduler(const SchedulerConfig& cfg,
                    KVCacheManager*        kv_mgr,
                    const ModelLayout&     model_layout,
                    PrefixCache*           prefix_cache,
                    CapacityScheduler*     capacity_scheduler);

    // Minimal constructor used by legacy TurboMind paths that do not
    // yet provide model layout, prefix cache, or capacity scheduler.
    EngineScheduler(const SchedulerConfig& cfg, KVCacheManager* kv_mgr);
    
    void on_new_requests(const std::vector<std::shared_ptr<Request>>& infer_reqs,
                         const std::vector<std::shared_ptr<Request>>& kill_reqs);

        void update_sequence_state(uint64_t seq_id, SequencePhase new_phase, int prefilled_len, int generated_len);

    

        void update_metrics(const DriftMetrics& new_metrics); // New method

    

        // Decide which sequences to run this step given the token budget.

        void schedule_step(std::vector<PrefillChunk>& prefill_batch,

                           std::vector<std::shared_ptr<Request>>& decode_batch);

    // For testing
    double get_decode_token_ratio() const;

    int get_active_requests_count() const;
    int get_queued_requests_count() const;
    bool empty() const;

private:
    SchedulerConfig   cfg_;
    KVCacheManager*   kv_mgr_;
    ModelLayout       model_layout_;
    PrefixCache*      prefix_cache_;
    CapacityScheduler* capacity_scheduler_;
    std::unordered_map<uint64_t, SequenceState> seq_states_;
    std::unordered_map<uint64_t, std::shared_ptr<Request>> active_requests_;

    std::list<std::shared_ptr<Request>> prefill_request_queue_; // Requests waiting for prefill
    std::list<std::shared_ptr<Request>> decode_request_queue_;  // Requests waiting for decode

    DriftMetrics metrics_;

    // Adaptive tuning parameters
    double decode_token_ratio_; // For dynamic adjustment of decode token budget
    double prefill_token_ratio_;

    mutable std::mutex metrics_mutex_; // New member // For dynamic adjustment of prefill token budget

    int tokens_for_decode_step(const SequenceState& state) const;
    int tokens_for_prefill_chunk(const SequenceState& state) const;
};

}  // namespace turbomind
