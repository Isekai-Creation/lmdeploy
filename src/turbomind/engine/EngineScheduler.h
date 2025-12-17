
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
#include <unordered_set>
#include <vector>

namespace turbomind {

class CapacityScheduler;
struct ExecutionResult;

enum class SequencePhase : uint8_t { kPrefill, kDecode, kFinished };

// Tracks which queue a sequence currently resides in to avoid duplicate membership.
enum class SequenceQueue : uint8_t { kNone, kPrefill, kDecode };

struct SequenceState {
    uint64_t             seq_id{0};
    int                  prompt_len{0};       // total prompt tokens
    int                  prefilled_len{0};    // how many prompt tokens already processed
    int                  generated_len{0};    // how many new tokens generated
    int                  max_new_tokens{0};
    SequencePhase        phase{SequencePhase::kPrefill};
    SequenceQueue        queue_tag{SequenceQueue::kNone};
    int                  kv_reservation_handle{-1};  // Placeholder for KVCacheManager integration
    std::vector<int>     kv_page_ids;
    uint64_t             kv_cookie{0};
    int                  decode_planned_fallbacks{0};
    int                  prefill_planned_fallbacks{0};
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

    void schedule_step(std::vector<PrefillChunk>&               prefill_batch,
                       std::vector<std::shared_ptr<Request>>& decode_batch);

    void update_sequence_state(uint64_t seq_id, int prefilled_tokens_added, int generated_tokens_added);
    void update_metrics(const DriftMetrics& new_metrics);
 
     DriftMetrics snapshot_metrics() const;
 
// Post-execution hook: update sequence state and requeue after a
    // scheduled step has actually been executed by the backend. The
    // `pre_lengths` map contains the per-sequence lengths observed
    // immediately before the backend step so that we can derive actual
    // per-sequence token deltas from Request.sequence_length.
    void on_step_executed(const std::vector<PrefillChunk>&             prefill_batch,
                           const std::vector<std::shared_ptr<Request>>& decode_batch,
                           const std::unordered_map<uint64_t, int>&     pre_lengths,
                           const std::vector<ExecutionResult>&          exec_results);

    // Step D: Speculative decoding execution result handling
    void on_speculative_execution(const std::vector<turbomind::ExecutionResult>& results);

    // For testing
    double get_decode_token_ratio() const;
    int    get_active_requests_count() const;
    int    get_queued_requests_count() const;
    bool   empty() const;

    // OOM detection
    bool   has_oom_detected() const;
    void   clear_oom_detected();
 
 private:
     // TODO: Implement policy hooks for FCFS and short-prompt-first
     int tokens_for_decode_step(const SequenceState& state) const;
     int tokens_for_prefill_chunk(const SequenceState& state) const;
     bool extract_prompt_tokens(const std::shared_ptr<Request>& req, std::vector<int>& tokens_out, int max_tokens = -1) const;
 
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
    double prefill_token_ratio_; // For dynamic adjustment of prefill token budget

     mutable std::mutex             metrics_mutex_;
     bool                           oom_detected_ = false; // Flag to indicate if an OOM condition was detected (stub for future)
     int                            session_len_limit_{0};
    std::unordered_set<uint64_t>   released_seq_ids_;
    bool                           require_capacity_scheduler_{false};

    void release_kv(uint64_t seq_id, const char* reason);
};



}  // namespace turbomind
