#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/engine/drift_metrics.h"
#include "src/turbomind/engine/drift_engine_config.h" // Added for from_engine_config implementation
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/progress_logger.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/models/llama/LlamaBatch.h"  // For ExecutionResult
#include <algorithm> // For std::min, std::max
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace turbomind {

// Implementation of from_engine_config
SchedulerConfig SchedulerConfig::from_engine_config(const DriftEngineConfig& engine_config) {
    return engine_config.scheduler_config;
}

EngineScheduler::EngineScheduler(const SchedulerConfig& cfg,
                                 KVCacheManager*        kv_mgr,
                                 const ModelLayout&     model_layout,
                                 PrefixCache*           prefix_cache,
                                 CapacityScheduler*     capacity_scheduler):
    cfg_(cfg),
    kv_mgr_(kv_mgr),
    model_layout_(model_layout),
    prefix_cache_(prefix_cache),
    capacity_scheduler_(capacity_scheduler),
    metrics_{},
    decode_token_ratio_(0.5),
    prefill_token_ratio_(0.5)
{
    session_len_limit_            = model_layout_.max_seq_len;
    require_capacity_scheduler_   = (capacity_scheduler_ != nullptr);
}

EngineScheduler::EngineScheduler(const SchedulerConfig& cfg, KVCacheManager* kv_mgr):
    cfg_(cfg),
    kv_mgr_(kv_mgr),
    model_layout_{},
    prefix_cache_(nullptr),
    capacity_scheduler_(nullptr),
    metrics_{},
    decode_token_ratio_(0.5),
    prefill_token_ratio_(0.5)
{
    session_len_limit_          = 0;
    require_capacity_scheduler_ = false;
}

void EngineScheduler::on_new_requests(const std::vector<std::shared_ptr<Request>>& infer_reqs,
                                      const std::vector<std::shared_ptr<Request>>& kill_reqs)
{
    for (const auto& r : infer_reqs) {
        if (!r) {
            continue;
        }

        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kSchedulerAdmit};
            evt.pct        = 10;
            evt.seq_id     = r->session.id;
            evt.session_id = r->session.id;
            evt.unique_id  = r->unique_id;
            evt.rank       = -1;
            evt.msg        = r->session.start_flag ? "start" : "cont";
            ProgressLogger::Log(evt);
        }

        // Enforce SessionParam legal state transitions
        if (r->session.start_flag) {
            // Allow start + end=true for single-shot requests; only
            // disallow non-zero step or kill on a start.
            if (r->session.step != 0 || r->session.kill_flag) {
                TM_LOG_WARNING(
                    "[EngineScheduler] Rejecting new request %llu due to inconsistent SessionParam: start_flag=true implies step=0 and kill_flag=false. (step=%d, kill=%d)",
                    r->session.id,
                    r->session.step,
                    r->session.kill_flag);
                r->ec = Request::kInconsistency;
                continue;
            }
        }
        else {
            if (r->session.step == 0 && !r->session.end_flag && !r->session.kill_flag) {
                TM_LOG_WARNING(
                    "[EngineScheduler] Rejecting continuation request %llu due to inconsistent SessionParam: step=0 for a non-start, non-end, non-kill request.",
                    r->session.id);
                r->ec = Request::kInconsistency;
                continue;
            }
            auto seq_it = seq_states_.find(r->session.id);
            if (seq_it == seq_states_.end()) {
                TM_LOG_WARNING("[EngineScheduler] Continuation request %llu for unknown sequence. Rejecting.", r->session.id);
                r->ec = Request::kInvalid;
                continue;
            }

            if (r->session.kill_flag) {
                // Treat as kill
                prefill_request_queue_.remove_if([&](const std::shared_ptr<Request>& req){ return req->session.id == r->session.id; });
                decode_request_queue_.remove_if([&](const std::shared_ptr<Request>& req){ return req->session.id == r->session.id; });
                seq_states_.erase(r->session.id);
                active_requests_.erase(r->session.id);
                release_kv(r->session.id, "kill_continuation");

                r->ec = Request::kCancel;
                continue;
            }

            if (r->session.end_flag) {
                // Graceful finish: mark sequence finished and release resources.
                update_sequence_state(r->session.id, 0, seq_it->second.max_new_tokens - seq_it->second.generated_len);
                continue;
            }
        }

        if (r->session.start_flag) {
            released_seq_ids_.erase(r->session.id);
            const int prompt_len = r->inputs.at("input_ids").shape(0);

            if (kv_mgr_) {
                std::vector<int> prompt_tokens;
                std::vector<int> pre_existing_page_ids;
                int              matched_tokens = 0;

                if (prefix_cache_ && cfg_.enable_prefix_caching && extract_prompt_tokens(r, prompt_tokens)) {
                    PrefixKey          key{prompt_tokens, 0};
                    PrefixMatchResult  result = prefix_cache_->match(key);
                    if (result.matched_tokens > 0) {
                        matched_tokens        = result.matched_tokens;
                        pre_existing_page_ids = result.page_indices;
                        TM_LOG_DEBUG("[EngineScheduler] PrefixCache matched %d tokens for sequence %llu.",
                                     matched_tokens,
                                     r->session.id);
                    }
                }

                const KVUsageEstimate est = KVCacheManager::estimate_usage(model_layout_, prompt_len, r->gen_cfg.max_new_tokens);

                KVReservation reservation{};
                bool          reservation_success = false;

                if (capacity_scheduler_) {
                    reservation_success =
                        capacity_scheduler_->try_start_request(r->session.id, est, &reservation, pre_existing_page_ids);
                    if (!reservation_success) {
                        TM_LOG_WARNING(
                            "[EngineScheduler] Rejecting request %llu due to insufficient KV capacity (needed %zu pages) via CapacityScheduler.",
                            r->session.id,
                            est.pages_needed);
                        r->ec = Request::kTooLong;
                        continue;
                    }
                }
                else {
                    reservation_success = kv_mgr_->reserve(r->session.id, est, &reservation, pre_existing_page_ids);
                    if (!reservation_success) {
                        TM_LOG_WARNING(
                            "[EngineScheduler] Rejecting request %llu due to insufficient KV capacity (needed %zu pages) via KVCacheManager.",
                            r->session.id,
                            est.pages_needed);
                        r->ec = Request::kTooLong;
                        continue;
                    }
                }

                if (reservation.page_ids.empty()) {
                    TM_LOG_ERROR("[EngineScheduler] Reservation for %llu succeeded but returned no pages; rejecting.",
                                 r->session.id);
                    if (ProgressLogger::Enabled()) {
                        ProgressEvent evt{ProgressStage::kError};
                        evt.pct        = 100;
                        evt.seq_id     = r->session.id;
                        evt.session_id = r->session.id;
                        evt.msg        = "reserve_ok_but_no_pages";
                        ProgressLogger::Log(evt);
                    }
                    release_kv(r->session.id, "reservation_no_pages");
                    r->ec = Request::kFail;
                    continue;
                }

                FT_CHECK_WITH_INFO(seq_states_.find(r->session.id) == seq_states_.end(),
                                   "Attempting to add new sequence that already exists in seq_states_.");

                SequenceState state{};
                state.seq_id                = r->session.id;
                state.prompt_len            = prompt_len;
                state.prefilled_len         = matched_tokens;
                state.generated_len         = 0;
                state.max_new_tokens        = r->gen_cfg.max_new_tokens;
                state.phase                 = (matched_tokens == prompt_len) ? SequencePhase::kDecode : SequencePhase::kPrefill;
                state.queue_tag             = (state.phase == SequencePhase::kPrefill) ? SequenceQueue::kPrefill : SequenceQueue::kDecode;
                state.kv_reservation_handle = reservation.first_page;
                state.kv_page_ids           = reservation.page_ids;
                state.kv_cookie             = reservation.kv_cookie;

                r->kv_page_ids = reservation.page_ids;
                r->kv_cookie   = reservation.kv_cookie;

                if (ProgressLogger::Enabled()) {
                    ProgressEvent evt{ProgressStage::kSchedulerAdmit};
                    evt.pct           = 45;
                    evt.seq_id        = state.seq_id;
                    evt.session_id    = state.seq_id;
                    evt.kv_pages_seq  = static_cast<int>(state.kv_page_ids.size());
                    evt.kv_map_cookie = state.kv_cookie;
                    evt.msg           = "kv_meta_attached";
                    ProgressLogger::Log(evt);
                }

                released_seq_ids_.erase(state.seq_id);
                seq_states_[state.seq_id] = std::move(state);
                if (seq_states_[r->session.id].phase == SequencePhase::kPrefill) {
                    prefill_request_queue_.push_back(r);
                }
                else {
                    decode_request_queue_.push_back(r);
                }
            }
            else {
                // No KV manager configured; admit the sequence without KV tracking.
                FT_CHECK_WITH_INFO(seq_states_.find(r->session.id) == seq_states_.end(),
                                   "Attempting to add new sequence that already exists in seq_states_ (no KV manager).");
                SequenceState state{};
                state.seq_id         = r->session.id;
                state.prompt_len     = prompt_len;
                state.prefilled_len  = 0;
                state.generated_len  = 0;
                state.max_new_tokens = r->gen_cfg.max_new_tokens;
                state.phase          = SequencePhase::kPrefill;
                state.queue_tag      = SequenceQueue::kPrefill;
                released_seq_ids_.erase(state.seq_id);
                seq_states_[state.seq_id] = state;
                prefill_request_queue_.push_back(r);
            }
        }

        if (!r->ec) {
            active_requests_[r->session.id] = r;
        }
    }

    // Handle killed requests.
    for (const auto& r : kill_reqs) {
        if (!r) {
            continue;
        }

        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kSchedulerAdmit};
            evt.pct        = 10;
            evt.seq_id     = r->session.id;
            evt.session_id = r->session.id;
            evt.unique_id  = r->unique_id;
            evt.msg        = "kill";
            ProgressLogger::Log(evt);
        }

        // Erase from queues
        prefill_request_queue_.remove_if([&](const std::shared_ptr<Request>& req){ return req->session.id == r->session.id; });
        decode_request_queue_.remove_if([&](const std::shared_ptr<Request>& req){ return req->session.id == r->session.id; });

        // Assert that the request is no longer in the queues
        FT_CHECK_WITH_INFO(std::find_if(prefill_request_queue_.begin(), prefill_request_queue_.end(),
                                       [&](const auto& req_ptr){ return req_ptr->session.id == r->session.id; }) == prefill_request_queue_.end(),
                           "Killed request still in prefill queue after remove_if.");
        FT_CHECK_WITH_INFO(std::find_if(decode_request_queue_.begin(), decode_request_queue_.end(),
                                       [&](const auto& req_ptr){ return req_ptr->session.id == r->session.id; }) == decode_request_queue_.end(),
                           "Killed request still in decode queue after remove_if.");


        seq_states_.erase(r->session.id);
        FT_CHECK_WITH_INFO(seq_states_.find(r->session.id) == seq_states_.end(),
                           "Killed request still in seq_states_ after erase.");
        active_requests_.erase(r->session.id);
        release_kv(r->session.id, "kill_queue");
    }
}


void EngineScheduler::update_sequence_state(uint64_t seq_id, int prefilled_tokens_added, int generated_tokens_added)
{
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        TM_LOG_WARNING("[EngineScheduler] update_sequence_state for unknown sequence ID: %lu", seq_id);
        return;
    }

    SequenceState& state = it->second;
    
    state.prefilled_len += prefilled_tokens_added;
    state.generated_len += generated_tokens_added;

    if (state.phase == SequencePhase::kPrefill && state.prefilled_len >= state.prompt_len) {
        state.phase     = SequencePhase::kDecode;
        state.queue_tag = SequenceQueue::kNone;
        TM_LOG_DEBUG("[EngineScheduler] Sequence %lu completed prefill, moving to decode phase.", seq_id);
        
        // After prefill completes, insert the prefix into the cache
        if (prefix_cache_ && kv_mgr_) {
            auto req_it = active_requests_.find(seq_id);
            if (req_it != active_requests_.end()) {
                const auto& req = req_it->second;
                const int   page_size = kv_mgr_->get_layout().page_size;
                if (page_size <= 0) {
                    TM_LOG_WARNING("[EngineScheduler] Cannot insert prefix for seq %lu: invalid page_size %d.", seq_id, page_size);
                }
                else {
                    const int aligned_tokens = (state.prompt_len / page_size) * page_size;
                    if (aligned_tokens <= 0) {
                        TM_LOG_DEBUG("[EngineScheduler] Skipping prefix insert for seq %lu: prompt too small for alignment.", seq_id);
                    }
                    else {
                        std::vector<int> prompt_tokens;
                        if (!extract_prompt_tokens(req, prompt_tokens, aligned_tokens)) {
                            TM_LOG_WARNING("[EngineScheduler] Unable to extract prompt tokens for seq %lu; skipping prefix insert.", seq_id);
                        }
                        else {
                            prompt_tokens.resize(aligned_tokens);
                            std::vector<int> live_page_ids = kv_mgr_->get_sequence_page_ids(seq_id);
                            if (live_page_ids != state.kv_page_ids) {
                                TM_LOG_WARNING("[EngineScheduler] KV page mismatch for seq %lu during prefix insert (cached=%zu live=%zu). Skipping.",
                                               seq_id,
                                               state.kv_page_ids.size(),
                                               live_page_ids.size());
                                if (ProgressLogger::Enabled()) {
                                    ProgressEvent evt{ProgressStage::kError};
                                    evt.pct           = 100;
                                    evt.seq_id        = seq_id;
                                    evt.session_id    = seq_id;
                                    evt.kv_pages_seq  = static_cast<int>(live_page_ids.size());
                                    evt.kv_map_cookie = state.kv_cookie;
                                    evt.msg           = "prefix_insert_kv_page_mismatch";
                                    ProgressLogger::Log(evt);
                                }
                            }
                            else if (live_page_ids.empty()) {
                                TM_LOG_WARNING("[EngineScheduler] Sequence %lu completed prefill but has no allocated KV pages. Not inserting to PrefixCache.", seq_id);
                            }
                            else {
                                PrefixKey key;
                                key.tokens       = prompt_tokens;
                                key.namespace_id = 0;
                                prefix_cache_->insert(key, live_page_ids, 0, seq_id);
                                TM_LOG_DEBUG("[EngineScheduler] Inserted prefix for sequence %lu into PrefixCache (aligned_tokens=%d).",
                                             seq_id,
                                             aligned_tokens);
                                if (ProgressLogger::Enabled()) {
                                    ProgressEvent evt{ProgressStage::kPrefixMatch};
                                    evt.pct           = 60;
                                    evt.seq_id        = seq_id;
                                    evt.session_id    = seq_id;
                                    evt.kv_pages_seq  = static_cast<int>(live_page_ids.size());
                                    evt.kv_map_cookie = state.kv_cookie;
                                    evt.chunk_len     = aligned_tokens;
                                    evt.msg           = "prefix_insert";
                                    ProgressLogger::Log(evt);
                                }
                            }
                        }
                    }
                }
            } else {
                TM_LOG_WARNING("[EngineScheduler] Sequence %lu completed prefill, but request not found in active_requests_. Cannot insert to PrefixCache.", seq_id);
            }
        }

    }

    if (state.generated_len >= state.max_new_tokens) {
        state.phase     = SequencePhase::kFinished;
        state.queue_tag = SequenceQueue::kNone;
    }

    if (state.phase == SequencePhase::kFinished) {
        TM_LOG_DEBUG("[EngineScheduler] Sequence %lu finished. Releasing resources.", seq_id);
        // KV reservation ownership:
        //  - When a CapacityScheduler is present, it is responsible for
        //    calling kv_mgr_->release(seq_id) exactly once via
        //    finish_request.
        //  - When there is no CapacityScheduler, fall back to releasing
        //    directly via KVCacheManager.
        release_kv(seq_id, "finished");
        seq_states_.erase(seq_id);
        active_requests_.erase(seq_id);

        prefill_request_queue_.remove_if([&](const auto& r_ptr){ return r_ptr->session.id == seq_id; });
        decode_request_queue_.remove_if([&](const auto& r_ptr){ return r_ptr->session.id == seq_id; });
    }
}



void EngineScheduler::update_metrics(const DriftMetrics& new_metrics)
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = new_metrics;

    // Basic adaptive tuning logic
    double latency_degradation_factor = metrics_.ema_p95_latency_ms / cfg_.target_latency_ms_p95;

    if (latency_degradation_factor > 1.1) { // Latency is 10% higher than target, reduce decode priority
        decode_token_ratio_ -= 0.05; // Adjust aggressively
    } else if (latency_degradation_factor < 0.9) { // Latency is 10% lower, increase decode priority
        decode_token_ratio_ += 0.02; // Adjust moderately
    }

    // Clamp decode_token_ratio_ to reasonable bounds
    decode_token_ratio_ = std::max(0.1, std::min(1.0, decode_token_ratio_));
    // For now, simple inverse for prefill_token_ratio_
    prefill_token_ratio_ = 1.0 - decode_token_ratio_;

    TM_LOG_DEBUG("[EngineScheduler] Metrics updated. P95 latency: %dms (target: %dms). New decode_token_ratio_: %.2f",
                 (int)metrics_.ema_p95_latency_ms, cfg_.target_latency_ms_p95, decode_token_ratio_);
}

DriftMetrics EngineScheduler::snapshot_metrics() const
{
    DriftMetrics snapshot{};

    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        snapshot.active_requests = active_requests_.size();
        snapshot.queued_prefill  = prefill_request_queue_.size();
        snapshot.queued_decode   = decode_request_queue_.size();
    }

    if (kv_mgr_) {
        snapshot.kv_total_pages = kv_mgr_->total_pages();
        snapshot.kv_used_pages  = kv_mgr_->used_pages();
        snapshot.kv_free_pages  = kv_mgr_->free_pages();
    }

    if (capacity_scheduler_) {
        snapshot.kv_blocked  = capacity_scheduler_->blocked_due_to_capacity();
        snapshot.kv_rejected = capacity_scheduler_->rejected_never_schedulable();
    }

    if (prefix_cache_) {
        snapshot.prefix_hits          = prefix_cache_->get_hit_count();
        snapshot.prefix_misses        = prefix_cache_->get_miss_count();
        snapshot.prefix_evictions     = prefix_cache_->get_eviction_count();
        snapshot.prefix_bytes_evicted = prefix_cache_->get_bytes_evicted();
    }

    snapshot.ema_tokens_per_second = metrics_.ema_tokens_per_second;
    snapshot.ema_p50_latency_ms    = metrics_.ema_p50_latency_ms;
    snapshot.ema_p95_latency_ms    = metrics_.ema_p95_latency_ms;

    return snapshot;
}
double EngineScheduler::get_decode_token_ratio() const
{
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return decode_token_ratio_;
}

int EngineScheduler::get_active_requests_count() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_); // Protects access to active_requests_
    return active_requests_.size();
}

int EngineScheduler::get_queued_requests_count() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_); // Protects access to prefill_request_queue_ and decode_request_queue_
    return prefill_request_queue_.size() + decode_request_queue_.size();
}

bool EngineScheduler::empty() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_); // Protects access to queues and active_requests_
    return active_requests_.empty() && prefill_request_queue_.empty() && decode_request_queue_.empty();
}

bool EngineScheduler::has_oom_detected() const {
    return oom_detected_;
}

void EngineScheduler::clear_oom_detected() {
    oom_detected_ = false;
}
int EngineScheduler::tokens_for_decode_step(const SequenceState& state) const
{
    // Non-spec mode: always 1 token per decode step.
    return 1;
}

int EngineScheduler::tokens_for_prefill_chunk(const SequenceState& state) const {
    int remaining_prompt_tokens = state.prompt_len - state.prefilled_len;
    if (remaining_prompt_tokens <= 0) return 0;

    // Use max_num_batched_tokens_per_seq from SchedulerConfig
    int chunk_size = cfg_.max_num_batched_tokens_per_seq();
    
    // Ensure chunk_size is at least 1 if there are remaining tokens.
    // Also, ensure it doesn't exceed the total max_num_batched_tokens.
    chunk_size = std::min(chunk_size, cfg_.max_num_batched_tokens);
    chunk_size = std::max(1, chunk_size); // Minimum 1 token per chunk if possible

    return std::min(remaining_prompt_tokens, chunk_size);
}

bool EngineScheduler::extract_prompt_tokens(const std::shared_ptr<Request>& req,
                                            std::vector<int>&              tokens_out,
                                            int                            max_tokens) const
{
    if (!req) {
        return false;
    }
    auto it = req->inputs.find("input_ids");
    if (it == req->inputs.end()) {
        return false;
    }
    const auto& tensor = it->second;
    const auto  device = tensor.device().type;
    if (device != DeviceType::kCPU && device != DeviceType::kCPUpinned) {
        TM_LOG_DEBUG("[EngineScheduler] input_ids for seq %llu not on host-accessible memory; skipping token extraction.", req->session.id);
        return false;
    }

    int prompt_len = 0;
    if (tensor.ndim() == 1) {
        prompt_len = tensor.shape(0);
    }
    else if (tensor.ndim() == 2) {
        if (tensor.shape(0) != 1) {
            TM_LOG_WARNING("[EngineScheduler] input_ids tensor for seq %llu has unsupported batch dimension %lld.",
                           req->session.id,
                           static_cast<long long>(tensor.shape(0)));
            return false;
        }
        prompt_len = tensor.shape(1);
    }
    else {
        TM_LOG_WARNING("[EngineScheduler] input_ids tensor for seq %llu has unsupported rank %d.", req->session.id, tensor.ndim());
        return false;
    }

    if (prompt_len <= 0) {
        return false;
    }

    const int copy_len = (max_tokens > 0) ? std::min(prompt_len, max_tokens) : prompt_len;
    if (copy_len <= 0) {
        return false;
    }

    tokens_out.resize(copy_len);
    try {
        std::memcpy(tokens_out.data(), tensor.data<int>(), sizeof(int) * copy_len);
    }
    catch (...) {
        TM_LOG_ERROR("[EngineScheduler] Failed to read input_ids for seq %llu.", req->session.id);
        tokens_out.clear();
        return false;
    }
    return true;
}


void EngineScheduler::schedule_step(std::vector<PrefillChunk>&             prefill_batch,
                                     std::vector<std::shared_ptr<Request>>& decode_batch)
{
    prefill_batch.clear();
    decode_batch.clear();

    // Apply scheduling policy to the prefill queue
    if (cfg_.schedule_policy == SchedulerConfig::SchedulePolicy::kSmallFirst) {
        prefill_request_queue_.sort([&](const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) {
            auto it_a = seq_states_.find(a->session.id);
            auto it_b = seq_states_.find(b->session.id);
            if (it_a != seq_states_.end() && it_b != seq_states_.end()) {
                return it_a->second.prompt_len < it_b->second.prompt_len;
            }
            return false; // Should not happen if queues are consistent
        });
    }
    // FCFS is the default order of std::list

    int current_num_seqs         = 0;
    int current_tokens_prefill   = 0;
    int current_tokens_decode    = 0;
    int total_tokens_in_batch    = 0;

    // Two-pass scheduling:
    // 1. Decode pass
    // 2. Prefill pass
    // The `prefer_decode_over_prefill` flag determines which pass runs first.

    int max_decode_tokens_budget = static_cast<int>(cfg_.max_num_batched_tokens * get_decode_token_ratio());
    int max_prefill_tokens_budget = cfg_.max_num_batched_tokens - max_decode_tokens_budget;

    max_decode_tokens_budget = std::max(0, max_decode_tokens_budget);
    max_prefill_tokens_budget = std::max(0, max_prefill_tokens_budget);
    if (max_decode_tokens_budget + max_prefill_tokens_budget > cfg_.max_num_batched_tokens) {
        max_prefill_tokens_budget = cfg_.max_num_batched_tokens - max_decode_tokens_budget;
    }

    std::vector<std::list<std::shared_ptr<Request>>::iterator> decode_selected;
    std::vector<std::list<std::shared_ptr<Request>>::iterator> prefill_selected;

    auto schedule_decode_pass = [&]() {
        auto it = decode_request_queue_.begin();
        while (it != decode_request_queue_.end()) {
            if (current_num_seqs >= cfg_.max_num_seqs ||
                total_tokens_in_batch >= cfg_.max_num_batched_tokens ||
                current_tokens_decode >= max_decode_tokens_budget) {
                break;
            }

            auto& req      = *it;
            auto  state_it = seq_states_.find(req->session.id);

            if (state_it == seq_states_.end() || state_it->second.phase != SequencePhase::kDecode) {
                it = decode_request_queue_.erase(it);
                continue;
            }

            int tokens_needed = tokens_for_decode_step(state_it->second);
            if (tokens_needed <= 0) {
                it = decode_request_queue_.erase(it);
                continue;
            }

            if (total_tokens_in_batch + tokens_needed <= cfg_.max_num_batched_tokens &&
                current_tokens_decode + tokens_needed <= max_decode_tokens_budget) {
                decode_batch.push_back(req);
                total_tokens_in_batch += tokens_needed;
                current_tokens_decode += tokens_needed;
                current_num_seqs++;
                state_it->second.queue_tag = SequenceQueue::kNone;
                decode_selected.push_back(it++);
            }
            else {
                ++it;
            }
        }
    };

    auto schedule_prefill_pass = [&]() {
        auto it = prefill_request_queue_.begin();
        while (it != prefill_request_queue_.end()) {
            if (current_num_seqs >= cfg_.max_num_seqs ||
                total_tokens_in_batch >= cfg_.max_num_batched_tokens ||
                current_tokens_prefill >= max_prefill_tokens_budget) {
                break;
            }

            auto& req = *it;
            auto state_it = seq_states_.find(req->session.id);

            if (state_it == seq_states_.end() || state_it->second.phase != SequencePhase::kPrefill) {
                it = prefill_request_queue_.erase(it);
                continue;
            }

            auto& state = state_it->second;
            int chunk_len = tokens_for_prefill_chunk(state);

            if (chunk_len > 0 &&
                total_tokens_in_batch + chunk_len <= cfg_.max_num_batched_tokens &&
                current_tokens_prefill + chunk_len <= max_prefill_tokens_budget) {
                prefill_batch.push_back({req, state.prefilled_len, chunk_len});
                total_tokens_in_batch += chunk_len;
                current_tokens_prefill += chunk_len;
                current_num_seqs++;
                state.queue_tag = SequenceQueue::kNone;
                prefill_selected.push_back(it++);
            }
            else {
                ++it;
            }
        }
    };

    if (cfg_.prefer_decode_over_prefill) {
        schedule_decode_pass();
        schedule_prefill_pass();
    }
    else {
        schedule_prefill_pass();
        schedule_decode_pass();
    }

    for (auto it_sel : decode_selected) {
        decode_request_queue_.erase(it_sel);
    }
    for (auto it_sel : prefill_selected) {
        prefill_request_queue_.erase(it_sel);
    }
 
     FT_CHECK_WITH_INFO(total_tokens_in_batch <= cfg_.max_num_batched_tokens, "Total tokens scheduled exceeds max_num_batched_tokens.");
     FT_CHECK_WITH_INFO(static_cast<int>(prefill_batch.size() + decode_batch.size()) <= cfg_.max_num_seqs, "Total sequences scheduled exceeds max_num_seqs.");
 
     TM_LOG_DEBUG("[EngineScheduler] scheduled step: prefill_seqs=%zu decode_seqs=%zu prefill_tokens=%d decode_tokens=%d total_tokens=%d",
                  prefill_batch.size(),
                  decode_batch.size(),
                  current_tokens_prefill,
                  current_tokens_decode,
                  total_tokens_in_batch);
 }


void EngineScheduler::on_step_executed(const std::vector<PrefillChunk>&             prefill_batch,
                                       const std::vector<std::shared_ptr<Request>>& decode_batch,
                                       const std::unordered_map<uint64_t, int>&     pre_lengths,
                                       const std::vector<ExecutionResult>&          exec_results)
{
    auto get_pre_len = [&](uint64_t seq_id) -> int {
        if (auto it = pre_lengths.find(seq_id); it != pre_lengths.end()) {
            return it->second;
        }
        auto st_it = seq_states_.find(seq_id);
        if (st_it == seq_states_.end()) {
            TM_LOG_ERROR("[EngineScheduler] Missing seq_state for seq %lu during on_step_executed", seq_id);
            return 0;
        }
        return st_it->second.prefilled_len + st_it->second.generated_len;
    };


    auto remove_from_queues = [&](uint64_t seq_id) {
        prefill_request_queue_.remove_if([&](const auto& r_ptr) { return r_ptr->session.id == seq_id; });
        decode_request_queue_.remove_if([&](const auto& r_ptr) { return r_ptr->session.id == seq_id; });
    };

    auto fail_sequence = [&](uint64_t seq_id, const char* reason, const std::shared_ptr<Request>& req) {
        const char* reason_text = reason ? reason : "delta_failure";
        uint64_t    kv_cookie   = 0;
        int         kv_pages    = 0;
        if (auto st_it = seq_states_.find(seq_id); st_it != seq_states_.end()) {
            kv_cookie = st_it->second.kv_cookie;
            kv_pages  = static_cast<int>(st_it->second.kv_page_ids.size());
        }
        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kError};
            evt.pct           = 100;
            evt.seq_id        = seq_id;
            evt.session_id    = seq_id;
            evt.kv_pages_seq  = kv_pages;
            evt.kv_map_cookie = kv_cookie;
            evt.msg           = reason_text;
            ProgressLogger::Log(evt);
        }
        if (auto it_req = active_requests_.find(seq_id); it_req != active_requests_.end() && it_req->second) {
            UpdateState(*it_req->second, Request::kFail, get_pre_len(seq_id));
        }
        else if (req) {
            UpdateState(*req, Request::kFail, get_pre_len(seq_id));
        }
        release_kv(seq_id, reason_text);
        seq_states_.erase(seq_id);
        active_requests_.erase(seq_id);
        remove_from_queues(seq_id);
    };

    std::unordered_map<uint64_t, int> post_lengths;
    post_lengths.reserve(exec_results.size());
    for (const auto& result : exec_results) {
        if (result.final_sequence_length >= 0) {
            post_lengths[result.sequence_id] = result.final_sequence_length;
        }
    }

    auto log_post_len = [&](uint64_t seq_id, bool is_decode, int pre_len, int post_len, int delta) {
        if (!ProgressLogger::Enabled()) {
            return;
        }
        ProgressEvent evt{is_decode ? ProgressStage::kDecodeExec : ProgressStage::kPrefillExec};
        evt.pct        = is_decode ? 86 : 66;
        evt.seq_id     = seq_id;
        evt.session_id = seq_id;
        std::ostringstream oss;
        oss << "post_len_reported pre=" << pre_len << " post=" << post_len << " delta=" << delta;
        evt.msg = oss.str();
        ProgressLogger::Log(evt);
    };

    auto get_post_len = [&](uint64_t seq_id) -> std::optional<int> {
        if (auto it = post_lengths.find(seq_id); it != post_lengths.end()) {
            return it->second;
        }
        return std::nullopt;
    };

    auto compute_delta = [&](const std::shared_ptr<Request>& req, bool is_decode) -> std::pair<int, bool> {
        if (!req) {
            return {0, false};
        }
        const uint64_t seq_id  = req->session.id;
        const int      pre_len = get_pre_len(seq_id);
        auto           post    = get_post_len(seq_id);
        if (!post.has_value()) {
            TM_LOG_ERROR("[EngineScheduler] Missing post length for seq %lu", seq_id);
            fail_sequence(seq_id, is_decode ? "missing_decode_post_length" : "missing_prefill_post_length", req);
            return {0, false};
        }
        int post_len = post.value();
        int delta    = post_len - pre_len;
        log_post_len(seq_id, is_decode, pre_len, post_len, delta);
 
        bool used_planned = false;
        if (delta <= 0) {
            TM_LOG_ERROR("[EngineScheduler] Non-positive delta for seq %lu (pre=%d post=%d)", seq_id, pre_len, post_len);
            fail_sequence(seq_id,
                          is_decode ? "decode_delta_nonpositive" : "prefill_delta_nonpositive",
                          req);
            return {0, false};
        }
 
        return {delta, used_planned};
    };


    constexpr int kMaxDecodeFallbacks  = 2;
    constexpr int kMaxPrefillFallbacks = 3;

    auto handle_fallback = [&](uint64_t seq_id,
                               SequenceState& state,
                               bool           is_decode,
                               bool           used_planned,
                               const std::shared_ptr<Request>& req) -> bool {
        (void)state;
        (void)seq_id;
        (void)is_decode;
        (void)used_planned;
        (void)req;
        return true;
    };

    for (const auto& req : decode_batch) {
        auto [delta, used_planned] = compute_delta(req, /*is_decode=*/true);
        if (!req || delta == 0) {
            continue;
        }
        const uint64_t seq_id = req->session.id;
        auto           st_it  = seq_states_.find(seq_id);
        if (st_it == seq_states_.end()) {
            continue;
        }
        if (!handle_fallback(seq_id, st_it->second, /*is_decode=*/true, used_planned, req)) {
            continue;
        }
        update_sequence_state(seq_id, 0, delta);
        if (auto fresh_it = seq_states_.find(seq_id); fresh_it != seq_states_.end()) {
            TM_LOG_DEBUG("[EngineScheduler] seq %lu decode advanced by %d tokens (prefilled=%d generated=%d phase=%d)",
                         seq_id,
                         delta,
                         fresh_it->second.prefilled_len,
                         fresh_it->second.generated_len,
                         static_cast<int>(fresh_it->second.phase));
        }
    }
    for (const auto& chunk : prefill_batch) {
        if (!chunk.req) {
            continue;
        }
        auto [delta, used_planned] = compute_delta(chunk.req, /*is_decode=*/false);
        if (delta == 0) {
            continue;
        }
        const uint64_t seq_id = chunk.req->session.id;
        auto           st_it  = seq_states_.find(seq_id);
        if (st_it == seq_states_.end()) {
            continue;
        }
        if (!handle_fallback(seq_id, st_it->second, /*is_decode=*/false, used_planned, chunk.req)) {
            continue;
        }
        update_sequence_state(seq_id, delta, 0);
        if (auto fresh_it = seq_states_.find(seq_id); fresh_it != seq_states_.end()) {
            TM_LOG_DEBUG("[EngineScheduler] seq %lu prefill advanced by %d tokens (prefilled=%d/%d phase=%d)",
                         chunk.req->session.id,
                         delta,
                         fresh_it->second.prefilled_len,
                         fresh_it->second.prompt_len,
                         static_cast<int>(fresh_it->second.phase));
        }
    }

    for (const auto& req : decode_batch) {
        auto st_it = seq_states_.find(req->session.id);
        if (st_it != seq_states_.end() && st_it->second.phase == SequencePhase::kDecode) {
            st_it->second.queue_tag = SequenceQueue::kDecode;
            decode_request_queue_.push_back(req);
        }
    }
    for (const auto& chunk : prefill_batch) {
        if (!chunk.req) {
            continue;
        }
        auto st_it = seq_states_.find(chunk.req->session.id);
        if (st_it == seq_states_.end()) {
            continue;
        }
        if (st_it->second.phase == SequencePhase::kPrefill && st_it->second.prefilled_len < st_it->second.prompt_len) {
            st_it->second.queue_tag = SequenceQueue::kPrefill;
            prefill_request_queue_.push_back(chunk.req);
        }
        else if (st_it->second.phase == SequencePhase::kDecode) {
            st_it->second.queue_tag = SequenceQueue::kDecode;
            decode_request_queue_.push_back(chunk.req);
        }
    }
}

void EngineScheduler::on_speculative_execution(const std::vector<turbomind::ExecutionResult>& results)
{
    // Speculative decoding disabled in v1.
    (void)results;
}

void EngineScheduler::release_kv(uint64_t seq_id, const char* reason)
{
    const char* reason_text = reason ? reason : "unspecified";
    if (!released_seq_ids_.insert(seq_id).second) {
        TM_LOG_WARNING("[EngineScheduler] Duplicate KV release attempt for seq %llu (reason=%s).", seq_id, reason_text);
        return;
    }

    if (capacity_scheduler_) {
        TM_LOG_DEBUG("[EngineScheduler] KV release owned by CapacityScheduler: seq=%llu reason=%s", seq_id, reason_text);
        capacity_scheduler_->finish_request(seq_id, reason_text);
        return;
    }

    if (require_capacity_scheduler_) {
        TM_LOG_ERROR("[EngineScheduler] CapacityScheduler missing during release for seq %llu (reason=%s).",
                     seq_id,
                     reason_text);
        throw std::runtime_error("CapacityScheduler required for DriftEngine KV lifecycle");
    }

    if (kv_mgr_) {
        TM_LOG_DEBUG("[EngineScheduler] KV release owned directly by EngineScheduler: seq=%llu reason=%s", seq_id, reason_text);
        kv_mgr_->release(seq_id);
        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kRelease};
            evt.pct        = 100;
            evt.seq_id     = seq_id;
            evt.session_id = seq_id;
            evt.msg        = std::string("direct_release:") + reason_text;
            ProgressLogger::Log(evt);
        }
        return;
    }

    TM_LOG_DEBUG("[EngineScheduler] KV release requested for seq %llu but no KV manager configured (reason=%s).",
                 seq_id,
                 reason_text);
}

}  // namespace turbomind
