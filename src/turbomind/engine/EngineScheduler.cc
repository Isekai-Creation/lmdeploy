#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/engine/drift_metrics.h"
#include "src/turbomind/utils/logger.h"
#include <algorithm> // For std::min, std::max

namespace turbomind {

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
    decode_token_ratio_(1.0),
    prefill_token_ratio_(0.0)
{
}

EngineScheduler::EngineScheduler(const SchedulerConfig& cfg, KVCacheManager* kv_mgr):
    cfg_(cfg),
    kv_mgr_(kv_mgr),
    model_layout_{},
    prefix_cache_(nullptr),
    capacity_scheduler_(nullptr),
    metrics_{},
    decode_token_ratio_(1.0),
    prefill_token_ratio_(0.0)
{
}

void EngineScheduler::on_new_requests(const std::vector<std::shared_ptr<Request>>& infer_reqs,
                                      const std::vector<std::shared_ptr<Request>>& kill_reqs)
{
    // Admit new requests and perform basic KV capacity checks.
    for (const auto& r : infer_reqs) {
        if (!r) {
            continue;
        }

        // Enforce SessionParam legal state transitions
        if (r->session.start_flag) {
            // New session must start with step 0, and not be marked as end or kill
            if (r->session.step != 0 || r->session.end_flag || r->session.kill_flag) {
                TM_LOG_WARNING("[EngineScheduler] Rejecting new request %llu due to inconsistent SessionParam: start_flag=true implies step=0, end_flag=false, kill_flag=false. (step=%d, end=%d, kill=%d)",
                               r->session.id, r->session.step, r->session.end_flag, r->session.kill_flag);
                r->ec = Request::kInconsistency;
                continue;
            }
        } else {
            // Continuation request must have a step > 0 (or look up existing sequence)
            // And can have end_flag or kill_flag set.
            if (r->session.step == 0 && !r->session.end_flag && !r->session.kill_flag) {
                // If not starting, not ending, not killing, and step is 0, this is likely an error.
                TM_LOG_WARNING("[EngineScheduler] Rejecting continuation request %llu due to inconsistent SessionParam: step=0 for a non-start, non-end, non-kill request.", r->session.id);
                r->ec = Request::kInconsistency;
                continue;
            }
            // Further checks on consistency with existing sequence state will be done later.
        }

        // Capacity check only for new sequences.
        if (kv_mgr_ && r->session.start_flag) {
            int prompt_len = r->inputs.at("input_ids").shape(0);
            std::vector<int> pre_existing_page_ids;
            int matched_tokens = 0;

            if (prefix_cache_ && cfg_.enable_prefix_caching) {
                PrefixKey key;
                key.tokens.assign(r->inputs.at("input_ids").data<int>(), r->inputs.at("input_ids").data<int>() + prompt_len);
                key.namespace_id = 0; // Default namespace for now

                PrefixMatchResult result = prefix_cache_->match(key);
                if (result.matched_tokens > 0) {
                    matched_tokens = result.matched_tokens;
                    pre_existing_page_ids = result.page_indices;
                    TM_LOG_DEBUG("[EngineScheduler] PrefixCache matched %d tokens for sequence %llu.", matched_tokens, r->session.id);
                }
            }

            // Estimate usage considering potentially pre-existing pages
            const KVUsageEstimate est = KVCacheManager::estimate_usage(
                model_layout_, prompt_len, r->gen_cfg.max_new_tokens); // est.pages_needed should be total pages for the full sequence

            KVReservation reservation{};
            bool reservation_success = false;

            if (capacity_scheduler_) { // Use capacity_scheduler if available
                reservation_success = capacity_scheduler_->try_start_request(r->session.id, est, &reservation, pre_existing_page_ids);
                if (!reservation_success) {
                    TM_LOG_WARNING(
                        "[EngineScheduler] Rejecting request %llu due to insufficient KV capacity "
                        "(needed %zu pages) via CapacityScheduler.",
                        r->session.id,
                        est.pages_needed);
                    r->ec = Request::kTooLong; // Set error code for rejection
                    continue;
                }
            } else if (kv_mgr_) { // Fallback to direct KVCacheManager if no CapacityScheduler
                reservation_success = kv_mgr_->reserve(r->session.id, est, &reservation, pre_existing_page_ids);
                if (!reservation_success) {
                    TM_LOG_WARNING(
                        "[EngineScheduler] Rejecting request %llu due0to insufficient KV capacity "
                        "(needed %zu pages) via KVCacheManager.",
                        r->session.id,
                        est.pages_needed);
                    r->ec = Request::kTooLong; // Set error code for rejection
                    continue;
                }
            } else {
                TM_LOG_WARNING(
                    "[EngineScheduler] No KV manager or capacity scheduler. Request %llu admitted without KV checks.",
                    r->session.id);
                reservation_success = true; // Assume success if no checks can be performed
            }

            if (reservation_success) {
                FT_CHECK_WITH_INFO(seq_states_.find(r->session.id) == seq_states_.end(),
                                   "Attempting to add new sequence that already exists in seq_states_.");
                SequenceState state{};
                state.seq_id          = r->session.id;
                state.prompt_len      = prompt_len;
                state.prefilled_len   = matched_tokens; // Start with matched tokens as prefilled
                state.generated_len   = 0;
                state.max_new_tokens  = r->gen_cfg.max_new_tokens;
                state.phase           = (matched_tokens == prompt_len) ? SequencePhase::kDecode : SequencePhase::kPrefill;
                state.prefill_chunk_size = 0; // Will be determined during scheduling
                
                seq_states_[state.seq_id] = state;
                if (state.phase == SequencePhase::kPrefill) {
                    FT_CHECK_WITH_INFO(std::find(prefill_request_queue_.begin(), prefill_request_queue_.end(), r) == prefill_request_queue_.end(),
                                       "New request already exists in prefill queue.");
                    prefill_request_queue_.push_back(r);
                    FT_CHECK_WITH_INFO(std::find(decode_request_queue_.begin(), decode_request_queue_.end(), r) == decode_request_queue_.end(),
                                       "New request added to prefill queue is also in decode queue.");
                } else { // Matched full prompt, goes to decode
                    FT_CHECK_WITH_INFO(std::find(decode_request_queue_.begin(), decode_request_queue_.end(), r) == decode_request_queue_.end(),
                                       "New request already exists in decode queue.");
                    decode_request_queue_.push_back(r);
                    FT_CHECK_WITH_INFO(std::find(prefill_request_queue_.begin(), prefill_request_queue_.end(), r) == prefill_request_queue_.end(),
                                       "New request added to decode queue is also in prefill queue.");
                }
            }
        } else if (r->session.start_flag) { // No KV manager, but still a start flag
            FT_CHECK_WITH_INFO(seq_states_.find(r->session.id) == seq_states_.end(),
                               "Attempting to add new sequence that already exists in seq_states_ (no KV manager).");
            SequenceState state{};
            state.seq_id          = r->session.id;
            state.prompt_len      = r->inputs.at("input_ids").shape(0);
            state.prefilled_len   = 0;
            state.generated_len   = 0;
            state.max_new_tokens  = r->gen_cfg.max_new_tokens;
            state.phase           = SequencePhase::kPrefill;
            state.prefill_chunk_size = 0;
            seq_states_[state.seq_id] = state;
            prefill_request_queue_.push_back(r);
            FT_CHECK_WITH_INFO(std::find(decode_request_queue_.begin(), decode_request_queue_.end(), r) == decode_request_queue_.end(),
                               "New request added to prefill queue is also in decode queue (no KV manager).");
        } else {
            // Continuation request
            auto it = seq_states_.find(r->session.id);
            if (it != seq_states_.end()) {
                if (it->second.phase == SequencePhase::kPrefill) {
                    prefill_request_queue_.push_back(r);
                    FT_CHECK_WITH_INFO(std::find(decode_request_queue_.begin(), decode_request_queue_.end(), r) == decode_request_queue_.end(),
                                       "Continuation request added to prefill queue is also in decode queue.");
                } else if (it->second.phase == SequencePhase::kDecode) { // Already in decode phase
                    decode_request_queue_.push_back(r);
                    FT_CHECK_WITH_INFO(std::find(prefill_request_queue_.begin(), prefill_request_queue_.end(), r) == prefill_request_queue_.end(),
                                       "Continuation request added to decode queue is also in prefill queue.");
                } else { // Should be kFinished, implies an issue if it's a continuation
                     TM_LOG_WARNING("[EngineScheduler] Continuation request %llu for finished sequence. Rejecting.", r->session.id);
                     r->ec = Request::kInvalid;
                }
            } else {
                TM_LOG_WARNING("[EngineScheduler] Continuation request %llu for unknown sequence. Rejecting.", r->session.id);
                r->ec = Request::kInvalid;
            }
        }

        // Add to active requests map if not rejected
        if (!r->ec) {
            active_requests_[r->session.id] = r;
        }
    }

    // Handle killed requests.
    for (const auto& r : kill_reqs) {
        if (!r) {
            continue;
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
        if (kv_mgr_) {
            kv_mgr_->release(r->session.id);
        }
        if (capacity_scheduler_) {
            capacity_scheduler_->finish_request(r->session.id);
        }
    }
}


void EngineScheduler::update_sequence_state(uint64_t      seq_id,
                                            SequencePhase new_phase,
                                            int           prefilled_len,
                                            int           generated_len)
{
    auto it = seq_states_.find(seq_id);
    if (it == seq_states_.end()) {
        TM_LOG_WARNING("[EngineScheduler] update_sequence_state for unknown sequence ID: %lu", seq_id);
        return;
    }

    SequenceState& state = it->second;
    state.phase          = new_phase;
    state.prefilled_len  = prefilled_len;
    state.generated_len  = generated_len;

    if (new_phase == SequencePhase::kFinished) {
        TM_LOG_DEBUG("[EngineScheduler] Sequence %lu finished. Releasing resources.", seq_id);
        if (kv_mgr_) {
            kv_mgr_->release(seq_id);
        }
        if (capacity_scheduler_) {
            capacity_scheduler_->finish_request(seq_id);
        }
        seq_states_.erase(seq_id);
        active_requests_.erase(seq_id);

        // Assert that the sequence is no longer in any queues
        FT_CHECK_WITH_INFO(std::find_if(prefill_request_queue_.begin(), prefill_request_queue_.end(),
                                       [&](const auto& r_ptr){ return r_ptr->session.id == seq_id; }) == prefill_request_queue_.end(),
                           "Finished sequence still in prefill queue.");
        FT_CHECK_WITH_INFO(std::find_if(decode_request_queue_.begin(), decode_request_queue_.end(),
                                       [&](const auto& r_ptr){ return r_ptr->session.id == seq_id; }) == decode_request_queue_.end(),
                           "Finished sequence still in decode queue.");

    } else if (state.phase == SequencePhase::kPrefill && state.prefilled_len >= state.prompt_len) {
        // Transition from prefill to decode
        TM_LOG_DEBUG("[EngineScheduler] Sequence %lu completed prefill, moving to decode phase.", seq_id);
        // Find the request in the prefill queue.
        auto req_it = active_requests_.find(seq_id);
        if (req_it != active_requests_.end()) {
            FT_CHECK_WITH_INFO(std::find(prefill_request_queue_.begin(), prefill_request_queue_.end(), req_it->second) != prefill_request_queue_.end(),
                               "Request to move from prefill to decode not found in prefill queue.");
            prefill_request_queue_.remove(req_it->second); // Remove from prefill
            
            FT_CHECK_WITH_INFO(std::find(decode_request_queue_.begin(), decode_request_queue_.end(), req_it->second) == decode_request_queue_.end(),
                               "Request already in decode queue before moving from prefill.");
            decode_request_queue_.push_back(req_it->second); // Add to decode
            
            FT_CHECK_WITH_INFO(std::find(prefill_request_queue_.begin(), prefill_request_queue_.end(), req_it->second) == prefill_request_queue_.end(),
                               "Request still in prefill queue after moving to decode.");
        }
        state.phase = SequencePhase::kDecode; // Set new phase AFTER successful queue manipulation.
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

double EngineScheduler::get_decode_token_ratio() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return decode_token_ratio_;
}

int EngineScheduler::get_active_requests_count() const {
    return active_requests_.size();
}

int EngineScheduler::get_queued_requests_count() const {
    return prefill_request_queue_.size() + decode_request_queue_.size();
}

bool EngineScheduler::empty() const {
    return active_requests_.empty() && prefill_request_queue_.empty() && decode_request_queue_.empty();
}

int EngineScheduler::tokens_for_decode_step(const SequenceState& state) const {
    // For standard decoding, 1 token per step.
    // Future: Support for speculative decoding where we might budget more slots.
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


void EngineScheduler::schedule_step(std::vector<PrefillChunk>&             prefill_batch,
                                    std::vector<std::shared_ptr<Request>>& decode_batch)
{
    prefill_batch.clear();
    decode_batch.clear();

    int current_num_seqs = 0;
    int current_tokens_prefill = 0;
    int current_tokens_decode = 0;

    // Calculate token budgets based on ratios
    int decode_budget_tokens;
    int prefill_budget_tokens;

    // Ensure atomic access to ratios if they can be updated by another thread (update_metrics)
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        decode_budget_tokens = static_cast<int>(cfg_.max_num_batched_tokens * decode_token_ratio_);
        prefill_budget_tokens = cfg_.max_num_batched_tokens - decode_budget_tokens;
    }

    // Prioritize decode requests if configured
    if (cfg_.prefer_decode_over_prefill) {
        // Fill decode_batch first
        for (auto it = decode_request_queue_.begin(); it != decode_request_queue_.end(); ) {
            const auto& req = *it;
            auto state_it = seq_states_.find(req->session.id);

            // Handle kFinished or unknown sequences
            if (state_it == seq_states_.end() || state_it->second.phase == SequencePhase::kFinished) {
                TM_LOG_DEBUG("[EngineScheduler][schedule_step] Removing finished/unknown decode request %llu from queue.", req->session.id);
                it = decode_request_queue_.erase(it);
                continue;
            }

            if (current_num_seqs >= cfg_.max_num_seqs) break;
            if (current_tokens_decode >= decode_budget_tokens) break;

            const auto& state = state_it->second;
            int tokens_needed = tokens_for_decode_step(state);

            if (current_tokens_decode + tokens_needed <= decode_budget_tokens) {
                decode_batch.push_back(req);
                current_tokens_decode += tokens_needed;
                current_num_seqs++;
                it = decode_request_queue_.erase(it); // Remove from queue and advance iterator
            } else {
                ++it; // Cannot fit this request, try next
            }
        }

        // Fill prefill_batch with remaining capacity
        for (auto it = prefill_request_queue_.begin(); it != prefill_request_queue_.end(); ) {
            const auto& req = *it;
            auto state_it = seq_states_.find(req->session.id);

            // Handle kFinished or unknown sequences
            if (state_it == seq_states_.end() || state_it->second.phase == SequencePhase::kFinished) {
                TM_LOG_DEBUG("[EngineScheduler][schedule_step] Removing finished/unknown prefill request %llu from queue.", req->session.id);
                it = prefill_request_queue_.erase(it);
                continue;
            }

            if (current_num_seqs >= cfg_.max_num_seqs) break;
            if (current_tokens_prefill >= prefill_budget_tokens) break;

            const auto& state = state_it->second;

            if (state.phase != SequencePhase::kPrefill) { // Should already be in decode_request_queue_ or finished
                TM_LOG_DEBUG("[EngineScheduler][schedule_step] Prefill request %llu in unexpected phase %d. Removing.", req->session.id, (int)state.phase);
                it = prefill_request_queue_.erase(it);
                continue;
            }

            // Chunked prefill logic
            int chunk_len = tokens_for_prefill_chunk(state);
            if (chunk_len == 0) { // Prefill complete, move to decode queue
                TM_LOG_DEBUG("[EngineScheduler] Sequence %lu prefill complete, calling update_sequence_state to transition to decode phase.", req->session.id);
                // Call update_sequence_state to handle the transition and queue movement
                update_sequence_state(req->session.id, SequencePhase::kDecode, state.prompt_len, state.generated_len);
                // The request will be moved to decode_request_queue_ and erased from prefill_request_queue_ by update_sequence_state.
                it = prefill_request_queue_.erase(it);
                continue;
            }

            if (current_tokens_prefill + chunk_len <= prefill_budget_tokens) {
                // Check if this is a "long" prefill and respect max_num_partial_prefills
                if (cfg_.max_num_partial_prefills > 0 && 
                    state.prompt_len > cfg_.long_prefill_token_threshold) 
                {
                    // Count how many long prefill requests are already in batch
                    int long_prefill_count = 0;
                    for(const auto& pc : prefill_batch) {
                        const auto& pc_state_it = seq_states_.find(pc.req->session.id);
                        if (pc_state_it != seq_states_.end() && pc_state_it->second.prompt_len > cfg_.long_prefill_token_threshold) {
                            long_prefill_count++;
                        }
                    }
                    if (long_prefill_count >= cfg_.max_num_partial_prefills) {
                        ++it; // Skip this long prefill for now, try next
                        continue;
                    }
                }

                prefill_batch.push_back({req, state.prefilled_len, chunk_len});
                current_tokens_prefill += chunk_len;
                current_num_seqs++;
                // Update the state for the scheduled chunk (this will be finalized by LlamaBatch)
                seq_states_[req->session.id].prefilled_len += chunk_len; 
                it = prefill_request_queue_.erase(it); // Remove from queue and advance iterator
            } else {
                ++it; // Cannot fit this chunk, try next
            }
        }
    } else { // No prioritization, or prefer prefill (not implemented yet, default to balance)
        // For now, a simple round-robin or first-come-first-served approach.
        // This section will be refined.
        // For simplicity, let's still try to fill decode first, then prefill, up to limits.
        // Re-use logic above without explicit preference.
        
        // Temporarily copy requests to a combined list to simplify iteration
        std::list<std::shared_ptr<Request>> combined_queue;
        combined_queue.splice(combined_queue.end(), decode_request_queue_);
        combined_queue.splice(combined_queue.end(), prefill_request_queue_);

        for (auto it = combined_queue.begin(); it != combined_queue.end(); ) {
            const auto& req = *it;
            auto state_it = seq_states_.find(req->session.id);

            // Handle kFinished or unknown sequences
            if (state_it == seq_states_.end() || state_it->second.phase == SequencePhase::kFinished) {
                TM_LOG_DEBUG("[EngineScheduler][schedule_step] Removing finished/unknown combined request %llu from queue.", req->session.id);
                it = combined_queue.erase(it);
                continue;
            }

            if (current_num_seqs >= cfg_.max_num_seqs) break;

            const auto& state = state_it->second;

            if (state.phase == SequencePhase::kPrefill) {
                int chunk_len = tokens_for_prefill_chunk(state);
                if (chunk_len == 0) { // Prefill complete, move to decode queue
                    TM_LOG_DEBUG("[EngineScheduler] Sequence %lu prefill complete, calling update_sequence_state to transition to decode phase.", req->session.id);
                    // Call update_sequence_state to handle the transition and queue movement
                    update_sequence_state(req->session.id, SequencePhase::kDecode, state.prompt_len, state.generated_len);
                    // The request will be moved to decode_request_queue_ and erased from combined_queue by update_sequence_state.
                    it = combined_queue.erase(it);
                    continue;
                }

                if (current_tokens_prefill + chunk_len <= prefill_budget_tokens) {
                    prefill_batch.push_back({req, state.prefilled_len, chunk_len});
                    current_tokens_prefill += chunk_len;
                    current_num_seqs++;
                    seq_states_[req->session.id].prefilled_len += chunk_len;
                    it = combined_queue.erase(it);
                } else {
                    ++it;
                }
            } else if (state.phase == SequencePhase::kDecode) {
                int tokens_needed = tokens_for_decode_step(state);

                if (current_tokens_decode + tokens_needed <= decode_budget_tokens) {
                    decode_batch.push_back(req);
                    current_tokens_decode += tokens_needed;
                    current_num_seqs++;
                    it = combined_queue.erase(it);
                } else {
                    ++it;
                }
            } else { // kFinished or invalid state
                TM_LOG_DEBUG("[EngineScheduler][schedule_step] Combined request %llu in unexpected phase %d. Removing.", req->session.id, (int)state.phase);
                it = combined_queue.erase(it);
            }
        }
        // Return remaining requests to their original queues.
        // This is a simplified approach, a more robust solution would be needed
        // to handle requests that couldn't be scheduled.
        for(const auto& req : combined_queue) {
            const auto& state_it = seq_states_.find(req->session.id);
            if (state_it != seq_states_.end()) {
                if (state_it->second.phase == SequencePhase::kPrefill) {
                    prefill_request_queue_.push_back(req);
                } else if (state_it->second.phase == SequencePhase::kDecode) {
                    decode_request_queue_.push_back(req);
                }
            }
        }
    }
    // Assert invariants at the end of schedule_step
    FT_CHECK_WITH_INFO(current_tokens_prefill + current_tokens_decode <= cfg_.max_num_batched_tokens,
                       "Total tokens scheduled exceeds max_num_batched_tokens.");
    FT_CHECK_WITH_INFO(prefill_batch.size() + decode_batch.size() <= cfg_.max_num_seqs,
                       "Total sequences scheduled exceeds max_num_seqs.");
    
    // Check for invariant violations regarding empty queues with non-empty seq_states_
    FT_CHECK_WITH_INFO((!prefill_request_queue_.empty() || !decode_request_queue_.empty() || seq_states_.empty()),
                       "Queues are empty but seq_states_ is not. Potentially stuck sequences.");
}

}  // namespace turbomind