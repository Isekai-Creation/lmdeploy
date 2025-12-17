#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/core/prefix_cache.h" // Include PrefixCache header

namespace turbomind {

// Define max eviction attempts to prevent infinite loops in extreme cases
static constexpr int MAX_EVICTION_ATTEMPTS = 5;

CapacityScheduler::CapacityScheduler(KVCacheManager* kv_mgr, PrefixCache* prefix_cache) 
    : kv_mgr_(kv_mgr), prefix_cache_(prefix_cache)
{
    TM_LOG_INFO("[CapacityScheduler] Initialized with KVCacheManager = %p, PrefixCache = %p", (void*)kv_mgr_, (void*)prefix_cache_);
}

bool CapacityScheduler::try_start_request(uint64_t seq_id,
                                          const KVUsageEstimate& est,
                                          KVReservation* out,
                                          const std::vector<int>& pre_existing_page_ids)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (active_reservations_.count(seq_id)) {
        TM_LOG_WARNING("[CapacityScheduler] Attempted to start already active request %llu.", seq_id);
        return false;
    }

    if (est.pages_needed > kv_mgr_->total_pages()) {
        rejected_never_schedulable_++;
        TM_LOG_WARNING("[CapacityScheduler] Request %llu needs %zu pages > total %zu. Never schedulable.",
                       seq_id, est.pages_needed, kv_mgr_->total_pages());
        return false;
    }

    KVReservation reservation{};
    bool success = kv_mgr_->reserve(seq_id, est, &reservation, pre_existing_page_ids);

    if (!success && prefix_cache_) {
        TM_LOG_DEBUG("[CapacityScheduler] Initial KV reservation failed for seq %llu, attempting eviction.", seq_id);
        for (int i = 0; i < MAX_EVICTION_ATTEMPTS; ++i) {
            prefix_cache_->evict_lru_entry();
            success = kv_mgr_->reserve(seq_id, est, &reservation, pre_existing_page_ids);
            if (success) {
                TM_LOG_DEBUG("[CapacityScheduler] KV reservation succeeded for seq %llu after %d evictions.", seq_id, i + 1);
                break;
            }
        }
    }

    if (success) {
        active_reservations_[seq_id] = reservation;
        if (out) {
            *out = reservation;
        }
        TM_LOG_DEBUG("[CapacityScheduler] Successfully started request %llu, reserved %d pages.", seq_id, reservation.num_pages);
    }
    else {
        blocked_due_to_capacity_++;
        TM_LOG_DEBUG("[CapacityScheduler] Failed to start request %llu after attempts. Insufficient KV capacity or invalid pre-existing pages.", seq_id);
    }
    return success;
}

void CapacityScheduler::finish_request(uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_reservations_.find(seq_id);
    if (it != active_reservations_.end()) {
        kv_mgr_->release(seq_id);
        if (prefix_cache_) {
            prefix_cache_->erase(seq_id);
        }
        active_reservations_.erase(it);
        TM_LOG_DEBUG("[CapacityScheduler] Finished request %llu, released resources.", seq_id);
    } else {
        TM_LOG_WARNING("[CapacityScheduler] Attempted to finish non-existent or inactive request %llu.", seq_id);
    }
}

size_t CapacityScheduler::blocked_due_to_capacity() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return blocked_due_to_capacity_;
}

size_t CapacityScheduler::rejected_never_schedulable() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return rejected_never_schedulable_;
}

size_t CapacityScheduler::active_reservation_count() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return active_reservations_.size();
}

} // namespace turbomind
