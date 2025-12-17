#pragma once

#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h" // Include PrefixCache header
#include <cstdint>
#include <unordered_map>
#include <mutex>

namespace turbomind {

class CapacityScheduler {
public:
    explicit CapacityScheduler(KVCacheManager* kv_mgr, PrefixCache* prefix_cache);

    virtual ~CapacityScheduler() = default;

    virtual bool try_start_request(uint64_t seq_id,
                                   const KVUsageEstimate& est,
                                   KVReservation* out,
                                   const std::vector<int>& pre_existing_page_ids = {});
    virtual void finish_request(uint64_t seq_id);

    size_t blocked_due_to_capacity() const;
    size_t rejected_never_schedulable() const;
    size_t active_reservation_count() const;

private:
    KVCacheManager* kv_mgr_;
    PrefixCache*    prefix_cache_; // New member
    std::unordered_map<uint64_t, KVReservation> active_reservations_;
    size_t blocked_due_to_capacity_{0};
    size_t rejected_never_schedulable_{0};
    mutable std::mutex mutex_;
};

} // namespace turbomind