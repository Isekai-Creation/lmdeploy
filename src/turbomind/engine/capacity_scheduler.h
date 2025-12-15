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

    bool try_start_request(uint64_t seq_id,
                           const KVUsageEstimate& est,
                           KVReservation* out,
                           const std::vector<int>& pre_existing_page_ids = {});
    void finish_request(uint64_t seq_id);
private:
    KVCacheManager* kv_mgr_;
    PrefixCache*    prefix_cache_; // New member
    // Keep track of active reservations (seq_id -> KVReservation)
    std::unordered_map<uint64_t, KVReservation> active_reservations_; 
    mutable std::mutex mutex_;
};

} // namespace turbomind