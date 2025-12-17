
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/utils/logger.h"

#include <algorithm>   // For std::min
#include <chrono>      // For timestamps
#include <limits>      // For std::numeric_limits

namespace turbomind {

namespace {

inline uint64_t now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

std::vector<int32_t> normalize_tokens(const std::vector<int32_t>& tokens, int page_size)
{
    if (page_size <= 0 || tokens.empty()) {
        return {};
    }
    const size_t aligned = (tokens.size() / page_size) * page_size;
    if (aligned == 0) {
        return {};
    }
    return std::vector<int32_t>(tokens.begin(), tokens.begin() + aligned);
}

} // namespace

PrefixCache::PrefixCache(int page_size, KVCacheManager* kv_cache_manager)
    : page_size_(page_size), kv_cache_manager_(kv_cache_manager)
{
    TM_LOG_INFO("[PrefixCache] Initialized with page_size = %d, KVCacheManager = %p", page_size_, (void*)kv_cache_manager_);
}

PrefixMatchResult PrefixCache::match(const PrefixKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    PrefixMatchResult result{};
    result.matched_tokens = 0;

    std::vector<int32_t> aligned_tokens = normalize_tokens(key.tokens, page_size_);
    if (aligned_tokens.empty()) {
        miss_count_++;
        return result;
    }

    PrefixKey aligned_key{aligned_tokens, key.namespace_id};
    auto      it = cache_map_.find(aligned_key);
    if (it != cache_map_.end()) {
        result.page_indices      = it->second.page_indices;
        result.matched_tokens    = static_cast<int>(aligned_tokens.size());
        it->second.last_access_ts = now_ms();
        hit_count_++;
        return result;
    }

    miss_count_++;
    return result;
}

void PrefixCache::insert(const PrefixKey& key,
                           const std::vector<int32_t>& page_indices,
                           int priority,
                           uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int32_t> truncated_tokens = normalize_tokens(key.tokens, page_size_);
    if (truncated_tokens.empty()) {
        // Short or non page-aligned prefixes are expected in normal operation
        // (e.g. early chunks or prompts shorter than one page). Treat this as
        // a no-op rather than a warning to avoid log spam.
        TM_LOG_DEBUG("[PrefixCache] Skipping insert for seq_id %llu due to non page-aligned or short prefix.", seq_id);
        return;
    }
    PrefixKey            effective_key{std::move(truncated_tokens), key.namespace_id};

    auto it = cache_map_.find(effective_key);
    uint64_t now_ts = now_ms();

    if (it != cache_map_.end()) {
        it->second.page_indices   = page_indices;
        it->second.priority       = priority;
        it->second.last_access_ts = now_ts;
        it->second.seq_id         = seq_id;
        seq_id_to_prefix_key_[seq_id] = effective_key;
        TM_LOG_INFO("[PrefixCache] Updated existing prefix for seq_id %llu.", seq_id);
        return;
    }

    CacheEntry entry;
    entry.page_indices   = page_indices;
    entry.priority       = priority;
    entry.last_access_ts = now_ts;
    entry.seq_id         = seq_id;

    cache_map_[effective_key]     = entry;
    seq_id_to_prefix_key_[seq_id] = effective_key;

    TM_LOG_INFO("[PrefixCache] Inserted prefix for seq_id %llu with %zu aligned tokens and %zu page indices. Total entries: %zu",
                seq_id,
                effective_key.tokens.size(),
                page_indices.size(),
                cache_map_.size());
}

void PrefixCache::erase(uint64_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it_seq_key = seq_id_to_prefix_key_.find(seq_id);
    if (it_seq_key != seq_id_to_prefix_key_.end()) {
        const PrefixKey& key_to_erase = it_seq_key->second;
        cache_map_.erase(key_to_erase);
        seq_id_to_prefix_key_.erase(it_seq_key);
    }
}

void PrefixCache::evict_lru_entry() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_map_.empty()) {
        return;
    }

    uint64_t oldest_ts = std::numeric_limits<uint64_t>::max();
    auto     lru_it    = cache_map_.end();

    for (auto it = cache_map_.begin(); it != cache_map_.end(); ++it) {
        if (it->second.last_access_ts < oldest_ts) {
            oldest_ts = it->second.last_access_ts;
            lru_it    = it;
        }
    }

    if (lru_it != cache_map_.end()) {
        uint64_t seq_id_to_evict   = lru_it->second.seq_id;
        size_t   num_pages_evicted = lru_it->second.page_indices.size();

        if (kv_cache_manager_) {
            // Only track bytes evicted for metrics; KV page lifetime is
            // owned by the scheduler / CapacityScheduler, not the
            // PrefixCache.
            bytes_evicted_ += num_pages_evicted * kv_cache_manager_->page_bytes();
        }

        cache_map_.erase(lru_it);
        seq_id_to_prefix_key_.erase(seq_id_to_evict);
        eviction_count_++;
        TM_LOG_INFO("[PrefixCache] Evicted LRU entry for seq_id %llu. Eviction count: %zu", seq_id_to_evict, eviction_count_);
    }
}


size_t PrefixCache::get_hit_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hit_count_;
}

size_t PrefixCache::get_miss_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return miss_count_;
}

size_t PrefixCache::get_eviction_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return eviction_count_;
}

size_t PrefixCache::get_bytes_evicted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_evicted_;
}

}  // namespace turbomind
