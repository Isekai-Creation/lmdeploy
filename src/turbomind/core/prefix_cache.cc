
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/utils/logger.h"

#include <algorithm>   // For std::min
#include <limits>      // For std::numeric_limits
#include <chrono>      // For timestamps

namespace turbomind {

PrefixCache::PrefixCache(int page_size, KVCacheManager* kv_cache_manager)
    : page_size_(page_size), kv_cache_manager_(kv_cache_manager)
{
    TM_LOG_INFO("[PrefixCache] Initialized with page_size = %d, KVCacheManager = %p", page_size_, (void*)kv_cache_manager_);
}

PrefixMatchResult PrefixCache::match(const PrefixKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    PrefixMatchResult result;
    result.matched_tokens = 0;
    
    // For now, only exact match or no match. Longest prefix matching is more complex.
    auto it = cache_map_.find(key);
    if (it != cache_map_.end()) {
        result.page_indices   = it->second.page_indices;
        result.matched_tokens = static_cast<int>(key.tokens.size());

        // Update last_access_ts for LRU
        it->second.last_access_ts =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch())
                .count();

        hit_count_++;
        TM_LOG_INFO(
            "[PrefixCache] Exact match found for key with %zu tokens, seq_id %llu. Hit count: %zu",
            key.tokens.size(),
            it->second.seq_id,
            hit_count_);
    }
    else {
        miss_count_++;
        TM_LOG_INFO("[PrefixCache] No exact match found for key with %zu tokens. Miss count: %zu",
                    key.tokens.size(),
                    miss_count_);
    }

    return result;
}

void PrefixCache::insert(const PrefixKey& key,
                          const std::vector<int32_t>& page_indices,
                          int priority,
                          uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // Truncate tokens to page boundaries before storing.
    std::vector<int32_t> truncated_tokens = key.tokens;
    if (page_size_ > 0 && truncated_tokens.size() > 0) {
        size_t effective_len = (truncated_tokens.size() / page_size_) * page_size_;
        // If the key has tokens but less than page_size_, we still want to store it (e.g. 1 token and page_size_=128)
        // This makes the first page always fill up to page_size tokens or all available tokens.
        if (effective_len == 0 && truncated_tokens.size() > 0) {
             effective_len = page_size_;
        }
        if (effective_len > 0) {
            truncated_tokens.resize(std::min(truncated_tokens.size(), effective_len));
        } else {
            truncated_tokens.clear();
        }
    }
    
    if (truncated_tokens.empty()) {
        TM_LOG_WARNING("[PrefixCache] Attempted to insert empty or too short prefix for seq_id %llu. Skipping.", seq_id);
        return;
    }

    PrefixKey effective_key = {truncated_tokens, key.namespace_id};

    // Check if an existing entry for this effective_key is already present.
    // If so, update it instead of re-inserting.
    auto it = cache_map_.find(effective_key);
    if (it != cache_map_.end()) {
        // Update existing entry
        it->second.page_indices = page_indices;
        it->second.priority = priority;
        it->second.last_access_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        it->second.seq_id = seq_id;
        seq_id_to_prefix_key_[seq_id] = effective_key;
        TM_LOG_INFO("[PrefixCache] Updated existing prefix for seq_id %llu.", seq_id);
        return;
    }

    // Insert new entry
    CacheEntry entry;
    entry.page_indices = page_indices;
    entry.priority = priority;
    entry.last_access_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    entry.seq_id = seq_id;

    cache_map_[effective_key] = entry;
    seq_id_to_prefix_key_[seq_id] = effective_key;

    TM_LOG_INFO("[PrefixCache] Inserted prefix for seq_id %llu with %zu effective tokens and %zu page indices. Total entries: %zu", seq_id, effective_key.tokens.size(), page_indices.size(), cache_map_.size());
}

void PrefixCache::erase(uint64_t seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it_seq_key = seq_id_to_prefix_key_.find(seq_id);
    if (it_seq_key != seq_id_to_prefix_key_.end()) {
        const PrefixKey& key_to_erase = it_seq_key->second;
        auto it_cache = cache_map_.find(key_to_erase);
        if (it_cache != cache_map_.end()) {
            if (kv_cache_manager_) {
                // Release KV pages associated with this prefix
                // This assumes KVCacheManager::release can handle releasing pages
                // that were reserved by the prefix cache.
                // It will require an adaptation of KVCacheManager::release to use seq_id
                // that represents the prefix.
                // For now, kv_cache_manager_->release is called with original seq_id.
                kv_cache_manager_->release(seq_id);
            }
            cache_map_.erase(it_cache);
            TM_LOG_INFO("[PrefixCache] Erased prefix for seq_id %llu. Released KV pages. Total entries: %zu", seq_id, cache_map_.size());
        }
        seq_id_to_prefix_key_.erase(it_seq_key);
    } else {
        TM_LOG_WARNING("[PrefixCache] Attempted to erase non-existent prefix for seq_id %llu.", seq_id);
    }
}

void PrefixCache::evict_lru_entry() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cache_map_.empty()) {
        return;
    }

    uint64_t oldest_ts = std::numeric_limits<uint64_t>::max();
    auto lru_it = cache_map_.end();

    for (auto it = cache_map_.begin(); it != cache_map_.end(); ++it) {
        if (it->second.last_access_ts < oldest_ts) {
            oldest_ts = it->second.last_access_ts;
            lru_it = it;
        }
    }

    if (lru_it != cache_map_.end()) {
        uint64_t seq_id_to_evict = lru_it->second.seq_id;
        size_t num_pages_evicted = lru_it->second.page_indices.size();
        
        if (kv_cache_manager_) {
            kv_cache_manager_->release(seq_id_to_evict);
            bytes_evicted_ += num_pages_evicted * kv_cache_manager_->page_bytes();
        }
        
        cache_map_.erase(lru_it);
        seq_id_to_prefix_key_.erase(seq_id_to_evict); // Also remove from seq_id_to_prefix_key_
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
