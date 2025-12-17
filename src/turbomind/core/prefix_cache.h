
#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <memory> // For std::shared_ptr, if needed internally.
#include <mutex>  // For thread safety

#include "src/turbomind/core/kv_cache_manager.h" // Add include

namespace turbomind {

// Forward declarations for internal use if needed.
// class KVCacheManager; 

struct PrefixKey {
    std::vector<int32_t> tokens;
    uint64_t             namespace_id;  // e.g., adapter id or 0

    // Define comparison operators for use in std::map or std::unordered_map
    bool operator==(const PrefixKey& other) const {
        return tokens == other.tokens && namespace_id == other.namespace_id;
    }
    struct Hasher {
        size_t operator()(const PrefixKey& pk) const {
            size_t seed = std::hash<uint64_t>{}(pk.namespace_id);
            // Simple hash combine for the token sequence.
            for (int32_t token : pk.tokens) {
                seed ^= std::hash<int32_t>{}(token) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
};

// Simple hash for std::vector<int32_t> for now, as std::hash for vector is not standard until C++20.
// This should be part of PrefixKey::Hasher or a global specialization.
// template<> struct hash<std::vector<int32_t>> {
//     size_t operator()(const std::vector<int32_t>& v) const {
//         size_t seed = v.size();
//         for(int32_t i : v) {
//             seed ^= hash<int32_t>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         }
//         return seed;
//     }
// };


struct PrefixMatchResult {
    std::vector<int32_t> page_indices;  // KV pages covering matched prefix
    int                  matched_tokens;
};

class PrefixCache {
public:
    explicit PrefixCache(int page_size, KVCacheManager* kv_cache_manager); // Add KVCacheManager

    virtual ~PrefixCache() = default;

    virtual PrefixMatchResult match(const PrefixKey& key) const;
    virtual void insert(const PrefixKey& key,
                        const std::vector<int32_t>& page_indices,
                        int priority = 0,
                        uint64_t seq_id = 0);
    virtual void erase(uint64_t seq_id); // Erase by sequence ID, not PrefixKey

    // Eviction logic
    void evict_lru_entry(); // Evict the least recently used entry

    // Metrics
    size_t get_hit_count() const;
    size_t get_miss_count() const;
    size_t get_eviction_count() const;
    size_t get_bytes_evicted() const;

private:
    int page_size_;
    KVCacheManager* kv_cache_manager_; // New member

    // Internal storage for cache entries
    // Key: PrefixKey, Value: CacheEntry (containing page_indices and priority/timestamp for eviction)
    struct CacheEntry {
        std::vector<int32_t> page_indices;
        int                  priority;
        uint64_t             last_access_ts; // For LRU eviction
        uint64_t             seq_id;         // To map back to sequence for erase
    };
    mutable std::unordered_map<PrefixKey, CacheEntry, PrefixKey::Hasher> cache_map_;
    std::unordered_map<uint64_t, PrefixKey> seq_id_to_prefix_key_; // Map sequence ID to its PrefixKey for erase operation

    // Metrics
    mutable size_t hit_count_      = 0;
    mutable size_t miss_count_     = 0;
    mutable size_t eviction_count_ = 0;
    mutable size_t bytes_evicted_  = 0;

    mutable std::mutex mutex_; // For thread safety
};

}  // namespace turbomind
