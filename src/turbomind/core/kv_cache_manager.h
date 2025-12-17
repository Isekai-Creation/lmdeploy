#pragma once

#include "src/turbomind/models/common/model_layout.h"
#include <cstdint>
#include <vector>
#include <unordered_map> // For sequence to page mapping
#include <queue>         // For free pages list
#include <mutex>         // For thread safety

namespace turbomind {

// Forward declarations to avoid including heavy headers
// struct SchedulerConfig; // No longer needed directly here

struct KVUsageEstimate {
    size_t pages_needed;
    size_t bytes_needed;
};

struct KVLayout {
    int        num_layers{0};
    int        num_kv_heads{0};
    int        head_dim{0};
    int        page_size{0};      // tokens per page
    int        bytes_per_value{0}; // derived from kv_dtype
    KVDataType kv_dtype{KVDataType::kFP16};
};

struct KVReservation {
    uint64_t seq_id;
    int      first_page; // Index of the first page allocated to this sequence
    int      num_pages;  // Total number of pages allocated to this sequence
};

// Internal representation of a page in the KV cache
struct KVCachePage {
    int   id;         // Unique ID of the page
    void* data;       // Pointer to the allocated memory for this page
    bool  is_free;    // Whether this page is currently free
    // Additional metadata for LRU/priority eviction could go here
};

class KVCacheManager {
public:
    KVCacheManager(const KVLayout& layout,
                   size_t total_capacity_bytes);

    virtual ~KVCacheManager(); // Destructor to free memory

    size_t total_pages() const;
    size_t used_pages() const;
    size_t free_pages() const; // New helper

    size_t page_bytes() const { return page_bytes_; }

    bool   can_reserve(uint64_t seq_id,
                       const KVUsageEstimate& est) const;
    virtual bool   reserve(uint64_t seq_id,
                           const KVUsageEstimate& est,
                           KVReservation* out);
    virtual bool   reserve(uint64_t seq_id,
                           const KVUsageEstimate& est,
                           KVReservation* out,
                           const std::vector<int>& pre_existing_page_ids);
    virtual void   release(uint64_t seq_id);

    // Translate (seq_id, position) -> physical page index(es)
    virtual int    page_for(uint64_t seq_id, int position) const;
    virtual void*  get_page_data_ptr(int page_id) const; // New method
    virtual const KVLayout& get_layout() const { return layout_; } // New method
    virtual std::vector<int> get_sequence_page_ids(uint64_t seq_id) const; // New method

    // For testing
    int get_page_ref_count(int page_id) const;

    static KVUsageEstimate estimate_usage(const ModelLayout& layout,
                                          int prompt_len,
                                          int max_new_tokens);
private:
    KVLayout layout_;
    size_t total_capacity_bytes_;
    size_t page_bytes_; // Size of a single page in bytes

    std::vector<KVCachePage> all_pages_; // All pages in the cache
    std::queue<int> free_page_ids_;     // Queue of IDs of free pages

    // Map from sequence ID to its allocated pages
    std::unordered_map<uint64_t, KVReservation> seq_reservations_;
    // Map from sequence ID and its relative position to physical page ID
    std::unordered_map<uint64_t, std::vector<int>> seq_page_map_;

    std::vector<int> page_ref_counts_; // New member for reference counting
    mutable std::mutex mutex_; // For thread safety
};

}  // namespace turbomind
