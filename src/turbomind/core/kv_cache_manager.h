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
    int        page_size{0};       // tokens per page
    int        bytes_per_value{0}; // derived from kv_dtype
    int        kv_factor{2};       // default K + V
    KVDataType kv_dtype{KVDataType::kFP16};
};


struct KVReservation {
    uint64_t         seq_id{0};
    int              first_page{-1};  // Index of the first page allocated to this sequence
    int              num_pages{0};    // Total number of pages allocated to this sequence
    std::vector<int> page_ids;        // Concrete pages reserved for this sequence
    uint64_t         kv_cookie{0};    // Hash/cookie for tracing/debug
};

// Internal representation of a page in the KV cache
struct KVCachePage {
    int   id;         // Unique ID of the page
    void* data;       // Pointer to the allocated memory for this page
    // Additional metadata for LRU/priority eviction could go here
};

class KVCacheManager {
public:
    KVCacheManager(const KVLayout& layout, size_t total_capacity_bytes);

    virtual ~KVCacheManager();

    // Page accounting.
    size_t total_pages() const;
    size_t used_pages() const;
    size_t free_pages() const;

    size_t page_bytes() const { return page_bytes_; }
    int    kv_factor() const { return kv_factor_; }

    // Capacity checks and reservation API.
    bool   can_reserve(uint64_t seq_id, const KVUsageEstimate& est) const;
    virtual bool reserve(uint64_t seq_id, const KVUsageEstimate& est, KVReservation* out);
    virtual bool reserve(uint64_t                     seq_id,
                         const KVUsageEstimate&       est,
                         KVReservation*              out,
                         const std::vector<int>&     pre_existing_page_ids);
    virtual void release(uint64_t seq_id);

    // Translate (seq_id, position) -> physical page index(es) and data ptrs.
    virtual int   page_for(uint64_t seq_id, int position) const;
    virtual void* get_page_data_ptr(int page_id) const;

    virtual const KVLayout& get_layout() const { return layout_; }
    virtual std::vector<int> get_sequence_page_ids(uint64_t seq_id) const;

    // Batch mapping helpers for executor mode.
    virtual bool   build_page_ptr_table(const std::vector<int>& page_ids,
                                        std::vector<void*>&     out_ptrs) const;
    virtual void*  base_pointer() const { return base_ptr_; }
    virtual size_t storage_bytes() const { return storage_bytes_; }
    virtual const KVLayout& layout() const { return layout_; }

    // For testing.
    int get_page_ref_count(int page_id) const;

    // Helper to estimate KV usage for a given model layout and
    // prompt/max_new_tokens pair. This mirrors the block-level KV
    // sizing used by the attention kernels and is used by the
    // DriftEngine guardrails.
    static KVUsageEstimate estimate_usage(const ModelLayout& layout,
                                          int                prompt_len,
                                          int                max_new_tokens);

private:
    KVLayout layout_;
    size_t   total_capacity_bytes_{0};

    // Size of a single logical page (block) in bytes.
    size_t page_bytes_{0};

    // Contiguous KV storage backing all pages.
    void*  base_ptr_{nullptr};
    size_t storage_bytes_{0};

    // Physical pages.
    std::vector<KVCachePage> all_pages_;
    std::queue<int>          free_page_ids_;

    // Map from sequence ID to its allocated pages.
    std::unordered_map<uint64_t, KVReservation> seq_reservations_;
    std::unordered_map<uint64_t, std::vector<int>> seq_page_map_;

    // Refcount per page for shared-prefix KV reuse.
    std::vector<int> page_ref_counts_;

    mutable std::mutex mutex_;

    // K/V factor (typically 2 for K+V).
    int kv_factor_{2};
};


}  // namespace turbomind
