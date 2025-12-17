#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/progress_logger.h"
#include <cuda_runtime_api.h> // For cudaMalloc and cudaFree
#include <cassert>
#include <cstdint>
#include <numeric> // For std::iota
#include <unordered_set>

namespace turbomind {
namespace {

uint64_t hash_page_ids(uint64_t seq_id, const std::vector<int>& page_ids)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= seq_id + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    for (const int page_id : page_ids) {
        hash ^= static_cast<uint64_t>(page_id) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    return hash;
}

}  // namespace

// Helper functions for CUDA memory management
static void* allocate_device_memory(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        TM_LOG_ERROR("CUDA memory allocation failed: %s", cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

static void free_device_memory(void* ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            TM_LOG_ERROR("CUDA memory free failed: %s", cudaGetErrorString(err));
        }
    }
}

KVCacheManager::KVCacheManager(const KVLayout& layout, size_t total_capacity_bytes)
    : layout_(layout), total_capacity_bytes_(total_capacity_bytes)
{
    FT_CHECK_WITH_INFO(layout_.num_layers > 0, "KVLayout.num_layers must be positive");
    FT_CHECK_WITH_INFO(layout_.num_kv_heads > 0, "KVLayout.num_kv_heads must be positive");
    FT_CHECK_WITH_INFO(layout_.head_dim > 0, "KVLayout.head_dim must be positive");
    FT_CHECK_WITH_INFO(layout_.page_size > 0, "KVLayout.page_size must be positive");

    int effective_bpv       = layout_.bytes_per_value > 0 ? layout_.bytes_per_value : bytes_per_value_from_dtype(layout_.kv_dtype);
    layout_.bytes_per_value = effective_bpv;

    const int kv_factor = layout_.kv_factor > 0 ? layout_.kv_factor : 2;
    layout_.kv_factor   = kv_factor;
    kv_factor_          = kv_factor;

    // Each logical "page" stores both K and V activations for all
    // layers and KV heads at `page_size` token positions.
    page_bytes_ = static_cast<size_t>(layout_.num_layers)
                   * layout_.num_kv_heads
                   * layout_.head_dim
                   * layout_.page_size
                   * effective_bpv
                   * kv_factor;

    TM_LOG_INFO(
        "[KVCacheManager] layout: layers=%d kv_heads=%d head_dim=%d page_size=%d dtype=%d bpv=%d kv_factor=%d page_bytes=%zu",
        layout_.num_layers,
        layout_.num_kv_heads,
        layout_.head_dim,
        layout_.page_size,
        static_cast<int>(layout_.kv_dtype),
        effective_bpv,
        kv_factor,
        page_bytes_);

    if (page_bytes_ == 0) {
        TM_LOG_ERROR("KVCacheManager page_bytes computed as 0. Invalid layout.");
        return;
    }

    size_t num_pages = total_capacity_bytes_ / page_bytes_;
    if (num_pages == 0) {
        TM_LOG_WARNING("KVCacheManager initialized with 0 pages. total_capacity_bytes: %lu, page_bytes: %lu", total_capacity_bytes_, page_bytes_);
        return;
    }

    // Allocate a single contiguous device buffer for all KV data. Before
    // doing so, capture a GPU memory snapshot so any OOM here can be
    // attributed precisely in logs.
    size_t     free_bytes  = 0;
    size_t     total_bytes = 0;
    auto       info_err    = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (info_err == cudaSuccess) {
        TM_LOG_INFO(
            "[KVCacheManager] about_to_alloc bytes=%zu free=%zu total=%zu",
            num_pages * page_bytes_,
            free_bytes,
            total_bytes);
    }
    else {
        TM_LOG_WARNING("[KVCacheManager] cudaMemGetInfo failed before alloc: %s",
                       cudaGetErrorString(info_err));
    }

    // Real allocation
    void* all_kv_data_ptr = allocate_device_memory(num_pages * page_bytes_);
    if (!all_kv_data_ptr) {
        TM_LOG_ERROR("Failed to allocate contiguous device memory for KV cache.");
        // Mark manager as invalid or throw exception if desired
        return;
    }

    all_pages_.reserve(num_pages);
    page_ref_counts_.assign(num_pages, 0);  // Initialize ref counts
    base_ptr_      = all_kv_data_ptr;
    storage_bytes_ = num_pages * page_bytes_;

    for (size_t i = 0; i < num_pages; ++i) {
        KVCachePage page;
        page.id   = static_cast<int>(i);
        page.data = static_cast<char*>(all_kv_data_ptr) + (i * page_bytes_);
        all_pages_.push_back(page);
        free_page_ids_.push(static_cast<int>(i));  // Add all pages to the free list
    }
    TM_LOG_INFO("KVCacheManager initialized with %zu pages, total capacity %lu bytes.", num_pages, total_capacity_bytes_);
}

KVCacheManager::~KVCacheManager()
{
    // Free the contiguous device buffer
    if (base_ptr_) {
        free_device_memory(base_ptr_);
        base_ptr_ = nullptr;
    }
    TM_LOG_INFO("KVCacheManager destroyed, memory freed.");
}

size_t KVCacheManager::total_pages() const
{
    return all_pages_.size();
}

size_t KVCacheManager::used_pages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return all_pages_.size() - free_page_ids_.size();
}

size_t KVCacheManager::free_pages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return free_page_ids_.size();
}

bool KVCacheManager::can_reserve(uint64_t seq_id, const KVUsageEstimate& est) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    // It's a const method, so we should check if seq_id already exists to prevent
    // double-reservation, though reserve() also checks.
    if (seq_reservations_.count(seq_id)) {
        TM_LOG_WARNING("Attempted to check reservation for already reserved sequence ID: %lu", seq_id);
        return false;
    }
    return est.pages_needed <= free_page_ids_.size();
}

// Existing reserve calls the new overloaded version
bool KVCacheManager::reserve(uint64_t seq_id, const KVUsageEstimate& est, KVReservation* out)
{
    return reserve(seq_id, est, out, {}); // Call the overloaded version with no pre-existing pages
}

// New overloaded reserve method to handle pre-existing pages from prefix cache
bool KVCacheManager::reserve(uint64_t seq_id,
                             const KVUsageEstimate& est,
                             KVReservation* out,
                             const std::vector<int>& pre_existing_page_ids)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (seq_reservations_.count(seq_id)) {
        TM_LOG_WARNING("Attempted to reserve KV for already reserved sequence ID: %lu", seq_id);
        return false;
    }
    if (est.pages_needed == 0 && pre_existing_page_ids.empty()) {
        TM_LOG_DEBUG("Attempted to reserve 0 pages for sequence ID: %lu. Returning true.", seq_id);
        return true; // Nothing to reserve
    }

    std::vector<int> allocated_page_ids;
    allocated_page_ids.reserve(est.pages_needed); // Pre-reserve capacity

    // First, incorporate pre-existing pages from the prefix cache
    std::unordered_set<int> seen_pages;
    for (int page_id : pre_existing_page_ids) {
        if (page_id < 0 || page_id >= static_cast<int>(all_pages_.size())) {
            TM_LOG_ERROR("Pre-existing page ID %d for sequence %lu is invalid. Failing reservation.", page_id, seq_id);
            return false;
        }
        if (page_ref_counts_[page_id] <= 0) {
            TM_LOG_ERROR("Pre-existing page ID %d for sequence %lu has non-positive refcount (%d). Failing reservation.",
                         page_id,
                         seq_id,
                         page_ref_counts_[page_id]);
            return false;
        }
        if (!seen_pages.insert(page_id).second) {
            TM_LOG_ERROR("Duplicate pre-existing page ID %d for sequence %lu.", page_id, seq_id);
            return false;
        }
        // Increment ref count for pre-existing pages (shared KV pages).
        page_ref_counts_[page_id]++;
        allocated_page_ids.push_back(page_id);
    }

    if (!pre_existing_page_ids.empty()) {
        TM_LOG_DEBUG("Reusing %zu prefix-cache pages for sequence ID: %lu", pre_existing_page_ids.size(), seq_id);
        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kKVReserve};
            evt.pct           = 35;
            evt.seq_id        = seq_id;
            evt.session_id    = seq_id;
            evt.kv_pages_seq  = static_cast<int>(pre_existing_page_ids.size());
            evt.msg           = "use_prefix_pages";
            ProgressLogger::Log(evt);
        }
    }

    // Calculate how many more pages are needed
    size_t remaining_pages_needed = est.pages_needed > allocated_page_ids.size() ? est.pages_needed - allocated_page_ids.size() : 0;

    auto revert_shared_page = [&](int page_id) {
        if (page_id < 0 || page_id >= static_cast<int>(page_ref_counts_.size())) {
            return;
        }
        if (page_ref_counts_[page_id] <= 0) {
            return;
        }
        page_ref_counts_[page_id]--;
        if (page_ref_counts_[page_id] == 0) {
            free_page_ids_.push(page_id);
        }
    };

    if (remaining_pages_needed > free_page_ids_.size()) {
        TM_LOG_DEBUG("Insufficient free pages to reserve for sequence ID: %lu. Needed: %zu, Available: %zu (after pre-existing: %zu)",
                  seq_id, est.pages_needed, free_page_ids_.size(), remaining_pages_needed);
        // If not enough pages are available, revert the ref count increments
        for (int page_id : pre_existing_page_ids) {
            revert_shared_page(page_id);
        }
        return false;
    }
    if (allocated_page_ids.size() > est.pages_needed) {
        TM_LOG_ERROR("Pre-existing pages (%zu) exceed requested pages (%zu) for sequence %lu",
                     allocated_page_ids.size(), est.pages_needed, seq_id);
        for (int page_id : pre_existing_page_ids) {
            revert_shared_page(page_id);
        }
        return false;
    }
    
    // Allocate remaining pages from the free list
    for (size_t i = 0; i < remaining_pages_needed; ++i) {
        int page_id = free_page_ids_.front();
        free_page_ids_.pop();
        page_ref_counts_[page_id] = 1;  // New page starts with ref count 1
        allocated_page_ids.push_back(page_id);
    }

    KVReservation reservation;
    reservation.seq_id    = seq_id;
    reservation.first_page = allocated_page_ids.empty() ? -1 : allocated_page_ids[0];
    reservation.num_pages  = allocated_page_ids.size();
    reservation.page_ids   = allocated_page_ids;
    reservation.kv_cookie  = hash_page_ids(seq_id, allocated_page_ids);
 
    seq_reservations_[seq_id] = reservation;
    seq_page_map_[seq_id]     = allocated_page_ids;


    if (out) {
        *out = reservation;
    }
    TM_LOG_DEBUG("Reserved %zu pages (including %zu pre-existing) for sequence ID: %lu. First page: %d", 
              allocated_page_ids.size(), pre_existing_page_ids.size(), seq_id, reservation.first_page);
    return true;
}

void KVCacheManager::release(uint64_t seq_id)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto res_it = seq_reservations_.find(seq_id);
    if (res_it == seq_reservations_.end()) {
        TM_LOG_WARNING("Attempted to release non-existent reservation for sequence ID: %lu", seq_id);
        return;
    }

    auto page_map_it = seq_page_map_.find(seq_id);
    if (page_map_it == seq_page_map_.end()) { // Should not happen if res_it is found
        TM_LOG_ERROR("Internal error: Reservation found but page map missing for sequence ID: %lu", seq_id);
        return;
    }

    const std::vector<int>& allocated_page_ids = page_map_it->second;
    for (int page_id : allocated_page_ids) {
        if (page_id < 0 || page_id >= static_cast<int>(all_pages_.size())) {
            TM_LOG_ERROR("Internal error: Invalid page ID %d in allocated_page_ids for sequence %lu", page_id, seq_id);
            continue;
        }
        if (page_ref_counts_[page_id] <= 0) {
            TM_LOG_ERROR("Page ref count underflow for page %d during release of seq %lu", page_id, seq_id);
            continue;
        }
        page_ref_counts_[page_id]--;

        if (page_ref_counts_[page_id] == 0) {
            free_page_ids_.push(page_id);
        }
    }

    seq_reservations_.erase(res_it);
    seq_page_map_.erase(page_map_it);
    TM_LOG_DEBUG("Released %zu pages for sequence ID: %lu", allocated_page_ids.size(), seq_id);
}

int KVCacheManager::page_for(uint64_t seq_id, int position) const
{
    std::lock_guard<std::mutex> lock(mutex_); // Lock for read access
    auto it = seq_page_map_.find(seq_id);
    if (it == seq_page_map_.end()) {
        TM_LOG_ERROR("No pages allocated for sequence ID: %lu", seq_id);
        return -1; // Or throw exception
    }

    int page_idx_in_seq = position / layout_.page_size;
    if (page_idx_in_seq < 0 || page_idx_in_seq >= it->second.size()) {
        TM_LOG_ERROR("Position %d out of bounds for sequence %lu. Page index in seq: %d, num allocated pages: %zu",
                  position, seq_id, page_idx_in_seq, it->second.size());
        return -1; // Or throw exception
    }
    return it->second[page_idx_in_seq];
}

void* KVCacheManager::get_page_data_ptr(int page_id) const
{
    std::lock_guard<std::mutex> lock(mutex_); // Lock for read access
    if (page_id < 0 || page_id >= static_cast<int>(all_pages_.size())) {
        TM_LOG_ERROR("Invalid page ID: %d", page_id);
        return nullptr;
    }
    return all_pages_[page_id].data;
}

std::vector<int> KVCacheManager::get_sequence_page_ids(uint64_t seq_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = seq_page_map_.find(seq_id);
    if (it == seq_page_map_.end()) {
        return {};
    }
    return it->second;
}

int KVCacheManager::get_page_ref_count(int page_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (page_id < 0 || page_id >= static_cast<int>(page_ref_counts_.size())) {
        return -1; // Invalid page_id
    }
    return page_ref_counts_[page_id];
}

KVUsageEstimate KVCacheManager::estimate_usage(const ModelLayout& layout,
                                              int prompt_len,
                                              int max_new_tokens)
{
    int total_tokens = prompt_len + max_new_tokens;
    int pages        = (total_tokens > 0 && layout.page_size > 0)
                           ? (total_tokens + layout.page_size - 1) / layout.page_size
                           : 0;

    if (pages == 0 && total_tokens > 0 && layout.page_size > 0) {
        pages = 1;
    }

    const int bytes_per_value_eff = bytes_per_value_from_dtype(layout.kv_dtype);
    const int kv_factor           = 2;

    size_t bytes = static_cast<size_t>(pages)
                 * layout.num_layers
                 * layout.num_kv_heads
                 * layout.head_dim
                 * layout.page_size
                 * bytes_per_value_eff
                 * kv_factor;

    return {static_cast<size_t>(pages), bytes};
}

bool KVCacheManager::build_page_ptr_table(const std::vector<int>& page_ids,
                                          std::vector<void*>&     out_ptrs) const

{
    std::lock_guard<std::mutex> lock(mutex_);
    out_ptrs.clear();
    out_ptrs.reserve(page_ids.size());
    const uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr_);
    const uintptr_t end  = base + storage_bytes_;
    for (size_t i = 0; i < page_ids.size(); ++i) {
        int page_id = page_ids[i];
        if (page_id < 0 || page_id >= static_cast<int>(all_pages_.size())) {
            TM_LOG_ERROR("build_page_ptr_table: invalid page_id %d at index %zu", page_id, i);
            return false;
        }
        void* page_ptr = all_pages_[page_id].data;
        if (base_ptr_ && page_ptr) {
            const uintptr_t val = reinterpret_cast<uintptr_t>(page_ptr);
            if (val < base || val >= end) {
                TM_LOG_ERROR("build_page_ptr_table: pointer %p out of range [base=%p, size=%zu] (page_id=%d)",
                             page_ptr,
                             base_ptr_,
                             storage_bytes_,
                             page_id);
                return false;
            }
        }
        out_ptrs.push_back(page_ptr);
    }
    return true;
}

}  // namespace turbomind
