#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/utils/logger.h"
#include <numeric> // For std::iota
#include <cuda_runtime_api.h> // For cudaMalloc and cudaFree

namespace turbomind {

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
    page_bytes_ = static_cast<size_t>(layout_.num_layers) * layout_.num_kv_heads * layout_.head_dim * layout_.page_size * layout_.bytes_per_value;

    size_t num_pages = total_capacity_bytes_ / page_bytes_;
    if (num_pages == 0) {
        TM_LOG_WARNING("KVCacheManager initialized with 0 pages. total_capacity_bytes: %lu, page_bytes: %lu", total_capacity_bytes_, page_bytes_);
        return;
    }

    // Allocate a single contiguous device buffer for all KV data
    void* all_kv_data_ptr = allocate_device_memory(num_pages * page_bytes_);
    if (!all_kv_data_ptr) {
        TM_LOG_ERROR("Failed to allocate contiguous device memory for KV cache.");
        // Mark manager as invalid or throw exception if desired
        return;
    }

    all_pages_.reserve(num_pages);
    page_ref_counts_.assign(num_pages, 0); // Initialize ref counts
    for (size_t i = 0; i < num_pages; ++i) {
        KVCachePage page;
        page.id = i;
        page.data = static_cast<char*>(all_kv_data_ptr) + (i * page_bytes_);
        page.is_free = true;
        all_pages_.push_back(page);
        free_page_ids_.push(i); // Add all pages to the free list
    }
    TM_LOG_INFO("KVCacheManager initialized with %zu pages, total capacity %lu bytes.", num_pages, total_capacity_bytes_);
}

KVCacheManager::~KVCacheManager()
{
    // Free the contiguous device buffer
    if (!all_pages_.empty() && all_pages_[0].data) {
        // Only free the base pointer once
        free_device_memory(all_pages_[0].data);
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
    for (int page_id : pre_existing_page_ids) {
        if (page_id < 0 || page_id >= all_pages_.size()) {
            TM_LOG_ERROR("Pre-existing page ID %d for sequence %lu is invalid. Failing reservation.", page_id, seq_id);
            return false;
        }
        // Increment ref count for pre-existing pages
        page_ref_counts_[page_id]++;
        allocated_page_ids.push_back(page_id);
    }

    // Calculate how many more pages are needed
    size_t remaining_pages_needed = est.pages_needed > allocated_page_ids.size() ? est.pages_needed - allocated_page_ids.size() : 0;

    if (remaining_pages_needed > free_page_ids_.size()) {
        TM_LOG_DEBUG("Insufficient free pages to reserve for sequence ID: %lu. Needed: %zu, Available: %zu (after pre-existing: %zu)",
                  seq_id, est.pages_needed, free_page_ids_.size(), remaining_pages_needed);
        // If not enough pages are available, revert the ref count increments
        for (int page_id : pre_existing_page_ids) {
            page_ref_counts_[page_id]--;
        }
        return false;
    }
    
    // Allocate remaining pages from the free list
    for (size_t i = 0; i < remaining_pages_needed; ++i) {
        int page_id = free_page_ids_.front();
        free_page_ids_.pop();
        all_pages_[page_id].is_free = false;
        page_ref_counts_[page_id] = 1; // New page starts with ref count 1
        allocated_page_ids.push_back(page_id);
    }

    KVReservation reservation;
    reservation.seq_id = seq_id;
    reservation.first_page = allocated_page_ids.empty() ? -1 : allocated_page_ids[0];
    reservation.num_pages = allocated_page_ids.size();

    seq_reservations_[seq_id] = reservation;
    seq_page_map_[seq_id] = allocated_page_ids;

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
        if (page_id < 0 || page_id >= all_pages_.size()) {
            TM_LOG_ERROR("Internal error: Invalid page ID %d in allocated_page_ids for sequence %lu", page_id, seq_id);
            continue;
        }
        
        page_ref_counts_[page_id]--;
        if (page_ref_counts_[page_id] == 0) {
            all_pages_[page_id].is_free = true;
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
    if (page_id < 0 || page_id >= all_pages_.size()) {
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
    if (page_id < 0 || page_id >= page_ref_counts_.size()) {
        return -1; // Invalid page_id
    }
    return page_ref_counts_[page_id];
}

KVUsageEstimate KVCacheManager::estimate_usage(const ModelLayout& layout,
                                              int prompt_len,
                                              int max_new_tokens)
{
    // As defined in ENGINE_TODOS.md section 4.1.1
    int total_tokens = prompt_len + max_new_tokens;
    // Ensure at least one page is allocated even for 0 tokens if page_size > 0
    int pages = (total_tokens > 0 && layout.page_size > 0) ? (total_tokens + layout.page_size - 1) / layout.page_size : 0;
    if (pages == 0 && total_tokens > 0 && layout.page_size > 0) { // If total_tokens > 0 but pages rounded to 0
        pages = 1; // At least one page to store any token data
    }
    if (pages == 0 && total_tokens == 0) { // No tokens, no pages needed
        pages = 0;
    }
    
    // Default bytes_per_value to 2 for half (fp16/bf16)
    // ModelLayout doesn't have bytes_per_value field, so we use a constant
    int bytes_per_value_eff = 2;

    size_t bytes = static_cast<size_t>(pages)
                 * layout.num_layers
                 * layout.num_kv_heads
                 * layout.head_dim
                 * bytes_per_value_eff; // Use effective bytes_per_value

    return {static_cast<size_t>(pages), bytes};
}

}  // namespace turbomind
