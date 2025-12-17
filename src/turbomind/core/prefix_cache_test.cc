#include "catch2/catch_test_macros.hpp"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/utils/cuda_utils.h"
#include <numeric>

using namespace turbomind;

namespace {

KVLayout make_kv_layout_from_model(const ModelLayout& ml)
{
    KVLayout kv{};
    kv.num_layers      = ml.num_layers;
    kv.num_kv_heads    = ml.num_kv_heads;
    kv.head_dim        = ml.head_dim;
    kv.page_size       = ml.page_size;
    kv.kv_dtype        = ml.kv_dtype;
    kv.bytes_per_value = bytes_per_value_from_dtype(ml.kv_dtype);
    return kv;
}

bool has_cuda()
{
    return getDeviceCount() > 0;
}

} // namespace

TEST_CASE("PrefixCache reuse and eviction releases KV", "[prefix_cache]")
{
    if (!has_cuda()) {
        WARN("Skipping prefix cache test: no CUDA device available.");
        return;
    }

    ModelLayout ml = make_test_layout();
    KVLayout kv    = make_kv_layout_from_model(ml);

    const size_t page_bytes = static_cast<size_t>(kv.num_layers) * kv.num_kv_heads * kv.head_dim * kv.page_size
                            * kv.bytes_per_value;
    const size_t num_pages       = 6;
    const size_t capacity_bytes  = page_bytes * num_pages;

    KVCacheManager mgr(kv, capacity_bytes);
    PrefixCache    cache(kv.page_size, &mgr);

    // Seed seq0 and insert into cache.
    KVUsageEstimate est0{2, 2 * page_bytes};
    REQUIRE(mgr.reserve(100, est0, nullptr));
    auto pages0 = mgr.get_sequence_page_ids(100);
    REQUIRE(pages0.size() == 2);

    PrefixKey key0{};
    key0.tokens.resize(kv.page_size * 2);
    std::iota(key0.tokens.begin(), key0.tokens.end(), 0);
    key0.namespace_id = 0;

    // Miss then insert and hit.
    auto miss = cache.match(key0);
    REQUIRE(miss.matched_tokens == 0);
    cache.insert(key0, pages0, 0, 100);
    auto hit = cache.match(key0);
    REQUIRE(hit.matched_tokens == kv.page_size * 2);
    REQUIRE(hit.page_indices == pages0);

    // Reuse cached pages for new seq1 via pre-existing page IDs.
    KVUsageEstimate est1{3, 3 * page_bytes};
    REQUIRE(mgr.reserve(101, est1, nullptr, hit.page_indices));
    REQUIRE(mgr.get_page_ref_count(pages0[0]) == 2);

    // Evict cached prefix; should release seq100 pages.
    cache.evict_lru_entry();
    REQUIRE(cache.get_eviction_count() == 1);
    REQUIRE(mgr.get_page_ref_count(pages0[0]) == 1);

    // Release seq1; all pages should return to free pool.
    mgr.release(101);
    REQUIRE(mgr.used_pages() == 0);
    REQUIRE(mgr.free_pages() == num_pages);
}
