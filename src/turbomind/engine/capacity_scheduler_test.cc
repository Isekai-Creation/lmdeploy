#include "catch2/catch_test_macros.hpp"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/utils/cuda_utils.h"

using namespace turbomind;

namespace {

KVLayout make_kv_layout_small()
{
    ModelLayout ml{};
    ml.num_layers   = 2;
    ml.num_kv_heads = 2;
    ml.head_dim     = 16;
    ml.page_size    = 32;
    ml.kv_dtype     = KVDataType::kFP16;

    KVLayout kv{};
    kv.num_layers      = ml.num_layers;
    kv.num_kv_heads    = ml.num_kv_heads;
    kv.head_dim        = ml.head_dim;
    kv.page_size       = ml.page_size;
    kv.kv_dtype        = ml.kv_dtype;
    kv.bytes_per_value = bytes_per_value_from_dtype(ml.kv_dtype);
    return kv;
}

bool has_cuda() { return getDeviceCount() > 0; }

} // namespace

TEST_CASE("CapacityScheduler blocks/reserves under tight KV budget", "[capacity_scheduler]")
{
    if (!has_cuda()) {
        WARN("Skipping capacity scheduler test: no CUDA device available.");
        return;
    }

    KVLayout kv = make_kv_layout_small();
    const size_t page_bytes = static_cast<size_t>(kv.num_layers) * kv.num_kv_heads * kv.head_dim * kv.page_size * kv.bytes_per_value;
    const size_t capacity_bytes = page_bytes * 3; // only 3 pages total

    KVCacheManager kv_mgr(kv, capacity_bytes);
    PrefixCache    prefix_cache(kv.page_size, &kv_mgr);
    CapacityScheduler cap_sched(&kv_mgr, &prefix_cache);

    KVUsageEstimate est_small{1, page_bytes};
    KVUsageEstimate est_big{3, 3 * page_bytes};

    // First request reserves successfully.
    KVReservation res0{};
    REQUIRE(cap_sched.try_start_request(1, est_small, &res0));
    REQUIRE(cap_sched.active_reservation_count() == 1);

    // Second large request cannot be scheduled (needs all pages).
    KVReservation res1{};
    REQUIRE_FALSE(cap_sched.try_start_request(2, est_big, &res1));
    REQUIRE(cap_sched.blocked_due_to_capacity() == 1);

    // Finish first; then big request should succeed.
    cap_sched.finish_request(1);
    REQUIRE(cap_sched.active_reservation_count() == 0);
    REQUIRE(cap_sched.try_start_request(2, est_big, &res1));
    REQUIRE(cap_sched.active_reservation_count() == 1);

    cap_sched.finish_request(2);
    REQUIRE(cap_sched.active_reservation_count() == 0);
}
