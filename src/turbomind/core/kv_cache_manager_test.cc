#include "src/turbomind/core/kv_cache_manager.h"
#include "catch2/catch_test_macros.hpp"

using namespace turbomind;

TEST_CASE("KVCacheManager Tests", "[kv_cache_manager]")
{
    KVLayout layout;
    layout.num_layers = 2;
    layout.num_kv_heads = 4;
    layout.head_dim = 64;
    layout.page_size = 16;
    layout.bytes_per_value = 2; // FP16

    size_t page_bytes = layout.num_layers * layout.num_kv_heads * layout.head_dim * layout.page_size * layout.bytes_per_value;
    size_t num_pages = 100;
    size_t total_capacity_bytes = num_pages * page_bytes;

    KVCacheManager kv_mgr(layout, total_capacity_bytes);

    REQUIRE(kv_mgr.total_pages() == num_pages);
    REQUIRE(kv_mgr.free_pages() == num_pages);
    REQUIRE(kv_mgr.used_pages() == 0);

    SECTION("Estimate usage includes page_size and dtype") {
        ModelLayout ml{};
        ml.num_layers    = layout.num_layers;
        ml.num_kv_heads  = layout.num_kv_heads;
        ml.head_dim      = layout.head_dim;
        ml.page_size     = layout.page_size;
        ml.kv_dtype      = KVDataType::kFP16;

        auto est = KVCacheManager::estimate_usage(ml, layout.page_size, layout.page_size);
        REQUIRE(est.pages_needed == 2);
        REQUIRE(est.bytes_needed == 2 * page_bytes);
    }

    SECTION("Simple reserve and release")
    {
        uint64_t seq_id = 1;
        KVUsageEstimate est = {10, 10 * page_bytes};
        KVReservation res;

        bool success = kv_mgr.reserve(seq_id, est, &res);
        REQUIRE(success);
        REQUIRE(kv_mgr.free_pages() == num_pages - 10);
        REQUIRE(kv_mgr.used_pages() == 10);
        REQUIRE(res.num_pages == 10);

        auto page_ids = kv_mgr.get_sequence_page_ids(seq_id);
        REQUIRE(page_ids.size() == 10);
        for (int page_id : page_ids) {
            REQUIRE(kv_mgr.get_page_ref_count(page_id) == 1);
        }

        kv_mgr.release(seq_id);
        REQUIRE(kv_mgr.free_pages() == num_pages);
        REQUIRE(kv_mgr.used_pages() == 0);

        for (int page_id : page_ids) {
            REQUIRE(kv_mgr.get_page_ref_count(page_id) == 0);
        }
    }

    SECTION("Double reserve")
    {
        uint64_t seq_id = 2;
        KVUsageEstimate est = {5, 5 * page_bytes};
        KVReservation res;

        kv_mgr.reserve(seq_id, est, &res);
        REQUIRE(kv_mgr.used_pages() == 5);

        // This should fail
        bool success = kv_mgr.reserve(seq_id, est, &res);
        REQUIRE_FALSE(success);
        REQUIRE(kv_mgr.used_pages() == 5);
    }

    SECTION("Double release")
    {
        uint64_t seq_id = 3;
        KVUsageEstimate est = {5, 5 * page_bytes};
        KVReservation res;

        kv_mgr.reserve(seq_id, est, &res);
        REQUIRE(kv_mgr.used_pages() == 5);

        kv_mgr.release(seq_id);
        REQUIRE(kv_mgr.used_pages() == 0);

        // This should be a no-op with a warning
        kv_mgr.release(seq_id);
        REQUIRE(kv_mgr.used_pages() == 0);
    }
    
    SECTION("Reserve with pre-existing pages")
    {
        uint64_t seq_id_1 = 10;
        KVUsageEstimate est1 = {5, 5 * page_bytes};
        KVReservation res1;
        kv_mgr.reserve(seq_id_1, est1, &res1);
        auto pages_for_seq1 = kv_mgr.get_sequence_page_ids(seq_id_1);

        // Simulate a cache hit
        uint64_t seq_id_2 = 11;
        std::vector<int> pre_existing_pages = {pages_for_seq1[0], pages_for_seq1[1]}; // hit 2 pages
        
        KVUsageEstimate est2 = {7, 7 * page_bytes}; // needs 5 new pages
        KVReservation res2;
        
        bool success = kv_mgr.reserve(seq_id_2, est2, &res2, pre_existing_pages);
        REQUIRE(success);
        REQUIRE(res2.num_pages == 7);
        REQUIRE(kv_mgr.used_pages() == 5 + 5); // 5 from seq1, 5 new for seq2
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[0]) == 2);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[1]) == 2);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[2]) == 1);

        kv_mgr.release(seq_id_1);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[0]) == 1);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[1]) == 1);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[2]) == 0);
        REQUIRE(kv_mgr.used_pages() == 7);

        kv_mgr.release(seq_id_2);
        REQUIRE(kv_mgr.used_pages() == 0);
        REQUIRE(kv_mgr.get_page_ref_count(pages_for_seq1[0]) == 0);
    }
}
