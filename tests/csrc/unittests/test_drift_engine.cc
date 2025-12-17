#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/scheduler_config.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/core/data_type.h" // Added for DataType::kInt32
#include "src/turbomind/core/tensor.h" // Added for turbomind::core::Tensor
#include "src/turbomind/core/allocator.h" // Added for DeviceType::kDEVICE

#include <iostream>
#include <cassert>
#include <vector>
#include <memory>

// Mock classes for dependencies
class MockKVCacheManager : public turbomind::KVCacheManager {
public:
    MockKVCacheManager() : KVCacheManager(turbomind::KVLayout(), 0) {}
    // Override methods as needed for testing
    bool reserve(uint64_t seq_id, const turbomind::KVUsageEstimate& est, turbomind::KVReservation* out, const std::vector<int>& pre_existing_page_ids) override {
        // Always succeed for mock
        if (out) {
            out->seq_id = seq_id;
            out->num_pages = est.pages_needed;
            out->first_page = 0; // Dummy value
        }
        return true;
    }
    void release(uint64_t seq_id) override {
        // Do nothing for mock
    }
    std::vector<int> get_sequence_page_ids(uint64_t seq_id) const override {
        return {}; // Dummy
    }
};

class MockPrefixCache : public turbomind::PrefixCache {
public:
    MockPrefixCache() : PrefixCache(0, nullptr) {}
    // Override methods as needed for testing
    turbomind::PrefixMatchResult match(const turbomind::PrefixKey& key) const override {
        return {}; // Dummy
    }
    void insert(const turbomind::PrefixKey& key, const std::vector<int32_t>& page_indices, int priority, uint64_t seq_id) override {
        // Do nothing for mock
    }
    void erase(uint64_t seq_id) override {
        // Do nothing for mock
    }
};

class MockCapacityScheduler : public turbomind::CapacityScheduler {
public:
    MockCapacityScheduler(turbomind::KVCacheManager* kv_mgr, turbomind::PrefixCache* prefix_cache)
        : CapacityScheduler(kv_mgr, prefix_cache) {}
    // Override methods as needed for testing
    bool try_start_request(uint64_t seq_id, const turbomind::KVUsageEstimate& est, turbomind::KVReservation* out, const std::vector<int>& pre_existing_page_ids) override {
        return true; // Always succeed for mock
    }
    void finish_request(uint64_t seq_id) override {
        // Do nothing for mock
    }
};


void test_engine_scheduler_init_and_new_request() {
    std::cout << "Running test_engine_scheduler_init_and_new_request..." << std::endl;

    turbomind::SchedulerConfig cfg;
    MockKVCacheManager mock_kv_mgr;
    MockPrefixCache mock_prefix_cache;
    MockCapacityScheduler mock_capacity_scheduler(&mock_kv_mgr, &mock_prefix_cache);
    
    turbomind::EngineScheduler scheduler(cfg, &mock_kv_mgr, turbomind::make_gpt_oss_120b_layout(), &mock_prefix_cache, &mock_capacity_scheduler);

    // Create a mock request
    auto req = std::make_shared<turbomind::Request>();
    req->session.id = 1;
    req->session.start_flag = true;
    req->session.step = 0;
    req->inputs["input_ids"] = turbomind::core::Tensor(turbomind::core::Layout({10}), turbomind::DataType::kInt32, turbomind::core::Device(turbomind::DeviceType::kDEVICE));
    req->gen_cfg.max_new_tokens = 20;

    std::vector<std::shared_ptr<turbomind::Request>> infer_reqs = {req};
    std::vector<std::shared_ptr<turbomind::Request>> kill_reqs;

    scheduler.on_new_requests(infer_reqs, kill_reqs);

    assert(!scheduler.empty() && "Scheduler should not be empty after adding a request");
    assert(scheduler.get_queued_requests_count() == 1 && "There should be one request in the queue");

    std::cout << "test_engine_scheduler_init_and_new_request passed." << std::endl;
}

int main() {
    test_engine_scheduler_init_and_new_request();
    // Add more tests here

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
