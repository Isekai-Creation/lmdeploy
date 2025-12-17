#include "src/turbomind/engine/EngineScheduler.h"
#include <gtest/gtest.h>
#include "src/turbomind/utils/logger.h" // For TM_LOG_INFO
#include "src/turbomind/core/tensor.h"
#include <numeric> // For std::iota
#include "src/turbomind/core/allocator.h" // For Allocator and DeviceType
#include "src/turbomind/utils/cuda_utils.h" // For cudaMemcpy

namespace turbomind::test { // Correct namespace declaration

using namespace turbomind::core; // Use core namespace for Tensor, Allocator, etc.

// Helper function to create a Tensor from a std::vector<int>
Tensor from_vector(const std::vector<int>& vec) {
    Allocator allocator(turbomind::DeviceType::kDEVICE); // Correct Allocator instantiation
    Layout layout = Layout{(ssize_t)vec.size()};
    Buffer buffer(vec.size(), turbomind::kInt32, allocator); // Correct DataType enum
    cudaMemcpy(buffer.data<int>(), vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice); // Correct copy
    return Tensor(buffer, layout);
}

// A mock request for testing
std::shared_ptr<Request> CreateTestRequest(uint64_t seq_id, int prompt_len, int max_new_tokens, bool start_flag = true) {
    auto req = std::make_shared<Request>();
    req->session.id = seq_id;
    req->session.start_flag = start_flag;
    req->session.end_flag = false;
    req->session.kill_flag = false;
    req->gen_cfg.max_new_tokens = max_new_tokens;
    std::vector<int> input_ids(prompt_len);
    std::iota(input_ids.begin(), input_ids.end(), 0);
    req->inputs["input_ids"] = from_vector(input_ids);
    return req;
}

class EngineSchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize any common resources for the tests
        turbomind::Logger::getLogger().setLevel(turbomind::Logger::INFO);
    }

    // You can define helper functions or member variables here
    SchedulerConfig cfg_;
    // We use nullptr for KVCacheManager for isolated testing for now
    KVCacheManager* kv_mgr_ = nullptr;
};

TEST_F(EngineSchedulerTest, AdmissionAndFCFS) {
    cfg_.max_num_batched_tokens = 100;
    cfg_.max_num_seqs = 4;
    cfg_.schedule_policy = SchedulerConfig::SchedulePolicy::kFcfs;
    cfg_.prefer_decode_over_prefill = false; // Simplify test

    EngineScheduler scheduler(cfg_, kv_mgr_);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    
    // 1. Admit a single request
    infer_reqs.push_back(CreateTestRequest(1, 50, 10));
    scheduler.on_new_requests(infer_reqs, kill_reqs);

    ASSERT_EQ(scheduler.get_active_requests_count(), 1);
    ASSERT_EQ(scheduler.get_queued_requests_count(), 1);

    // 2. Schedule a step
    std::vector<PrefillChunk> prefill_batch;
    std::vector<std::shared_ptr<Request>> decode_batch;
    scheduler.schedule_step(prefill_batch, decode_batch);

    // 3. Verify the output
    ASSERT_EQ(prefill_batch.size(), 1);
    ASSERT_EQ(decode_batch.size(), 0);
    EXPECT_EQ(prefill_batch[0].req->session.id, 1);
    EXPECT_EQ(prefill_batch[0].len, 50);

    // After scheduling, the request should be moved out of the queue
    ASSERT_EQ(scheduler.get_queued_requests_count(), 0);

    // After state update (which is done in the test explicitly for now)
    // the sequence should move to decode queue.
    scheduler.update_sequence_state(1, 50, 0); 
    ASSERT_EQ(scheduler.get_queued_requests_count(), 1); // now in decode queue

    // 4. Schedule another step (now it should be a decode request)
    prefill_batch.clear();
    decode_batch.clear();
    scheduler.schedule_step(prefill_batch, decode_batch);

    ASSERT_EQ(prefill_batch.size(), 0);
    ASSERT_EQ(decode_batch.size(), 1);
    EXPECT_EQ(decode_batch[0]->session.id, 1);
}

TEST_F(EngineSchedulerTest, TokenBudgeting) {
    cfg_.max_num_batched_tokens = 70;
    cfg_.max_num_seqs = 2;
    cfg_.schedule_policy = SchedulerConfig::SchedulePolicy::kFcfs;
    cfg_.prefer_decode_over_prefill = false;

    EngineScheduler scheduler(cfg_, kv_mgr_);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;

    infer_reqs.push_back(CreateTestRequest(1, 50, 10));
    infer_reqs.push_back(CreateTestRequest(2, 30, 10)); // This one won't fit
    scheduler.on_new_requests(infer_reqs, kill_reqs);

    ASSERT_EQ(scheduler.get_active_requests_count(), 2);
    ASSERT_EQ(scheduler.get_queued_requests_count(), 2);

    std::vector<PrefillChunk> prefill_batch;
    std::vector<std::shared_ptr<Request>> decode_batch;
    scheduler.schedule_step(prefill_batch, decode_batch);

    // Only the first request should be scheduled
    ASSERT_EQ(prefill_batch.size(), 1);
    EXPECT_EQ(prefill_batch[0].req->session.id, 1);
    EXPECT_EQ(prefill_batch[0].len, 50);

    ASSERT_EQ(scheduler.get_queued_requests_count(), 1); // The second request remains
}

TEST_F(EngineSchedulerTest, ShortPromptFirst) {
    cfg_.max_num_batched_tokens = 100;
    cfg_.max_num_seqs = 2;
    cfg_.schedule_policy = SchedulerConfig::SchedulePolicy::kSmallFirst;
    cfg_.prefer_decode_over_prefill = false;

    EngineScheduler scheduler(cfg_, kv_mgr_);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;

    infer_reqs.push_back(CreateTestRequest(1, 80, 10)); // Long prompt
    infer_reqs.push_back(CreateTestRequest(2, 20, 10)); // Short prompt
    scheduler.on_new_requests(infer_reqs, kill_reqs);

    ASSERT_EQ(scheduler.get_active_requests_count(), 2);
    ASSERT_EQ(scheduler.get_queued_requests_count(), 2);

    std::vector<PrefillChunk> prefill_batch;
    std::vector<std::shared_ptr<Request>> decode_batch;
    scheduler.schedule_step(prefill_batch, decode_batch);

    // The short prompt should be scheduled first
    ASSERT_EQ(prefill_batch.size(), 2);
    EXPECT_EQ(prefill_batch[0].req->session.id, 2);
    EXPECT_EQ(prefill_batch[0].len, 20);
    EXPECT_EQ(prefill_batch[1].req->session.id, 1);
    EXPECT_EQ(prefill_batch[1].len, 80);

    ASSERT_EQ(scheduler.get_queued_requests_count(), 0);
}

TEST_F(EngineSchedulerTest, SnapshotMetricsReflectsQueues) {
    cfg_.max_num_batched_tokens = 32;
    cfg_.max_num_seqs = 4;

    EngineScheduler scheduler(cfg_, kv_mgr_);

    std::vector<std::shared_ptr<Request>> infer_reqs;
    std::vector<std::shared_ptr<Request>> kill_reqs;
    infer_reqs.push_back(CreateTestRequest(1, 10, 5));
    infer_reqs.push_back(CreateTestRequest(2, 8, 4));

    scheduler.on_new_requests(infer_reqs, kill_reqs);

    DriftMetrics m = scheduler.snapshot_metrics();
    EXPECT_EQ(m.active_requests, 2u);
    EXPECT_EQ(m.queued_prefill, 2u);
    EXPECT_EQ(m.queued_decode, 0u);
    EXPECT_EQ(m.kv_total_pages, 0u);
    EXPECT_EQ(m.kv_used_pages, 0u);
    EXPECT_EQ(m.kv_free_pages, 0u);
    EXPECT_EQ(m.kv_blocked, 0u);
    EXPECT_EQ(m.kv_rejected, 0u);
    EXPECT_EQ(m.prefix_hits, 0u);
    EXPECT_EQ(m.prefix_misses, 0u);
}

} // namespace turbomind::test
