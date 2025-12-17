#include "catch2/catch_test_macros.hpp"
#include "src/turbomind/engine/EngineScheduler.h"
#include "src/turbomind/engine/capacity_scheduler.h"
#include "src/turbomind/core/kv_cache_manager.h"
#include "src/turbomind/core/prefix_cache.h"
#include "src/turbomind/models/common/model_layout.h"

using namespace turbomind;

namespace {

KVLayout make_kv_layout()
{
    ModelLayout ml{};
    ml.num_layers   = 2;
    ml.num_kv_heads = 2;
    ml.head_dim     = 16;
    ml.page_size    = 32;
    ml.max_seq_len  = 2048;
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

Request make_request(uint64_t id, int prompt_tokens, int max_new_tokens)
{
    Request r{};
    r.session.id         = id;
    r.session.start_flag = true;
    r.session.step       = 0;
    r.gen_cfg.max_new_tokens = max_new_tokens;

    Tensor input{{prompt_tokens}, data_type_v<int>, Device{kCPU}};
    std::vector<int> data(prompt_tokens, 1);
    input.assign(data.data());
    r.inputs.emplace("input_ids", input);
    return r;
}

} // namespace

TEST_CASE("EngineScheduler respects token and sequence budgets", "[engine_scheduler]")
{
    SchedulerConfig cfg{};
    cfg.max_num_batched_tokens = 8;
    cfg.max_num_seqs           = 2;
    cfg.enable_chunked_prefill = true;
    cfg.enable_prefix_caching  = false;
    cfg.prefer_decode_over_prefill = true;

    KVLayout kv = make_kv_layout();
    const size_t page_bytes = static_cast<size_t>(kv.num_layers) * kv.num_kv_heads * kv.head_dim * kv.page_size * kv.bytes_per_value;
    KVCacheManager kv_mgr(kv, page_bytes * 8);
    PrefixCache prefix_cache(kv.page_size, &kv_mgr);
    CapacityScheduler cap_sched(&kv_mgr, &prefix_cache);

    EngineScheduler sched(cfg, &kv_mgr, make_gpt_oss_120b_layout(), &prefix_cache, &cap_sched);

    auto r1 = std::make_shared<Request>(make_request(1, 6, 4));
    auto r2 = std::make_shared<Request>(make_request(2, 3, 4));

    std::vector<std::shared_ptr<Request>> infer{r1, r2};
    std::vector<std::shared_ptr<Request>> kill;

    sched.on_new_requests(infer, kill);

    std::vector<PrefillChunk> prefill;
    std::vector<std::shared_ptr<Request>> decode;
    sched.schedule_step(prefill, decode);

    // With prefer_decode and default ratio 0.5, both decode and prefill share budget.
    REQUIRE(prefill.size() + decode.size() <= cfg.max_num_seqs);
    int total_tokens = 0;
    for (const auto& pc : prefill) total_tokens += pc.len;
    total_tokens += static_cast<int>(decode.size()); // 1 token per decode
    REQUIRE(total_tokens <= cfg.max_num_batched_tokens);
}
