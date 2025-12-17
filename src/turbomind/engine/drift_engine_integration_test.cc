#include "catch2/catch_test_macros.hpp"
#include "src/turbomind/engine/drift_engine.h"
#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/utils/logger.h"

using namespace turbomind;

// Minimal stub Gateway that feeds a fixed set of requests then signals abort.
class StubGateway : public Gateway {
public:
    explicit StubGateway(const std::vector<std::shared_ptr<Request>>& infer) : infer_(infer) {}

    void pop(std::vector<std::shared_ptr<Request>>& infer,
             std::vector<std::shared_ptr<Request>>& kill,
             int /*free_slots*/,
             bool /*is_empty*/,
             bool& abort_flag,
             int /*rank*/) override
    {
        if (drained_) {
            abort_flag = true;
            return;
        }
        infer = infer_;
        kill.clear();
        drained_ = true;
    }

private:
    std::vector<std::shared_ptr<Request>> infer_;
    bool drained_{false};
};

static std::shared_ptr<Request> make_req(uint64_t id, int prompt_len, int max_new_tokens)
{
    auto r = std::make_shared<Request>();
    r->session.id         = id;
    r->session.start_flag = true;
    r->session.step       = 0;
    r->gen_cfg.max_new_tokens = max_new_tokens;
    Tensor input{{prompt_len}, data_type_v<int>, Device{kCPU}};
    std::vector<int> data(prompt_len, 1);
    input.assign(data.data());
    r->inputs.emplace("input_ids", input);
    return r;
}

TEST_CASE("DriftEngine minimal integration does not deadlock", "[drift_engine_integration]")
{
    // Small KV layout and capacity for test.
    ModelLayout ml = make_test_layout();
    KVLayout kv{};
    kv.num_layers      = ml.num_layers;
    kv.num_kv_heads    = ml.num_kv_heads;
    kv.head_dim        = ml.head_dim;
    kv.page_size       = ml.page_size;
    kv.kv_dtype        = ml.kv_dtype;
    kv.bytes_per_value = bytes_per_value_from_dtype(kv.kv_dtype);

    DriftEngineConfig cfg{};
    cfg.model_layout      = ml;
    cfg.kv_layout         = kv;
    cfg.kv_capacity_bytes = static_cast<size_t>(kv.num_layers) * kv.num_kv_heads * kv.head_dim * kv.page_size * kv.bytes_per_value * 8;

    auto kv_mgr = std::make_shared<KVCacheManager>(kv, cfg.kv_capacity_bytes);
    auto prefix = std::make_shared<PrefixCache>(kv.page_size, kv_mgr.get());
    std::vector<std::shared_ptr<Request>> infer = {make_req(1, ml.page_size, 4), make_req(2, ml.page_size / 2, 2)};
    auto gateway = std::make_shared<StubGateway>(infer);

    DriftEngine engine(cfg, gateway, kv_mgr, prefix);

    // Bind a dummy LlamaBatch to satisfy executor hooks; execute scheduled is no-op.
    LlamaBatch dummy_batch;
    engine.bind_llama_batch(&dummy_batch);

    engine.run(/*rank=*/0);
    // If we reach here without hang, test passes.
    SUCCEED();
}
