#pragma once

#include <cstdint>
#include <vector>

namespace turbomind {

using ReqId = uint64_t;
using TokenId = int32_t;

struct SuffixSpecParams {
    int   max_spec_tokens{0};
    float max_spec_factor{1.0f};
    float max_spec_offset{0.0f};
    float min_token_prob{0.1f};
    bool  use_tree_spec{false};
};

struct SuffixDraft {
    std::vector<TokenId> token_ids;
    std::vector<int32_t> parents;
    std::vector<float>   probs;
    float                score{0.0f};
    int32_t              match_len{0};
};

class SuffixDecodeCache {
public:
    SuffixDecodeCache(int /*max_tree_depth*/, int /*max_cached_requests*/) {}

    void start_request(ReqId /*req_id*/, const TokenId* /*prompt_tokens*/, int /*prompt_len*/) {}
    void stop_request(ReqId /*req_id*/) {}
    void add_active_response(ReqId /*req_id*/, const TokenId* /*tokens*/, int /*num_tokens*/) {}
    void evict_cached_response(ReqId /*req_id*/) {}

    SuffixDraft speculate(ReqId /*req_id*/, const TokenId* /*context*/, int /*context_len*/, const SuffixSpecParams& /*params*/)
    {
        return SuffixDraft{};
    }
};

class SuffixDraftProvider {
public:
    explicit SuffixDraftProvider(SuffixDecodeCache* cache): cache_(cache) {}

    SuffixDraft propose(ReqId req_id, const TokenId* context, int context_len, const SuffixSpecParams& params)
    {
        if (!cache_) {
            return SuffixDraft{};
        }
        return cache_->speculate(req_id, context, context_len, params);
    }

private:
    SuffixDecodeCache* cache_{nullptr};
};

}  // namespace turbomind
