#pragma once

#include <cstdint>
#include <vector>

namespace turbomind {

using TokenId = int32_t;

struct DraftNode {
    TokenId token_id{0};
    int32_t parent{-1};
    float   logit_or_score{0.0f};
};

struct DraftForest {
    std::vector<DraftNode> nodes;
    std::vector<int32_t>   seq_offsets;  // size = batch_size + 1
};

class IDraftProvider {
public:
    virtual ~IDraftProvider() = default;
    virtual void propose(const std::vector<uint64_t>& seq_ids, DraftForest& out) = 0;
};

}  // namespace turbomind
