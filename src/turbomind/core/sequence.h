
#pragma once

#include <cstdint>
#include <vector>
#include <string> // For std::ostream and std::stringstream
#include <ostream>
#include <numeric> // For std::iota if needed

namespace turbomind {

// Define BlockIds and UniqueIds here or include their definitions
using BlockIds  = std::vector<int>;
using UniqueIds = std::vector<uint64_t>;

struct Sequence {

    enum Status
    {
        kCached = 0,
        kLocked,
        kActive
    };

    uint64_t id;
    Status   status = kCached;

    BlockIds  blocks; // To be deprecated, or repurposed for external block IDs if needed
    UniqueIds block_unique_ids; // To be deprecated

    int input_length = 0;  // the number of tokens to be processed in each forward iter

    mutable std::vector<int> prompt;

    mutable std::vector<int> tokens;  // update by user or when the sequence is finished

    mutable int cache_len = 0;

    // additional data kept round-to-round
    mutable std::vector<std::byte> random_state;  // update by user

    mutable float rope_theta = 0.f;

    // embedding data
    mutable std::vector<std::vector<std::byte>> input_embeddings;
    mutable std::vector<std::pair<int, int>>    input_embedding_ranges;

    // New member: KV page IDs allocated by KVCacheManager for this sequence
    std::vector<int> kv_page_ids;
    // The KVReservation for this sequence is managed by KVCacheManager.

    explicit Sequence(uint64_t _id): id(_id) {}

    friend std::ostream& operator<<(std::ostream& os, const Sequence& seq);
};

using Sequences = std::vector<const Sequence*>;

inline std::ostream& operator<<(std::ostream& os, const Sequence& seq)
{
    os << "id=" << seq.id << ", status=" << static_cast<int>(seq.status) << ", token_count=" << seq.tokens.size()
       << ", block_count=" << seq.blocks.size() << ", cache_len=" << seq.cache_len
       << ", random_state_size=" << seq.random_state.size() << ", input_length=" << seq.input_length
       << ", kv_page_ids.size()=" << seq.kv_page_ids.size();
    return os;
}

}  // namespace turbomind
