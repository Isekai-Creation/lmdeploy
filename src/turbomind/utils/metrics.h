#pragma once

#include <chrono>
#include <cstdint>
#include <ostream>

namespace turbomind {

struct ScheduleMetrics {
    // sequences
    int total_seqs;    // the number of received sequence
    int active_seqs;   // the number of active sequence
    int waiting_seqs;  // the number of waiting sequence

    // kv block usage
    int total_blocks;   // the number of kv blocks
    int active_blocks;  // the number of active kv blocks
    int cached_blocks;  // the number of cached kv blocks
    int free_blocks;    // the number of free kv blocks
};

struct RequestMetrics {
    int64_t enque_time{};      // when a request is enqued
    int64_t scheduled_time{};  // when a request is scheduled for inference

    // EAGLE speculative decoding metrics (TurboMind backend), aggregated
    // over the lifetime of a request.
    int64_t eagle_total_draft_tokens{};      // total number of draft tokens proposed
    int64_t eagle_total_accepted_tokens{};   // total number of draft tokens accepted
    int64_t eagle_steps{};                   // number of speculative steps taken
    int64_t eagle_total_rewound_tokens{};    // total number of tokens rewound from KV cache
    int64_t eagle_rewind_steps{};            // number of steps where KV rewind was applied
    int64_t eagle_max_tokens_per_seq{};      // maximum tokens_per_seq observed across steps
    int64_t eagle_max_accepted_len{};        // maximum accepted_len observed across steps
    int64_t eagle_steps_accept_ge2{};        // number of steps where accepted_len >= 2
    int64_t eagle_total_committed_extras{};  // sum(max(0, accepted_len - 1)) across steps

    // Target-tree decode specific metrics (EAGLE3). These counters are
    // only populated when the TurboMind engine enables the target-tree
    // path; when disabled they remain zero so downstream tooling can
    // distinguish baseline speculative runs from tree-aware ones.
    int64_t eagle_tree_draft_tokens{};       // total number of tree draft tokens proposed
    int64_t eagle_tree_target_tokens{};      // total number of tree target tokens evaluated
    int64_t eagle_tree_accepted_tokens{};    // total number of accepted tokens along tree paths

    static int64_t timestamp()
    {
        // Get current timestamp in microseconds since Unix epoch
        // system_clock uses wall-clock time (matches Python's time.time())
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
};

inline std::ostream& operator<<(std::ostream& os, const ScheduleMetrics& m)
{
    os << "ScheduleMetrics { ";
    os << "total_seqs=" << m.total_seqs;
    os << ", active_seqs=" << m.active_seqs;
    os << ", waiting_seqs=" << m.waiting_seqs;
    os << ", total_blocks=" << m.total_blocks;
    os << ", cached_blocks=" << m.cached_blocks;
    os << ", free_blocks=" << m.free_blocks;
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const RequestMetrics& m)
{
    os << "RequestMetrics { ";
    os << "enque_time=" << m.enque_time;
    os << ", scheduled_time=" << m.scheduled_time;
    os << ", eagle_total_draft_tokens=" << m.eagle_total_draft_tokens;
    os << ", eagle_total_accepted_tokens=" << m.eagle_total_accepted_tokens;
    os << ", eagle_steps=" << m.eagle_steps;
    os << ", eagle_total_rewound_tokens=" << m.eagle_total_rewound_tokens;
    os << ", eagle_rewind_steps=" << m.eagle_rewind_steps;
    os << ", eagle_max_tokens_per_seq=" << m.eagle_max_tokens_per_seq;
    os << ", eagle_max_accepted_len=" << m.eagle_max_accepted_len;
    os << ", eagle_steps_accept_ge2=" << m.eagle_steps_accept_ge2;
    os << ", eagle_total_committed_extras=" << m.eagle_total_committed_extras;
    os << ", eagle_tree_draft_tokens=" << m.eagle_tree_draft_tokens;
    os << ", eagle_tree_target_tokens=" << m.eagle_tree_target_tokens;
    os << ", eagle_tree_accepted_tokens=" << m.eagle_tree_accepted_tokens;
    os << " }";
    return os;
}

}  // namespace turbomind
