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

    // EAGLE speculative decoding metrics (TurboMind backend)
    // Aggregated over the lifetime of a request.
    int64_t eagle_total_draft_tokens{};      // total number of draft tokens proposed
    int64_t eagle_total_accepted_tokens{};   // total number of draft tokens accepted
    int64_t eagle_steps{};                   // number of speculative steps taken
    int64_t eagle_total_rewound_tokens{};    // total number of tokens rewound from KV cache
    int64_t eagle_rewind_steps{};            // number of steps where KV rewind was applied

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
    os << " }";
    return os;
}

}  // namespace turbomind
