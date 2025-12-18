#pragma once

#include <cstdint>
#include <string>

namespace turbomind {

enum class ProgressStage : uint16_t {
    kRequestEnqueue = 0,
    kGatewayPop,
    kSchedulerAdmit,
    kKVReserve,
    kPrefixMatch,
    kPrefillSchedule,
    kPrefillExec,
    kKVWrite,
    kDecodeSchedule,
    kDecodeExec,
    kSampling,
    kKVExtend,
    kOutputAppend,
    kCallbackNotify,
    kFinish,
    kRelease,
    kError,
};

struct ProgressEvent {
    ProgressStage stage;
    uint8_t       pct{0};
    uint64_t      seq_id{0};
    uint64_t      session_id{0};
    uint64_t      unique_id{0};
    int           rank{-1};
    int           step{-1};
    int           prefilled_len{-1};
    int           prompt_len{-1};
    int           generated_len{-1};
    int           max_new_tokens{-1};
    int           token_pos{-1};
    int           chunk_start{-1};
    int           chunk_len{-1};
    int           batch_size{-1};
    int           kv_pages_total{-1};
    int           kv_pages_used{-1};
    int           kv_pages_free{-1};
    int           kv_pages_seq{-1};
    uint64_t      kv_map_cookie{0};
    std::string   msg;
};

class ProgressLogger {
public:
    // Core query / logging API.
    static bool Enabled();
    static bool Verbose();
    static void Log(const ProgressEvent& event);

    // Dump recent events without exiting the process (used from
    // OOM paths and terminate handlers).
    static void DumpRecent();

    // Force-enable progress logging for DriftEngine regardless of env.
    static void ForceEnableForDrift(bool enable);

    // Install a crash handler that dumps recent progress events on abort.
    static void InstallCrashHandler();
};

}  // namespace turbomind
