#include "src/turbomind/utils/progress_logger.h"

#include <array>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <sstream>
#include <string_view>
#include <string>

#include "src/turbomind/utils/logger.h"

namespace turbomind {
namespace {

struct ProgressConfig {
    bool enabled{false};
    bool verbose{false};
};

inline bool env_enabled(const char* name, bool default_value)
{
    if (const char* env = std::getenv(name)) {
        return std::atoi(env) != 0;
    }
    return default_value;
}

ProgressConfig init_config()
{
    ProgressConfig cfg;
    cfg.enabled = env_enabled("TM_DRIFT_PROGRESS", false);
    cfg.verbose = env_enabled("TM_DRIFT_PROGRESS_VERBOSE", false);
    return cfg;
}

ProgressConfig& config()
{
    static ProgressConfig cfg = init_config();
    return cfg;
}

// Force-enable flag for DriftEngine regardless of env.
std::atomic<bool> g_force_enabled_for_drift{false};

// Simple ring buffer of recent log lines for crash dumps.
constexpr size_t                               kRingCapacity = 512;
std::array<std::string, kRingCapacity>         g_ring;
std::atomic<size_t>                            g_ring_index{0};
std::atomic<bool>                              g_crash_handler_installed{false};

constexpr std::array<std::string_view, 17> kStageNames = {
    "RequestEnqueue",
    "GatewayPop",
    "SchedulerAdmit",
    "KVReserve",
    "PrefixMatch",
    "PrefillSchedule",
    "PrefillExec",
    "KVWrite",
    "DecodeSchedule",
    "DecodeExec",
    "Sampling",
    "KVExtend",
    "OutputAppend",
    "CallbackNotify",
    "Finish",
    "Release",
    "Error",
};

std::string_view to_string(ProgressStage stage)
{
    const auto idx = static_cast<size_t>(stage);
    if (idx < kStageNames.size()) {
        return kStageNames[idx];
    }
    return "Unknown";
}

void append_field(std::ostringstream& oss, std::string_view key, int64_t value)
{
    oss << ' ' << key << '=' << value;
}

void append_string(std::ostringstream& oss, std::string_view key, const std::string& value)
{
    if (!value.empty()) {
        oss << ' ' << key << "=\"";
        for (const char ch : value) {
            if (ch == '\"') {
                oss << '\\' << ch;
            }
            else {
                oss << ch;
            }
        }
        oss << '\"';
    }
}

void append_to_ring(const std::string& line)
{
    // Best-effort, lock-free append into circular buffer.
    const size_t idx = g_ring_index.fetch_add(1, std::memory_order_relaxed);
    g_ring[idx % kRingCapacity] = line;
}

void crash_dump_handler(int signo)
{
    TM_LOG_ERROR("[DRIFT][PROG] crash signal=%d, dumping recent progress events", signo);
    const size_t end   = g_ring_index.load(std::memory_order_relaxed);
    const size_t start = (end > kRingCapacity) ? end - kRingCapacity : 0;
    for (size_t i = start; i < end; ++i) {
        const std::string& line = g_ring[i % kRingCapacity];
        if (!line.empty()) {
            TM_LOG_ERROR("%s", line.c_str());
        }
    }
    std::_Exit(signo);
}

}  // namespace

bool ProgressLogger::Enabled()
{
    return config().enabled || g_force_enabled_for_drift.load(std::memory_order_relaxed);
}

bool ProgressLogger::Verbose()
{
    return config().verbose;
}

void ProgressLogger::ForceEnableForDrift(bool enable)
{
    g_force_enabled_for_drift.store(enable, std::memory_order_relaxed);
}

void ProgressLogger::InstallCrashHandler()
{
    bool expected = false;
    if (!g_crash_handler_installed.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return;
    }
    std::signal(SIGABRT, crash_dump_handler);
    std::signal(SIGSEGV, crash_dump_handler);
}

void ProgressLogger::Log(const ProgressEvent& event)
{
    if (!Enabled()) {
        return;
    }

    std::ostringstream oss;
    oss << "[DRIFT][PROG] stage=" << to_string(event.stage) << " pct=" << int(event.pct);

    if (event.rank >= 0) {
        append_field(oss, "rank", event.rank);
    }
    if (event.step >= 0) {
        append_field(oss, "step", event.step);
    }
    if (event.seq_id != 0) {
        append_field(oss, "seq", static_cast<int64_t>(event.seq_id));
    }
    if (event.session_id != 0 && event.session_id != event.seq_id) {
        append_field(oss, "session", static_cast<int64_t>(event.session_id));
    }
    if (event.unique_id != 0) {
        append_field(oss, "uid", static_cast<int64_t>(event.unique_id));
    }
    if (event.prefilled_len >= 0) {
        append_field(oss, "prefilled", event.prefilled_len);
    }
    if (event.prompt_len >= 0) {
        append_field(oss, "prompt", event.prompt_len);
    }
    if (event.generated_len >= 0) {
        append_field(oss, "gen", event.generated_len);
    }
    if (event.max_new_tokens >= 0) {
        append_field(oss, "max_new", event.max_new_tokens);
    }
    if (event.token_pos >= 0) {
        append_field(oss, "token_pos", event.token_pos);
    }
    if (event.chunk_start >= 0) {
        append_field(oss, "chunk_start", event.chunk_start);
    }
    if (event.chunk_len >= 0) {
        append_field(oss, "chunk_len", event.chunk_len);
    }
    if (event.batch_size >= 0) {
        append_field(oss, "batch", event.batch_size);
    }
    if (event.kv_pages_total >= 0) {
        append_field(oss, "kv_total", event.kv_pages_total);
    }
    if (event.kv_pages_used >= 0) {
        append_field(oss, "kv_used", event.kv_pages_used);
    }
    if (event.kv_pages_free >= 0) {
        append_field(oss, "kv_free", event.kv_pages_free);
    }
    if (event.kv_pages_seq >= 0) {
        append_field(oss, "kv_seq", event.kv_pages_seq);
    }
    if (event.kv_map_cookie != 0) {
        append_field(oss, "kv_cookie", static_cast<int64_t>(event.kv_map_cookie));
    }

    if (Verbose() || !event.msg.empty()) {
        append_string(oss, "msg", event.msg);
    }

    const std::string line = oss.str();
    append_to_ring(line);
    TM_LOG_INFO("%s", line.c_str());
}

}  // namespace turbomind
