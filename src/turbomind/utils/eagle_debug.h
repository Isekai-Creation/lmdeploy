/*
 * EAGLE debug/trace logging helpers.
 *
 * These helpers gate EAGLE-specific logging behind environment variables so
 * that offline experiments can enable detailed traces without recompiling,
 * while keeping default logs minimal.
 */

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace turbomind {

inline bool isEnvVarEnabled(const char* name)
{
    const char* v = std::getenv(name);
    if (!v || !v[0]) {
        return false;
    }
    // Treat "0" as disabled; any other non-empty value enables the flag.
    return !(v[0] == '0' && v[1] == '\0');
}

// Global EAGLE debug/metrics flags, controlled by SpeculativeConfig via
// setEagleDebugFlags(). Default to false until explicitly enabled.
inline bool& eagleDebugFlag()
{
    static bool flag = false;
    return flag;
}

inline bool& eagleMetricsDebugFlag()
{
    static bool flag = false;
    return flag;
}

inline bool isEagleDebugEnabled()
{
    return eagleDebugFlag();
}

inline bool isEagleKVDebugEnabled()
{
    static int cached = -1;
    if (cached == -1) {
        // LMDEPLOY_EAGLE_KV_DEBUG controls KV rewind-specific logging. This
        // remains env-driven for now as it is orthogonal to decode-mode
        // selection.
        cached = isEnvVarEnabled("LMDEPLOY_EAGLE_KV_DEBUG") ? 1 : 0;
    }
    return cached == 1;
}

inline bool isEagleMetricsDebugEnabled()
{
    return eagleMetricsDebugFlag();
}

inline bool isEagleForceSingleTokenEnabled()
{
    return false;
}

inline bool isEagleDisableMultiTokenEnabled()
{
    return false;
}

inline void setEagleDebugFlags(bool eagle_debug, bool eagle_metrics_debug)
{
    eagleDebugFlag()        = eagle_debug;
    eagleMetricsDebugFlag() = eagle_metrics_debug;
}

// -----------------------------------------------------------------------
// GEMM shape logging and summary
//
// When LMDEPLOY_EAGLE_GEMM_SHAPE_LOG is enabled, selected call sites
// record GEMM shapes relevant to Eagle3 (e.g. draft FFN and LM head).
// Shapes are logged in a machine-parseable format and deduplicated in
// a process-global map. A compact "Top GEMM shapes" summary is emitted
// once at process exit.
// -----------------------------------------------------------------------

struct EagleGemmShapeKey {
    std::string tag;
    int         M{};
    int         K{};
    int         N{};
    int         dtype{};
    std::string layout;

    bool operator<(const EagleGemmShapeKey& other) const noexcept
    {
        if (tag != other.tag) {
            return tag < other.tag;
        }
        if (M != other.M) {
            return M < other.M;
        }
        if (K != other.K) {
            return K < other.K;
        }
        if (N != other.N) {
            return N < other.N;
        }
        if (dtype != other.dtype) {
            return dtype < other.dtype;
        }
        return layout < other.layout;
    }
};

inline std::map<EagleGemmShapeKey, std::uint64_t>& eagleGemmShapeCounts()
{
    static std::map<EagleGemmShapeKey, std::uint64_t> counts;
    return counts;
}

struct EagleSpinLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

inline EagleSpinLock& eagleGemmShapeLock()
{
    static EagleSpinLock lock;
    return lock;
}

class EagleSpinGuard {
public:
    explicit EagleSpinGuard(EagleSpinLock& lock): lock_(lock)
    {
        while (lock_.flag.test_and_set(std::memory_order_acquire)) {
        }
    }

    ~EagleSpinGuard()
    {
        lock_.flag.clear(std::memory_order_release);
    }

private:
    EagleSpinLock& lock_;
};

inline void EagleGemmShapeSummaryAtExit()
{
    if (!isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG")) {
        return;
    }

    auto& counts = eagleGemmShapeCounts();
    if (counts.empty()) {
        return;
    }

    std::vector<std::pair<EagleGemmShapeKey, std::uint64_t>> entries;
    {
        EagleSpinGuard guard(eagleGemmShapeLock());
        entries.assign(counts.begin(), counts.end());
    }

    std::sort(entries.begin(),
              entries.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    constexpr int kTopN = 16;
    std::fprintf(stderr,
                 "[EAGLE3][GEMM_SHAPE][SUMMARY] Top GEMM shapes (tag,M,K,N,dtype,layout,count):\n");
    for (int i = 0; i < static_cast<int>(entries.size()) && i < kTopN; ++i) {
        const auto& key   = entries[i].first;
        const auto  count = entries[i].second;
        std::fprintf(stderr,
                     "  tag=%s M=%d K=%d N=%d dtype=%d layout=%s count=%llu\n",
                     key.tag.c_str(),
                     key.M,
                     key.K,
                     key.N,
                     key.dtype,
                     key.layout.c_str(),
                     static_cast<unsigned long long>(count));
    }

    // Additionally export a machine-readable shape summary for GEMM
    // tuning when running in PERF_MODE so that external scripts can
    // drive gemm2 tuning for the exact shapes observed in perf runs.
    if (isEnvVarEnabled("LMDEPLOY_EAGLE_PERF_MODE")) {
#ifndef _WIN32
        ::mkdir("build", 0777);
#endif
        const char* path = "build/eagle3_gemm_shapes_sm120.json";
        if (FILE* f = std::fopen(path, "w")) {
            std::fprintf(f, "[\n");
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& key   = entries[i].first;
                const auto  count = entries[i].second;
                std::fprintf(f,
                             "  {\"tag\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,"
                             "\"dtype\":%d,\"layout\":\"%s\",\"count\":%llu}%s\n",
                             key.tag.c_str(),
                             key.M,
                             key.K,
                             key.N,
                             key.dtype,
                             key.layout.c_str(),
                             static_cast<unsigned long long>(count),
                             (i + 1 < entries.size() ? "," : ""));
            }
            std::fprintf(f, "]\n");
            std::fclose(f);
        }
    }
}

inline void registerEagleGemmShapeSummary()
{
    static bool registered = false;
    if (!registered) {
        std::atexit(EagleGemmShapeSummaryAtExit);
        registered = true;
    }
}

inline void logEagleGemmShape(const char* tag,
                              int         M,
                              int         K,
                              int         N,
                              int         dtype,
                              const char* layout)
{
    if (!isEnvVarEnabled("LMDEPLOY_EAGLE_GEMM_SHAPE_LOG")) {
        return;
    }

    registerEagleGemmShapeSummary();

    const char* resolved_tag    = tag && tag[0] ? tag : "UNKNOWN";
    const char* resolved_layout = layout && layout[0] ? layout : "row_major";

    std::fprintf(stderr,
                 "[EAGLE3][GEMM_SHAPE] tag=%s M=%d K=%d N=%d dtype=%d layout=%s\n",
                 resolved_tag,
                 M,
                 K,
                 N,
                 dtype,
                 resolved_layout);

    EagleGemmShapeKey key;
    key.tag    = resolved_tag;
    key.M      = M;
    key.K      = K;
    key.N      = N;
    key.dtype  = dtype;
    key.layout = resolved_layout;

    auto& counts = eagleGemmShapeCounts();
    {
        EagleSpinGuard guard(eagleGemmShapeLock());
        counts[key] += 1;
    }
}

// -----------------------------------------------------------------------
// Thread-local GEMM tag for fallback attribution.
//
// EagleGemmTagGuard is used at higher-level call sites (e.g. Eagle3
// draft FFN and LM head) to mark GEMMs in the Eagle3 path. Fallback
// handlers can query the current tag to decide whether to abort in
// PERF_MODE.
// -----------------------------------------------------------------------

inline const char*& eagleCurrentGemmTagRef()
{
    thread_local const char* tag = nullptr;
    return tag;
}

inline const char* eagleCurrentGemmTag()
{
    return eagleCurrentGemmTagRef();
}

struct EagleGemmTagGuard {
    const char* prev_;

    explicit EagleGemmTagGuard(const char* tag): prev_{eagleCurrentGemmTagRef()}
    {
        eagleCurrentGemmTagRef() = tag;
    }

    ~EagleGemmTagGuard()
    {
        eagleCurrentGemmTagRef() = prev_;
    }
};

}  // namespace turbomind
