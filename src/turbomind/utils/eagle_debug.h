/*
 * EAGLE debug/trace logging helpers.
 *
 * These helpers gate EAGLE-specific logging behind environment variables so
 * that offline experiments can enable detailed traces without recompiling,
 * while keeping default logs minimal.
 */

#pragma once

#include <cstdlib>

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

}  // namespace turbomind
