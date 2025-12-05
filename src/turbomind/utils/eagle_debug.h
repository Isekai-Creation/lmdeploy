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

inline bool isEagleDebugEnabled()
{
    static int cached = -1;
    if (cached == -1) {
        cached = isEnvVarEnabled("LMDEPLOY_EAGLE_DEBUG") ? 1 : 0;
    }
    return cached == 1;
}

inline bool isEagleKVDebugEnabled()
{
    static int cached = -1;
    if (cached == -1) {
        // LMDEPLOY_EAGLE_KV_DEBUG overrides the generic EAGLE flag.
        if (isEnvVarEnabled("LMDEPLOY_EAGLE_KV_DEBUG")) {
            cached = 1;
        }
        else {
            cached = isEagleDebugEnabled() ? 1 : 0;
        }
    }
    return cached == 1;
}

inline bool isEagleMetricsDebugEnabled()
{
    static int cached = -1;
    if (cached == -1) {
        if (isEnvVarEnabled("LMDEPLOY_EAGLE_METRICS_DEBUG")) {
            cached = 1;
        }
        else {
            cached = isEagleDebugEnabled() ? 1 : 0;
        }
    }
    return cached == 1;
}

}  // namespace turbomind

