#pragma once

#include <stdexcept>

#include "src/turbomind/utils/logger.h"

#define DRIFT_NO_STUB_FAIL(msg)                                                                                         \
    do {                                                                                                                \
        TM_LOG_ERROR("[DRIFT][NO-STUB] %s", msg);                                                                       \
        throw std::runtime_error(msg);                                                                                  \
    } while (0)
