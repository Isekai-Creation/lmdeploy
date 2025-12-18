// Copyright (c) OpenMMLab. All rights reserved.

#include <memory>

#include "src/turbomind/engine/gateway.h"
#include "src/turbomind/engine/request_queue.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/utils/progress_logger.h"

namespace turbomind {

Gateway::Gateway(int groups, int group_size, std::function<std::shared_ptr<void>()> ctx_factory):
    size_{groups * group_size},
    group_size_{group_size},
    queues_(size_),
    flags_(groups),
    ctx_factory_{ctx_factory},
    next_{0}
{
    for (int i = 0; i < groups; ++i) {
        flags_[i] = std::make_unique<std::atomic<uint64_t>>(0);
    }

    for (int i = 0; i < size_; ++i) {
        const int group_id = i / group_size;
        queues_[i]         = std::make_unique<RequestQueue>(flags_[group_id].get());
    }

    signal_thread_ = std::thread(&Gateway::signal_thread_entry, this);
}

void Gateway::shutdown()
{
    for (auto& q : queues_) {
        q->close();
    }

    signal_buffer_.close();
    signal_thread_.join();
}

void Gateway::signal_thread_entry() noexcept
{
    try {
        while (true) {
            bool                abort{};
            std::vector<Signal> signals = signal_buffer_.take_all(abort);
            if (abort) {
                break;
            }
            else {
                auto ctx = ctx_factory_();
                (void)ctx;
                for (const auto& s : signals) {
                    s();
                }
            }
        }
    }
    catch (const std::exception& e) {
        TM_LOG_ERROR("[Gateway] signal_thread exception: %s", e.what());
        if (turbomind::ProgressLogger::Enabled()) {
            turbomind::ProgressEvent evt{turbomind::ProgressStage::kError};
            evt.pct = 100;
            evt.msg = std::string("gateway_signal_exception:") + e.what();
            turbomind::ProgressLogger::Log(evt);
        }
    }
    catch (...) {
        TM_LOG_ERROR("[Gateway] signal_thread unknown exception");
        if (turbomind::ProgressLogger::Enabled()) {
            turbomind::ProgressEvent evt{turbomind::ProgressStage::kError};
            evt.pct = 100;
            evt.msg = "gateway_signal_exception:unknown";
            turbomind::ProgressLogger::Log(evt);
        }
    }
}

}  // namespace turbomind
