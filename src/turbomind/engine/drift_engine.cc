#include "src/turbomind/engine/drift_engine.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace {

inline turbomind::ModelLayout resolve_model_layout(const turbomind::DriftEngineConfig& cfg)
{
    if (cfg.model_layout.num_layers > 0 && cfg.model_layout.num_kv_heads > 0 && cfg.model_layout.head_dim > 0 && cfg.model_layout.page_size > 0) {
        return cfg.model_layout;
    }
    auto layout          = turbomind::make_gpt_oss_120b_layout();
    layout.max_seq_len   = cfg.session_len > 0 ? cfg.session_len : layout.max_seq_len;
    return layout;
}

inline std::function<void(const std::vector<turbomind::PrefillChunk>&,
                          const std::vector<std::shared_ptr<turbomind::Request>>&)> make_llama_batch_executor(turbomind::LlamaBatch* batch)
{
    return [batch](const std::vector<turbomind::PrefillChunk>& prefill,
                   const std::vector<std::shared_ptr<turbomind::Request>>& decode) {
        if (batch) {
            batch->ExecuteScheduled(prefill, decode);
        }
    };
}

inline std::function<void()> make_llama_batch_shutdown(turbomind::LlamaBatch* batch)
{
    return [batch]() {
        if (batch) {
            // Best-effort interruption: rely on existing gateway-driven shutdown semantics.
            TM_LOG_INFO("[DriftEngine] Shutdown invoked for LlamaBatch executor.");
        }
    };
}

inline size_t auto_kv_capacity_bytes_from_env()
{
    // Mirror TurboMind BlockManager semantics: treat TM_CACHE_MAX_ENTRY_COUNT
    // as an upper bound on the fraction of *free* device memory usable for
    // KV, then apply an internal safety cap (or TM_KV_EFFECTIVE_RATIO) to
    // avoid over-allocation on large models such as 120B.
    double ratio = 0.75;
    if (const char* env = std::getenv("TM_CACHE_MAX_ENTRY_COUNT")) {
        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if (end != env && v > 0.0 && v <= 1.0) {
            ratio = v;
        }
    }

    double effective_ratio = ratio;
    if (const char* env = std::getenv("TM_KV_EFFECTIVE_RATIO")) {
        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if (end != env && v > 0.0 && v <= 1.0) {
            if (v < effective_ratio) {
                effective_ratio = v;
                TM_LOG_WARNING(
                    "[DriftEngine] TM_KV_EFFECTIVE_RATIO=%s clamping cache_max_entry_count from %.3f to %.3f",
                    env,
                    ratio,
                    effective_ratio);
            }
        }
    }
    else {
        constexpr double kMaxSafeRatio = 0.70;
        if (effective_ratio > kMaxSafeRatio) {
            TM_LOG_WARNING(
                "[DriftEngine] Clamping cache_max_entry_count from %.3f to %.3f to avoid KV overallocation.",
                ratio,
                kMaxSafeRatio);
            effective_ratio = kMaxSafeRatio;
        }
    }

    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    cudaError_t err    = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        TM_LOG_WARNING("[DriftEngine] cudaMemGetInfo failed when deriving KV capacity: %s",
                       cudaGetErrorString(err));
        // Fallback to 2GB if we cannot query.
        free_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    }

    size_t capacity = static_cast<size_t>(static_cast<double>(free_bytes) * effective_ratio);
    TM_LOG_WARNING(
        "[DriftEngine] Auto KV capacity: free=%zu bytes, ratio=%.3f (effective=%.3f) -> kv_capacity_bytes=%zu",
        free_bytes,
        ratio,
        effective_ratio,
        capacity);
    return capacity;
}
}

namespace turbomind {

DriftEngine::DriftEngine(const DriftEngineConfig& cfg,
                         std::shared_ptr<Gateway> gateway,
                         std::shared_ptr<KVCacheManager> kv_mgr,
                         std::shared_ptr<PrefixCache> prefix_cache)
    : cfg_(cfg),
      gateway_(std::move(gateway)),
      kv_mgr_(std::move(kv_mgr)),
      prefix_cache_(std::move(prefix_cache)),
      capacity_sched_(nullptr),
      scheduler_(nullptr)
{
    cfg_.model_layout = resolve_model_layout(cfg_);
    cfg_.kv_layout    = derive_kv_layout(cfg_.model_layout, cfg_.kv_layout);

    // Enforce v1 constraints: non-speculative, TP=1-only (checked in bindings), no CUDA graphs.
    if (cfg_.scheduler_config.enable_speculative_decoding) {
        TM_LOG_WARNING("[DriftEngine] Disabling speculative decoding for v1 non-spec mode.");
        cfg_.scheduler_config.enable_speculative_decoding = false;
    }
    if (cfg_.tp > 1) {
        TM_LOG_WARNING("[DriftEngine] TP>1 not supported in v1 DriftEngine; forcing TP=1 behaviour.");
    }

    // If no explicit KV capacity was provided, derive it from the current
    // free device memory and TM_CACHE_MAX_ENTRY_COUNT after weights have
    // been loaded by TurboMind. This keeps DriftEngine aligned with the
    // BlockManager-based KV sizing used by the legacy engine.
    if (cfg_.kv_capacity_bytes == 0) {
        cfg_.kv_capacity_bytes = auto_kv_capacity_bytes_from_env();
    }

    if (!kv_mgr_) {
        if (cfg_.kv_capacity_bytes > 0) {
            kv_mgr_ = std::make_shared<KVCacheManager>(cfg_.kv_layout, cfg_.kv_capacity_bytes);
            TM_LOG_INFO("[DriftEngine] Created KVCacheManager with %zu bytes", cfg_.kv_capacity_bytes);
        }
        else {
            TM_LOG_WARNING("[DriftEngine] No KVCacheManager provided and kv_capacity_bytes is 0; KV will be unavailable.");
        }
    }

    if (!prefix_cache_ && kv_mgr_) {
        prefix_cache_ = std::make_shared<PrefixCache>(cfg_.kv_layout.page_size, kv_mgr_.get());
    }

    capacity_sched_ = std::make_unique<CapacityScheduler>(kv_mgr_.get(), prefix_cache_.get());
    scheduler_      = std::make_unique<EngineScheduler>(cfg_.scheduler_config, kv_mgr_.get(), cfg_.model_layout, prefix_cache_.get(), capacity_sched_.get());

    // Guardrail: fail fast when configured capacity cannot hold a full batch at session_len.
    if (cfg_.kv_capacity_bytes > 0 && cfg_.session_len > 0 && cfg_.max_batch_size > 0) {
        const auto est          = KVCacheManager::estimate_usage(cfg_.model_layout, cfg_.session_len, 0);
        const size_t total_need = est.bytes_needed * static_cast<size_t>(cfg_.max_batch_size);
        if (total_need > cfg_.kv_capacity_bytes) {
            TM_LOG_ERROR("[DriftEngine] Configured KV capacity (%zu bytes) is insufficient for session_len=%d "
                         "and max_batch_size=%d (requires ~%zu bytes). Reduce session_len/max_batch_size or "
                         "increase kv_capacity_bytes.",
                         cfg_.kv_capacity_bytes,
                         cfg_.session_len,
                         cfg_.max_batch_size,
                         total_need);
            throw std::runtime_error("KV capacity too small for requested session_len * max_batch_size");
        }
    }

    TM_LOG_INFO("DriftEngine created with new architecture");
}

KVLayout DriftEngine::derive_kv_layout(const ModelLayout& model_layout, const KVLayout& provided) const
{
    KVLayout kv = provided;
    kv.num_layers      = kv.num_layers > 0 ? kv.num_layers : model_layout.num_layers;
    kv.num_kv_heads    = kv.num_kv_heads > 0 ? kv.num_kv_heads : model_layout.num_kv_heads;
    kv.head_dim        = kv.head_dim > 0 ? kv.head_dim : model_layout.head_dim;
    kv.page_size       = kv.page_size > 0 ? kv.page_size : model_layout.page_size;
    kv.kv_dtype        = kv.kv_dtype != KVDataType::kFP16 ? kv.kv_dtype : model_layout.kv_dtype;
    kv.bytes_per_value = kv.bytes_per_value > 0 ? kv.bytes_per_value : bytes_per_value_from_dtype(kv.kv_dtype);

    TM_LOG_INFO("Derived KVLayout: num_layers=%d, num_kv_heads=%d, head_dim=%d, page_size=%d, bytes_per_value=%d", 
        kv.num_layers, kv.num_kv_heads, kv.head_dim, kv.page_size, kv.bytes_per_value);

    assert(kv.num_layers > 0 && "num_layers must be positive");
    assert(kv.num_kv_heads > 0 && "num_kv_heads must be positive");
    assert(kv.head_dim > 0 && "head_dim must be positive");
    assert(kv.page_size > 0 && "page_size must be positive");
    assert(kv.bytes_per_value > 0 && "bytes_per_value must be positive");

    return kv;
}

void DriftEngine::run(int rank) {
    TM_LOG_INFO("DriftEngine worker running on rank %d", rank);
    worker_loop(rank);
}

void DriftEngine::shutdown() {
    TM_LOG_INFO("DriftEngine shutting down.");
    abort_.store(true);
    if (batch_shutdown_) {
        batch_shutdown_();
    }
}

void DriftEngine::set_batch_executor(std::function<void(const std::vector<PrefillChunk>&,
                                                        const std::vector<std::shared_ptr<Request>>&)> executor,
                                     std::function<void()> shutdown_cb)
{
    batch_executor_  = std::move(executor);
    batch_shutdown_  = std::move(shutdown_cb);
}

DriftMetrics DriftEngine::metrics() const
{
    return scheduler_ ? scheduler_->snapshot_metrics() : DriftMetrics{};
}

void DriftEngine::bind_llama_batch(LlamaBatch* batch)
{
    llama_batch_ = batch;
    if (llama_batch_) {
        // Bridge KV cache manager and prefix cache into LlamaBatch so that
        // executor mode uses the same KV pages and prefix reuse logic as
        // DriftEngine's scheduler and KVCacheManager.
        if (kv_mgr_ || prefix_cache_) {
            llama_batch_->bridge_kv_cache_manager(kv_mgr_.get(), prefix_cache_.get());
        }
        llama_batch_->set_executor_mode();
        TM_LOG_INFO("[DriftEngine] LlamaBatch bound in executor mode (CUDA graphs disabled for v1).");
    }
    set_batch_executor(make_llama_batch_executor(batch), make_llama_batch_shutdown(batch));
}

void DriftEngine::configure_speculative_decoding(bool enable, const std::string& method, int max_draft_tokens)
{
    if (scheduler_) {
        // Access scheduler's config through reflection or add config methods
        // For now, log the configuration (real implementation would update config)
        TM_LOG_INFO("[DriftEngine] Configuring speculative decoding: enabled=%s, method=%s, max_draft=%d", 
                    enable ? "true" : "false", method.c_str(), max_draft_tokens);
        
        // Note: Real implementation would update scheduler_->cfg_.enable_speculative_decoding
        // This is the integration point for Step D completion
    }
}

void DriftEngine::worker_loop(int rank) {
    TM_LOG_INFO("DriftEngine internal worker_loop on rank %d. Abort flag: %d", rank, abort_.load());

    while (!abort_.load()) {
        std::vector<std::shared_ptr<Request>> infer_reqs;
        std::vector<std::shared_ptr<Request>> kill_reqs;

        // 1. Fetch new/kill requests from Gateway if available.
        if (gateway_) {
            const int free_slots = cfg_.scheduler_config.max_num_seqs
                - (scheduler_ ? scheduler_->get_active_requests_count() + scheduler_->get_queued_requests_count() : 0);
            bool is_empty = scheduler_ ? scheduler_->empty() : true;
            gateway_->pop(infer_reqs, kill_reqs, free_slots, is_empty, abort_, rank);
            if (abort_.load()) {
                TM_LOG_INFO("[DriftEngine] Gateway signaled abort; exiting worker.");
                break;
            }

            // Register new/kill requests with both the executor-facing
            // LlamaBatch (for SequenceManager / KV wiring) and the
            // scheduler exactly once per pop.
            if (llama_batch_ && (!infer_reqs.empty() || !kill_reqs.empty())) {
                llama_batch_->attach_new_requests(infer_reqs, kill_reqs);
            }
            if (scheduler_ && (!infer_reqs.empty() || !kill_reqs.empty())) {
                scheduler_->on_new_requests(infer_reqs, kill_reqs);
            }
        }

        // Check if scheduler detected an OOM and if engine is configured to abort on OOM
        if (scheduler_ && scheduler_->has_oom_detected() && cfg_.abort_on_oom) {
            TM_LOG_ERROR("[DriftEngine] OOM detected by scheduler and abort_on_oom is true. Shutting down engine.");
            shutdown();
            scheduler_->clear_oom_detected(); // Clear the flag after handling
            continue; // Continue to loop condition, which should now be false
        }

        // Check if there are any active or queued requests, otherwise sleep
        if (scheduler_ && scheduler_->empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Prevent busy-waiting
            continue;
        }

        // 3. Schedule a step
        std::vector<PrefillChunk> prefill_batch;
        std::vector<std::shared_ptr<Request>> decode_batch;
        scheduler_->schedule_step(prefill_batch, decode_batch);

        size_t step_prefill_tokens = 0;
        for (const auto& chunk : prefill_batch) {
            step_prefill_tokens += static_cast<size_t>(chunk.len);
        }
        size_t step_decode_tokens = decode_batch.size(); // One token per sequence for now

        DriftMetrics metrics = scheduler_->snapshot_metrics();
        metrics.step_prefill_tokens = step_prefill_tokens;
        metrics.step_decode_tokens  = step_decode_tokens;
        scheduler_->update_metrics(metrics);

        if (prefill_batch.empty() && decode_batch.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Capture pre-step sequence lengths so that the scheduler can
        // derive actual per-sequence token deltas from
        // Request.sequence_length after the backend executes the step.
        std::unordered_map<uint64_t, int> pre_lengths;
        auto capture_len = [&pre_lengths](const std::shared_ptr<Request>& req) {
            if (!req) {
                return;
            }
            int len = 0;
            auto it = req->inputs.find("input_ids");
            if (it != req->inputs.end()) {
                len = it->second.shape(0);
            }
            pre_lengths[req->session.id] = len;
        };
        for (const auto& chunk : prefill_batch) {
            capture_len(chunk.req);
        }
        for (const auto& req : decode_batch) {
            capture_len(req);
        }
 
        TM_LOG_DEBUG("[DriftEngine] Scheduled prefill=%zu tokens (%zu seqs) decode=%zu tokens (%zu seqs)",
                     step_prefill_tokens,
                     prefill_batch.size(),
                     step_decode_tokens,
                     decode_batch.size());

        // Allow the backend to see the scheduler's selection for this
        // step. Executor-mode LlamaBatch currently relies on its
        // internal scheduling but we keep this hook for future biasing.
        if (llama_batch_) {
            llama_batch_->attach_scheduled(prefill_batch, decode_batch);
        }
 
        // 4. Execute batches on the model (hooked via batch_executor_)
        if (batch_executor_) {
            batch_executor_(prefill_batch, decode_batch);
        }
        else {
            TM_LOG_DEBUG("[DriftEngine] No batch executor bound; skipping model execution.");
        }

// 5. After execution, update sequence states using actual
        //    per-sequence token deltas derived from
        //    Request.sequence_length where available.
        if (scheduler_) {
            scheduler_->on_step_executed(prefill_batch, decode_batch, pre_lengths);
        }

        if (abort_.load()) {
            TM_LOG_INFO("[DriftEngine] Abort requested during executor run; exiting worker loop.");
            break;
        }
    }
    TM_LOG_INFO("DriftEngine worker_loop on rank %d exited.", rank);
}


}  // namespace turbomind
