#include "src/turbomind/engine/drift_engine.h"
#include "src/turbomind/utils/logger.h"
#include "src/turbomind/models/common/model_layout.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/models/llama/LlamaBatch.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/utils/progress_logger.h"
#include <chrono>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <cstdlib>
#include <string>
#include <cuda_runtime_api.h>


namespace {

inline turbomind::ModelLayout resolve_model_layout(const turbomind::DriftEngineConfig& cfg)
{
    // Prefer an explicit model_layout provided via bindings (derived
    // from the TurboMind model config) when all key fields are set.
    if (cfg.model_layout.num_layers > 0 && cfg.model_layout.num_kv_heads > 0 && cfg.model_layout.head_dim > 0
        && cfg.model_layout.page_size > 0) {
        TM_LOG_INFO(
            "[DriftEngine] Using explicit ModelLayout override: layers=%d kv_heads=%d head_dim=%d page_size=%d "
            "max_seq_len=%d kv_dtype=%d",
            cfg.model_layout.num_layers,
            cfg.model_layout.num_kv_heads,
            cfg.model_layout.head_dim,
            cfg.model_layout.page_size,
            cfg.model_layout.max_seq_len,
            static_cast<int>(cfg.model_layout.kv_dtype));
        return cfg.model_layout;
    }

    auto layout        = turbomind::make_gpt_oss_120b_layout();
    layout.max_seq_len = cfg.session_len > 0 ? cfg.session_len : layout.max_seq_len;
    TM_LOG_INFO(
        "[DriftEngine] Using default GPT-OSS-120B ModelLayout: layers=%d kv_heads=%d head_dim=%d page_size=%d "
        "max_seq_len=%d kv_dtype=%d",
        layout.num_layers,
        layout.num_kv_heads,
        layout.head_dim,
        layout.page_size,
        layout.max_seq_len,
        static_cast<int>(layout.kv_dtype));
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
    // Treat TM_CACHE_MAX_ENTRY_COUNT as an upper bound on the fraction of
    // *free* device memory usable for KV. For DriftEngine v1 we apply a
    // safety clamp and headroom reservation by default so that KV does not
    // starve weights/workspaces, while still allowing power‑users to
    // override the clamp via TM_DRIFT_KV_NO_CLAMP.
    double ratio = 0.75;
    if (const char* env = std::getenv("TM_CACHE_MAX_ENTRY_COUNT")) {
        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if (end != env && v > 0.0 && v <= 1.0) {
            ratio = v;
        }
    }

    double effective_ratio = ratio;

    // Optional environment override to tighten the KV budget without
    // changing engine configs. When set to a value in (0,1], this caps
    // the internal ratio used for sizing KVCacheManager before the
    // Drift‑specific clamp is applied.
    if (const char* env = std::getenv("TM_KV_EFFECTIVE_RATIO")) {
        char*  end = nullptr;
        double v   = std::strtod(env, &end);
        if (end != env && v > 0.0 && v <= 1.0) {
            if (v < effective_ratio) {
                effective_ratio = v;
                TM_LOG_WARNING(
                    "[DriftEngine] TM_KV_EFFECTIVE_RATIO=%s overrides cache_max_entry_count from %.3f to %.3f",
                    env,
                    ratio,
                    effective_ratio);
            }
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

    // By default clamp the effective ratio to 0.70 to leave room for
    // model weights and workspaces unless the user explicitly disables
    // the clamp via TM_DRIFT_KV_NO_CLAMP=1.
    bool   clamp_enabled = true;
    if (const char* env = std::getenv("TM_DRIFT_KV_NO_CLAMP")) {
        if (std::atoi(env) != 0) {
            clamp_enabled = false;
        }
    }
    double clamped_ratio = effective_ratio;
    if (clamp_enabled && clamped_ratio > 0.70) {
        clamped_ratio = 0.70;
    }

    // Reserve headroom for runtime workspaces. Keep at least 10% of
    // total memory or 8GB, whichever is larger.
    const size_t reserve_bytes = std::max<size_t>(8ULL * 1024ULL * 1024ULL * 1024ULL,
                                                  static_cast<size_t>(static_cast<double>(total_bytes) * 0.10));

    size_t capacity_from_ratio = static_cast<size_t>(static_cast<double>(free_bytes) * clamped_ratio);
    size_t capacity            = capacity_from_ratio;
    if (free_bytes > reserve_bytes) {
        size_t max_allowed = free_bytes - reserve_bytes;
        if (capacity > max_allowed) {
            capacity = max_allowed;
        }
    }
    else {
        // If we are already below the headroom threshold, be conservative.
        capacity = static_cast<size_t>(static_cast<double>(free_bytes) * 0.5);
    }

    TM_LOG_INFO(
        "[DriftEngine] Auto KV capacity: free=%zu total=%zu bytes, "
        "ratio=%.3f (effective=%.3f, clamped=%.3f) reserve_bytes=%zu -> kv_capacity_bytes=%zu",
        free_bytes,
        total_bytes,
        ratio,
        effective_ratio,
        clamped_ratio,
        reserve_bytes,
        capacity);
    return capacity;
}

inline void log_progress_mem_snapshot(const char* scope, uint8_t pct)
{
    if (!turbomind::ProgressLogger::Enabled()) {
        return;
    }
    if (!scope) {
        scope = "drift_scope";
    }
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return;
    }
    turbomind::ProgressEvent evt{turbomind::ProgressStage::kKVReserve};
    evt.pct = pct;
    std::ostringstream oss;
    oss << scope << " free=" << free_bytes << " total=" << total_bytes;
    evt.msg = oss.str();
    turbomind::ProgressLogger::Log(evt);
}

inline void log_progress_message(uint8_t pct, const std::string& msg)
{
    if (!turbomind::ProgressLogger::Enabled()) {
        return;
    }
    turbomind::ProgressEvent evt{turbomind::ProgressStage::kKVReserve};
    evt.pct = pct;
    evt.msg = msg;
    turbomind::ProgressLogger::Log(evt);
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
    // Always enable Drift progress trace and crash handler for DriftEngine.
    ProgressLogger::ForceEnableForDrift(true);
    ProgressLogger::InstallCrashHandler();
    log_progress_mem_snapshot("drift_engine_ctor_entry", 2);

    cfg_.model_layout = resolve_model_layout(cfg_);
    cfg_.kv_layout    = derive_kv_layout(cfg_.model_layout, cfg_.kv_layout);
    log_progress_message(3, "drift_engine_model_layout_ready");

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

    log_progress_message(4, "kv_capacity_selected bytes=" + std::to_string(cfg_.kv_capacity_bytes));

    if (!kv_mgr_) {
        if (cfg_.kv_capacity_bytes > 0) {
            log_progress_message(5, "kv_cache_manager_build_begin bytes=" + std::to_string(cfg_.kv_capacity_bytes));
            kv_mgr_ = std::make_shared<KVCacheManager>(cfg_.kv_layout, cfg_.kv_capacity_bytes);
            TM_LOG_INFO("[DriftEngine] Created KVCacheManager with %zu bytes", cfg_.kv_capacity_bytes);
            log_progress_mem_snapshot("kv_cache_manager_ready", 7);
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

    log_progress_message(9, "drift_scheduler_ready");

    // Guardrail: fail fast when configured capacity cannot hold a full batch at session_len.
    if (cfg_.kv_capacity_bytes > 0 && cfg_.session_len > 0 && cfg_.max_batch_size > 0) {
        const auto  est = KVCacheManager::estimate_usage(cfg_.model_layout, cfg_.session_len, 0);

        // Use the quantization-aware page_bytes override when present so
        // that the capacity check reflects the actual per-page KV size
        // (INT8/INT4) instead of the conservative FP16/BF16 layout.
        size_t bytes_per_seq = est.bytes_needed;
        if (cfg_.kv_layout.page_bytes_override > 0 && est.pages_needed > 0) {
            bytes_per_seq = cfg_.kv_layout.page_bytes_override * est.pages_needed;
        }

        const size_t total_need = bytes_per_seq * static_cast<size_t>(cfg_.max_batch_size);
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

    log_progress_mem_snapshot("drift_engine_ctor_done", 10);
    log_progress_message(15, "drift_engine_ctor_complete");
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

    // Align the KV page byte size with TurboMind's block-level KV
    // layout so that each DriftEngine page maps 1:1 to a TurboMind KV
    // block. For non-quantized KV (the v1 DriftEngine default), derive
    // the block size via get_cache_block_size so that
    // KVCacheManager::page_bytes() matches BlockManager's block_size for
    // the same (layers, kv_heads, head_dim, page_size, dtype).
    // get_cache_block_size is defined in kv_cache_utils_v2.cu; forward
    // declare it here to avoid pulling heavy Cutlass headers into this
    // translation unit.
    extern size_t get_cache_block_size(DataType dtype,
                                       DataType kvtype,
                                       int      layer_num,
                                       int      head_num,
                                       int      head_dim,
                                       int      block_seq_len);

    auto kv_dtype_to_datatype = [](KVDataType dt) -> DataType {
        switch (dt) {
        case KVDataType::kFP16:
            return DataType::kFloat16;
        case KVDataType::kBF16:
            return DataType::kBfloat16;
        case KVDataType::kNVFP4:
            // FP4 payload; block layout uses 4‑bit values with an
            // external scale pool. Treat as kFloat4_e2m1 for sizing.
            return DataType::kFloat4_e2m1;
        }
        return DataType::kFloat16;
    };

    if (cfg_.quant_policy == 0) {
        const DataType dt = kv_dtype_to_datatype(kv.kv_dtype);
        const size_t   block_bytes =
            get_cache_block_size(dt, dt, kv.num_layers, kv.num_kv_heads, kv.head_dim, kv.page_size);
        if (block_bytes > 0) {
            kv.page_bytes_override = block_bytes;
            TM_LOG_INFO(
                "[DriftEngine] KV page_bytes_override (non‑quantized) via get_cache_block_size: %zu "
                "(layers=%d kv_heads=%d head_dim=%d page_size=%d)",
                kv.page_bytes_override,
                kv.num_layers,
                kv.num_kv_heads,
                kv.head_dim,
                kv.page_size);
        }
    }
    else {
        // Derive a quantization-aware page_bytes override when INT4/INT8
        // KV cache is enabled via quant_policy. This mirrors the block
        // layout used by SequenceManager + BlockManager so that each
        // DriftEngine KV "page" maps 1:1 to a TurboMind KV block and the
        // attention kernels see identical geometry.
        int device_id = 0;
        cudaGetDevice(&device_id);

        int           sm_version = 0;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
            sm_version = prop.major * 10 + prop.minor;
        }

        const KvCacheMode kv_mode = GetKvCacheMode(cfg_.quant_policy, sm_version);

        // Only integer KV cache modes (INT8 / INT4) change the logical
        // block size today. FP4/NVFP4 remains mapped to the unquantized
        // layout until the dedicated FP4 scale pool is wired end-to-end.
        if (kv_mode == KvCacheMode::kInt8 || kv_mode == KvCacheMode::kInt4) {
            const int dbits = kv.bytes_per_value * 8;

            int q_bits = dbits;
            int t_bits = 0;

            switch (kv_mode) {
            case KvCacheMode::kInt8:
                q_bits = 8;
                t_bits = dbits;
                break;
            case KvCacheMode::kInt4:
                q_bits = 4;
                t_bits = dbits;
                break;
            default:
                break;
            }

            const int    block_len        = kv.page_size;
            const int    head_dim         = kv.head_dim;
            const int    kv_heads         = kv.num_kv_heads;
            const int    layers           = kv.num_layers;
            const int    token_data_size  = q_bits * head_dim / 8;
            const int    token_param_size = t_bits * 2 / 8;
            const size_t head_data_size   = static_cast<size_t>(block_len) * static_cast<size_t>(token_data_size);
            const size_t head_param_size  = static_cast<size_t>(block_len) * static_cast<size_t>(token_param_size);
            const size_t layer_size       = static_cast<size_t>(kv_heads) * 2U * (head_data_size + head_param_size);
            const size_t block_size       = layer_size * static_cast<size_t>(layers);

            if (block_size > 0) {
                kv.page_bytes_override = block_size;
                TM_LOG_INFO(
                    "[DriftEngine] KV quantization active: quant_policy=%d kv_mode=%d "
                    "q_bits=%d t_bits=%d page_size=%d -> page_bytes_override=%zu",
                    cfg_.quant_policy,
                    static_cast<int>(kv_mode),
                    q_bits,
                    t_bits,
                    kv.page_size,
                    kv.page_bytes_override);
            }
        }
    }

    TM_LOG_INFO(
        "Derived KVLayout: num_layers=%d, num_kv_heads=%d, head_dim=%d, page_size=%d, bytes_per_value=%d, "
        "kv_factor=%d, page_bytes_override=%zu",
        kv.num_layers,
        kv.num_kv_heads,
        kv.head_dim,
        kv.page_size,
        kv.bytes_per_value,
        kv.kv_factor,
        kv.page_bytes_override);

    assert(kv.num_layers > 0 && "num_layers must be positive");
    assert(kv.num_kv_heads > 0 && "num_kv_heads must be positive");
    assert(kv.head_dim > 0 && "head_dim must be positive");
    assert(kv.page_size > 0 && "page_size must be positive");
    assert(kv.bytes_per_value > 0 && "bytes_per_value must be positive");

    return kv;
}

void DriftEngine::run(int rank) {
    TM_LOG_INFO("DriftEngine worker running on rank %d", rank);
    if (ProgressLogger::Enabled()) {
        ProgressEvent evt{ProgressStage::kGatewayPop};
        evt.pct  = 85;
        evt.rank = rank;
        evt.msg  = "worker_loop_launch";
        ProgressLogger::Log(evt);
    }
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
        if (ProgressLogger::Enabled()) {
            ProgressEvent evt{ProgressStage::kGatewayPop};
            evt.pct = 30;
            evt.msg = "llama_batch_bound";
            ProgressLogger::Log(evt);
        }
    }
    set_batch_executor(make_llama_batch_executor(batch), make_llama_batch_shutdown(batch));
    if (ProgressLogger::Enabled()) {
        ProgressEvent evt{ProgressStage::kGatewayPop};
        evt.pct = 40;
        evt.msg = "batch_executor_ready";
        ProgressLogger::Log(evt);
    }
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

    bool logged_ready = false;
 
     while (!abort_.load()) {
         if (!logged_ready) {
             if (ProgressLogger::Enabled() && !progress_ready_logged_.exchange(true)) {
                 ProgressEvent evt_start{ProgressStage::kGatewayPop};
                 evt_start.pct  = 90;
                 evt_start.rank = rank;
                 evt_start.msg  = "worker_loop_active";
                 ProgressLogger::Log(evt_start);
                 ProgressEvent evt_ready{ProgressStage::kRelease};
                 evt_ready.pct  = 100;
                 evt_ready.rank = rank;
                 evt_ready.msg  = "drift_engine_ready_for_requests";
                 ProgressLogger::Log(evt_ready);
             }
             logged_ready = true;
         }
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
 
        if (ProgressLogger::Enabled()) {
            for (const auto& chunk : prefill_batch) {
                if (!chunk.req) {
                    continue;
                }
                ProgressEvent evt{ProgressStage::kPrefillSchedule};
                evt.pct         = 45;
                evt.seq_id      = chunk.req->session.id;
                evt.session_id  = chunk.req->session.id;
                evt.unique_id   = chunk.req->unique_id;
                evt.chunk_start = chunk.start_pos;
                evt.chunk_len   = chunk.len;
                evt.rank        = rank;
                evt.msg         = "scheduled";
                ProgressLogger::Log(evt);
            }
            for (const auto& req : decode_batch) {
                if (!req) {
                    continue;
                }
                ProgressEvent evt{ProgressStage::kDecodeSchedule};
                evt.pct        = 70;
                evt.seq_id     = req->session.id;
                evt.session_id = req->session.id;
                evt.unique_id  = req->unique_id;
                evt.rank       = rank;
                if (auto it = pre_lengths.find(req->session.id); it != pre_lengths.end()) {
                    evt.token_pos = it->second;
                }
                evt.msg = "scheduled";
                ProgressLogger::Log(evt);
            }
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
        //    per-sequence token deltas reported by the executor.
        if (scheduler_) {
            std::vector<ExecutionResult> exec_results;
            if (llama_batch_) {
                exec_results = llama_batch_->get_execution_results();
            }
            scheduler_->on_step_executed(prefill_batch, decode_batch, pre_lengths, exec_results);
            if (ProgressLogger::Enabled()) {
                ProgressEvent evt{ProgressStage::kCallbackNotify};
                evt.pct   = 96;
                evt.rank  = rank;
                evt.msg   = "step_complete";
                evt.batch_size = static_cast<int>(decode_batch.size() + prefill_batch.size());
                ProgressLogger::Log(evt);
            }
        }

        if (abort_.load()) {
            TM_LOG_INFO("[DriftEngine] Abort requested during executor run; exiting worker loop.");
            break;
        }
    }
    TM_LOG_INFO("DriftEngine worker_loop on rank %d exited.", rank);
}


}  // namespace turbomind
