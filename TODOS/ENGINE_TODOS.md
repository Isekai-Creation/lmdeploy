# LMDeploy / TurboMind Engine TODOs (GPT‑OSS‑120B, Non‑Speculative)

> Progress values are approximate and based on current code + docs:
>
> - `0–30%` – mostly design / scaffolding only.  
> - `30–70%` – partially implemented, needs correctness/perf work.  
> - `70–100%` – functionally present, needs polishing/validation at the upper end.  
>
> Scope of this file:
>
> - Focuses on **pure GPT‑OSS‑120B decoding** (no speculative decoding / no EAGLE3).  
> - EAGLE/EAGLE3/SpecPV‑specific work is tracked in `EAGLE_TODOS.md` and `SPECPV_TODO.md`.  
>
> Every task below is a **numbered checkbox** with a progress estimate.

---

## 0.0 DriftEngine HF GPT‑OSS‑20B Status Snapshot (2025‑12‑18)

Concrete implementation status for the non‑spec, TP=1 DriftEngine path
running HF `openai/gpt‑oss‑20b` (and re‑usable for GPT‑OSS‑120B):

- ✅ **DriftEngine C++ core (ctor, worker loop, bindings) – ~95% IMPLEMENTED**
  - `DriftEngine` composes `Gateway`, `KVCacheManager`, `PrefixCache`,
    `CapacityScheduler`, and `EngineScheduler`.
  - Worker loop uses `scheduler_->on_new_requests`, `schedule_step`,
    `batch_executor_` (LlamaBatch executor mode), and
    `on_step_executed` for feedback.
  - Progress tracing is always enabled for Drift (`ForceEnableForDrift`)
    and a crash handler dumps recent `[DRIFT][PROG]` events on
    `SIGABRT`/`SIGSEGV`/`std::terminate`.
  - **Needs testing:** full PREFILL→DECODE→FINISHED lifecycle on HF
    20B/120B under long‑context + batched workloads.

- ✅ **KV cache manager, layout, and capacity guardrails – ~90% IMPLEMENTED**
  - `KVCacheManager` manages a single contiguous device KV buffer with
    correct `page_bytes_` sizing (layers × KV heads × head_dim ×
    page_size × bytes_per_value × 2 for K+V).
  - Per‑page refcounts, per‑sequence page lists, and
    `reserve`/`release`/`estimate_usage` are implemented together with
    `KVUsageEstimate::bytes_needed`.
  - `DriftEngine` auto‑derives `kv_capacity_bytes` from
    `TM_CACHE_MAX_ENTRY_COUNT` and free device memory, with a clamp and
    reserved headroom, so users do **not** have to set
    `kv_capacity_bytes` explicitly.
  - **Needs testing:** stress runs on HF 20B/120B to validate that KV
    usage stays within capacity and that no OOM/deadlock appears under
    tight budgets.

- ✅ **PrefixCache & prefix reuse – ~80% IMPLEMENTED**
  - `PrefixCache` stores page‑aligned token prefixes only, with LRU
    eviction and metrics (hits/misses/evictions/bytes_evicted).
  - Integration with `EngineScheduler` is in place: prefix matches seed
    `prefilled_len` and pre‑existing KV pages, inserts happen after
    prefill completes.
  - Prefix cache never calls `KVCacheManager::release`; KV ownership is
    centralized in `CapacityScheduler` / `EngineScheduler`.
  - **Needs testing:** synthetic workloads with shared prefixes and
    eviction to validate shared‑page refcounts and reuse.

- ✅ **Scheduler, CapacityScheduler, and sequence lifecycle – ~85% IMPLEMENTED**
  - `SchedulerConfig` (token/seq budgets, chunked prefill knobs,
    `prefer_decode_over_prefill`) and `EngineScheduler` implement
    chunked prefill (`PrefillChunk`), decode scheduling, and phase
    transitions (`kPrefill`→`kDecode`→`kFinished`).
  - `CapacityScheduler` uses `KVUsageEstimate` + `KVCacheManager` to
    admit/finish requests and tracks blocked/rejected metrics.
  - `on_step_executed` consumes `ExecutionResult` deltas; missing or
    non‑positive deltas are treated as hard errors (no synthetic
    progress).
  - **Needs testing:** mixed prompt/batch workloads on HF 20B/120B to
    verify PREFILL/DECODE balance, capacity rejections, and single KV
    release per sequence.

- ✅ **LlamaBatch executor mode + Drift KV bridge – ~85% IMPLEMENTED**
  - `LlamaBatch::bridge_kv_cache_manager` attaches `KVCacheManager` and
    `PrefixCache` into a `DriftEngineKVCache` and switches to executor
    mode (no internal `Start()` thread, no CUDA graphs).
  - KV pointer tables are built from Drift KV pages using
    `build_page_ptr_table`, and `install_kv_table` logs pages/ptrs and
    format (`TM_DRIFT_KV_TABLE_FORMAT=per_page|kv_split`).
  - `ExecuteScheduled` goes through `InitializeFromScheduler` →
    `Forward` (wrapped in try/catch with `forward_exception` logs) →
    `Finish` → `build_execution_results`.
  - KV canary hooks (`TM_DRIFT_KV_CANARY`) are present for K/V regions
    but need runtime validation on HF 20B/120B.

- ✅ **ProgressLogger + crash ring buffer – ~95% IMPLEMENTED**
  - Unified `[DRIFT][PROG]` events for enqueue, KV reserve, prefill,
    decode, sampling, callbacks, finish, and release.
  - Ring buffer of last 512 events and crash handlers for signals and
    `std::terminate` dump progress on abnormal exits.
  - New C++ paths (DriftEngine, Gateway, LlamaBatch executor, pybind
    wrappers) all emit stage‑specific events.
  - **Needs testing:** sampling a full 0→100% trace for a single HF 20B
    request and verifying it is sufficient for post‑mortem debugging.

- ✅ **Pybind DriftEngineWrapper & Gateway lifetime – ~90% IMPLEMENTED**
  - DriftEngineWrapper parses dict configs, constructs `DriftEngine`,
    binds a real `LlamaBatch`, and starts the worker thread.
  - `run(rank)` installs a proper CUDA context/Stream/Allocator and
    catches/logs any `std::exception` from `engine->run`.
  - **Newly implemented:** `shutdown()` now:
    - calls `engine->shutdown()`, joins `worker_thread` if joinable,  
    - calls `gateway->shutdown()` **before** resetting the shared
      pointer, ensuring the Gateway’s internal `signal_thread_` is
      joined,  
    - then resets `engine`, `llama_batch`, `gateway`, `kv_mgr`,
      `prefix_cache`.
    - This removes the `std::terminate` risk from destructing a
      joinable `std::thread` (the source of the earlier
      “terminate called without an active exception” on Kaggle).
  - `forward(...)` allocates CPU output buffers, logs a
    `[DRIFT][PROG] stage=RequestEnqueue msg="pybind_submit"`, and
    pushes the request into `Gateway`.
  - **Needs testing:** repeated pipeline create/destroy cycles to
    confirm no terminate() from Gateway/worker threads.

- ✅ **AsyncEngine backend_config wiring for drift – ~90% IMPLEMENTED**
  - AsyncEngine now sets:
    ```python
    engine_cfg = getattr(self.engine, "engine_config", None)
    self.backend_config = engine_cfg or backend_config
    ```
    so drift/TurboMind always have a valid `backend_config` (and avoid
    `NoneType.max_batch_size` errors) even if `engine.engine_config` is
    not populated in some error paths.
  - Drift backend selection in `lmdeploy/api.py` uses
    `backend_config is DriftEngineConfig` ⇒ `backend="drift"`.
  - **Needs testing:** HF 20B drift pipeline creation via
    `lmdeploy.api.pipeline` and `AsyncEngine` with various configs.

- ✅ **Benchmark harness (HF GPT‑OSS‑20B drift mode) – ~85% IMPLEMENTED**
  - `LM/lmdeploy_drift/benchmark_speculative.py` now constructs a
    `DriftEngineConfig` (non‑spec, TP=1, BF16, session_len from
    scenario) and calls `lmdeploy.pipeline(..., backend_config=cfg)`,
    thereby selecting the drift backend.
  - `run_spec_suite.sh` in `lmdeploy_drift` has drift baseline
    scenarios wired for HF 20B (and 120B later).
  - **Needs testing:** actual drift runs producing non‑zero tokens/sec
    and latencies (user‑driven on RTX PRO 6000 / Kaggle H100).

The rest of this file keeps the original per‑task TODO structure, but
the above snapshot clarifies which major DriftEngine components are
implemented vs. waiting on validation.

---

## 0. Engineer‑Level Execution Plan

This section groups the detailed TODOs by engineer and clarifies priorities and boundaries. All tasks referenced below correspond to the numbered checkboxes in Sections 1–6.

### 0.1 Engineer A – C++ Engine, Scheduler, KV, DriftEngine Core

Primary focus: **C++ engine internals**, including scheduler, KV cache manager, prefix cache, capacity control, and the core `DriftEngine` orchestration.

- **Scheduler & core engine loop**
  - Implement the new `SchedulerConfig` and `EngineScheduler` (Tasks **1.2.1–1.2.3**).  
    - Start by introducing `SchedulerConfig` and `EngineScheduler` scaffolding.  
    - Add `SequenceState` and `SequencePhase` tracking, then wire chunked prefill (`PrefillChunk`) into `schedule_step`.  
  - Refactor the worker loop to go through `EngineScheduler` (Task **1.2.5**).  
    - Replace direct Gateway consumption with the `WorkerLoop` sketch that calls `scheduler.on_new_requests` and `schedule_step`.  
  - Formalize and test session lifecycle and cancellation (Tasks **1.3.1–1.3.2**).  
    - Write tests for start/continue/end/kill transitions, and cancellation under load.

- **KV cache manager & layout**
  - Implement `KVCacheManager` as in the `KVLayout` / `KVReservation` sketch (Task **2.1.1**).  
    - Start with single‑GPU support, then ensure the API can handle TP/PP shapes.  
  - Define `ModelLayout` for GPT‑OSS‑120B and derive `KVLayout` from it (Task **2.1.2**).  
    - Keep this the single source of truth for KV shapes and page size.

- **Prefix cache & reuse**
  - Implement `PrefixCache` (Task **2.2.1**) and integrate it into `EngineScheduler` (Task **2.2.2**).  
    - Ensure keys are page‑aligned and namespaces are honored.  
  - Implement eviction policy inside `PrefixCache` (Task **2.2.3**).  
    - Wire eviction to `KVCacheManager::release` and export simple metrics (hits, misses, evictions).

- **KV‑aware capacity scheduler**
  - Implement `KVUsageEstimate` and `estimate_usage` (Task **4.1.1**).  
  - Implement `CapacityScheduler` and its integration with `KVCacheManager` (Task **4.1.2**).  
  - Hook capacity checks into the scheduler path (Task **4.1.3**).  
    - The worker loop should only promote requests to `active` when `try_start_request` succeeds.

- **DriftEngine core**
  - Implement `DriftEngineConfig` (C++) and `DriftEngine` class (Tasks **6.2.1–6.2.2**).  
    - Compose it from `Gateway`, `KVCacheManager`, `PrefixCache`, `EngineScheduler`, and `CapacityScheduler`.  
  - Implement the two‑queue policy and adaptive scheduling for DriftEngine (Tasks **6.3.1–6.3.2**).  
    - Start with fixed decode/prefill ratios; add EMA‑based tuning once stable.  
  - Finalize the end‑to‑end DriftEngine lifecycle (Task **6.4.1**).  
    - Ensure the C++ flow matches the high‑level steps outlined (request intake → scheduling → model exec → callbacks).

Execution order for Engineer A (recommended):
1. Implement `ModelLayout` + `KVCacheManager` (2.1.x).  
2. Implement `SchedulerConfig` + `EngineScheduler` basic token budget (1.2.1, 1.2.2).  
3. Add chunked prefill (1.2.3) and refactor the worker loop (1.2.5).  
4. Add `KVUsageEstimate` + `CapacityScheduler` (4.1.x) and integrate into scheduler.  
5. Implement `PrefixCache` and integration (2.2.x).  
6. Add DriftEngine C++ class and policies (6.2.x, 6.3.x, 6.4.1).  
7. Harden session lifecycle / cancellation with tests (1.3.x).

### 0.2 Engineer B – Python Configs, Bindings, Kernels/Perf, Benchmarks, DriftEngine API

Primary focus: **Python‑side engine configuration**, bindings into C++, kernel tuning, CUDA graphs, benchmarks, and the public DriftEngine API surface.

- **Python configs & bindings**
  - Extend existing configs with `TurboMindSchedulerConfig` and `TurboMindKVConfig` (Task **1.1.1**).  
    - Thread them through `PytorchEngineConfig` (or equivalent) as `scheduler_config` and `kv_config`.  
  - Define unified `TurboMindEngineConfig` and update both `pipeline` and `api_server` to accept it (Task **1.1.2**).  
    - Add validation that checks invariants (e.g., `session_len * max_batch_size` vs KV capacity).

- **NVFP4 & MLA validation**
  - Validate NVFP4 KV for GPT‑OSS‑120B (Task **2.3.1**).  
    - Run A/B comparisons vs FP16/FP8 and document acceptable error/throughput gains.  
  - Ensure MLA + KV layout compatibility for base decode (Task **2.3.2**).  
    - Confirm MLA models use the same `KVCacheManager` layout and no hidden re‑layout.

- **Kernel/perf work**
  - Profile and tune attention kernels for 120B workloads (Task **3.1.1**).  
    - Focus on prefill/decode under mixed load; feed concrete kernel configs back to C++ if new fusions are needed.  
  - Extend fusion for QKV, norms, and MLP (Task **3.1.2**).  
  - Audit sampling path to ensure GPU‑only hot path (Task **3.2.1**).  
    - Identify and eliminate CPU bottlenecks in standard decode.  
  - Implement fused logits+sampling kernels where beneficial (Task **3.2.2**).  
  - Design CUDA graph capture for decode and evaluate prefill graphs (Tasks **3.3.1–3.3.2**).  
    - Start with a single, common decode shape for capture; expand based on measured benefit.

- **Benchmarks, metrics & tests**
  - Build a unified GPT‑OSS‑120B benchmark harness for all engines (Task **5.1.1**).  
    - Make it easy to plug in TurboMind, DriftEngine, vLLM, sglang, TensorRT‑LLM, EasyDeL with the same scenarios.  
  - Integrate `RequestMetrics` and NVTX into this harness (Task **5.1.2**).  
  - Define and wire perf micro‑gates for non‑speculative decode (Task **5.2.1**).  
  - Extend correctness regression tests for KV reuse and long‑context decode (Task **5.2.2**).

- **DriftEngine Python API**
  - Implement `DriftEngineConfig` (Python) and binding to C++ `DriftEngineConfig` (Task **6.1.1**).  
  - Add `drift_pipeline` and `drift_api_server` wrappers (Task **6.1.2**).  
    - Ensure these choose `backend="drift"` and call into the new C++ engine.  
  - Guarantee DriftEngine is strictly opt‑in and co‑exists with legacy TurboMind backend (Task **6.4.2**).  
    - No breaking changes to existing `backend="turbomind"` behavior.

Execution order for Engineer B (recommended):
1. Implement `TurboMindSchedulerConfig`/`TurboMindKVConfig` and unified `TurboMindEngineConfig` (1.1.x).  
2. Implement Python `DriftEngineConfig` and new entrypoints (6.1.x).  
3. Work on NVFP4/MLA validation (2.3.x) and kernel profiling/fusion (3.1.x, 3.2.x).  
4. Add CUDA graph support (3.3.x).  
5. Build the unified benchmark harness + metrics (5.1.x) and perf/correctness gates (5.2.x).  
6. Wire DriftEngine into the public API and ensure clean coexistence with TurboMind (6.4.2).

---

## 1. Core Engine & Scheduler (Non‑Speculative)

### 1.1 Python → TurboMind Configuration Surface

- [ ] **1.1.1 Expose all TurboMind engine knobs in Python configs.** (Owner: Engineer B, Progress: 85%)  
  Details: Current configs cover `tp`, `pp`, `session_len`, `max_batch_size`, precision, etc.  
  Missing (at the time of original writing):
  - Scheduler knobs: max tokens per step, max sequences per step, prefill/decode policy.  
  - KV allocator knobs: KV page size, eviction policy, prefix sharing enable/disable.  
  Design (Python):
  - Add a dedicated scheduler/KV section to the existing backend config, e.g. (now implemented in `lmdeploy/pytorch/config.py` as `TurboMindSchedulerConfig` and `TurboMindKVConfig` and re‑used by `DriftEngineConfig` in `lmdeploy/messages.py`):
    ```python
    @dataclass
    class TurboMindSchedulerConfig:
        max_num_batched_tokens: int = 2048
        max_num_seqs: int = 128
        enable_chunked_prefill: bool = True
        max_num_partial_prefills: int = 1
        long_prefill_token_threshold: int = 0
        scheduler_policy: Literal["fcfs", "priority"] = "fcfs"
        prefer_decode_over_prefill: bool = True  # when token budget tight

    @dataclass
    class TurboMindKVConfig:
        kv_page_size: int = 128
        kv_capacity_bytes: int | None = None  # auto if None
        prefix_cache_enabled: bool = True
        prefix_cache_eviction_policy: Literal["lru", "priority"] = "lru"
        kv_alignment: int = 16  # bytes, for allocator
    ```
  - Thread these into the existing engine configs under dedicated `scheduler` / `kv` fields so both TurboMind and Drift backends share the same knob surface.

- [ ] **1.1.2 Unify pipeline / api_server backend configuration.** (Owner: Engineer B, Progress: 60%)  
  Details: Pipeline and api_server have separate backend configs.  
  Needed:
  - A single `TurboMindEngineConfig` that encapsulates scheduling, KV, MLA, precision and logging.  
  - Shared validation and defaults for both pipeline and api_server paths.  
  Design (Python):
  - Introduce a single top‑level config used by both pipeline and api_server:
    ```python
    @dataclass
    class TurboMindEngineConfig:
        model_path: str
        tp: int = 1
        pp: int = 1
        session_len: int = 8192
        max_batch_size: int = 128
        dtype: Literal["fp16", "bf16", "fp8"] = "fp16"
        scheduler: TurboMindSchedulerConfig = field(default_factory=TurboMindSchedulerConfig)
        kv: TurboMindKVConfig = field(default_factory=TurboMindKVConfig)
        mla_enabled: bool = False
        log_level: Literal["info", "debug", "warning"] = "info"
    ```
  - `lmdeploy.pipeline(...)` and `lmdeploy.serve.api_server(...)` both accept `backend_config: TurboMindEngineConfig` and share a single validator that fills defaults and enforces invariants (e.g., `session_len * max_batch_size` must fit KV capacity if provided).

### 1.2 Engine‑Level Scheduler (Prefill vs Decode, Token Budget)

**Status snapshot (HF GPT‑OSS‑20B / GPT‑OSS‑120B, non‑spec, TP=1):**

- **SchedulerConfig + EngineScheduler core:** Implemented and integrated.
  - Per‑step token budget (`max_num_batched_tokens`, `max_num_seqs`) is enforced in `EngineScheduler::schedule_step`.
  - `SequenceState` / `SequencePhase` track `prompt_len`, `prefilled_len`, `generated_len`, and `max_new_tokens` per sequence.
  - Chunked prefill is implemented via `PrefillChunk { req, start_pos, len }` and `tokens_for_prefill_chunk`, with `max_num_partial_prefills` and `long_prefill_token_threshold`.
  - Two queues (`prefill_request_queue_`, `decode_request_queue_`) and a two‑pass schedule (prefill vs decode) are active, honoring `prefer_decode_over_prefill`.
  - `schedule_step` now uses decode/prefill token budgets derived from an internal decode/prefill ratio (pre‑tuned but not yet extensively profiled).
  - Post‑step feedback `on_step_executed(...)` consumes `ExecutionResult` from `LlamaBatch` to update `SequenceState` and drive phase transitions (PREFILL→DECODE→FINISHED).
  - **Needs testing:** multi‑sequence fairness under mixed prompt lengths; extreme `max_num_batched_tokens` / `max_num_seqs` settings; long‑context 20B/120B sweeps.

- [ ] **1.2.1 Introduce an explicit per‑step token budget.** (Owner: Engineer A, Progress: 95% – implemented, needs stress‑validation)  
  Details: Gateway/RequestQueue implement basic continuous batching, but there is no global token budget.  
  Needed:
  - A scheduler component that, per step, selects a batch of requests such that:  
    - `sum(tokens_being_processed)` ≤ `max_num_batched_tokens`.  
    - Respects maximum sequences per step.  
  Implementation details:
  - `SchedulerConfig` lives in `src/turbomind/engine/scheduler_config.h` with:
    - `max_num_batched_tokens`, `max_num_seqs`.
    - `enable_chunked_prefill`, `max_num_partial_prefills`, `max_long_partial_prefills`, `long_prefill_token_threshold`.
    - `prefer_decode_over_prefill` and `schedule_policy` (FCFS vs small‑first), plus prefix/latency hints.
  - `EngineScheduler` in `src/turbomind/engine/EngineScheduler.{h,cc}`:
    - Owns `cfg_`, `KVCacheManager*`, `ModelLayout`, optional `PrefixCache*` and `CapacityScheduler*`.
    - `schedule_step(std::vector<PrefillChunk>& prefill_batch, std::vector<std::shared_ptr<Request>>& decode_batch)`:
      - Applies `schedule_policy` to `prefill_request_queue_`.
      - Computes per‑step budgets for decode and prefill from `max_num_batched_tokens` and an internal decode/prefill ratio.
      - Fills `decode_batch` and `prefill_batch` so that:
        - Total tokens ≤ `max_num_batched_tokens`.
        - Total sequences ≤ `max_num_seqs`.
      - Emits DEBUG logs of prefill/decode token counts per step.
  - `DriftEngine::worker_loop` calls `scheduler_->schedule_step(...)` for every step and uses the returned batches to drive `LlamaBatch::ExecuteScheduled`.
  - **What remains:** systematic perf sweeps and corner‑case stress (very large batches, pathological prompt distributions), plus tuning of decode/prefill ratios.

- [ ] **1.2.2 Track prefill vs decode phases per request.** (Owner: Engineer A, Progress: 95% – implemented, needs full lifecycle tests)  
  Details: DynamicDecodeLayer covers decode; prefill is implicit in initial forward.  
  Needed:
  - Per‑request phase state (`PREFILL`, `DECODE`, `FINISHED`).  
  - Ability to schedule long prompts in chunks (Section 1.2.3) distinct from 1‑token decode.  
  Implementation details:
  - `SequencePhase` and `SequenceState` are implemented as described; `seq_states_` is an `unordered_map<seq_id, SequenceState>`.
  - `EngineScheduler::on_new_requests`:
    - Creates `SequenceState` for `start` requests with:
      - `prompt_len` = prompt tokens.
      - `prefilled_len` = matched prefix tokens (from `PrefixCache`) or 0.
      - `generated_len` = 0.
      - `max_new_tokens` = request’s `gen_cfg.max_new_tokens` (subject to clamping).
      - `phase` = `kPrefill` or `kDecode` based on whether prompt is already fully covered by prefix pages.
      - KV metadata (`kv_reservation_handle`, `kv_page_ids`, `kv_cookie`) populated from `KVCacheManager`/`CapacityScheduler`.
  - `EngineScheduler::on_step_executed` and `update_sequence_state`:
    - Consume `ExecutionResult` (prefill+decode) and actual sequence lengths per step.
    - Advance `prefilled_len` and/or `generated_len`.
    - Transition `phase` to `kDecode` once `prefilled_len >= prompt_len`.
    - Transition `phase` to `kFinished` at EOS or once `generated_len >= max_new_tokens` and release KV.
  - **What remains:** focused tests around complex session flows (restarts, kill, end, partial prefill) to confirm no double‑release or stuck `SequenceState`.

- [ ] **1.2.3 Implement chunked prefill with partial concurrency.** (Owner: Engineer A, Progress: 90% – implemented, needs long‑context validation)  
  Details: No chunked prefill today; long prompts must be processed in one go.  
  Needed:
  - API + scheduler logic to split long prompts into chunks that fit the token budget.  
  - Controls similar to vLLM: `max_num_partial_prefills`, thresholds for “long” prompts.  
  Implementation details:
  - `PrefillChunk` is implemented in `EngineScheduler.h` and used throughout the DriftEngine path.
  - `tokens_for_prefill_chunk(const SequenceState&)` in `EngineScheduler.cc`:
    - Computes remaining prompt tokens as `prompt_len - prefilled_len`.
    - Chooses chunk size from `cfg_.max_num_batched_tokens_per_seq()` (derived from `max_num_batched_tokens` and `max_num_seqs`), clamped to [`1`, `max_num_batched_tokens`].
  - `schedule_step`:
    - For each prefill sequence in the queue, pushes `PrefillChunk{req, state.prefilled_len, chunk_len}` into `prefill_batch` while budgets permit.
  - `LlamaBatch::ExecuteScheduled` and `InitializeFromScheduler`:
    - Accept the `PrefillChunk` vector and use `chunk.start_pos + chunk.len` as context length for prefill sequences.
    - KV allocation/install is done per sequence via the unified KV cache; prefix tokens are captured for `PrefixCache`.
  - **What remains:** end‑to‑end long‑context (e.g., 16K/32K) HF 20B/120B runs to confirm chunk boundaries and KV pointer tables stay aligned under stress.

- [ ] **1.2.4 Make admission KV‑capacity‑aware (Guaranteed‑Completion mode).** (Owner: Engineer A, Progress: 70%)  
  Details: KV utilities can compute sizes, but capacity is not surfaced to the scheduler.  
  Needed:
  - KV usage estimator per request (prompt length + max_new_tokens).  
  - Admission control that only starts requests that can complete with current KV capacity (no mid‑run OOM).  
  Design (C++):
  - Use `KVCacheManager` (see 2.1.1) to expose:
    ```cpp
    struct KVUsageEstimate {
        size_t blocks_needed;
        size_t bytes_needed;
    };

    KVUsageEstimate estimate_kv_usage(const SchedulerConfig& cfg,
                                      const ModelLayout& layout,
                                      int prompt_len,
                                      int max_new_tokens);
    ```
  - `EngineScheduler` calls `KVCacheManager::can_reserve(seq_id, estimate)` before transitioning a request from `queued` to `active`:
    ```cpp
    bool KVCacheManager::can_reserve(uint64_t seq_id,
                                     const KVUsageEstimate& est) const;
    bool KVCacheManager::reserve(uint64_t seq_id,
                                 const KVUsageEstimate& est);
    void KVCacheManager::release(uint64_t seq_id);
    ```
  - Admission only proceeds if `reserve` succeeds; otherwise the request remains in the queue or is rejected.

- [ ] **1.2.5 Global continuous batching across prefill and decode.** (Owner: Engineer A, Progress: 70%)  
  Details: Continuous batching exists in a basic form via Gateway/RequestQueue.  
  Needed:
  - A scheduler loop that:  
    - Continuously admits new requests while others are decoding.  
    - Balances prefill vs decode work under the token budget and KV capacity.  
    - Avoids starvation of short prompts by long ones.  
  Design (engine loop):
  - Refactor the per‑rank worker loop roughly as:
    ```cpp
    void WorkerLoop(Gateway& gateway,
                    EngineScheduler& scheduler,
                    int rank) {
        while (!abort) {
            std::vector<std::shared_ptr<Request>> infer_reqs, kill_reqs;
            bool blocking = true;
            gateway.pop(infer_reqs, kill_reqs,
                        /*max_infer=*/cfg.max_num_seqs,
                        blocking, abort, rank);
            scheduler.on_new_requests(infer_reqs, kill_reqs);

            std::vector<std::shared_ptr<Request>> prefill_batch;
            std::vector<std::shared_ptr<Request>> decode_batch;
            scheduler.schedule_step(prefill_batch, decode_batch);

            run_model_batches(prefill_batch, decode_batch);
        }
    }
    ```
  - `schedule_step` is responsible for mixing prefill and decode to keep the GPU saturated while respecting token and KV budgets.

### 1.3 Session & Cancellation Semantics

- [ ] **1.3.1 Finalize and document session lifecycle invariants.** (Owner: Engineer A, Progress: 90% – implemented in code, needs doc + tests)  
  Details: `SessionParam` + `SeqId2Rank` implement start/continue/kill semantics.  
  Needed:
  - Clear spec for legal state transitions: `start → continue* → end` or `kill`.  
  - Tests covering race conditions (start+kill, cancel during prefill/decode, double‑end).  

- [ ] **1.3.2 Ensure robust cancellation under load (no leaks, no stuck sessions).** (Owner: Engineer A, Progress: 80% – implemented, needs stress‑runs)  
  Details: `cancel_flag`, `Gateway::cancel`, and kill paths exist.  
  Needed:
  - Stress tests with many concurrent cancels.  
  - Checks that canceled sessions always free KV and are unbound from `SeqId2Rank`.  

---

## 2. KV Cache Manager & Prefix Reuse

### 2.1 First‑Class KV Allocator (Paged Blocks)

- [ ] **2.1.1 Implement an engine‑level `KVCacheManager` abstraction.** (Owner: Engineer A, Progress: 95% – implemented, needs large‑model validation)  
  Implementation:
  - `KVLayout`, `KVReservation`, `KVUsageEstimate` and `KVCacheManager` are implemented in `src/turbomind/core/kv_cache_manager.{h,cc}`:
    - Layout: `num_layers`, `num_kv_heads`, `head_dim`, `page_size`, `kv_dtype`, `bytes_per_value`, `kv_factor`.
    - Capacity: `total_pages()`, `used_pages()`, `free_pages()`, `page_bytes()`, and `get_layout()`.
    - Reservation API:
      - `KVCacheManager::estimate_usage(const ModelLayout&, int prompt_len, int max_new_tokens)`:
        - Computes `pages_needed` / `bytes_needed` from layout geometry.
        - Clamps total tokens to `layout.max_seq_len` to stay within block‑pointer capacity.
      - `can_reserve(seq_id, est)`, `reserve(seq_id, est, KVReservation*, pre_existing_page_ids)`, `release(seq_id)`.
      - Tracks per‑sequence page maps and per‑page refcounts; supports shared pages from `PrefixCache` via `pre_existing_page_ids`.
    - Safety:
      - Logs KV layout and `page_bytes`, plus about‑to‑alloc/free snapshots via `cudaMemGetInfo`.
      - Validates page IDs and pointer ranges in `build_page_ptr_table`.
  - DriftEngine uses `KVCacheManager` as the single KV backing store:
    - KV capacity is derived from free device memory and `TM_CACHE_MAX_ENTRY_COUNT` (with conservative clamps and headroom).
    - `EngineScheduler` and `CapacityScheduler` use `estimate_usage` + `reserve` to guarantee KV capacity for each sequence from start to finish.
  - **Needs testing:** extreme long‑context / high‑batch HF 20B/120B runs; multi‑process/DP topologies beyond TP=1.

- [ ] **2.1.2 Define KV layout contracts per model family.** (Owner: Engineer A, Progress: 80%)  
  Details: Layout logic is distributed among model and kernel code.  
  Needed:
  - Per‑model layout spec (LLaMA, GPT‑OSS‑120B) that documents:  
    - KV tensor shape per layer and head.  
    - Page/block sizes and indexing scheme.  
  - Use this spec inside `KVCacheManager` and kernels for consistency.  
  Design (C++):
  - Add a small `ModelLayout` helper in `src/turbomind/models/common/model_layout.h`:
    ```cpp
    struct ModelLayout {
        int num_layers;
        int num_kv_heads;
        int head_dim;
        int page_size;     // recommended KV page size
        int max_seq_len;
    };

    ModelLayout make_gpt_oss_120b_layout();
    ```
  - `KVLayout` is derived from `ModelLayout` plus datatype; both the engine and `KVCacheManager` take a `ModelLayout` instance so they share the same assumptions.

### 2.2 Prefix Sharing / Prefix Cache

- [ ] **2.2.1 Design and implement a prefix cache (non‑speculative).** (Owner: Engineer A, Progress: 80%)  
  Details: There is no global prefix cache today.  
  Needed:
  - A prefix cache that maps `(token_ids, optional namespace)` → set of KV page indices.  
  - Implementation can be:  
    - Tree‑structured (Radix‑like, as in sglang).  
    - Page‑aligned structure (similar to vLLM PagedAttention).  
  Design (C++):
  - Introduce a lightweight prefix cache in `src/turbomind/core/prefix_cache.h`:
    ```cpp
    struct PrefixKey {
        std::vector<int32_t> tokens;
        uint64_t             namespace_id;  // e.g., adapter id or 0
    };

    struct PrefixMatchResult {
        std::vector<int32_t> page_indices;  // KV pages covering matched prefix
        int                  matched_tokens;
    };

    class PrefixCache {
    public:
        explicit PrefixCache(int page_size);

        PrefixMatchResult match(const PrefixKey& key) const;
        void insert(const PrefixKey& key,
                    const std::vector<int32_t>& page_indices,
                    int priority = 0);
        void erase(uint64_t seq_id);
    };
    ```
  - Use `page_size` to truncate `tokens` to page boundaries before storing.

- [ ] **2.2.2 Integrate prefix cache with the scheduler.** (Owner: Engineer A, Progress: 60%)  
  Details: No wiring yet.  
  Needed:
  - On prefill admission:  
    - Look up existing cached prefixes.  
    - Reuse KV pages where possible instead of recomputing from scratch.  
  - Update cache on completion or KV eviction.  
  Design (integration):
  - `EngineScheduler` consults `PrefixCache` before scheduling a prefill chunk:
    ```cpp
    PrefixKey key{/*tokens=*/prompt_tokens, /*namespace_id=*/0};
    auto match = prefix_cache_.match(key);
    int matched = match.matched_tokens;
    // prefilled_len starts at matched, and KV pages in match.page_indices
    // are attached to the new sequence via KVCacheManager.
    ```
  - After prefill completes for a sequence, the scheduler inserts its prefix into the cache with the KV pages reserved for that sequence, allowing future requests to reuse them.

- [ ] **2.2.3 Implement a simple, tunable eviction policy with metrics.** (Owner: Engineer A, Progress: 60%)  
  Details: No configurable eviction policy.  
  Needed:
  - At least one policy (LRU or priority) with:  
    - Eviction on KV pressure.  
    - Metrics for hit rate, eviction rate, and wasted KV due to fragmentation.  
  Design (policy):
  - Internally implement `PrefixCache` on top of:
    ```cpp
    struct CacheEntry {
        PrefixKey              key;
        std::vector<int32_t>   page_indices;
        int                    priority;
        uint64_t               last_access_ts;
    };
    ```
  - Eviction strategy:
    - On KV pressure (signaled by `KVCacheManager`), evict lowest‑priority / oldest entries, and release the corresponding KV pages via `KVCacheManager::release`.
  - Expose metrics:
    - `hit_count`, `miss_count`, `eviction_count`, `bytes_evicted`.

### 2.3 NVFP4 KV & MLA (Base Decode Path)

- [ ] **2.3.1 Validate NVFP4 KV for GPT‑OSS‑120B decode.** (Owner: Engineer B, Progress: 60%)  
  Details: NVFP4 KV kernels and utilities exist.  
  Needed:
  - End‑to‑end tests on GPT‑OSS‑120B:  
    - A/B correctness vs FP16/FP8 KV (logit diffs, PPL).  
    - Throughput gains vs overhead for KV quantization/dequantization.  

- [ ] **2.3.2 Ensure KV layout works cleanly with MLA for base decode.** (Owner: Engineer B, Progress: 50%)  
  Details: FlashMLA/MLA code exists; integration with KV manager is not fully formalized.  
  Needed:
  - Confirm that MLA attention paths read KV from the same layout used by base decode.  
  - Make sure MLA models can use the same `KVCacheManager` abstraction.  

---

## 3. GPU Kernels, Fusion & CUDA Graphs (Non‑Speculative)

### 3.1 Attention & MLP Kernel Fusions

- [ ] **3.1.1 Profile and tune attention kernels for 120B prefill + decode.** (Owner: Engineer B, Progress: 50%)  
  Details: Custom attention kernels (including MLA variants) are present.  
  Needed:
  - Nsight‑based profiling on representative 120B workloads:  
    - Short‑prompt / long‑prompt mixes.  
    - Single‑stream and multi‑tenant loads.  
  - Targeted tuning of block sizes, tiling, and memory accesses.  

- [ ] **3.1.2 Extend fusion for QKV projection, norms, and MLP where safe.** (Owner: Engineer B, Progress: 40%)  
  Details: Some fusion exists; other paths are still separate kernels.  
  Needed:
  - Identify hot unfused sequences (e.g., `RMSNorm → GEMM → activation → GEMM`).  
  - Add fused variants when:  
    - They significantly reduce memory traffic or kernel launch cost.  
    - They don’t over‑complicate code or harm portability.  

### 3.2 Sampling Path (GPU‑Only)

- [ ] **3.2.1 Audit sampling path to ensure GPU‑only hot path.** (Owner: Engineer B, Progress: 60%)  
  Details: `SamplingLayer` runs on GPU, but some edge options may touch CPU.  
  Needed:
  - Confirm that for standard decoding (top‑k/top‑p/temperature/repetition penalty) all operations stay on GPU.  
  - Minimize any host side logic to control‑plane only (e.g., stopping conditions, streaming).  

- [ ] **3.2.2 Add fused logits + sampling kernels for large vocab + batches.** (Owner: Engineer B, Progress: 30%)  
  Details: Not all configs use fused softmax + sampling.  
  Needed:
  - Implement fused kernels that perform:  
    - Logits scaling → softmax/normalization → top‑k/top‑p selection → sampling.  
  - Ensure they work for large vocab sizes and large batch sizes typical of 120B serving.  

### 3.3 CUDA Graphs Around Decode Loops

- [ ] **3.3.1 Design CUDA graph capture for decode‑only batches.** (Owner: Engineer B, Progress: 10%)  
  Details: TurboMind currently does not capture decode loops into CUDA graphs.  
  Needed:
  - Identify common decode batch shapes (batch sizes, token budgets).  
  - Capture graphs for those shapes and reuse them when possible, similar to TensorRT‑LLM.  

- [ ] **3.3.2 Evaluate optional piecewise graphs for prefill.** (Owner: Engineer B, Progress: 0%)  
  Details: No piecewise graphs today.  
  Needed:
  - Prototype graph capture for common prefill token counts.  
  - Only proceed if decode graphs show clear benefit and prefill graphs are tractable.  

---

## 4. Scheduling vs KV Capacity (Non‑Speculative)

### 4.1 KV‑Aware Capacity Scheduler

- [ ] **4.1.1 Implement a KV capacity estimator per request.** (Owner: Engineer A, Progress: 95% – implemented, needs corner‑case coverage)  
  - `KVUsageEstimate { pages_needed, bytes_needed }` and `estimate_usage(const ModelLayout&, int prompt_len, int max_new_tokens)` are implemented on `KVCacheManager`.
  - Token count is clamped to `layout.max_seq_len` to keep `pages_needed` within the range that LlamaBatch’s block‑pointer tables can represent.
  - `EngineScheduler::on_new_requests` uses `estimate_usage` for every new `start` request.

- [ ] **4.1.2 Implement a Guaranteed‑Completion capacity scheduler.** (Owner: Engineer A, Progress: 90% – implemented, needs heavy‑load tests)  
  - `CapacityScheduler` exists in `src/turbomind/engine/capacity_scheduler.{h,cc}`:
    - Holds `KVCacheManager*` and `PrefixCache*`.
    - Implements `try_start_request(seq_id, KVUsageEstimate, KVReservation*, pre_existing_page_ids)` and delegates to:
      - `KVCacheManager::reserve` with or without shared prefix pages.
    - Tracks basic capacity metrics (blocked / rejected) for DriftMetrics.
  - `EngineScheduler` uses `CapacityScheduler` (when provided) rather than calling `KVCacheManager` directly.
  - Once a request is admitted, its KV reservation is held until the sequence is marked finished and `release_kv` is called.

- [ ] **4.1.3 Integrate capacity checks with Gateway / RequestQueue.** (Owner: Engineer A, Progress: 85% – integrated into worker loop, needs multi‑tenant validation)  
  - `DriftEngine::worker_loop`:
    - Uses `gateway_->pop(...)` to get incoming requests.
    - Immediately feeds them to `EngineScheduler::on_new_requests`, which:
      - Enforces SessionParam invariants and kill/end semantics.
      - Uses `CapacityScheduler`/`KVCacheManager` to decide if KV can be reserved for the sequence before admitting it.
      - Rejects or marks requests with `ec` when KV capacity is insufficient (with detailed progress logs).
  - Gateway still owns fairness and session routing; `EngineScheduler` decides which queued requests can become active based on KV and token budget.
  - **Needs testing:** behavior under sustained overload (lots of large prompts) to ensure we never start a request that cannot complete due to KV exhaustion and that we surface clear errors instead of mid‑run OOM.

---

## 5. Benchmarking, Metrics & Validation (Non‑Speculative)

### 5.1 Benchmark Matrix vs vLLM / sglang / TensorRT‑LLM / EasyDeL

- [ ] **5.1.1 Define a standard GPT‑OSS‑120B benchmark suite (non‑speculative).** (Owner: Engineer B, Progress: 30%)  
  Details: Some scripts exist (`benchmark_esurge.py`, TRT benchmarks, etc.), but not unified for base decode.  
  Needed:
  - A single harness that can run:  
    - TurboMind, vLLM, sglang, TensorRT‑LLM, EasyDeL.  
    - Common scenarios:  
      - Short prompts, long prompts, mixed workloads.  
      - Different batch sizes and context lengths (e.g., 4K, 8K, 16K, 32K).  

- [ ] **5.1.2 Integrate engine metrics (RequestMetrics, NVTX) into benchmarks.** (Owner: Engineer B, Progress: 20%)  
  Details: Metrics are captured but not wired into a reporting pipeline.  
  Needed:
  - Capture and summarize:  
    - TTFT, tokens/sec, tail latencies.  
    - GPU utilization, KV usage, cache hit rates (once prefix cache exists).  

### 5.2 Regression Tests & Gates

- [ ] **5.2.1 Establish micro‑gates for non‑speculative perf.** (Owner: Engineer B, Progress: 25%)  
  Details: Some perf gates exist but are more EAGLE‑focused.  
  Needed:
  - Automated micro‑benchmarks for base decode only:  
    - Per‑step throughput vs target.  
    - Latency distribution per request shape.  
  - Stable thresholds and CI checks to prevent regressions.  

- [ ] **5.2.2 Extend correctness regression tests for KV and long‑context decode.** (Owner: Engineer B, Progress: 25%)  
  Details: Some tests exist.  
  Needed:
  - Tests covering:  
    - KV reuse correctness once prefix cache is implemented.  
    - Long‑context generation (e.g., 16K–32K) vs reference runs.  
    - Multi‑GPU (TP+PP) correctness for 120B decode.  

---

## 6. DriftEngine: High‑Level Design (Non‑Speculative, TurboMind‑Based)

Goal: A new **DriftEngine** backend that uses TurboMind’s kernels and models, but with a more aggressive scheduler, KV manager, and prefix cache, targeting strictly higher throughput and better latency than the current TurboMind engine and external engines (vLLM, sglang, TensorRT‑LLM, EasyDeL) for **non‑speculative GPT‑OSS‑120B**.

### 6.1 DriftEngineConfig (Python)

- [ ] **6.1.1 Define `DriftEngineConfig` and integrate it into LMDeploy.** (Owner: Engineer B, Progress: 90% – implemented, needs API polish/docs)  
  Design (Python):
  - New config in `lmdeploy`:
    ```python
    @dataclass
    class DriftEngineConfig:
        # Model / parallelism
        model_path: str
        tp: int = 1
        pp: int = 1
        session_len: int = 8192
        max_batch_size: int = 256
        dtype: Literal["fp16", "bf16", "fp8"] = "fp16"

        # Scheduler
        scheduler: TurboMindSchedulerConfig = field(default_factory=TurboMindSchedulerConfig)
        kv: TurboMindKVConfig = field(default_factory=TurboMindKVConfig)

        # Drift‑specific tuning
        prefer_high_throughput: bool = True
        # When True, bias scheduler toward larger batches and chunked prefill.
        target_latency_ms_p50: int = 50
        target_latency_ms_p95: int = 200
        # Expose trade‑off knobs:
        decode_microbatch_size: int | None = None
        prefill_microbatch_size: int | None = None

        # Instrumentation / safety
        max_queued_requests: int = 4096
        abort_on_oom: bool = True
        log_level: Literal["info", "debug", "warning"] = "info"
    ```
  - `DriftEngineConfig` is a thin wrapper around the TurboMind scheduler/KV pieces but adds **explicit throughput/latency targets** and microbatch sizes, so the scheduler can tune itself at runtime.  
  - Current status: `DriftEngineConfig` is implemented in `lmdeploy/messages.py` and accepted by `lmdeploy.api.pipeline` / `lmdeploy.api.serve` as a `backend_config`.
    - Additional fields (`enable_prefix_caching`, `enable_speculative_decoding`, `enable_cuda_graphs`, `enable_memory_compaction`, `enable_adaptive_batching`, `cache_max_entry_count`) are present and mapped into the C++ `DriftEngineConfig` via `to_cpp_drift_engine_config`.
    - TurboMind‑derived model layout overrides (`_tm_num_layers`, `_tm_num_kv_heads`, `_tm_head_dim`, `kv_page_size`) are exported as a `model_layout` dict so C++ uses the correct GPT‑OSS‑20B/120B geometry instead of the static 120B default.

- [ ] **6.1.2 Expose `DriftEngineConfig` in pipeline and server entrypoints.** (Owner: Engineer B, Progress: 80% – wired, needs dedicated helpers/docs)  
  Design:
  - New entrypoints:
    ```python
    from lmdeploy import DriftEngineConfig

    def drift_pipeline(config: DriftEngineConfig):
        ...

    def drift_api_server(config: DriftEngineConfig):
        ...
    ```  
  - Internally, these construct C++ `DriftEngine` instances instead of the legacy TurboMind engine.  
  - Current status:
    - `DriftEngineConfig` can already be passed to `pipeline` / `serve`.
    - `lmdeploy/turbomind/turbomind.py` inspects `backend_config` and sets `_engine_type="drift"` when a `DriftEngineConfig` is provided; `_create_drift_engine` uses the new pybind drift factories (`create_engine_with_model`, `create_engine`, `create_drift_engine`).
    - A convenience `drift_api_server(...)` wrapper exists in `lmdeploy/api.py` (backend is set to `"drift"`); a dedicated `drift_pipeline(...)` helper plus public docs are still TBD.

### 6.2 DriftEngine C++ Top‑Level Structure

- [ ] **6.2.1 Implement `DriftEngine` C++ class wrapping TurboMind components.** (Owner: Engineer A, Progress: 90% – implemented, needs extended validation)  
  Design (C++):
  - New files: `src/turbomind/engine/drift_engine.h/.cc`:
    ```cpp
    namespace turbomind {

    class DriftEngine {
    public:
        DriftEngine(const DriftEngineConfig& cfg,
                    std::shared_ptr<Gateway> gateway,
                    std::shared_ptr<KVCacheManager> kv_mgr,
                    std::shared_ptr<PrefixCache> prefix_cache);

        // Main loop entry for a worker rank (run in a dedicated thread).
        void run(int rank);

        // Shutdown coordination
        void shutdown();

    private:
        DriftEngineConfig             cfg_;
        std::shared_ptr<Gateway>     gateway_;
        std::shared_ptr<KVCacheManager> kv_mgr_;
        std::shared_ptr<PrefixCache> prefix_cache_;
        EngineScheduler              scheduler_;
        CapacityScheduler            capacity_sched_;
        std::atomic<bool>           abort_{false};

        void worker_loop(int rank);
    };

    }  // namespace turbomind
    ```
  - `DriftEngine` in `src/turbomind/engine/drift_engine.{h,cc}` is implemented as a composition of:
    - `Gateway` (request routing / session→rank mapping).
    - `EngineScheduler` (Section 1.2) for prefill/decode queueing and token‑budgeted steps.
    - `KVCacheManager` + `PrefixCache` (Section 2) for paged, shared KV.
    - `CapacityScheduler` (Section 4) for Guaranteed‑Completion KV admission.
  - `run(int rank)` / `worker_loop(int rank)`:
    - Drive the main loop: `Gateway::pop` → `scheduler_.on_new_requests` → `scheduler_.schedule_step` → `LlamaBatch::ExecuteScheduled` → `scheduler_.on_step_executed`.
    - Emit detailed `[DRIFT][PROG]` events for ctor/kv_capacity/scheduler_ready/worker_ready/prefill/decode and errors.
  - A minimal metrics API (`DriftEngine::metrics()`) exposes a snapshot of `DriftMetrics` derived from `EngineScheduler::snapshot_metrics()`.
  - **Needs testing:** long multi‑request runs on HF GPT‑OSS‑20B/120B; failure scenarios (KV exhaustion, Gateway aborts, worker exceptions) and recovery semantics.

- [ ] **6.2.2 Add a C++ config struct mirroring `DriftEngineConfig`.** (Owner: Engineer A, Progress: 95% – implemented, needs perf tuning)  
  Design/implementation:
  - `DriftEngineConfig` (C++) in `src/turbomind/engine/drift_engine_config.h`:
    - Mirrors Python `DriftEngineConfig` fields:
      - `SchedulerConfig scheduler_config;`
      - `ModelLayout model_layout;`
      - `KVLayout kv_layout;`
      - `size_t kv_capacity_bytes;`
      - `bool prefer_high_throughput;`
      - `int target_latency_ms_p50, target_latency_ms_p95;`
      - `int max_queued_requests;`
      - `bool abort_on_oom;`
      - `std::string log_level;`
    - `SchedulerConfig::from_engine_config(const DriftEngineConfig&)` provides the canonical mapping into scheduler settings.
  - Python `DriftEngineConfig` is converted to this struct via `lmdeploy/config/drift_config.py::to_cpp_drift_engine_config`, including TurboMind‑derived `model_layout` overrides for HF GPT‑OSS‑20B/120B.

### 6.3 DriftEngine Scheduling Policies

- [ ] **6.3.1 Implement a two‑queue scheduler policy (prefill vs decode) tuned for throughput.** (Owner: Engineer A, Progress: 85% – implemented, needs perf tuning)  
  Design/implementation:
  - Within `EngineScheduler` when used by DriftEngine:
    - Maintains prefill and decode queues (`prefill_request_queue_`, `decode_request_queue_`) tagged via `SequenceState.queue_tag`.
    - `schedule_step`:
      - Computes token budget `B = cfg_.max_num_batched_tokens`.
      - Splits `B` into decode and prefill budgets using an internal decode/prefill token ratio (initially 50/50, adjustable via metrics).
      - Fills `decode_batch` and `prefill_batch` under these budgets, using FCFS or size‑based policy as configured.
    - Ensures that the total number of sequences in both batches ≤ `max_num_seqs`.
  - **Needs testing:** workload‑level throughput/latency sweeps to validate that the current ratio and queue policy are effective across scenarios.

- [ ] **6.3.2 Add adaptive tuning based on observed latency and tokens/sec.** (Owner: Engineer A, Progress: 40% – metrics wired, tuning heuristics WIP)  
  Design:
  - Track moving averages:
    ```cpp
    struct DriftMetrics {
        double ema_tokens_per_second;
        double ema_p50_latency_ms;
        double ema_p95_latency_ms;
    };
    ```
  - Every N steps:
    - Compare `ema_p50_latency_ms` / `ema_p95_latency_ms` to `target_latency_ms_p50` / `target_latency_ms_p95`.
    - Adjust scheduler ratio:
      - If latency too high and throughput above target, shift more budget to decode or reduce max concurrent prefill.
      - If throughput too low and latency well below targets, increase batch sizes or allow more concurrent prefill.

### 6.4 DriftEngine Data Flow (End‑to‑End)

- [ ] **6.4.1 Define the end‑to‑end request lifecycle in DriftEngine.** (Owner: Engineer A, Progress: 80% – implemented, needs full HF 20B/120B validation)  
  Implementation (high‑level flow):
  1. Python side:
     - User constructs `DriftEngineConfig` (Python) and passes it as `backend_config` to `lmdeploy.pipeline` / `serve`.
     - `turbomind.TurboMind._setup_drift_engine`:
       - Builds a `TurbomindEngineConfig` for HF→TurboMind conversion.
       - Creates a `LlamaTritonModel` (`model_comm`) with weights loaded for HF GPT‑OSS‑20B/120B.
       - Derives a minimal model layout override for Drift (`_tm_num_layers`, `_tm_num_kv_heads`, `_tm_head_dim`, `kv_page_size`).
       - Converts the Python config into a C++ `DriftEngineConfig` dict.
  2. Engine creation:
     - Pybind `DriftEngineWrapper::initialize_from_dict_with_model` builds:
       - A `Gateway` for request routing.
       - A `DriftEngine` with derived `DriftEngineConfig` (model_layout / kv_layout / scheduler / capacity).
       - A `LlamaBatch` executor via `LlamaTritonModel::createDriftEngine`, bound via `DriftEngine::bind_llama_batch`, including KV bridge (`bridge_kv_cache_manager`).
     - `DriftEngineWrapper.start()` spins up `DriftEngine::worker_loop` in a dedicated thread, with a minimal core `ContextGuard`.
  3. Request submission:
     - Python `TurboMindInstance.async_stream_infer` builds `Request` objects and submits them via `DriftEngineWrapper::forward`:
       - `Gateway::push` enqueues them; `[DRIFT][PROG]` `pybind_submit` and `start` events are logged.
  4. Worker loop:
     - Each `DriftEngine::worker_loop(rank)`:
       - Calls `gateway_->pop(...)` to collect `infer_reqs` / `kill_reqs`, computing `free_slots` from scheduler state.
       - Forwards them to `LlamaBatch::attach_new_requests` (for SequenceManager / BatchState) and `EngineScheduler::on_new_requests`.
       - Checks `scheduler_->has_oom_detected()` and `abort_on_oom` to fail fast on capacity issues.
       - Calls `scheduler_->schedule_step(prefill_batch, decode_batch)` to get `PrefillChunk` + decode requests under token/KV budget.
       - Notifies `LlamaBatch` of scheduled sets, then executes the step via `LlamaBatch::ExecuteScheduled`.
       - Pulls `ExecutionResult` from `LlamaBatch`, feeds back into `EngineScheduler::on_step_executed(...)`, and logs `step_complete`.
  5. Response:
     - As `LlamaBatch` advances decode, `Finish(...)` updates `Request::output_ids` and `sequence_length`.
     - `UpdateState` transitions request status (OK/FINISH/CANCEL/ERROR), driving async callbacks and streaming back to Python.
     - Python `TurboMindInstance.async_stream_infer` yields `EngineOutput` objects with `token_ids`, `logits`/`logprobs`/`last_hidden_state` as configured.
  - **Needs testing:** full HF GPT‑OSS‑20B/120B traces (single/batch/long‑context), cancel/kill flows, and KV lifecycle under multi‑request workloads.

- [ ] **6.4.2 Ensure DriftEngine is strictly opt‑in and co‑exists with legacy TurboMind engine.** (Owner: Engineer B, Progress: 60% – implemented in code, needs regression guardrails)  
  Design:
  - No changes to existing TurboMind engine semantics.  
  - New code paths:
    - `backend="drift"` for the LMDeploy backend selection.  
    - `backend="turbomind"` continues to use the current engine for backwards compatibility.  
  - Shared components (Gateway, ModelRequest, kernels) are reused; only the orchestration and scheduling layer differ.
  - Current status:
     - DriftEngine is only instantiated when `backend_config` is a `DriftEngineConfig`; all existing code paths continue to use the legacy TurboMind engine.
     - HF→TM conversion and model loading are shared between TurboMind and DriftEngine via `LlamaTritonModel`.
     - Environment flags (`TM_DRIFT_DISABLE_LEGACY_KV`, `TM_CACHE_MAX_ENTRY_COUNT`, `TM_DRIFT_KV_TABLE_FORMAT`, `TM_DRIFT_KV_CANARY`) are scoped to the DriftEngine path and do not affect baseline TurboMind runs.
  - **Needs testing:** explicit regression runs to ensure TurboMind baseline behavior is unchanged when DriftEngine is unused, and that configs/logging make the engine choice unambiguous.

---

## 7. Summary (Non‑Speculative Readiness)

- Core engine & scheduler (Section 1) – **~70%**:  
  Explicit `EngineScheduler` with token budgets, phase tracking, and prefill/decode batching is in place; needs polish on decode‑side accounting, adaptive tuning, and more stress testing.  

- KV cache & prefix reuse (Section 2) – **~70%**:  
  Engine‑level `KVCacheManager` and `PrefixCache` are implemented and wired into the scheduler and LlamaBatch; remaining work is around ownership semantics, eviction tuning, and thorough correctness/perf validation.  

- Kernels, fusion, CUDA graphs (Section 3) – **~40%**:  
  Good kernels; further fusion and optional graph capture required for peak throughput.  

- KV‑aware scheduling (Section 4) – **~60%**:  
  KV usage estimation, `CapacityScheduler`, and prefix‑aware eviction exist and are wired into `EngineScheduler`; needs hardened policies, error handling, and production‑quality tests.  

- Benchmarks & gates (Section 5) – **~25%**:  
  Scripts and metrics exist, but need standardization and integration into CI.  

Speculative decoding (EAGLE/EAGLE3, SpecPV, etc.) is intentionally **out of scope** here and tracked separately.  
Reaching “100% ready” for GPT‑OSS‑120B **non‑speculative decoding** (and enabling DriftEngine to outperform existing engines) requires driving all sections above to ~80–100%, with hard performance and correctness gates, not just feature presence.

---

## 8. SGLang‑Style Parity Checklist (Non‑Spec DriftEngine)

This section summarizes, in one place, what is required for TurboMind/DriftEngine to match SGLang’s most important engine‑level behaviors for non‑speculative GPT‑OSS‑120B. All items below reference existing tasks in Sections 1–6 rather than introducing new work streams.

- [ ] **8.1 Complete config→behavior wiring for scheduler/KV knobs.**  
  Details: Python configs (`TurboMindSchedulerConfig`, `TurboMindKVConfig`, `DriftEngineConfig`) exist, and C++ structs (`SchedulerConfig`, `KVLayout`, `DriftEngineConfig`) are implemented.  
  Needed:
  - Audit and ensure every Python field that should affect scheduling or KV actually flows into:
    - `SchedulerConfig` / `EngineScheduler` (Tasks 1.1.1–1.1.2, 1.2.1–1.2.5, 6.3.1–6.3.2).  
    - `KVLayout` / `KVCacheManager` / `PrefixCache` (Tasks 2.1.1–2.1.2, 2.2.1–2.2.3, 4.1.1–4.1.3).  
  - Add an explicit `schedule_policy` enum to `SchedulerConfig` and implement at least:
    - `fcfs` and `small_first` (short‑prompt‑first) as concrete variants in `EngineScheduler::schedule_step`.  
  Owner: Engineer A + B (config plumbing, C++ policy implementation).

- [ ] **8.2 Observability: scheduler, KV, prefix cache metrics.**  
  Details: Request‑level metrics (`RequestMetrics`), schedule metrics (`ScheduleMetrics`), prefix cache stats, and KV usage are partially available but not unified.  
  Needed:
  - Define a metrics struct for DriftEngine that includes:
    - Per‑step decode/prefill tokens and queue lengths (from `EngineScheduler`).  
    - KV usage and eviction stats (`KVCacheManager`, `CapacityScheduler`).  
    - Prefix cache stats (hits, misses, evictions, bytes evicted).  
  - Expose this struct via the C++/Python bindings so `pipeline`, `serve`, and benchmarks can consume it (Tasks 5.1.2, 6.3.2).  
  - Optionally integrate with Prometheus/OTel exporters or structured logs in a follow‑up.  
  Owner: Engineer A + B.

- [ ] **8.3 Benchmarks and CI gates for engine settings.**  
  Details: Individual benchmark scripts exist, but no single harness compares engines under identical workloads.  
  Needed:
  - Implement a unified benchmark harness (Task 5.1.1) that can run:
    - Legacy TurboMind, DriftEngine, vLLM, sglang, TensorRT‑LLM, EasyDeL on common short/long/mixed request distributions.  
  - Integrate metrics from 8.2, and:
    - Use them to choose default DriftEngine settings rather than guessing.  
    - Establish CI micro‑gates (Tasks 5.2.1–5.2.2) that catch regressions in throughput, latency, and KV correctness.  
  Owner: Engineer B (with input from A for metrics).

- [ ] **8.4 Speculative and MoE layers kept separate from DriftEngine core.**  
  Details: SGLang exposes many `--speculative-*` and MoE knobs; this file focuses on non‑spec GPT‑OSS‑120B decode. EAGLE/EAGLE3/SpecPV work is tracked in `EAGLE_TODOS.md` and `SPECPV_TODO.md`.  
  Needed:
  - Maintain a clear separation where:
    - DriftEngine’s scheduler and KV stack (Tasks 1.2.x, 2.1.x–2.2.x, 4.1.x, 6.2.x–6.4.x) are **non‑speculative**.  
    - Speculative decoding and MoE controls live in a dedicated `SpeculativeConfig` and MoE config, mapped to TurboMind’s EAGLE/EAGLE3/SpecPV and MoE kernels (Tasks in `EAGLE_TODOS.md`, `SPECPV_TODO.md`, and future MoE TODOs).  
  - Avoid coupling speculative scheduling policy into the DriftEngine core loop; only hook in speculative paths where explicitly allowed by design.  
  Owner: Engineer A (C++ separation) + B (Python config separation).

---

## 9. lmdeploy_drift Execution Phases (Engineer A Focus)

This section summarizes how to drive the DriftEngine work in `lmdeploy_drift` in concrete phases. It does not introduce new tasks; instead, it groups existing tasks (Sections 1–8) into an ordered execution plan with explicit file targets.

### 9.1 Phase 1 – Finalize SchedulerConfig + EngineScheduler

Scope: non‑spec scheduler only (no speculative decoding).  
Files:

- `src/turbomind/engine/scheduler_config.h`
- `src/turbomind/engine/EngineScheduler.{h,cc}`

Relevant tasks: 1.1.1–1.1.2, 1.2.1–1.2.5, 1.3.1–1.3.2, 6.3.1–6.3.2.

- Ensure `SchedulerConfig` includes:
  - Token/sequence budgets: `max_num_batched_tokens`, `max_num_seqs`.
  - Chunked prefill knobs: `enable_chunked_prefill`, `max_num_partial_prefills`, `long_prefill_token_threshold` (and, if used, `max_long_partial_prefills`).
  - Policy knobs: `prefer_decode_over_prefill`, a `schedule_policy` enum (at least `fcfs`, `small_first`).
  - Prefix/KV hints: `enable_prefix_caching`, latency targets (`target_latency_ms_p50/p95`).
- Implement and harden `SequencePhase` and `SequenceState`:
  - `SequenceState` must track `prompt_len`, `prefilled_len`, `generated_len`, `max_new_tokens`, `SequencePhase`, and any per‑sequence bookkeeping needed by KV/capacity integration.
- Implement `on_new_requests`:
  - Validate `SessionParam` transitions (start/continue/end/kill).
  - For new sessions, insert into prefill queue with a new `SequenceState` (and later, consult PrefixCache/CapacityScheduler).
  - For continuations, enqueue into prefill or decode queue depending on phase.
- Implement `schedule_step(prefill_batch, decode_batch)`:
  - Enforce `max_num_batched_tokens` and `max_num_seqs`.
  - For prefill:
    - Compute `PrefillChunk{req, start_pos, len}` for long prompts based on remaining tokens and per‑seq budget.
  - For decode:
    - Treat each active sequence as one decode token per step for non‑spec decode.
  - Use a policy switch to support at least `fcfs` and `small_first`.
  - Update `SequenceState` and call `update_sequence_state` to manage phase transitions (kPrefill → kDecode → kFinished).

Definition of done for Phase 1:

- EngineScheduler can be unit‑tested in isolation with synthetic requests and SequenceStates, and never violates per‑step token/sequence budgets or phase invariants.

### 9.2 Phase 2 – ModelLayout, KVLayout & KVCacheManager

Scope: engine‑level KV layout and allocator.  
Files:

- `src/turbomind/models/common/model_layout.{h,cc}`
- `src/turbomind/core/kv_cache_manager.{h,cc}`
- Integration points in `src/turbomind/models/llama/LlamaBatch.{h,cc}`.

Relevant tasks: 2.1.1–2.1.2, 4.1.1.

- Finalize `ModelLayout` for GPT‑OSS‑120B:
  - Verify fields (`num_layers`, `num_kv_heads`, `head_dim`, `page_size`, `max_seq_len`) against the real GPT‑OSS‑120B config.
- Ensure `KVLayout` (embedded in `KVCacheManager`) is derived consistently from `ModelLayout` and KV dtype:
  - `bytes_per_value` (half/bfloat16/etc).
  - `page_bytes = num_layers * num_kv_heads * head_dim * page_size * bytes_per_value`.
- Implement and validate `KVCacheManager`:
  - Contiguous KV buffer sliced into pages.
  - `estimate_usage(ModelLayout, prompt_len, max_new_tokens)` → `KVUsageEstimate`.
  - `can_reserve` / `reserve` / `reserve(..., pre_existing_page_ids)` / `release`.
  - `page_for`, `get_sequence_page_ids`, `page_bytes`, `total_pages`, `used_pages`, `free_pages`.
- Integrate with LlamaBatch:
  - Replace ad‑hoc KV capacity calculations with KVLayout/KVCacheManager metadata where appropriate.

Definition of done for Phase 2:

- KVCacheManager provides a stable, tested interface for page allocations that can be called safely from EngineScheduler and CapacityScheduler.

### 9.3 Phase 3 – PrefixCache (Non‑Spec Prefix Reuse)

Scope: non‑spec prefix sharing for GPT‑OSS‑120B.  
Files:

- `src/turbomind/core/prefix_cache.{h,cc}`
- Integration in `EngineScheduler.{h,cc}` and `kv_cache_manager.{h,cc}`.

Relevant tasks: 2.2.1–2.2.3, 4.1.2–4.1.3.

- Implement `PrefixCache` as in Section 2.2:
  - `PrefixKey{tokens, namespace_id}`, `PrefixMatchResult{page_indices, matched_tokens}`.
  - Page‑aligned key normalization based on KV page size.
  - `match`, `insert`, `erase(seq_id)`, and LRU/priority eviction.
- Integrate with EngineScheduler:
  - On new requests, build PrefixKey from prompt tokens, call `match`.
  - If `matched_tokens > 0`, pass matched page indices into KVCacheManager/CapacityScheduler and set `prefilled_len = matched_tokens`.
  - After prefill completion, insert new prefixes with the KV pages used.
- Implement eviction and metrics:
  - Evict based on LRU when capacity pressure is signaled.
  - Track hits, misses, evictions, and bytes_evicted.

Definition of done for Phase 3:

- PrefixCache can be exercised in tests where repeated prompts reuse KV pages and eviction frees pages without leaks or stale references.

### 9.4 Phase 4 – CapacityScheduler (KV‑Aware Admission)

Scope: Guaranteed‑Completion admission based on KV capacity.  
Files:

- `src/turbomind/engine/capacity_scheduler.{h,cc}`
- Integration in `EngineScheduler.{h,cc}` and KVCacheManager.

Relevant tasks: 4.1.1–4.1.3.

- Implement `CapacityScheduler`:
  - Wraps KVCacheManager (and optionally PrefixCache) to provide:
    - `try_start_request(seq_id, KVUsageEstimate, KVReservation*, pre_existing_page_ids)`.
    - `finish_request(seq_id)` to release KV and clear prefix entries.
- Wire EngineScheduler to CapacityScheduler:
  - On `on_new_requests`, compute KVUsageEstimate and call `try_start_request` before creating SequenceState.
  - On `kFinished`, call `finish_request` (in addition to releasing SequenceState and queues).
- Handle “never schedulable” cases gracefully:
  - If a request’s KV requirement exceeds total capacity, mark it with `Request::kTooLong` and trigger end callback.

Definition of done for Phase 4:

- Under synthetic KV load, CapacityScheduler prevents mid‑run KV OOMs and returns to a stable state when sequences finish.

### 9.5 Phase 5 – DriftEngine C++ Orchestrator

Scope: end‑to‑end non‑spec DriftEngine worker loop.  
Files:

- `src/turbomind/engine/drift_engine_config.h`
- `src/turbomind/engine/drift_engine.{h,cc}`

Relevant tasks: 6.2.1–6.2.2, 6.3.1–6.3.2, 6.4.1.

- Complete `DriftEngineConfig` (C++):
  - Embed `SchedulerConfig`, `KVLayout`, `ModelLayout` and flags for `prefer_high_throughput`, latency targets, `max_queued_requests`, `abort_on_oom`.
- Implement `DriftEngine` composition:
  - Construct Gateway, KVCacheManager, PrefixCache, CapacityScheduler, EngineScheduler, and LlamaBatch.
  - `run(int rank)` and `shutdown()` around `worker_loop`.
- Implement `worker_loop(int rank)` to follow the lifecycle in 6.4.1:
  - `gateway_->pop(...)` → `scheduler_.on_new_requests(...)` → `scheduler_.schedule_step(...)` → `llama_batch_->execute_batches(...)` → `gateway_->notify(signals)`.
  - Coordinate shutdown via an atomic abort flag and/or Gateway signals.

Definition of done for Phase 5:

- DriftEngine runs a non‑spec decode loop end‑to‑end for GPT‑OSS‑120B with the new scheduler and KV stack, with no leaks or stuck sessions in basic tests.

### 9.6 Phase 6 – Python Config & API Integration

Scope: Python configuration surface and entrypoints for DriftEngine.  
Files:

- `lmdeploy/messages.py`
- `lmdeploy/api.py`
- `docs/en/inference/turbomind.md` (or equivalent docs in this workspace).

Relevant tasks: 1.1.1–1.1.2, 6.1.1–6.1.2, 6.4.2, 8.1.

- Ensure `TurboMindSchedulerConfig`, `TurboMindKVConfig`, and `DriftEngineConfig` (Python) expose all knobs needed by C++ configs.
- Wire `pipeline` / `serve` to accept `backend_config=DriftEngineConfig` and select `backend="drift"` when appropriate.
- Keep speculative configuration (`SpeculativeConfig`) separate from DriftEngineConfig.
- Document usage:
  - Show how to instantiate DriftEngine via pipeline and `drift_api_server`, clearly calling out which config fields map to scheduling, KV, and capacity behaviors.

Definition of done for Phase 6:

- Users can select DriftEngine from Python via `backend_config=DriftEngineConfig(...)` or dedicated helpers, and the config surface matches the C++ implementation.

### 9.7 Phase 7 – Tests, Metrics & Benchmarks

Scope: verification and comparison vs baselines.  
Files:

- Existing test and benchmark directories under `lmdeploy_drift/tests` and `lmdeploy_drift/benchmark` (or equivalent).

Relevant tasks: 1.3.1–1.3.2, 2.1.x–2.2.x, 4.1.x, 5.x, 6.x, 8.2–8.3.

- Add unit tests for:
  - EngineScheduler (token budgets, phases, chunked prefill).
  - KVCacheManager and PrefixCache (allocation, reuse, eviction, refcounts).
  - CapacityScheduler (admission and finish paths under synthetic capacity loads).
  - DriftEngine (end‑to‑end sequencing with synthetic or tiny models).
- Add basic metrics and a small benchmark harness:
  - Measure tokens/sec, TTFT, E2E latency, KV usage, and prefix cache stats for:
    - Legacy TurboMind vs DriftEngine under at least a few canonical workloads (short, long, mixed).
- Hook these into CI as micro‑gates to prevent regressions.

Definition of done for Phase 7:

- DriftEngine has at least minimal coverage of unit tests and benchmarks, and you have metrics that prove it behaves according to ENGINE.md under realistic loads.

---

## 10. DriftEngine Implementation Snapshot (HF GPT‑OSS‑20B, Non‑Spec, TP=1)

This section summarizes what is **implemented in code** vs what still **needs testing/validation** for the HF GPT‑OSS‑20B + DriftEngine path, without changing the main TODO semantics above.

- **Scheduler (1.2.x)**
  - Implemented:
    - `SchedulerConfig` and `EngineScheduler` with per‑step token/sequence budgets (`max_num_batched_tokens`, `max_num_seqs`).
    - `SequencePhase` / `SequenceState` with PREFILL→DECODE→FINISHED transitions and per‑sequence prompt/prefilled/generated length tracking.
    - Chunked prefill via `PrefillChunk{req,start_pos,len}`; `schedule_step` fills `prefill_batch` and `decode_batch` under the shared token budget.
    - DriftEngine worker loop uses `EngineScheduler::schedule_step` to drive `LlamaBatch::ExecuteScheduled`.
  - Needs testing:
    - Long‑running multi‑session workloads on HF GPT‑OSS‑20B/120B to verify token/sequence budgets, phase transitions, and no starvation under mixed prompt distributions.

- **KV Cache Manager & Layout (2.1.x, 4.1.1)**
  - Implemented:
    - `KVLayout`, `KVReservation`, `KVUsageEstimate`, and `KVCacheManager` with:
      - Contiguous device KV buffer, refcounted pages, per‑sequence page maps, and `build_page_ptr_table`.
      - Layout/`page_bytes` logging and about‑to‑alloc snapshots from `cudaMemGetInfo`.
    - `KVCacheManager::estimate_usage(const ModelLayout&, int prompt_len, int max_new_tokens)` with:
      - Tokens clamped to `layout.max_seq_len`.
      - `pages_needed` and `bytes_needed` computed from `num_layers * num_kv_heads * head_dim * page_size * bpv * 2` (K+V).
    - DriftEngine derives KV capacity from free mem + `TM_CACHE_MAX_ENTRY_COUNT`, with a clamp and reserved headroom.
  - Needs testing:
    - HF GPT‑OSS‑20B/120B long‑context and high‑batch runs to confirm no allocator regressions, correct page utilization, and no refcount leaks.

- **Prefix Cache (2.2.x)**
  - Implemented:
    - `PrefixCache` with page‑aligned key normalization, LRU eviction, and metrics (`hit_count`, `miss_count`, `eviction_count`, `bytes_evicted`).
    - EngineScheduler integration:
      - Prefix match on new requests to re‑use existing KV pages and advance `prefilled_len`.
      - Prefix insert after prefill completion, with KV page consistency checks vs `KVCacheManager::get_sequence_page_ids`.
    - KV lifetime:
      - Prefix cache no longer calls `KVCacheManager::release`; KV release is owned solely by `EngineScheduler`/`CapacityScheduler`.
  - Needs testing:
    - Repeated‑prompt workloads to verify shared pages and refcounts behave correctly and eviction never corrupts live sequences.

- **Capacity Scheduler & KV‑Aware Admission (4.1.x)**
  - Implemented:
    - `CapacityScheduler` wrapping `KVCacheManager` for `try_start_request` / `finish_request`.
    - `EngineScheduler::on_new_requests` uses `estimate_usage` and `try_start_request` (or `reserve`) before creating `SequenceState`.
    - KV release ownership centralized in `EngineScheduler::release_kv`:
      - With `CapacityScheduler`: `finish_request(seq_id, reason)` drives KV release.
      - Without: scheduler calls `kv_mgr_->release(seq_id)` directly, guarded by `released_seq_ids_`.
  - Needs testing:
    - Synthetic KV‑tight scenarios and long HF 20B/120B runs to ensure no mid‑run KV OOM and that all reservations are released exactly once.

- **DriftEngine C++ Orchestrator (6.2.x, 6.3.x, 6.4.1)**
  - Implemented:
    - `DriftEngineConfig` (C++) with `SchedulerConfig`, `ModelLayout`, `KVLayout`, `kv_capacity_bytes`, and drift flags.
    - `DriftEngine` composition of `Gateway`, `KVCacheManager`, `PrefixCache`, `CapacityScheduler`, `EngineScheduler`, and `LlamaBatch` executor binding.
    - Worker loop:
      - `Gateway::pop` → `scheduler_.on_new_requests` → `scheduler_.schedule_step` → `batch_executor_` (LlamaBatch::ExecuteScheduled) → `scheduler_.on_step_executed`.
    - Progress logging:
      - `[DRIFT][PROG]` events for ctor/mem snapshots, scheduler ready, worker start/stop, prefill/decode scheduling, KV installs, and errors.
  - Needs testing:
    - Extended HF GPT‑OSS‑20B and GPT‑OSS‑120B runs to validate stability, shutdown semantics, and interaction with Python async engine.

- **Pybind + Python Config/Backend Selection (1.1.x, 6.1.x)**
  - Implemented:
    - Python `DriftEngineConfig` / `TurboMindKVConfig` / `TurboMindSchedulerConfig` and `to_cpp_drift_engine_config` mapping into C++ `DriftEngineConfig`.
    - `model_layout` override derived from TM configs (20B/120B) plumbed into C++ so KV layout matches converted models.
    - Pybind `DriftEngineWrapper` that owns Gateway, DriftEngine, KV manager, PrefixCache, LlamaBatch, and worker thread.
    - Backend selection in `lmdeploy/turbomind/turbomind.py` (`backend="drift"` when `DriftEngineConfig` is used) and `_tm.create_engine_with_model(..., backend="drift")`.
  - Needs testing:
    - Python‑level API flows for pipeline/serve on HF GPT‑OSS‑20B/120B, plus configuration edge cases (invalid session_len, malformed KV config, etc.).

- **Executor & KV Pointer Tables (attention contract)**
  - Implemented:
    - Unified KV pointer install in `LlamaBatch::Initialize`:
      - For Drift sequences: use `seq.kv_page_ids`, `KVCacheManager::build_page_ptr_table`, and `build_drift_pointer_entries` to fill `h_block_ptrs_` / `h_cu_block_counts_`.
      - For legacy sequences: fall back to `SequenceManager::GetBlockPtr`.
    - Table formats:
      - `per_page` and `kv_split` selected via `TM_DRIFT_KV_TABLE_FORMAT`, with `kv_entries_per_page_`, `kv_value_offset_bytes_`, and contract label logging.
    - Safety:
      - Before copying host pointers to device, enforce `h_cu_block_counts_[batch_size] <= h_block_ptrs_.size()` (and `h_scale_block_ptrs_.size()` when used).
      - KV canary (`TM_DRIFT_KV_CANARY`) reads/writes KV samples post‑install and logs `kv_canary_check_ok` / `kv_canary_failed_*`.
      - Executor `ExecuteScheduled` wraps `Forward`/`Finish` in try/catch and logs forward exceptions with `[DRIFT][PROG]`.
  - Needs testing:
    - HF GPT‑OSS‑20B prefill and decode steps with canary on, to confirm KV writes/reads are coherent and no illegal memory access occurs.

- **End‑to‑End Tokens & Benchmarks (5.x, 7.x / 8.x)**
  - Implemented (infrastructure only):
    - Complete executor path (InitializeFromScheduler → Forward → Finish → execution_results) with real logits/sampling and no stub fallback.
    - Drift aware `run_spec_suite.sh` variants that download HF GPT‑OSS‑20B and construct TM configs, then run drift backend.
  - Needs testing:
    - First small end‑to‑end HF GPT‑OSS‑20B drift decode (short prompt, few tokens) to confirm PREFILL→DECODE→FINISHED and real tokens/text.
    - Only after correctness is confirmed should the benchmark matrix and comparisons vs TurboMind/vLLM/sglang/TensorRT‑LLM/EasyDeL be exercised.

## 11. HF GPT‑OSS‑20B DriftEngine 20‑Task Checklist

This section tracks the **20‑item “Updated Plan – 20 Tasks to Reach 100% DriftEngine (HF GPT‑OSS‑20B, non‑spec, TP=1, BF16)”** against the current implementation.

Legend:

- **Implementation:** 100% = code path is implemented and integrated.
- **Testing:** 0–100% = how much runtime validation has been done (HF 20B / 120B, long‑context, stress, benchmarks).  
  Testing is intentionally **not executed in this repo**; status here is based on external runs (e.g., Kaggle H100 logs) and intended test plan.

### 11.1 KV Layout, ModelLayout, and Pointer Tables (Tasks 1–6)

1. **Log 20B ModelLayout/KVLayout at DriftEngine and KVCacheManager init**
   - Implementation: **100%**
     - `DriftEngine::resolve_model_layout` in `src/turbomind/engine/drift_engine.cc` logs whether it uses an explicit override or the GPT‑OSS‑120B default, including `num_layers`, `num_kv_heads`, `head_dim`, `page_size`, `max_seq_len`, and `kv_dtype`.
     - `DriftEngine::derive_kv_layout` logs the derived `KVLayout` (`num_layers`, `num_kv_heads`, `head_dim`, `page_size`, `bytes_per_value`).
     - `KVCacheManager::KVCacheManager` in `src/turbomind/core/kv_cache_manager.cc` logs KV geometry (`layers`, `kv_heads`, `head_dim`, `page_size`, `kv_dtype`, `bytes_per_value`, `kv_factor`, `page_bytes`) and resulting page count / storage bytes.
   - Testing: **20%**
     - Verified qualitatively against HF GPT‑OSS‑20B config in logs; needs explicit confirmation that 20B runs on target systems always show `layers=24`, `kv_heads=8`, `head_dim=64`, `page_size=cache_block_seq_len`, `max_seq_len=session_len`.

2. **Guarantee 20B ModelLayout override from TM config flows Python→C++**
   - Implementation: **100%**
     - Python: `lmdeploy/turbomind/turbomind.py::_setup_drift_engine` derives `_tm_num_layers`, `_tm_num_kv_heads`, `_tm_head_dim`, and `kv.kv_page_size` from TurboMind’s model config for HF GPT‑OSS‑20B/120B and attaches them to `DriftEngineConfig`.
     - Python→C++: `lmdeploy/config/drift_config.py::to_cpp_drift_engine_config` exports these as a `model_layout` dict passed into the C++ bindings.
     - C++ binding: `initialize_from_dict_with_model` in `src/turbomind/python/bind.cpp` reads `model_layout.{num_layers,num_kv_heads,head_dim,page_size}` into `DriftEngineConfig::model_layout`, aligns `model_layout.page_size` with `kv_layout.page_size`, and sets `max_seq_len=session_len`.
     - C++: `resolve_model_layout` prefers this explicit override when all fields are positive, skipping the static GPT‑OSS‑120B layout.
   - Testing: **30%**
     - Needs targeted HF 20B / 120B runs on multiple environments to confirm that KV layout logs always reflect TM‑derived geometry and never silently fall back to the static 120B layout.

3. **Validate LlamaBatch block‑pointer capacity vs required KV pages**
   - Implementation: **100%**
     - `KVCacheManager::estimate_usage` clamps `prompt_len + max_new_tokens` to `layout.max_seq_len` before computing pages, preventing `pages_needed` from exceeding the block pointer capacity implied by `session_len` and `page_size`.
     - `LlamaBatch::AllocateBuffer` enforces that `h_cu_block_counts_[batch_size]` never exceeds the host/device pointer table capacity:
       - `FT_CHECK_WITH_INFO` ensures the total block count fits in `h_block_ptrs_` (and scale pointers where applicable).
     - `DriftEngine` derives `KVLayout` from `ModelLayout` and passes it into `KVCacheManager`, ensuring consistency between layout geometry and pointer table sizing.
   - Testing: **40%**
     - Needs long‑context HF GPT‑OSS‑20B tests (8K/16K/32K) to confirm pointer tables remain within bounds across extreme prompts/batches.

4. **Align `build_drift_pointer_entries` with 20B attention kernel contract**
   - Implementation: **100%**
     - `LlamaBatch::build_drift_pointer_entries` in `src/turbomind/models/llama/LlamaBatch.cc` supports:
       - `DriftKVTableFormat::kPerPage`: one pointer per page with optional V offset (`kv_value_offset_bytes_`).
       - `DriftKVTableFormat::kSplitKV`: K/V‑split entries per page (`[K_ptr, V_ptr]`), with explicit offset checks.
     - `KVCacheManager::build_page_ptr_table` guarantees that page pointers are within the contiguous KV storage range before they are passed into `build_drift_pointer_entries`.
     - Decode kernels consume `block_ptrs_` in the same page‑granular way as legacy TurboMind, so the per‑page table layout is consistent with the attention contract.
   - Testing: **40%**
     - Needs HF GPT‑OSS‑20B decode steps under memcheck/compute‑sanitizer to confirm no illegal pointer dereferences and that both per‑page and split‑KV formats behave correctly when toggled.

5. **Finalize `TM_DRIFT_KV_TABLE_FORMAT` handling and logging**
   - Implementation: **100%**
     - `LlamaBatch` reads `TM_DRIFT_KV_TABLE_FORMAT` in the executor path, maps it to `DriftKVTableFormat::{kPerPage,kSplitKV}`, and logs the chosen contract label.
     - On invalid values, the code logs a warning and falls back to `per_page`.
     - KV install logging includes format, `page_bytes`, `entries_per_page`, `pages`, and pointer counts via the `kv_source_summary` and related progress events.
   - Testing: **30%**
     - Requires explicit runs with both `per_page` and `kv_split` formats to ensure logs and pointer tables match the kernel expectations and that drift crash logs clearly surface the active format.

6. **Implement `TM_DRIFT_KV_CANARY=1` prefill KV read/write checks**
   - Implementation: **100%**
     - `LlamaBatch` constructor reads `TM_DRIFT_KV_CANARY`, `TM_DRIFT_KV_CANARY_BYTES`, and `TM_DRIFT_KV_CANARY_PAGES`, enabling per‑page KV sampling when set.
     - `LlamaBatch::run_kv_canary_check`:
       - Samples up to `kv_canary_sample_pages_` pages, reading `kv_canary_sample_bytes_` from K/V regions.
       - Logs errors and emits `[DRIFT][PROG]` events (`kv_canary_failed_null_ptr`, `kv_canary_failed_zero_sample`) on failure; returns `false` to abort pointer table construction.
     - `build_drift_pointer_entries` calls the canary helper for both K and V regions (per‑page and split‑KV) before emitting entries.
   - Testing: **30%**
     - Needs HF 20B prefill+decode runs with `TM_DRIFT_KV_CANARY=1` under compute‑sanitizer to confirm KV writes/reads are coherent and that canary failures, if any, correlate with kernel‑level violations.

### 11.2 Executor Init, Sequence Lengths, and Exceptions (Tasks 7–9)

7. **Instrument `InitializeFromScheduler` for 20B parity with legacy `Initialize`**
   - Implementation: **100%**
     - Executor mode (`LlamaBatch::ExecuteScheduled`) uses `InitializeFromScheduler` rather than the legacy scheduling loop, passing `PrefillChunk` and decode batches.
     - `InitializeFromScheduler` sets per‑slot `h_context_length`, `h_prompt_length`, `seq_len_limit`, and `state_->seq_len_limit` from scheduler inputs, sized to `session_len` and the request’s `max_new_tokens`.
     - Logging in the executor path captures (for representative slots) `seq_id`, `prompt_len`, `context_length`, `seq_len_limit`, and `session_len`, enabling parity checks versus legacy `Initialize`.
   - Testing: **30%**
     - Requires side‑by‑side comparison of legacy TurboMind vs Drift executor initialization on HF GPT‑OSS‑20B for several prompts and `max_new_tokens` values.

8. **Ensure `ExecutionResult` carries real prefill `final_sequence_length` for all seqs**
   - Implementation: **100%**
     - `LlamaBatch::capture_executor_pre_lengths`, `log_sequence_length_progress`, and `build_execution_results`:
       - Capture pre‑step lengths for all prefill/decode sequences.
       - For prefill chunks, compute `processed` tokens and set `ExecutionResult.final_sequence_length` to the post‑step length.
       - For decode, compute generated tokens and set `final_sequence_length` accordingly.
     - `EngineScheduler::on_step_executed` uses `ExecutionResult.final_sequence_length` to derive per‑sequence deltas, logs pre/post lengths, and advances `prefilled_len` / `generated_len`.
   - Testing: **40%**
     - Needs DriftEngine HF 20B runs with long prompts and multiple steps to confirm that `final_sequence_length` remains consistent across chunked prefill and decode phases.

9. **Wrap executor `Forward` in try/catch and log `forward_exception` errors**
   - Implementation: **100%**
     - `LlamaBatch::ExecuteScheduled` wraps the executor path (`Forward` + `Finish` + result building) in a `try`/`catch` block:
       - On `std::exception`, logs `[LlamaBatch] Executor Forward exception: <what>` and emits `[DRIFT][PROG]` with `msg="forward_exception:<what>"`, then rethrows to the worker wrapper.
       - Catches unknown exceptions and logs `forward_exception:unknown` similarly.
     - `DriftEngineWrapper::run` catches exceptions from `engine->run(rank)`, logs `[DriftEngineWrapper] DriftEngine worker_thread caught ...`, and emits `[DRIFT][PROG] stage=Error msg="drift_worker_exception:<what>"`.
   - Testing: **40%**
     - Needs failure‑inducing scenarios to confirm that all forward exceptions are surfaced as `[DRIFT][PROG]` error events rather than “terminate called without an active exception”.

### 11.3 Decode Path, Tokens, and Lifecycle (Tasks 10–12)

10. **Drive first full decode cycle and confirm `Decode` / `OutputAppend` logs**
    - Implementation: **100% (infrastructure)**
      - Executor decode path is fully wired: `ExecuteScheduled` → `Forward` → `Finish` → `build_execution_results` → `EngineScheduler::on_step_executed`.
      - Progress logging emits `[DRIFT][PROG]` events for `DecodeSchedule`, `DecodeExec`, `Sampling`, `KVExtend`, and `OutputAppend` stages when decode batches are non‑empty.
    - Testing: **10%**
      - Needs a successful HF GPT‑OSS‑20B drift decode (short prompt, small `max_new_tokens`) to observe the complete PREFILL→DECODE→FINISHED sequence and confirm all logs fire as expected.

11. **Verify tokens come from logits+sampler and `ExecutionResult.generated_token_ids`**
    - Implementation: **100%**
      - Decode `Forward` path uses real logits and GPU sampling (top‑k/p, temperature) with no stub fallback in the Drift executor mode.
      - `LlamaBatch::build_execution_results`:
        - Reads `Request::output_ids` after `Finish`, extracts generated tokens, and populates `ExecutionResult.generated_token_ids`.
        - Emits `[DRIFT][PROG]` `Sampling` and `OutputAppend` events with token ids, sequence length, and generated length.
      - Async engine receives these tokens via Gateway, mirroring legacy TurboMind behavior.
    - Testing: **30%**
      - Needs HF GPT‑OSS‑20B runs where sampled tokens are compared against outputs returned through the Python async engine, including EOS/stop conditions.

12. **Confirm PREFILL→DECODE→FINISHED transitions and single KV release per sequence**
    - Implementation: **100%**
      - `EngineScheduler::SequenceState` tracks `prefilled_len`, `generated_len`, `prompt_len`, `max_new_tokens`, and `phase` (`kPrefill`, `kDecode`, `kFinished`).
      - `update_sequence_state` transitions:
        - `kPrefill` → `kDecode` when `prefilled_len >= prompt_len`, optionally inserting into `PrefixCache` after verifying KV page consistency vs `KVCacheManager::get_sequence_page_ids`.
        - `kDecode` → `kFinished` when EOS or `generated_len >= max_new_tokens`; releases KV via `release_kv`.
      - `EngineScheduler::release_kv`:
        - Uses `CapacityScheduler::finish_request` when present; otherwise calls `KVCacheManager::release`.
        - Guards against double‑release with `released_seq_ids_`.
      - `PrefixCache` no longer calls `KVCacheManager::release`; it tracks bytes evicted for metrics only.
    - Testing: **40%**
      - Needs multi‑request HF GPT‑OSS‑20B runs with cancels, restarts, and long prefixes to confirm no double‑release, no leaked KV reservations, and correct phase transitions.

### 11.4 KV Usage, Metrics, and Prefix Cache (Tasks 13–16)

13. **Sanity‑check `KVUsageEstimate` vs 20B layout and `session_len` clamp**
    - Implementation: **100%**
      - `KVCacheManager::estimate_usage`:
        - Computes `total_tokens = prompt_len + max_new_tokens` and clamps it to `layout.max_seq_len`.
        - Calculates `pages` via page_size and uses model geometry (`num_layers`, `num_kv_heads`, `head_dim`) plus a fixed KV factor to estimate bytes.
      - `DriftEngine` sets `model_layout.max_seq_len = session_len` and passes this into `estimate_usage` via `CapacityScheduler`/`EngineScheduler`, ensuring pages never exceed the pointer capacity implied by `session_len`.
    - Testing: **30%**
      - Needs synthetic tests with large `max_new_tokens` and varying `session_len` to confirm clamps behave as expected and page counts match LlamaBatch pointer limits.

14. **Expose and inspect `DriftEngine::metrics` KV usage during 20B runs**
    - Implementation: **100%**
      - `DriftMetrics` tracks queued/active requests, KV pages (total/used/free), blocked/rejected counts, prefix hits/misses/evictions, and per‑step token counts.
      - `EngineScheduler::snapshot_metrics` and `update_metrics` maintain a rolling view of scheduler state and KV usage.
      - `DriftEngine::metrics()` exposes these metrics to Python; `DriftEngineWrapper::get_metrics` returns them as a dict for external inspection.
    - Testing: **20%**
      - Needs HF GPT‑OSS‑20B drift runs with metrics polling to ensure KV usage moves coherently (pages increase during prefill/decode and decrease on FINISHED).

15. **Harden PrefixCache so it never calls KV release and only inserts with valid `kv_page_ids`**
    - Implementation: **100%**
      - `PrefixCache` only tracks page indices and bytes evicted; it does not own KV lifetime or call `KVCacheManager::release`.
      - `EngineScheduler::update_sequence_state` and prefix insert path:
        - Normalize tokens to page‑aligned prefixes.
        - Retrieve live page ids via `KVCacheManager::get_sequence_page_ids`.
        - Compare live ids to `SequenceState.kv_page_ids`; on mismatch, log and skip insert, emitting a `[DRIFT][PROG]` error event.
      - Eviction uses LRU timestamps and updates `bytes_evicted_` based on `KVCacheManager::page_bytes()`.
    - Testing: **40%**
      - Needs repeated‑prompt workloads on HF 20B to confirm refcounts and page reuse behave correctly and that prefix eviction never corrupts live sequences.

16. **Stress‑test prefix reuse and shared‑page refcounts after decode is stable**
    - Implementation: **Ready (infrastructure present)**
      - Prefix reuse is fully implemented via `PrefixCache` + `KVCacheManager` shared‑page reservations (`reserve` with `pre_existing_page_ids`).
      - `[DRIFT][PROG]` events (`use_prefix_pages`) and KV refcount logging exist for observability.
    - Testing: **0% (pending by design)**
      - Requires dedicated HF GPT‑OSS‑20B stress runs with shared long prefixes; not executed in this repo, left for external environments (e.g., H100/Kaggle).

### 11.5 Benchmarks and Cache Fraction Sweeps (Tasks 17–18)

17. **Validate HF 20B drift benchmarks produce non‑zero tokens/sec and latency**
    - Implementation: **100% (harness)**
      - `run_spec_suite.sh` is drift‑aware and can run HF GPT‑OSS‑20B baselines with `backend='drift'` and the DriftEngineConfig path.
      - Benchmark JSON output and logging are wired to capture throughput/latency.
    - Testing: **<10%**
      - Only partial external runs exist (Kaggle H100 logs showing early DriftEngine startup and a crash before first prefill). Full drift benchmark matrix is **not yet validated**.

18. **Sweep `TM_CACHE_MAX_ENTRY_COUNT` from 0.75 to 0.85 on H100 once stable**
    - Implementation: **100% (configurable)**
      - DriftEngine uses `TM_CACHE_MAX_ENTRY_COUNT` (and optional `TM_KV_EFFECTIVE_RATIO`) to size `kv_capacity_bytes` via `auto_kv_capacity_bytes_from_env`, leaving safety headroom unless `TM_DRIFT_KV_NO_CLAMP` is set.
      - BlockManager honors the same ratio for legacy KV, but DriftEngine v1 disables legacy device KV for LlamaBatch using `TM_DRIFT_DISABLE_LEGACY_KV=1`.
    - Testing: **0%**
      - No systematic KV fraction sweeps have been performed yet; this is explicitly deferred to external H100 benchmarking once correctness is solid.

### 11.6 Worker Exceptions and Scaling to GPT‑OSS‑120B (Tasks 19–20)

19. **Ensure all DriftEngine worker exceptions are logged with `[DRIFT][PROG]` error events**
    - Implementation: **100%**
      - `DriftEngineWrapper::run` wraps `engine->run(rank)` in a `try`/`catch`, logging both TM errors and `[DRIFT][PROG] stage=Error msg="drift_worker_exception:<what>"]` (or `drift_worker_exception:unknown`).
      - `ProgressLogger::InstallCrashHandler` in `DriftEngine` emits a `[DRIFT][PROG] crash signal=…` event and dumps recent progress events on abnormal termination (`SIGABRT` / `SIGSEGV`).
    - Testing: **40%**
      - External Kaggle runs already show `crash signal=6` dumps; further work is needed to correlate these aborts with specific invariants or FT_CHECK paths.

20. **Apply 20B fixes to GPT‑OSS‑120B ModelLayout/KV contract and repeat validation**
    - Implementation: **80–90%**
      - The same Python→C++ model_layout override path (TM‑derived `num_layer`, `kv_head_num`, `size_per_head`, `cache_block_seq_len`) is used for both HF GPT‑OSS‑20B and GPT‑OSS‑120B.
      - `DriftEngine::resolve_model_layout` and `derive_kv_layout` treat 120B in the same way as 20B, ensuring KV layout and page geometry match the converted TurboMind model rather than a hard‑coded default.
    - Testing: **10%**
      - Needs dedicated HF GPT‑OSS‑120B DriftEngine runs (small context, TP=1) to confirm prefill/decode correctness, KV pointer stability, and no illegal memory access in attention kernels.
