# LMDeploy / TurboMind Engine TODOs (GPT‑OSS‑120B, Non‑Speculative)

Progress values are approximate and based on current code + docs:

- `0–30%` – mostly design / scaffolding only.  
- `30–70%` – partially implemented, needs correctness/perf work.  
- `70–100%` – functionally present, needs polishing/validation at the upper end.  

Scope of this file:

- Focuses on **pure GPT‑OSS‑120B decoding** (no speculative decoding / no EAGLE3).  
- EAGLE/EAGLE3/SpecPV‑specific work is tracked in `EAGLE_TODOS.md` and `SPECPV_TODO.md`.  

Every task below is a **numbered checkbox** with a progress estimate.

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

- [ ] **1.2.1 Introduce an explicit per‑step token budget.** (Owner: Engineer A, Progress: 80%)  
  Details: Gateway/RequestQueue implement basic continuous batching, but there is no global token budget.  
  Needed:
  - A scheduler component that, per step, selects a batch of requests such that:  
    - `sum(tokens_being_processed)` ≤ `max_num_batched_tokens`.  
    - Respects maximum sequences per step.  
  Design (C++):
  - Add an engine‑level scheduler config struct (in `src/turbomind/engine/scheduler_config.h`):
    ```cpp
    struct SchedulerConfig {
        int  max_num_batched_tokens{2048};
        int  max_num_seqs{128};
        bool enable_chunked_prefill{true};
        int  max_num_partial_prefills{1};
        int  long_prefill_token_threshold{0};
        bool prefer_decode_over_prefill{true};
    };
    ```
  - Introduce an `EngineScheduler` that uses this:
    ```cpp
    class EngineScheduler {
    public:
        explicit EngineScheduler(const SchedulerConfig& cfg,
                                 KVCacheManager* kv_mgr);
        // Decide which sequences to run this step given the token budget.
        void schedule_step(std::vector<std::shared_ptr<Request>>& prefill_batch,
                           std::vector<std::shared_ptr<Request>>& decode_batch);
    private:
        SchedulerConfig cfg_;
        KVCacheManager* kv_mgr_;
        // internal per-sequence state map (see 1.2.2).
    };
    ```
  - The engine main loop calls `schedule_step` instead of directly consuming Gateway/RequestQueue results, and then forms GPU batches from `prefill_batch` and `decode_batch`.

- [ ] **1.2.2 Track prefill vs decode phases per request.** (Owner: Engineer A, Progress: 80%)  
  Details: DynamicDecodeLayer covers decode; prefill is implicit in initial forward.  
  Needed:
  - Per‑request phase state (`PREFILL`, `DECODE`, `FINISHED`).  
  - Ability to schedule long prompts in chunks (Section 1.2.3) distinct from 1‑token decode.  
  Design (C++):
  - Add a small per‑sequence state record managed by `EngineScheduler`:
    ```cpp
    enum class SequencePhase : uint8_t { kPrefill, kDecode, kFinished };

    struct SequenceState {
        uint64_t     seq_id;
        int          prompt_len;      // total prompt tokens
        int          prefilled_len;   // how many prompt tokens already processed
        int          generated_len;   // how many new tokens generated
        int          max_new_tokens;
        SequencePhase phase;
    };
    ```
  - Maintain `std::unordered_map<uint64_t, SequenceState> seq_states_` inside `EngineScheduler`:
    - On new `start` request: create `SequenceState` with `phase = kPrefill`.
    - After first full prompt processed: flip to `kDecode`.
    - After EOS or reaching `max_new_tokens`: mark `kFinished` and let scheduler free KV via `KVCacheManager`.

- [ ] **1.2.3 Implement chunked prefill with partial concurrency.** (Owner: Engineer A, Progress: 70%)  
  Details: No chunked prefill today; long prompts must be processed in one go.  
  Needed:
  - API + scheduler logic to split long prompts into chunks that fit the token budget.  
  - Controls similar to vLLM: `max_num_partial_prefills`, thresholds for “long” prompts.  
  Design (C++):
  - Extend `SequenceState` with `prefill_chunk_size` derived from `max_num_batched_tokens` and current load.
  - The scheduler computes per‑sequence remaining prefill length:
    ```cpp
    int remaining = state.prompt_len - state.prefilled_len;
    int chunk = std::min(remaining, cfg_.max_num_batched_tokens_per_seq());
    ```
  - Expose a lightweight descriptor for a prefill chunk:
    ```cpp
    struct PrefillChunk {
        std::shared_ptr<Request> req;
        int start_pos;  // inclusive, in prompt tokens
        int len;        // number of tokens in this chunk
    };
    ```
  - `schedule_step` fills `prefill_batch` with `PrefillChunk`s respecting:
    - Global token budget.
    - `max_num_partial_prefills` and `long_prefill_token_threshold` (limit concurrent long prompts).
  - The model execution path must accept `(start_pos, len)` and only process that portion of the prompt (input IDs slice + appropriate position IDs).

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

- [ ] **1.3.1 Finalize and document session lifecycle invariants.** (Owner: Engineer A, Progress: 90%)  
  Details: `SessionParam` + `SeqId2Rank` implement start/continue/kill semantics.  
  Needed:
  - Clear spec for legal state transitions: `start → continue* → end` or `kill`.  
  - Tests covering race conditions (start+kill, cancel during prefill/decode, double‑end).  

- [ ] **1.3.2 Ensure robust cancellation under load (no leaks, no stuck sessions).** (Owner: Engineer A, Progress: 80%)  
  Details: `cancel_flag`, `Gateway::cancel`, and kill paths exist.  
  Needed:
  - Stress tests with many concurrent cancels.  
  - Checks that canceled sessions always free KV and are unbound from `SeqId2Rank`.  

---

## 2. KV Cache Manager & Prefix Reuse

### 2.1 First‑Class KV Allocator (Paged Blocks)

- [ ] **2.1.1 Implement an engine‑level `KVCacheManager` abstraction.** (Owner: Engineer A, Progress: 80%)  
  Details: `kv_cache_utils_v2` and allocators exist at kernel level but not as a shared manager.  
  Needed:
  - A C++ `KVCacheManager` that:  
    - Manages KV blocks/pages for all layers.  
    - Exposes `allocate(sequence_id, length)`, `free(sequence_id)`, `capacity()`, `used()` APIs.  
    - Supports multi‑GPU topologies (TP/PP) consistently.  
  Design (C++):
  - New header `src/turbomind/core/kv_cache_manager.h`:
    ```cpp
    struct KVLayout {
        int num_layers;
        int num_kv_heads;
        int head_dim;
        int page_size;      // tokens per page
        int bytes_per_value;
    };

    struct KVReservation {
        uint64_t seq_id;
        int      first_page;
        int      num_pages;
    };

    class KVCacheManager {
    public:
        KVCacheManager(const KVLayout& layout,
                       size_t total_capacity_bytes);

        size_t total_pages() const;
        size_t used_pages() const;

        bool   can_reserve(uint64_t seq_id,
                           const KVUsageEstimate& est) const;
        bool   reserve(uint64_t seq_id,
                       const KVUsageEstimate& est,
                       KVReservation* out);
        void   release(uint64_t seq_id);

        // Translate (seq_id, position) → physical page index(es)
        int    page_for(uint64_t seq_id, int position) const;
    private:
        KVLayout layout_;
        // internal free list / bitmap / allocator structures
    };
    ```
  - The attention kernels continue to work with raw pointers + indices; they query page indices via `page_for`.

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

- [ ] **4.1.1 Implement a KV capacity estimator per request.** (Owner: Engineer A, Progress: 80%)  
  Details: KV utilities know per‑token KV sizes, but they’re not aggregated per request.  
  Needed:
  - Given a prompt length and `max_new_tokens`, estimate KV blocks required to complete a request.  
  - Include per‑layer and per‑head factors, and TP/PP topology.  
  Design (C++):
  - Add a small helper (possibly as static methods on `KVCacheManager`):
    ```cpp
    struct KVUsageEstimate {
        size_t pages_needed;
        size_t bytes_needed;
    };

    KVUsageEstimate estimate_usage(const ModelLayout& layout,
                                   int prompt_len,
                                   int max_new_tokens) {
        int total_tokens = prompt_len + max_new_tokens;
        int pages = (total_tokens + layout.page_size - 1) / layout.page_size;
        size_t bytes = static_cast<size_t>(pages)
                     * layout.num_layers
                     * layout.num_kv_heads
                     * layout.head_dim
                     * sizeof(half);  // or actual KV type
        return {pages, bytes};
    }
    ```
  - `EngineScheduler` uses this estimator when deciding whether to reserve KV for a new request.

- [ ] **4.1.2 Implement a Guaranteed‑Completion capacity scheduler.** (Owner: Engineer A, Progress: 70%)  
  Details: No dedicated capacity scheduler exists today.  
  Needed:
  - Scheduler mode that:  
    - Guarantees that once a request starts, KV capacity is reserved until completion.  
    - Rejects/delays new requests when insufficient capacity remains.  
  Design (C++):
  - Add a simple capacity scheduler, used inside `EngineScheduler`:
    ```cpp
    class CapacityScheduler {
    public:
        explicit CapacityScheduler(KVCacheManager* kv_mgr);

        bool try_start_request(uint64_t seq_id,
                               const KVUsageEstimate& est);
        void finish_request(uint64_t seq_id);
    private:
        KVCacheManager* kv_mgr_;
    };
    ```
  - `try_start_request` calls `kv_mgr_->reserve`; if it fails, the request stays in the queue (or is rejected by policy).

- [ ] **4.1.3 Integrate capacity checks with Gateway / RequestQueue.** (Owner: Engineer A, Progress: 60%)  
  Details: Gateway/RequestQueue currently accept all requests.  
  Needed:
  - On enqueue and scheduling, consult the capacity scheduler:  
    - Decide whether to queue, delay, or reject requests based on KV availability and token budget.  
  Design (integration):
  - Gateway/RequestQueue remain responsible for fairness and session routing.  
  - The worker loop (via `EngineScheduler` + `CapacityScheduler`) decides:
    - Which queued requests can transition to `active` based on KV capacity and token budget.  
    - Which requests must remain queued.  
  - For “hard” rejections (optional), add an error code and early callback path in `ModelRequest::Forward` when `CapacityScheduler` signals “never schedulable” under current KV limits.

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

- [ ] **6.1.1 Define `DriftEngineConfig` and integrate it into LMDeploy.** (Owner: Engineer B, Progress: 60%)  
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

- [ ] **6.1.2 Expose `DriftEngineConfig` in pipeline and server entrypoints.** (Owner: Engineer B, Progress: 50%)  
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
  - Current status: `DriftEngineConfig` can already be passed to `pipeline` / `serve`, and a convenience `drift_api_server(...)` wrapper exists in `lmdeploy/api.py` (backend is set to `"drift"`); a dedicated `drift_pipeline(...)` helper is still TBD.

### 6.2 DriftEngine C++ Top‑Level Structure

- [ ] **6.2.1 Implement `DriftEngine` C++ class wrapping TurboMind components.** (Owner: Engineer A, Progress: 40%)  
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
  - `DriftEngine` is a **composition** of:
    - `Gateway` (request routing).
    - `EngineScheduler` (Section 1.2) for prefill/decode mixing and chunking.
    - `KVCacheManager` + `PrefixCache` (Section 2).
    - `CapacityScheduler` (Section 4).

- [ ] **6.2.2 Add a C++ config struct mirroring `DriftEngineConfig`.** (Owner: Engineer A, Progress: 80%)  
  Design:
  - `DriftEngineConfig` (C++) in `src/turbomind/engine/drift_engine_config.h`:
    ```cpp
    struct DriftEngineConfig {
        SchedulerConfig scheduler;
        KVLayout        kv_layout;
        ModelLayout     model_layout;

        bool prefer_high_throughput{true};
        int  target_latency_ms_p50{50};
        int  target_latency_ms_p95{200};
        int  max_queued_requests{4096};
        bool abort_on_oom{true};
    };
    ```
  - Python `DriftEngineConfig` is converted to this struct at binding layer.

### 6.3 DriftEngine Scheduling Policies

- [ ] **6.3.1 Implement a two‑queue scheduler policy (prefill vs decode) tuned for throughput.** (Owner: Engineer A, Progress: 60%)  
  Design:
  - Within `EngineScheduler` when used by DriftEngine:
    - Maintain two priority queues:
      - `prefill_queue_`: sequences in `kPrefill` phase, holding `PrefillChunk`s.
      - `decode_queue_`: sequences in `kDecode` phase.
    - Scheduling policy:
      - Compute token budget for this step `B = cfg_.max_num_batched_tokens`.
      - Compute desired prefill vs decode ratio based on config:
        - If `prefer_high_throughput == true`:
          - Use a fixed ratio, e.g., 60–70% tokens for decode, 30–40% for prefill.
        - Else (latency‑optimized):
          - Prioritize decode for low‑latency sequences and short prompts.
      - Fill `decode_batch` first up to its budget, using:
        - Either FIFO or small‑latency‑first policy.
      - Use remaining tokens for `prefill_batch`, obeying `max_num_partial_prefills` and long‑prompt controls.

- [ ] **6.3.2 Add adaptive tuning based on observed latency and tokens/sec.** (Owner: Engineer A, Progress: 30%)  
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

- [ ] **6.4.1 Define the end‑to‑end request lifecycle in DriftEngine.** (Owner: Engineer A, Progress: 10%)  
  Design (high‑level flow):
  1. Python side:
     - User constructs `DriftEngineConfig` and starts a `drift_pipeline` or `drift_api_server`.  
     - This spins up one or more `DriftEngine` worker threads/processes.
  2. Request submission:
     - Requests arrive via Python binding, converted into TurboMind `Request` objects via `ModelRequest::Forward`.  
     - `Gateway::push` enqueues them in per‑rank `RequestQueue`s; `SeqId2Rank` maintains session affinity.
  3. Worker loop:
     - Each `DriftEngine::worker_loop(rank)`:
       - Calls `gateway_->pop(...)` to collect new `infer_reqs` and `kill_reqs`.  
       - Passes them to `scheduler_.on_new_requests(...)`.  
       - `CapacityScheduler` and `KVCacheManager` decide which new sequences can start based on KV capacity (Guaranteed‑Completion).  
       - `PrefixCache` is consulted to reuse existing KV pages for shared prefixes.  
       - `scheduler_.schedule_step(...)` generates `prefill_batch` and `decode_batch` for this step.  
       - The model is executed for both batches (prefill and decode), using existing TurboMind kernels.  
       - `SequenceState` is updated; finished sequences release KV via `KVCacheManager::release` and are removed.  
  4. Response:
     - As tokens are produced, `UpdateState` / `forward_cb` propagate updates back to Python, which streams tokens to the user.

- [ ] **6.4.2 Ensure DriftEngine is strictly opt‑in and co‑exists with legacy TurboMind engine.** (Owner: Engineer B, Progress: 0%)  
  Design:
  - No changes to existing TurboMind engine semantics.  
  - New code paths:
    - `backend="drift"` for the LMDeploy backend selection.  
    - `backend="turbomind"` continues to use the current engine for backwards compatibility.  
  - Shared components (Gateway, ModelRequest, kernels) are reused; only the orchestration and scheduling layer differ.

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
