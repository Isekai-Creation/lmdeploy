# DriftEngine Phase 2 & Phase 3 Scaffolding Design (No-Wiring)

## Scope & Constraints
- Keep Phase 1 behavior frozen. All new structs, kernels, and pipeline hooks land behind feature flags or unused stubs.
- Phase 2 covers `ModelLayout`/`KVLayout`/`KVCacheManager` scaffolding plus per-request state needed by CapacityScheduler.
- Phase 3 covers PrefixCache primitives, suffix/speculative cache hooks, and the CUDA plumbing that will later back reuse/prefetch.
- Deliverables: headers, CUDA kernel entry-points, and Python config plumbing ready for wiring. No runtime behavior may change.
- All work must coexist with HF GPT-OSS-20B/120B baselines on both DriftEngine and legacy TurboMind.

## Phase 2 – ModelLayout & KV Stack Scaffolding

### Objectives
1. Encode GPT-OSS-120B layout metadata (layers, KV heads, head dim, token-per-page) in a reusable structure.
2. Expose a typed `KVLayout` contract to C++ and Python so memory footprint can be derived before allocation.
3. Prepare CUDA kernels and `LlamaBatch` executor shims to consume per-page metadata without enabling new flows yet.

### Key Components & Files
| Component | Location | Notes |
| --- | --- | --- |
| `ModelLayout` extensions | `src/turbomind/models/common/model_layout.{h,cc}` | Add BF16/FP8 awareness, page-bytes override, spec hooks. |
| `KVLayout` + `KVCacheManager` | `src/turbomind/core/kv_cache_manager.{h,cc}` | Pre-calculate `page_bytes`, expose `page_bytes_override`, add lazy metrics. |
| Drift config plumbing | `src/turbomind/engine/drift_engine_config.h`, `lmdeploy/config/drift_config.py` | Mirror new layout fields, no behavior changes. |
| TurboMind pipeline hints | `lmdeploy/turbomind/turbomind.py` | Surface `_tm_num_layers`, `_tm_num_kv_heads`, `_tm_head_dim`, `_tm_page_size`, but keep scheduling logic unchanged. |
| CUDA consumer stubs | `src/turbomind/models/llama/LlamaBatch.{h,cc}`, `src/turbomind/models/llama/llama_kernels.cu` | Accept `KVLayout` metadata, guard behind `#if DRIFT_PHASE2` macros. |

### Detailed Steps
1. **ModelLayout metadata**
   - Introduce `ModelLayout::QuantProfile` capturing per-tensor dtype, KV quant policy, and NVFP4 compatibility.
   - Provide helper `ModelLayout::from_tm_config(const TmModelConfig&)` that maps HF import data onto DriftEngine layout, but only stores it; no consumers yet.
2. **KVLayout contract**
   - Add derived fields: `tokens_per_page`, `bytes_per_page`, `kv_factor`, `page_bytes_override`.
   - Provide `validate_against(model_layout)` to ensure mismatches are logged (but still allow Phase 1 layout to proceed).
3. **KVCacheManager scaffolding**
   - Add `trace_page_install(seq_id, span<const int>)` stub, `kv_cookie` tracking, and new metrics counters without hooking them.
   - Implement host-side helpers for `build_page_ptr_table` that accept `std::span` to avoid copies. Existing runtime still calls older overload until wiring.
4. **CUDA kernel entrypoints**
   - Stub `pack_kv_pages(const KVLayout&, void* dst, const PageDescriptor*)` in `llama_kernels.cu` returning immediately; used later for fused prefetch.
   - Add compile-time constants for `page_size`/`head_dim` that default to previous values when scaffolding is unused.
   - Mirror kernels in `eagle3_attention_kernels.cu` and `specpv_kv_cache.h` to accept optional `kv_layout` pointer (unused if null).
5. **Pipeline integration (no-op)**
   - Extend `DriftEngineConfig` (Python + C++) with `model_layout` and `kv_layout` dictionaries already populated by `_setup_drift_engine`.
   - `to_cpp_drift_engine_config` now includes these fields even though current C++ ignores them unless `DRIFT_ENABLE_PHASE2_SCHEME` is set.
6. **Testing scaffolds**
   - Add gtests under `tests/csrc/unittests/kv_cache_manager_phase2.cc` that instantiate the new structs but skip actual allocations when `DRIFT_PHASE2_TESTS` is false.

### Kernel & CUDA Inventory (Phase 2)
- `llama_kernels.cu`: add stubbed helpers for page-aware block strides, warp shapes, and NVFP4 conversions.
- `llama_utils.cu`: introduce `encode_kv_layout(DeviceContext*, const KVLayout&)` returning metadata buffers (unused for now).
- `eagle3_attention_kernels.cu`: accept optional page descriptors for later suffix speculative mode.
- `unified_attention_layer.cc`: store `KVLayout` pointer per layer but guard usage.

### Pipeline/Config Impact
- `DriftEngineConfig` Python dataclass gains optional `model_layout_override` block but `DriftEngine` constructor still derives the legacy layout when overrides are absent.
- `TurboMindSchedulerConfig` remains unchanged other than carrying spec hints already merged.
- Provide CLI/environment toggles (`DRIFT_ENABLE_PHASE2_LAYOUT_LOGS`) to dump scaffolding state without altering flow.

## Phase 3 – Prefix & Suffix Cache Scaffolding

### Objectives
1. Provide a fully-specified PrefixCache API (C++ side) plus the suffix/speculative draft providers without activating them.
2. Enumerate CUDA kernels required for prefix eviction/match, suffix tree speculation, and acceptance scoring, but keep them returning placeholders.
3. Ensure Python configs and pipeline entrypoints can describe suffix cache limits, spec method hints, and SuffixDecodeCache wiring for future phases.

### Key Components & Files
| Component | Location | Notes |
| --- | --- | --- |
| PrefixCache headers | `src/turbomind/core/prefix_cache.{h,cc}` | Flesh out structs, metrics, but keep integration flags false. |
| Suffix cache scaffolding | `src/turbomind/speculative/suffix_cache.h`, `src/turbomind/speculative/draft_provider.h` | Expand structs for multi-branch drafts, still return empty drafts. |
| Scheduler touch-points | `src/turbomind/engine/EngineScheduler.{h,cc}` | Prepare hooks (`on_prefix_match`, `on_suffix_draft`) with empty implementations. |
| CUDA kernels | `src/turbomind/models/llama/llama_kernels.cu`, `src/turbomind/models/llama/LlamaDraftLayer.cu`, `src/turbomind/models/llama/eagle3_attention_kernels.cu` | Add stub kernels for prefix materialization and suffix scoring. |
| Pipeline plumbing | `lmdeploy/messages.py`, `lmdeploy/config/drift_config.py`, `lmdeploy/turbomind/turbomind.py` | Surface suffix cache knobs + spec hints (already partially done) without enabling features. |

### Detailed Steps
1. **PrefixCache API**
   - Define `PrefixKey`, `PrefixEntry`, `PrefixMatchResult`, and LRU bookkeeping fields.
   - Add concurrency guards and instrumentation counters.
   - Provide `match(const PrefixKey&, PrefixMatchResult*)`, `insert(seq_id, const PrefixEntry&)`, `evict(seq_id)` bodies that only log operations until `enable_prefix_caching` flips at runtime.
2. **Scheduler hooks**
   - Add `EngineScheduler::maybe_match_prefix(const Request&)` and `EngineScheduler::maybe_record_prefix(seq_id)` that currently return false/no-op respectively.
   - Extend `SequenceState` with `std::vector<int> reused_page_ids` placeholder for future prefix replays.
3. **Suffix cache & speculative draft provider**
   - Flesh out `SuffixSpecParams`, `SuffixDraft`, `SuffixDecodeCache` internals (LRU maps, trie nodes, metrics) but keep `speculate` returning an empty draft while recording instrumentation.
   - Provide `SuffixDraftProvider` factory stubs inside DriftEngine that instantiate caches only when `enable_suffix_decoding` is true (default false).
4. **CUDA kernels**
   - Add placeholder kernels:
     - `prefix_cache_pack_pages<<<>>>` for copying page windows into staging buffers.
     - `suffix_tree_expand<<<>>>` for generating multi-branch speculative drafts.
     - `suffix_acceptance_score<<<>>>` for computing acceptance probabilities.
   - Each kernel should compile (no-op body) so unit tests can link against them once wiring begins.
5. **Pipeline impacts**
   - Extend CLI/config docs to describe new knobs (`suffix_cache_max_depth`, `suffix_min_token_prob`, `specpv_*`).
   - `_setup_drift_engine` stores user-requested spec flags under private `_requested_*` attributes so later phases can restore them.
6. **Testing scaffolds**
   - Add placeholder unit tests in `tests/csrc/unittests/prefix_cache_phase3.cc` that instantiate caches, perform fake matches/inserts, and assert instrumentation counters without touching real KV pages.
   - Provide Python smoke tests under `tests/test_prefix_suffix_config.py` that assert config round-trips through `to_cpp_drift_engine_config`.

### Kernel & CUDA Inventory (Phase 3)
- `llama_kernels.cu`: add `prefix_cache_pack_pages` and `prefix_cache_restore_pages` (both no-op) to be filled when wiring occurs.
- `LlamaDraftLayer.cu`: stub `suffix_tree_expand` harness hooking into Softmax/rescore loops.
- `eagle3_attention_kernels.cu`: add `SpecPVWindowInit` stub to pre-load suffix windows.
- `specpv_kv_cache.h`: carry new enums for suffix window states, unused until acceptance logic arrives.

### Pipeline/Config Impact
- `lmdeploy/messages.DriftEngineConfig` already includes suffix/speculative knobs; ensure docs clarify they are scaffolding-only.
- `lmdeploy/config/drift_bridge.py` exports these knobs even for legacy helpers so downstream experiments can introspect without enabling features.
- Provide docstrings/warnings in `lmdeploy/turbomind/turbomind.py::_setup_drift_engine` indicating speculative decoding stays disabled until compute-sanitizer validation passes.

## Implementation Checklist (No-Wiring)
1. **Struct & Config Updates**
   - [ ] C++: `ModelLayout`, `KVLayout`, `PrefixCache`, `SuffixDecodeCache` scaffolds.
   - [ ] Python: ensure serialization/deserialization handles all new knobs.
2. **CUDA/C++ Entry Points**
   - [ ] Add kernel declarations/definitions returning immediately.
   - [ ] Provide build flags (`DRIFT_ENABLE_PHASE2_SCAFFOLD`, `DRIFT_ENABLE_PHASE3_SCAFFOLD`) gating future wiring.
3. **Pipeline & Docs**
   - [ ] Update docs in `docs/en/inference/turbomind.md` plus new scaffolding doc (this file) describing semantics.
   - [ ] Document env toggles for logging/tracing new scaffolding without affecting runtime.
4. **Testing Hooks**
   - [ ] Placeholder unit tests compiled with `DRIFT_PHASE{2,3}_TESTS=ON` but skipped by default.
   - [ ] Python smoke tests verifying config translation and warning messages.

## Next Actions
1. Implement the scaffolding items above in small PRs (config → C++ structs → CUDA stubs → docs/tests), verifying no behavior changes in benchmarks.
2. After HF GPT-OSS-20B compute-sanitizer issues are resolved (Phase 1), progressively enable wiring using the pre-laid interfaces here.
