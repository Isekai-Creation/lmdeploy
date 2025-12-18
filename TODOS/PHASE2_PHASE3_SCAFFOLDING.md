# DriftEngine Phase 2 & Phase 3 Scaffolding Design (No-Wiring)

## Scope & Constraints
- Keep Phase 1 behavior frozen. All new structs, kernels, and pipeline hooks land behind feature flags or unused stubs.
- Phase 2 covers `ModelLayout`/`KVLayout`/`KVCacheManager` scaffolding plus per-request state needed by CapacityScheduler.
- Phase 3 covers PrefixCache primitives, suffix/speculative cache hooks, and the CUDA plumbing that will later back reuse/prefetch.
- Deliverables: headers, CUDA kernel entry-points, and Python config plumbing ready for wiring. No runtime behavior may change.
- All work must coexist with HF GPT-OSS-20B/120B baselines on both DriftEngine and legacy TurboMind.

## Cross-Engine Kernel Inventory (Reference Only)
| Area | Drift / TurboMind | TensorRT-LLM | Legacy LMDeploy Drops | sglang |
| --- | --- | --- | --- | --- |
| ModelLayout & KV | `src/turbomind/models/common/model_layout.{h,cc}`, `src/turbomind/core/kv_cache_manager.{h,cc}`, `src/turbomind/models/llama/LlamaBatch.{h,cc}`, `src/turbomind/models/llama/llama_kernels.cu`, `src/turbomind/kernels/attention/kv_cache_utils_v2.cu` | `cpp/include/tensorrt_llm/batch_manager/allocateKvCache.h`, `capacityScheduler.h`, `cpp/tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.cu`, `runtime/cache_transmission/cacheSplitConcat.cu` | `lmdeploy_8da9555d/src/turbomind/models/llama/llama_kernels.cu`, `lmdeploy_eagle/src/turbomind/kernels/gpt_kernels.cu` | `sgl-kernel/csrc/kvcacheio/transfer.cu`, `sgl-kernel/csrc/memory/store.cu` |
| Prefill/Decode Scheduling | `src/turbomind/engine/EngineScheduler.{h,cc}`, `src/turbomind/engine/scheduler_config.h`, `src/turbomind/models/llama/EagleDraftLayer.cu` | `cpp/tensorrt_llm/batch_manager/sequenceSlotManager.h`, `microBatchScheduler.h`, `layers/beamSearchLayer.cu` | `lmdeploy_8da9555d/src/turbomind/kernels/attention/decoding.cu`, `lmdeploy_eagle/src/turbomind/models/llama/EagleDraftLayer.cu` | `sgl-kernel/csrc/attention/cascade.cu`, `sgl-kernel/csrc/spatial/greenctx_stream.cu` |
| Prefix/Suffix/Speculative | `src/turbomind/core/prefix_cache.{h,cc}`, `src/turbomind/speculative/{draft_provider,suffix_cache}.h`, `src/turbomind/models/llama/eagle3_attention_kernels.cu`, `src/turbomind/models/llama/EagleDraftLayer.cu` | `cpp/tensorrt_llm/kernels/speculativeDecoding/{draftTokenTreeKernels,externalDraftTokensKernels,medusaDecodingKernels,explicitDraftTokensKernels,eagleDecodingKernels}.cu`, `runtime/batch_manager/promptTuningBuffers.h` | `lmdeploy_eagle/src/turbomind/models/llama/eagle3_attention_kernels.cu`, `lmdeploy_8da9555d/src/turbomind/models/llama/EagleDraftLayer.cu` | `sgl-kernel/csrc/speculative/{speculative_sampling,eagle_utils,ngram_utils}.cu`, `sgl-kernel/csrc/grammar/apply_token_bitmask_inplace_cuda.cu` |

> These references are parity guides only. Nothing from TensorRT-LLM or sglang is copied; they inform the scaffolding requirements before Phase 2/3 wiring begins.

## Phase 2 – ModelLayout & KV Stack Scaffolding

### Objectives
1. Encode GPT-OSS-120B layout metadata (layers, KV heads, head dim, token-per-page) in a reusable structure.
2. Expose a typed `KVLayout` contract to C++ and Python so memory footprint can be derived before allocation.
3. Prepare CUDA kernels and `LlamaBatch` executor shims to consume per-page metadata without enabling new flows yet.

### Key Components & Files
| Component | Location | Notes | Reference Kernels / Code |
| --- | --- | --- | --- |
| `ModelLayout` extensions | `src/turbomind/models/common/model_layout.{h,cc}` | Add BF16/FP8 awareness, page-bytes override, spec hooks. | Compare to `TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/allocateKvCache.h` & `capacityScheduler.h`. |
| `KVLayout` + `KVCacheManager` | `src/turbomind/core/kv_cache_manager.{h,cc}` | Pre-calculate `page_bytes`, expose `page_bytes_override`, add lazy metrics. | Reference `TensorRT-LLM/cpp/tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.cu` for page stitching and `sgl-kernel/csrc/kvcacheio/transfer.cu` for IO patterns. |
| Drift config plumbing | `src/turbomind/engine/drift_engine_config.h`, `lmdeploy/config/drift_config.py` | Mirror new layout fields, no behavior changes. | Keep parity with `lmdeploy_8da9555d/lmdeploy/messages.py` and CLI knobs. |
| TurboMind pipeline hints | `lmdeploy/turbomind/turbomind.py` | Surface `_tm_num_layers`, `_tm_num_kv_heads`, `_tm_head_dim`, `_tm_page_size`, but keep scheduling logic unchanged. | Align with `_from_hf` layout derivation logic + `lmdeploy_eagle/.../llama_utils.cu`. |
| CUDA consumer stubs | `src/turbomind/models/llama/LlamaBatch.{h,cc}`, `src/turbomind/models/llama/llama_kernels.cu` | Accept `KVLayout` metadata, guard behind `#if DRIFT_PHASE2` macros. | Reference `TensorRT-LLM/cpp/tensorrt_llm/runtime/cache_transmission/cacheSplitConcat.cu` and `sgl-kernel/csrc/attention/cutlass_mla_kernel.cu`. |

### Detailed Steps
1. **ModelLayout metadata**
   - Introduce `ModelLayout::QuantProfile` capturing per-tensor dtype, KV quant policy, and NVFP4 compatibility inside `src/turbomind/models/common/model_layout.h`.
   - Provide helper `ModelLayout::from_tm_config(const TmModelConfig&)` mapped from `_setup_drift_engine` HF import data; compare with `TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/allocateKvCache.h` field derivations.
2. **KVLayout contract**
   - Add derived fields: `tokens_per_page`, `bytes_per_page`, `kv_factor`, `page_bytes_override` within `src/turbomind/core/kv_cache_manager.h` and ensure Python mirrors exist in `lmdeploy/messages.py`.
   - Provide `validate_against(model_layout)` to log mismatches; reference `capacityScheduler.h::validateKvConfig` for parity.
3. **KVCacheManager scaffolding**
   - Add `trace_page_install(seq_id, span<const int>)` stub, `kv_cookie` tracking, and new metrics counters modeled after `TensorRT-LLM/cpp/tensorrt_llm/batch_manager/capacityScheduler.h`.
   - Implement host-side helpers for `build_page_ptr_table` that accept `std::span` to avoid copies; align with `sgl-kernel/csrc/kvcacheio/transfer.cu` semantics.
4. **CUDA kernel entrypoints**
   - Stub `pack_kv_pages(const KVLayout&, void* dst, const PageDescriptor*)` in `src/turbomind/models/llama/llama_kernels.cu`; mimic tensor layouts seen in `TensorRT-LLM/cpp/tensorrt_llm/runtime/cache_transmission/cacheSplitConcat.cu`.
   - Add compile-time constants for `page_size`/`head_dim` that default to previous values when scaffolding is unused.
   - Mirror kernels in `src/turbomind/models/llama/eagle3_attention_kernels.cu` and future `specpv_kv_cache.h` helpers to accept optional `kv_layout` pointer (unused if null).
5. **Pipeline integration (no-op)**
   - Extend `DriftEngineConfig` (Python + C++) with `model_layout` and `kv_layout` dictionaries already populated by `_setup_drift_engine`; ensure `lmdeploy/turbomind/turbomind.py` stores `_tm_num_layers`, `_tm_page_size`.
   - `to_cpp_drift_engine_config` exports these fields but `DriftEngine` ignores them unless `DRIFT_ENABLE_PHASE2_SCHEME=1`.
6. **Testing scaffolds**
   - Add gtests under `tests/csrc/unittests/kv_cache_manager_phase2.cc` instantiating the new structs with fake allocators; follow `TensorRT-LLM/cpp/tests/unit_tests/kernels/sparseKvCacheTest.cu` patterns but guard behind `DRIFT_PHASE2_TESTS`.

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
| Component | Location | Notes | Reference Kernels / Code |
| --- | --- | --- | --- |
| PrefixCache headers | `src/turbomind/core/prefix_cache.{h,cc}` | Flesh out structs, metrics, but keep integration flags false. | Compare with `TensorRT-LLM/cpp/tensorrt_llm/kernels/lruKernel.cu` and `sgl-kernel/csrc/kvcacheio/transfer.cu`. |
| Suffix cache scaffolding | `src/turbomind/speculative/suffix_cache.h`, `src/turbomind/speculative/draft_provider.h` | Expand structs for multi-branch drafts, still return empty drafts. | Reference `TensorRT-LLM/cpp/tensorrt_llm/kernels/speculativeDecoding/draftTokenTreeKernels.cu`, `externalDraftTokensKernels.cu`, and `sgl-kernel/csrc/speculative/speculative_sampling.cu`. |
| Scheduler touch-points | `src/turbomind/engine/EngineScheduler.{h,cc}` | Prepare hooks (`on_prefix_match`, `on_suffix_draft`) with empty implementations. | Mirror behaviors from `TensorRT-LLM/cpp/include/tensorrt_llm/batch_manager/pauseRequests.h` and `sgl-kernel/csrc/attention/cascade.cu`. |
| CUDA kernels | `src/turbomind/models/llama/llama_kernels.cu`, `src/turbomind/models/llama/LlamaDraftLayer.cu`, `src/turbomind/models/llama/eagle3_attention_kernels.cu` | Add stub kernels for prefix materialization and suffix scoring. | Reference `TensorRT-LLM/cpp/tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.cu` and `sgl-kernel/csrc/speculative/eagle_utils.cu`. |
| Pipeline plumbing | `lmdeploy/messages.py`, `lmdeploy/config/drift_config.py`, `lmdeploy/turbomind/turbomind.py` | Surface suffix cache knobs + spec hints (already partially done) without enabling features. | Ensure parity with historical knobs in `lmdeploy_8da9555d/lmdeploy/messages.py` and CLI flags in `sglang/serve`.

### Detailed Steps
1. **PrefixCache API**
   - Define `PrefixKey`, `PrefixEntry`, `PrefixMatchResult`, and LRU bookkeeping fields inside `src/turbomind/core/prefix_cache.{h,cc}`.
   - Add concurrency guards and instrumentation counters (hits, misses, evictions) modeled after `TensorRT-LLM/cpp/tensorrt_llm/kernels/lruKernel.cu` and `sgl-kernel/csrc/kvcacheio/transfer.cu` logging patterns.
   - Provide `match(const PrefixKey&, PrefixMatchResult*)`, `insert(seq_id, const PrefixEntry&)`, `evict(seq_id)` bodies that only log operations until `enable_prefix_caching` flips at runtime.
2. **Scheduler hooks**
   - Add `EngineScheduler::maybe_match_prefix(const Request&)` and `EngineScheduler::maybe_record_prefix(seq_id)` (no-ops) referencing `src/turbomind/engine/EngineScheduler.{h,cc}` and cite TensorRT-LLM `pauseRequests.h` semantics for future implementation.
   - Extend `SequenceState` with `std::vector<int> reused_page_ids` placeholder for future prefix replays.
3. **Suffix cache & speculative draft provider**
   - Flesh out `SuffixSpecParams`, `SuffixDraft`, `SuffixDecodeCache` internals (LRU maps, trie nodes, metrics) in `src/turbomind/speculative/suffix_cache.h`; base the data model on TensorRT-LLM `draftTokenTreeKernels.cu` structures and sglang `speculative_sampling.cu` acceptance flows.
   - Provide `SuffixDraftProvider` factory stubs inside DriftEngine that instantiate caches only when `enable_suffix_decoding` is true (default false).
4. **CUDA kernels**
   - Add placeholder kernels:
     - `prefix_cache_pack_pages<<<>>>` for copying page windows into staging buffers (use `src/turbomind/models/llama/llama_kernels.cu` and reference TensorRT-LLM `speculativeDecoding/kvCacheUpdateKernels.cu`).
     - `suffix_tree_expand<<<>>>` inside `src/turbomind/models/llama/LlamaDraftLayer.cu`, referencing TensorRT-LLM `speculativeDecoding/draftTokenTreeKernels.cu`.
     - `suffix_acceptance_score<<<>>>` for acceptance probability computation (compare with `sgl-kernel/csrc/speculative/speculative_sampling.cu`).
   - Each kernel should compile (no-op body) so unit tests can link against them once wiring begins.
5. **Pipeline impacts**
   - Extend CLI/config docs to describe new knobs (`suffix_cache_max_depth`, `suffix_min_token_prob`, `specpv_*`), referencing `lmdeploy/messages.py` and `docs/en/inference/turbomind_config.md` plus TensorRT-LLM CLI analogs.
   - `_setup_drift_engine` stores user-requested spec flags under private `_requested_*` attributes so later phases can restore them.
6. **Testing scaffolds**
   - Add placeholder unit tests in `tests/csrc/unittests/prefix_cache_phase3.cc` that instantiate caches, perform fake matches/inserts, and assert instrumentation counters without touching real KV pages.
   - Provide Python smoke tests under `tests/test_prefix_suffix_config.py` that assert config round-trips through `to_cpp_drift_engine_config` and compare against `TensorRT-LLM/cpp/tests/unit_tests/kernels/speculativeDecoding` coverage goals.

### Kernel & CUDA Inventory (Phase 3)
- `llama_kernels.cu`: add `prefix_cache_pack_pages` and `prefix_cache_restore_pages` (both no-op) to be filled when wiring occurs.
- `LlamaDraftLayer.cu`: stub `suffix_tree_expand` harness hooking into Softmax/rescore loops.
- `eagle3_attention_kernels.cu`: add `SpecPVWindowInit` stub to pre-load suffix windows.
- `specpv_kv_cache.h`: carry new enums for suffix window states, unused until acceptance logic arrives.

### Pipeline/Config Impact
- `lmdeploy/messages.DriftEngineConfig` already includes suffix/speculative knobs; ensure docs clarify they are scaffolding-only.
- `lmdeploy/config/drift_bridge.py` exports these knobs even for legacy helpers so downstream experiments can introspect without enabling features.
- Provide docstrings/warnings in `lmdeploy/turbomind/turbomind.py::_setup_drift_engine` indicating speculative decoding stays disabled until compute-sanitizer validation passes.

## Implementation Tasks (Phase 2 & 3 – No-Wiring)
1. Mirror GPT-OSS-120B layout metadata into `ModelLayout::QuantProfile` (compare vs TensorRT-LLM `allocateKvCache.h`).
2. Implement `ModelLayout::from_tm_config` and `_setup_drift_engine` plumbing capturing `_tm_*` attributes.
3. Extend `KVLayout` struct with derived fields plus `validate_against` helper referencing `capacityScheduler.h`.
4. Add `page_bytes_override` + `tokens_per_page` logic to `KVCacheManager` ctor; log before allocation.
5. Scaffold `trace_page_install`/`kv_cookie` instrumentation inside `KVCacheManager` (no behavior changes).
6. Introduce no-op `pack_kv_pages` kernels in `llama_kernels.cu`, guarded by `#if DRIFT_ENABLE_PHASE2_SCAFFOLD`.
7. Add metadata stubs inside `LlamaBatch::bind_kv_layout` (no runtime use without flag).
8. Expand Python configs (`DriftEngineConfig`, `drift_bridge.py`) with `model_layout_override` and suffix knobs, ensuring serialization.
9. Write gating env/flags (`DRIFT_ENABLE_PHASE2_SCHEME`, `DRIFT_PHASE2_TESTS`) and document defaults (off).
10. Define `PrefixKey`/`PrefixEntry`/`PrefixMatchResult`/LRU skeleton inside `src/turbomind/core/prefix_cache.{h,cc}`.
11. Add `EngineScheduler::maybe_match_prefix` + `maybe_record_prefix` stubs referencing `SequenceState` updates.
12. Expand `SequenceState` with `reuse_page_ids` and `PrefetchPlan` placeholders (unused now).
13. Flesh out `SuffixSpecParams`, `SuffixDraft`, `SuffixDecodeCache` data structures plus logging, returning empty drafts.
14. Extend `SuffixDraftProvider` to respect `enable_suffix_decoding` flag but short-circuit to no-op caches.
15. Stub CUDA kernels `prefix_cache_pack_pages`, `suffix_tree_expand`, `suffix_acceptance_score` referencing TensorRT-LLM + sglang files.
16. Ensure `_setup_drift_engine` captures user spec flags into `_requested_*` attrs while forcing non-spec execution.
17. Update docs (`docs/en/inference/turbomind_config.md`) explaining scaffolding knobs and stating they are disabled by default.
18. Create placeholder C++ unit tests (`kv_cache_manager_phase2.cc`, `prefix_cache_phase3.cc`) that assert struct construction only when opt-in flags set.
19. Add Python smoke tests (`tests/test_prefix_suffix_config.py`) verifying config round-trips with new knobs.
20. Wire logging toggles (`DRIFT_ENABLE_PHASE2_LAYOUT_LOGS`, prefix cache hit/miss counters) without enabling runtime usage.

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

## Scaffolding Flags & Metrics
- **Build flags**: `DRIFT_ENABLE_PHASE2_SCAFFOLD`, `DRIFT_ENABLE_PHASE3_SCAFFOLD`, `DRIFT_PHASE2_TESTS`, `DRIFT_PHASE3_TESTS` remain **off by default**; define them in CMake/toolchain but keep wiring disabled.
- **Env toggles**: document `DRIFT_ENABLE_PHASE2_LAYOUT_LOGS`, `DRIFT_ENABLE_PREFIX_CACHE_TRACE`, `DRIFT_REQUESTED_SPEC_LOG` for verbose traces without behavior changes.
- **Metrics**: only register counters (prefix hits/misses, suffix draft attempts, KV cookie logs) when flags are on; otherwise guard with `if constexpr` or runtime checks to avoid overhead.

## Phase 1 Freeze Reminder
- All scaffolding must compile in the default build with zero runtime impact.
- Keep `enable_speculative_decoding=False` in both Python and C++; store requested values on `_requested_*` members only.
- Do not enable prefix/suffix cache insertion or KV re-use until HF GPT-OSS-20B compute-sanitizer and pointer-contract tasks from Phase 1 are resolved.

## Next Actions
1. Implement the scaffolding items above in small PRs (config → C++ structs → CUDA stubs → docs/tests), verifying no behavior changes in benchmarks.
2. After HF GPT-OSS-20B compute-sanitizer issues are resolved (Phase 1), progressively enable wiring using the pre-laid interfaces here.
