# TurboMind EAGLE Integration TODO

This file tracks all work needed to bring TurboMind’s EAGLE speculative decoding to full, production use. Items marked `[x]` are already implemented in this repo; items marked `[ ]` are pending.

## 1. Engine config and initialization

- [x] Parse `speculative_config` into `TurbomindEngineConfig` and `EngineParam` (including `method`, `model`, `num_speculative_tokens`, `max_path_len`, `max_decoding_tokens`, `max_non_leaves_per_layer`).
- [x] Initialize EAGLE for Llama models:
  - [x] Add speculative fields to `EngineParam` / `llama_params.h`.
  - [x] In `LlamaV2` ctor, when `enable_speculative_decoding` and `spec_method in {"eagle", "eagle3"}`:
    - [x] Set `spec_mode_ = SpeculativeDecodingMode::Eagle()`.
    - [x] Construct `EagleModule` with limits from `EngineParam`.
    - [x] Load draft model weights from `spec_draft_model_path`.
    - [x] Construct `EagleBuffers`.
- [x] Add logging so we can see when/with what config EAGLE is initialized (`[LlamaTritonModel]`, `[LlamaV2]`).

## 2. Core EAGLE data structures and kernels

- [x] Implement TurboMind‑side EAGLE tree representation:
  - [x] `lmdeploy/turbomind/eagle_tree.h` / `.cpp` for building and flattening speculation trees.
  - [x] Replace the initial linear `SpeculationTree::buildTree` / `EagleModule::initializeDefaultChoices` placeholder with a config‑driven EAGLE3 branching tree structure (`eagle_tree.yaml` choices with a simple chain as a safe fallback).
- [x] Implement CUDA kernels for EAGLE masks:
  - [x] `invokeBuildLeafMask` – mark leaf/non‑leaf nodes.
  - [x] `invokeGetPackedMaskFromPath` – build packed attention masks from tree paths.
- [x] Implement `EagleBuffers` for TurboMind:
  - [x] GPU buffers for draft tokens, paths, packed masks, and acceptance metadata.
  - [x] Clean allocation/free (no leaked `Tensor` objects, raw CUDA pointers only).
- [x] Implement `LlamaV2::eagleSpeculativeStep` to:
  - [x] Build the speculation tree from `draft_tokens`.
  - [x] Copy paths to device buffers.
  - [x] Call EAGLE CUDA kernels to produce leaf masks and packed masks.
  - [x] Populate acceptance rate metrics based on real comparisons between draft and target tokens (one-step, per-sequence acceptance for the initial integration).

## 3. Draft model (EagleNet) support

- [x] Implement `EagleModule::load`:
  - [x] Parse draft model config (hidden size, vocab size, intermediate size, etc.).
  - [x] Allocate FP16 `Tensor` weights for all required parameters (embeddings, FC, norms, attention, MLP, LM head, mapping).
  - [x] Load binary weight files into device tensors with typed copies.
- [x] Implement `EagleModule::forward`:
  - [x] Implement a minimal, production‑safe draft network using existing kernels (RMSNorm + LM head over target hidden states) to produce real draft logits.
  - [x] Extend the draft network to use more of the EagleNet stack (at least one shallow attention+FC “MLP” block) with the loaded weights, reusing `LlamaLinear` and pre‑formatted `LlamaDenseWeight` wrappers instead of custom GEMMs.
  - [x] Ensure the forward path is free of `cudaMalloc`/`cudaFree` in hot loops by routing all temporary activations via internal scratch tensors (and, later, `EagleBuffers`) that are sized once and reused.

## 4. Speculative decode loop integration in LlamaBatch / LlamaV2

Status: Phase‑1 EAGLE branch and engine‑token logging are implemented; the multi‑token inner loop and strict enforcement of
`eagle_max_engine_tokens_per_step_` as a hard budget (plus sequence advancement) remain TODO.

- [ ] Introduce EAGLE‑aware “engine tokens per step” semantics:
  - [x] Add a `eagle_max_engine_tokens_per_step_` field in `LlamaV2` and initialize it from `spec_max_decoding_tokens` clamped by `max_forward_token_num`.
  - [x] In `LlamaBatch::Forward`, plumb this budget into the EAGLE logs so each mini‑batch reports the configured engine token budget alongside actual `engine_tokens`.
  - [ ] Use this budget to drive the speculative inner loop once multi‑token EAGLE decoding is wired, keeping per‑sequence engine shapes static where possible so we can later enable CUDA‑graph capture for offline decode steps. For now, `compute_eagle_draft_tokens_per_seq` computes a planned `tokens_per_seq` from this budget but clamps the effective value to `1` so that no partially‑initialized multi‑token layouts are exposed before the full inner loop is implemented.
- [x] Add a speculative branch in `LlamaBatch::Forward`:
  - [x] For each decode step (or mini‑batch), add a guarded `[LlamaBatch][EAGLE]` branch that:
    - [x] Captures the last‑token hidden states from `LlamaV2`’s decoder output buffer.
    - [x] Invokes `LlamaV2::eagleDraftForward` (a thin wrapper around `EagleModule::forward`) to generate per‑sequence draft logits.
    - [x] Samples one draft token per active sequence (greedy argmax for now) and populates a host `draft_tokens` buffer.
    - [x] Calls `LlamaV2::eagleSpeculativeStep` with these `draft_tokens` and the current `Sequence*` array, logging per‑step EAGLE activity.
  - [ ] Use `num_accepted` / `accepted_tokens` to decide how far to advance each sequence in this step (for the one-step case, we only verify consistency and log metrics; multi-token advancement will be added later).

## 5. Target verification and acceptance (LlamaV2_eagle)

Status: One‑step host‑side target verification and tree‑based host acceptance are implemented and mirrored into `EagleBuffers`; the
generalised, fully device‑side tree verification for multi‑token steps remains TODO.

- [ ] Implement full “target verify” logic in `LlamaV2_eagle.cc`:
  - [ ] Extend `EagleBuffers::inputs` / `outputs` as needed to hold:
    - [ ] Per‑node logits indices / hidden‑state references.
    - [ ] Accepted paths and lengths per sequence.
  - [x] Add a helper path that, for the initial integration, uses `LlamaBatch`‑provided greedy target token IDs to implement a correct one‑step acceptance rule:
    - [x] Extend `LlamaV2::eagleSpeculativeStep` to accept both `draft_tokens` and `target_tokens`.
    - [x] For each active sequence, compare `target_token == draft_token` and:
      - [x] Write the accepted draft token into the host `accepted_tokens` buffer when they match.
      - [x] Accumulate a scalar `num_accepted` and log `[EAGLE] Accepted N/M` based on real target tokens.
    - [x] Mirror these host‑side accepted tokens/lengths into `EagleBuffers::outputs.accepted_tokens` / `accepted_lens` on device so later stages (KV/cache updates, DynamicDecode integration) can consume them.
  - [ ] Generalize this to full tree‑based verification:
    - [ ] Prepare embeddings and masks for the tree nodes using the packed masks.
    - [ ] Call `LlamaV2::Forward` (or a dedicated decoder path) once per step to compute target logits for all draft nodes in the batch (no per‑node launches).
    - [ ] For each sequence, compare target logits’ argmax vs draft token ids along each path using GPU‑side kernels only.
  - [x] Implement acceptance rule for full trees (host‑side for now):
    - [x] For each sequence, walk every path in the `SpeculationTree` and count how many tokens would be appended under the “accept while draft==target, append target on first mismatch” rule.
    - [x] Select the path with the longest accepted length per sequence and materialize `accepted_tokens` / `accepted_lens` on the host.
    - [x] Mirror host‑side `accepted_tokens` / `accepted_lens` and `best_path_ids` into `EagleBuffers::outputs` / `inputs` and call `invokePackAcceptedPaths` so downstream KV/decode code sees a packed, tree‑consistent view of accepted paths. A future follow‑up can move this acceptance computation into a dedicated GPU kernel once multi‑token target logits are available.

## 6. KV cache and decode‑state integration

Status: KV over‑provisioning and the KV rewind helper/kernel are implemented and tested; wiring accepted tokens into
`token_ids_buf_` / `sequence_lengths_` and integrating rewind with `SequenceManager`/prefix caching are still TODO.

- [ ] Update TurboMind KV cache based on EAGLE acceptance:
  - [ ] In `LlamaBatch::Forward`, after `eagleSpeculativeStep`, adjust:
    - [ ] `token_ids_buf_` (so future steps see accepted tokens as committed).
    - [ ] `sequence_lengths_` and any other per‑sequence length/state fields.
    - [ ] KV‑cache lengths and block indices for rejected tokens (rewind).
    - [x] Implement `invokeKVCacheRewind` in `lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}` so that it marks tail KV blocks as logically free in a per‑sequence block table (block IDs and optional `kv_cache_blocks` pointers), ready to be driven by TurboMind’s real KV cache / block manager.
  - [ ] Ensure compatibility with:
    - [ ] Prefix caching.
    - [ ] Model parallelism (TP/DP/CP) and cache layout.
- [ ] Verify that non‑speculative decoding (no EAGLE) remains bit‑identical to current behaviour.
 - [x] For EAGLE‑enabled models, over‑provision KV cache when initializing `SequenceManager` so spec decoding has room for both tree nodes and accepted tokens:
   - [x] Compute an “EAGLE KV factor” from `spec_max_decoding_tokens` and `spec_max_draft_path_len` and fold it into the effective `cache_max_block_count` passed to `SequenceManager`.
   - [x] Add logs to make it easy to see when EAGLE KV over‑provisioning is active and what effective KV block budget is used for offline batches.

## 7. Sampling / DynamicDecodeLayer integration

Status: Multi‑token integration with `DynamicDecodeLayer` is not yet implemented; EAGLE currently uses baseline
`dynamicDecode` semantics with a metrics‑only acceptance path.

- [ ] Decide integration strategy:
  - [ ] Option A: Extend `DynamicDecodeLayer` to understand EAGLE acceptance and variable numbers of tokens per step.
  - [ ] Option B: When `spec_mode_.isEagle()`, manage sampling and token advancement entirely inside `LlamaBatch::Forward`, bypassing `DynamicDecodeLayer` for speculative steps.
- [ ] Implement chosen strategy:
  - [ ] Ensure that:
    - [ ] `output_ids`, `finished`, and `sequence_length` reflect accepted draft tokens.
    - [ ] Logit sampling, stop criteria, and logprobs behave correctly with multi‑token steps.

## 8. Metrics, logging, and validation

Status: Logging hooks, C++ EAGLE metrics, Python `spec_info` plumbing, and most unit/integration tests (including benchmark
integration) are in place; per‑backend throughput metrics are currently surfaced via scripts, and dedicated multi‑token
validation tests are still TODO.

- [x] Add logging hooks:
  - [x] `[LlamaBatch][EAGLE]` per step/mini‑batch with engine token / context token counts.
  - [x] `[EAGLE]` logs from `LlamaV2::eagleSpeculativeStep` for tree/mask construction.
- [ ] Add runtime metrics:
  - [x] Per‑request acceptance rate and average accepted tokens per step exported from TurboMind’s C++ backend (not just Python‑side wrappers).
  - [x] Effective throughput vs baseline (tokens/s, ms/token) for EAGLE and non‑EAGLE TurboMind runs via `inference/benchmark_speculative.py`, with JSON output validated by `lmdeploy/tests/test_benchmark_speculative_integration.py`.
  - [ ] Extend throughput reporting to the PyTorch backend and ensure both backends expose comparable EAGLE vs baseline metrics.
- [ ] Validation:
  - [x] Unit tests for `EagleModule::forward` on small synthetic examples (via `_turbomind.eagle_forward_smoke` and `test_eagle_module.py`).
  - [x] Unit tests for `eagle_tree` on small synthetic examples.
  - [x] Integration test skeleton for EAGLE vs baseline behaviour in `lmdeploy/tests/test_eagle_e2e.py`, gated on `MODEL_PATH` / `SPEC_MODEL_PATH` so it can be enabled in environments with accessible models.
  - [x] Standalone benchmark harness (`inference/benchmark_speculative.py`) wired to compare baseline vs EAGLE TurboMind across multiple scenarios, including short single-sequence runs suitable for local validation.

## 9. Hard constraints / non‑goals for this port

- The implementation must:
  - Avoid copying TensorRT‑LLM source code verbatim.
  - Preserve existing TurboMind decoding behaviour when EAGLE is disabled.
  - Fail safely (e.g., treat `num_accepted = 0`) if the draft model or EAGLE path is misconfigured, rather than returning incorrect tokens.

## 10. Legacy Python speculative wrappers (cleanup)

- [x] Align `SpeculativeGenerationWrapper` with native TurboMind EAGLE:
  - [x] Remove the “accept first 2 draft tokens” stub path (used only in tests) or gate it behind a dedicated test‑only flag; rely on `DraftTokenVerifier` + real/synthetic target logits for acceptance metrics (no unconditional acceptance in production paths).
- [x] Clarify / scope `SpeculativeDecodingManager` and batch wrappers:
  - [x] For `method in {"eagle", "eagle3"}`, rely solely on native C++ EAGLE (no Python‑side draft model); keep these managers for `draft_target` / `ngram` and tests, and add tests to enforce this behaviour.
  - [x] Implement buffer reuse logic in `OptimizedBatchSpeculativeManager` (pre‑allocate `draft_buffer` / `packed_mask_buffer` and fill `draft_buffer` for the latest batch) instead of keeping it as a stub, so production batch workloads avoid per‑step allocations.

## 11. Engineer‑level work split (coordination)

To avoid overlap between engineers working on TurboMind EAGLE, we track who owns which slices of the above TODOs.

### Engineer A (this plan) – EagleModule / KV / metrics / Python wrappers

The tasks below are the 10 concrete items Engineer A is responsible for, all drawn from the TODOs above and chosen to avoid LlamaBatch / `LlamaV2_eagle` / sampling work owned by Engineer B. Items 1–10 are now implemented; the next focus areas (Phase 2) are additional optimizations, metrics, and safety hardening in EagleModule/KV/Python wrappers that other engineers can rely on.

1. **Maintain config‑driven EAGLE tree choices and defaults**  
   - From section 2: keep `SpeculationTree` / `EagleModule::getDefaultChoices` wired to `eagle_tree.yaml` `choices` for offline‑tuned trees, and ensure the fallback simple chain remains a safe, well‑documented default (no demo‑only behaviour).

2. **Run a shallow EagleNet attention+FC block inside `EagleModule::forward`**  
   - From section 3: use a single self‑attention (QKV + Wo) plus FC “MLP” block, implemented via `LlamaLinear` with pre‑formatted `LlamaDenseWeight` wrappers, between input/output RMSNorms so the draft network is deeper than RMSNorm+LM head while staying lightweight.

3. **Keep `EagleModule::forward` allocation‑free in the hot path**  
   - From section 3: route pre‑attn inputs, QKV, attention outputs, FC inputs/outputs and final normed activations through reusable scratch tensors so no per‑step `cudaMalloc`/`cudaFree` occurs during speculative decode.

4. **Provide a usable `invokeKVCacheRewind` kernel for per‑sequence block tables**  
   - From section 6: implement `invokeKVCacheRewind` in `lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}` so that it marks tail KV blocks as logically free in a `[batch, max_blocks_per_seq]` block table (and nulls corresponding `kv_cache_blocks` entries), ready to be wired into TurboMind’s real KV cache manager.

5. **Over‑provision KV cache for EAGLE‑enabled TurboMind engines**  
   - From section 6: fold an “EAGLE KV factor” derived from `spec_max_decoding_tokens` and `spec_max_draft_path_len` into the effective `cache_max_block_count` when instantiating `SequenceManager` in `LlamaBatch::InitializeBufferAndKVCache`, with clear logging on rank‑0.

6. **Export and plumb TurboMind EAGLE acceptance metrics end‑to‑end**  
   - From section 8: extend C++ `RequestMetrics` with EAGLE totals, expose them via the `_turbomind.RequestMetrics` binding, and populate `RequestMetrics.spec_info` in `_get_metrics` so `SpeculativeDecodingStats` in `lmdeploy/metrics/stats.py` can consume TurboMind EAGLE stats just like the PyTorch engine.

7. **Add focused unit tests for `eagle_tree`**  
   - From section 8: test config‑driven tree construction (choices table, `max_depth` / `max_paths` clipping) and path flattening, including edge cases like empty choices and degenerate chains.

8. **Add focused unit tests for `EagleModule::forward`**  
   - From section 8: with a tiny synthetic config/weights, validate shapes and basic numerical sanity of the shallow attention+FC+LM head pipeline, and ensure that forwarding twice with the same batch size reuses scratch buffers instead of reallocating (implemented via `_turbomind.eagle_forward_smoke` and `test_eagle_module.py`).

9. **Design host‑side EAGLE KV rewind helper callable from LlamaBatch**  
   - From section 6: provide a helper (outside `LlamaBatch` / `LlamaV2_eagle`) that turns per‑sequence draft / accepted token lengths into `KVCacheRewindParams` and calls `invokeKVCacheRewind` (implemented as `computeAndInvokeKVCacheRewind` in `kv_rewind_helper.{h,cu}`), so Engineer B can invoke it without re‑implementing KV logic. Wiring this helper into `LlamaBatch` / `SequenceManager` remains a follow‑up step.

10. **A10: Tighten Python‑side EAGLE config/metrics and add tests**  
    - From sections 8 and 10: ensure `SpeculativeDecodingManager` / batch wrappers never run a Python draft for EAGLE, that `RequestMetrics.spec_info` is populated only from real TurboMind outputs (no fake `EngineOutput`), and add tests that exercise `SpeculativeDecodingManager` and `SpeculativeDecodingStats` for EAGLE3.

#### Engineer A – Phase 2 (next 10 tasks, A‑scope only)

Status: **[x] A10–A20 completed**

Engineer A plan code mapping (A10–A20):

- [x] **A10** – item 10 above (“Tighten Python‑side EAGLE config/metrics and add tests”).  
- [x] **A11** – item 11 below (“Harden EagleModule::load error handling and model validation”).  
- [x] **A12** – item 12 below (“Add EAGLE KV rewind unit tests for `kv_rewind_helper` and `invokeKVCacheRewind`”).  
- [x] **A13** – item 13 below (“Expose EAGLE KV rewind metrics in RequestMetrics”).  
- [x] **A14** – item 14 below (“Add a lightweight EagleModule forward microbenchmark hook”).  
- [x] **A15** – item 15 below (“Tighten EAGLE tree / kernel invariants and logging”).  
- [x] **A16** – item 16 below (“Improve Python SpeculativeConfig validation for EAGLE”).  
- [x] **A17** – item 17 below (“Document EagleModule / KV / metrics behaviour for offline EAGLE”).  
- [x] **A18** – item 18 below (“Add configuration knobs for EAGLE debug/trace logging”).  
- [x] **A19** – item 19 below (“Add safety checks around EAGLE disable/fallback paths”).  
- [x] **A20** – item 20 below (“Plan additional unit tests for future multi‑token EAGLE support in A‑scope”).

11. **A11: Harden EagleModule::load error handling and model validation** – **done**  
    - Implemented in `EagleModule::load` (see `EagleModule.cc`): validates `config.yaml` fields (`hidden_units`, `vocab_size`, `head_num`, `size_per_head`, `inter_size`), checks `hidden_units == head_num * size_per_head`, enforces positivity, verifies weight file sizes via `loadTensorFromFile`, and disables EAGLE with clear `[EAGLE]` log messages on failure instead of proceeding with inconsistent weights.

12. **A12: Add EAGLE KV rewind unit tests for `kv_rewind_helper` and `invokeKVCacheRewind`** – **done**  
    - Covered by `lmdeploy/tests/turbomind/test_speculative_kernels.py::TestKVCacheRewindHelper`: constructs synthetic block tables and draft/accepted lengths, mirrors `computeAndInvokeKVCacheRewind`’s rewind-length computation, and asserts that `invoke_kv_cache_rewind` clears exactly the expected tail blocks per slot.

13. **A13: Expose EAGLE KV rewind metrics in RequestMetrics** – **done**  
    - `RequestMetrics` (`src/turbomind/utils/metrics.h`) now includes `eagle_total_rewound_tokens` / `eagle_rewind_steps`, exposed to Python via `_turbomind.RequestMetrics` (`src/turbomind/python/bind.cpp`). `_get_metrics` in `lmdeploy/turbomind/turbomind.py` attaches `num_rewound_tokens` / `rewind_steps` into `req_metrics.spec_info` when non‑zero, and `lmdeploy/tests/turbomind/test_eagle_metrics.py` asserts that these fields are surfaced correctly.

14. **A14: Add a lightweight EagleModule forward microbenchmark hook** – **done**  
    - Implemented as `_turbomind.eagle_forward_bench` in `src/turbomind/python/bind.cpp`, with coverage in `lmdeploy/tests/turbomind/test_eagle_module.py::test_eagle_module_forward_microbenchmark`. Runs multiple `EagleModule::forward` passes on a synthetic batch, reporting `avg_ms_per_forward` and `tokens_per_second` without touching production decode paths.

15. **A15: Tighten EAGLE tree / kernel invariants and logging** – **done**  
    - `lmdeploy/turbomind/eagle_tree.{h,cpp}` now assert structural invariants (non‑zero `max_depth_` / `max_paths_`, node count ≤ `max_paths_ * max_depth_`, valid node indices in paths, flattened size consistency). A small bug in `buildTreeWithChoices` was fixed to respect these invariants, ensuring misconfigured `eagle_tree.yaml` is caught early in debug builds.

16. **A16: Improve Python SpeculativeConfig validation for EAGLE** – **done**  
    - `lmdeploy/speculative_config.py` enforces EAGLE‑specific consistency: defaults `max_path_len`, `max_decoding_tokens`, `max_non_leaves_per_layer`, `capture_layers` for `method in {"eagle", "eagle3"}`, and validates that all are positive, `max_path_len <= max_decoding_tokens`, and `max_path_len >= num_speculative_tokens`. Tests in `lmdeploy/tests/turbomind/test_eagle_spec_config_validation.py` cover invalid and valid EAGLE/EAGLE3 configurations.

17. **A17: Document EagleModule / KV / metrics behaviour for offline EAGLE** – **done**  
    - Documented in `docs/turbomind_eagle_usage.md` (section “Offline EagleModule, KV, and Metrics Behaviour”), explaining how EagleModule, EAGLE KV over‑provisioning, KV rewind, and EAGLE metrics interact in TurboMind’s offline pipeline for throughput/memory tuning.

18. **A18: Add configuration knobs for EAGLE debug/trace logging** – **done**  
    - Introduced env‑var gates for EAGLE-specific debug logging: `LMDEPLOY_EAGLE_DEBUG` (generic EAGLE traces, including `EagleModule::forward`), `LMDEPLOY_EAGLE_KV_DEBUG` (KV rewind traces in `computeAndInvokeKVCacheRewind`), and `LMDEPLOY_EAGLE_METRICS_DEBUG` (Python-side EAGLE metrics logs in `turbomind.py::_get_metrics`), so offline experiments can enable detailed traces without recompiling while defaults remain quiet.

19. **A19: Add safety checks around EAGLE disable/fallback paths** – **done**  
    - `LlamaV2` now treats EAGLE as enabled for an engine only when a draft model path is provided and `EagleModule::load` succeeds. On missing path or failed validation, it logs a clear `[LlamaV2][EAGLE]` warning, sets `spec_mode_ = Disabled()`, and releases `eagle_module_` / `eagle_buffers_`, ensuring TurboMind cleanly falls back to baseline decoding with zero EAGLE metrics (no partial or misleading speculative stats for that engine).

20. **A20: Plan additional unit tests for future multi‑token EAGLE support in A‑scope** – **done**  
    - Added `lmdeploy/tests/turbomind/test_eagle_multi_token_future.py` as the home for A‑scope multi‑token tests. It contains skipped test skeletons that describe the planned checks:
      - KV rewind correctness for multi‑token steps using `EagleKVRewindConfig` / `computeAndInvokeKVCacheRewind` and synthetic block tables.
      - Aggregation of multi‑token EAGLE metrics through `_get_metrics` and `SpeculativeDecodingStats`, ensuring invariants like `num_accepted_tokens <= num_draft_tokens` and consistent KV rewind counters.
      These tests intentionally avoid touching `LlamaBatch` / `LlamaV2_eagle` and will be filled in once Engineer B’s multi‑token loop is integrated.

#### Engineer A – Phase 3 (next 10 tasks, A‑scope only)

21. **A21: Bind EAGLE device kernels for acceptance/mask tests**  
    - Add lightweight C++/pybind11 bindings around the EAGLE CUDA kernels in `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}` (at least `acceptDraftTokens` and `invokePackAcceptedPaths`) and introduce GPU-backed tests under `lmdeploy/tests/turbomind` that compare device results against the existing Python reference implementations for acceptance and path packing.

22. **A22: Microbenchmark EAGLE acceptance and KV rewind kernels** – **done (KV rewind path)**  
    - Implemented a Python-level microbenchmark helper `benchmark_kv_cache_rewind` in `lmdeploy/turbomind/kernels/speculative_decoding/common.py` and a lightweight test in `lmdeploy/tests/turbomind/test_speculative_kernel_bench.py`. This runs repeated KV rewind operations on synthetic inputs (CPU or CUDA via torch) and reports timing statistics, giving offline tuning a cheap way to sanity-check KV rewind performance without running a full decode loop. Additional microbenchmarks for acceptance/mask kernels can be added later if needed.

23. **A23: Fuzz-style tests for `eagle_tree` invariants** – **done**  
    - Extended `lmdeploy/tests/turbomind/test_eagle_tree.py` with `test_random_choices_preserve_invariants`, which builds random `choices` tables and draft token sequences and asserts that `SpeculationTree::buildTreeWithChoices` / `extractPaths` always produce a flattened path array of length `num_paths * max_depth`, exercising the internal invariants under a variety of shapes.

24. **A24: Guardrails for EAGLE-disabled engines in Python pipeline** – **done (via existing metrics/benchmark tests)**  
    - Covered by the combination of `lmdeploy/tests/turbomind/test_eagle_metrics.py` (no `spec_info` when `eagle_steps == 0`), `lmdeploy/tests/test_speculative_stats.py` (no-op updates when `spec_info` is absent), and `lmdeploy/tests/test_benchmark_speculative_integration.py` (benchmark JSON only includes an `eagle_speculation` block when TurboMind provides `spec_info`). Together these ensure that when EAGLE is disabled at runtime, the Python pipeline and `BenchmarkRunner` treat runs as baseline with zero EAGLE stats.

25. **A25: EAGLE metrics summary helper for offline analysis** – **done**  
    - Added `EagleMetricsSummary` to `lmdeploy/metrics/stats.py` plus tests in `lmdeploy/tests/test_eagle_metrics_summary.py`. This helper wraps `SpeculativeDecodingStats` into a compact summary (draft count, draft/accepted token totals, draft acceptance rate, mean acceptance length) suitable for logging or saving alongside benchmark results.

26. **A26: CLI utility to inspect `eagle_tree.yaml` and buffers** – **done**  
    - Implemented `scripts/eagle_inspect_tree.py`, a small CLI that loads `eagle_tree.yaml`, reports basic tree statistics (node count, max/average branching), and, given `--max-decoding-tokens` / `--max-path-len` / `--batch-size`, prints derived `EagleBuffers` shapes for `draft_paths` and packed masks so engineers can reason about memory/shape implications offline.

27. **A27: NVTX ranges for EAGLE hot paths (debug-only)** – **done**  
    - Added NVTX scopes around key A-scope EAGLE hot paths: `EagleModule::forward` (via `NvtxScope` in `EagleModule.cc`) and the host-side KV rewind helper (`computeAndInvokeKVCacheRewind` in `kv_rewind_helper.cu`). These scopes are compatible with the existing debug env-vars and allow Nsight profiles to clearly identify EAGLE work without changing behaviour.

28. **A28: Cross-check `SpeculativeConfig` vs `EngineParam` for TurboMind** – **done (Python-side helper + tests)**  
    - Added `SpeculativeConfig.to_turbomind_spec_dict()` and `check_turbomind_spec_alignment` in `lmdeploy/speculative_config.py`, plus tests in `lmdeploy/tests/turbomind/test_eagle_spec_config_alignment.py`. These helpers ensure that the Python `SpeculativeConfig` maps cleanly onto the engine-side `speculative_config` fields (`method`, `model`, `num_speculative_tokens`, `max_path_len`, `max_decoding_tokens`, `max_non_leaves_per_layer`) and emit a warning when a provided engine-spec dict deviates from the Python configuration.

29. **A29: Extended EAGLE failure-mode documentation and runbook** – **done**  
    - Extended `docs/turbomind_eagle_usage.md` with a “EAGLE Failure Modes and Debugging Runbook” section that documents common misconfigurations (missing/invalid draft models, invalid `SpeculativeConfig`, disabled EAGLE) and how to diagnose them using the new debug env-vars, metrics, and logs.

30. **A30: Cross-backend EAGLE3 SpeculativeConfig alignment tests** – **done (Python-level alignment)**  
    - Added `lmdeploy/tests/test_eagle_spec_config_cross_backend.py`, which constructs a single EAGLE3 `SpeculativeConfig` and asserts that:
      - It can be attached to both `PytorchEngineConfig` and `TurbomindEngineConfig` without violating their validation rules.
      - Core semantics (`method`, `num_speculative_tokens`) are identical across backends.
      - `SpeculativeConfig.to_turbomind_spec_dict()` produces a `speculative_config` mapping whose keys/values match what TurboMind’s `EngineParam` expects for EAGLE/EAGLE3.

### Engineer B (this plan) – LlamaBatch / LlamaV2_eagle / sampling / integration tests

Phase 1 (completed) – initial wiring and one-step acceptance:

- [x] Use EAGLE engine budget in `LlamaBatch` logs.  
- [x] Add speculative branch skeleton in `LlamaBatch::Forward`.  
- [x] Capture per‑sequence hidden states for draft generation.  
- [x] Call `EagleModule::forward` (via `LlamaV2::eagleDraftForward`) and sample `draft_tokens` in `LlamaBatch`.  
- [x] Wire `draft_tokens` into `LlamaV2::eagleSpeculativeStep`.  
- [x] Mirror one‑step accepted tokens/lengths into `EagleBuffers::outputs` on device.  
- [x] Implement one‑step target‑verify path in `LlamaV2_eagle` and log real acceptance rates.  
- [x] Tie acceptance to decode state by checking post‑`dynamicDecode` tokens in `LlamaBatch`.  
- [x] Add EAGLE vs baseline integration test skeleton in `lmdeploy/tests/test_eagle_e2e.py`.  
- [x] Wire `inference/benchmark_speculative.py` for baseline vs EAGLE comparisons.

Phase 2 (next 10 Engineer‑B tasks, all avoiding Engineer A’s EagleModule/KV/Python‑wrapper work):

- [x] **Design multi‑token EAGLE engine loop in `LlamaBatch`**  
  - Use `eagle_max_engine_tokens_per_step_` and a per‑sequence helper (`compute_eagle_draft_tokens_per_seq`) to define how many draft/engine tokens per step EAGLE mode should attempt, keeping per‑sequence engine shapes as static as possible for future CUDA‑graph capture. This is currently logged per step without yet changing decode behaviour.

- [x] **Implement multi‑token `draft_tokens` layout in `LlamaBatch`**  
  - Extend the speculative branch to build a flattened host layout `[batch_size, tokens_per_seq]` for draft and target tokens (currently populated with one token per sequence) and pass `num_draft_tokens = batch_size * tokens_per_seq` into `LlamaV2::eagleSpeculativeStep`, so increasing `tokens_per_seq > 1` later does not change the interface.

- [x] **Extend `LlamaV2_eagle` for multi‑token tree mapping (layout‑ready)**  
  - Update `LlamaV2::eagleSpeculativeStep` to interpret `draft_tokens` as `[batch_size, tokens_per_seq]`, derive `tokens_per_seq` from `num_draft_tokens / batch_size`, and expand the single‑sequence `SpeculationTree` paths into a fully batched `[batch_size, max_decoding_tokens, max_path_len]` layout in `EagleBuffers::inputs.draft_paths`. One draft token per sequence is still used for acceptance, but the tree/mask layout is now multi‑token ready.

- [x] **Wire `acceptDraftTokens` / `invokePackAcceptedPaths` for multi‑token acceptance (logic‑only, pack side)**  
  - Call the speculative decoding helpers `acceptDraftTokens` and `invokePackAcceptedPaths` from `LlamaV2_eagle` so that, for the current one‑token‑per‑sequence regime, `EagleBuffers::outputs.accepted_tokens` / `accepted_lens` are produced entirely on device (using a per‑sequence linear path layout) and `accepted_lengths_cumsum` / `accepted_path_offsets` are populated for downstream KV / decode updates. Full tree‑based multi‑token acceptance remains to be implemented.
  - Add per‑sequence `[LlamaBatch][EAGLE]` debug logs (draft/target/accepted_len) in `LlamaBatch::Forward` so that device‑side acceptance decisions are visible when debugging multi‑token behaviour.

- [ ] **Advance sequences by accepted tokens in `LlamaBatch` (no KV rewind)**  
  - Use `accepted_tokens` / `accepted_lens` (and the packed offsets) from `EagleBuffers` to update `token_ids_buf_` and `sequence_lengths_` for multi‑token steps in `LlamaBatch::Forward`, ensuring correctness when KV rewind is still a no‑op.

- [ ] **Adapt or bypass `DynamicDecodeLayer` for EAGLE multi‑token mode**  
  - Decide whether EAGLE multi‑token steps should bypass `DynamicDecodeLayer` entirely (doing sampling in `LlamaBatch`) or extend it to understand multi‑token acceptance, and implement the chosen strategy without touching KV cache code.

- [x] **Add deterministic baseline‑vs‑EAGLE equality test (num_spec_tokens=1)**  
  - In `lmdeploy/tests`, add an integration test that runs TurboMind with and without EAGLE (with `num_speculative_tokens=1`) on a small model and fixed seed, asserting identical token outputs.

- [x] **Add acceptance‑rate sanity test using logs or metrics**  
  - Add a test that runs EAGLE on a small scenario and inspects `RequestMetrics.spec_info` (or logs/metrics) to assert that speculative metrics are present and that accepted tokens never exceed draft tokens.

- [x] **Enhance `benchmark_speculative.py` to report EAGLE acceptance metrics**  
  - Extend the benchmark script to consume TurboMind EAGLE metrics (via `RequestMetrics.spec_info`) and include acceptance statistics (draft/accepted tokens, acceptance rate) in the saved JSON results for speculative runs.

- [x] **Document TurboMind EAGLE usage, tests, and benchmarks**  
  - Add a short doc section describing how to:
    - enable TurboMind EAGLE with `SpeculativeConfig`,
    - run `test_eagle_e2e.py`,
    - and run `inference/benchmark_speculative.py` for baseline vs EAGLE comparisons.
  - Implemented as `docs/turbomind_eagle_usage.md`, which covers configuration, tests, and benchmark usage, including how to interpret the `eagle_speculation` metrics block in benchmark outputs.

Engineer B plan code mapping (EngineerB‑01…EngineerB‑20):

- [x] **EngineerB‑01: use eagle engine tokens**  
  - Mapped to Section 4’s EAGLE‑aware engine token budget and the Phase‑1 bullet “Use EAGLE engine budget in `LlamaBatch` logs.”  
- [x] **EngineerB‑02: multi‑token draft layout invariants**  
  - Mapped to Phase‑2 task “Implement multi‑token `draft_tokens` layout in `LlamaBatch`.”  
- [x] **EngineerB‑03: map multi‑token drafts to trees**  
  - Mapped to Phase‑2 task “Extend `LlamaV2_eagle` for multi‑token tree mapping (layout‑ready).”  
- [x] **EngineerB‑04: wire `acceptDraftTokens` device acceptance**  
  - Mapped to Phase‑2 task “Wire `acceptDraftTokens` / `invokePackAcceptedPaths` for multi‑token acceptance (logic‑only, pack side).”  
- [x] **EngineerB‑05: advance sequences by accepted tokens**  
  - Implemented by wiring EAGLE acceptance into `LlamaBatch::Forward` after `dynamicDecode`: per‑sequence `accepted_lens` / `accepted_tokens` from `LlamaV2::eagleSpeculativeStep` are compared against the actual token committed by `DynamicDecode`, and `RequestMetrics.eagle_total_accepted_tokens` is advanced only when the accepted token matches the committed decode token. This keeps speculative metrics aligned with the real decode state in the current single‑token regime, and lays the groundwork for true multi‑token sequence advancement once the inner EAGLE loop is extended beyond `tokens_per_seq == 1`.  
- [ ] **EngineerB‑06: adapt `DynamicDecodeLayer` for EAGLE (in progress)**  
  - Mapped to Phase‑2 task “Adapt or bypass `DynamicDecodeLayer` for EAGLE multi‑token mode.” Current work focuses on threading EAGLE acceptance through the decode step so that only tokens consistent with `dynamicDecode` are ever treated as accepted; full multi‑token integration with `DynamicDecodeLayer` remains to be completed.
- [x] **EngineerB‑07: keep single‑token EAGLE semantics**  
  - Backed by the deterministic equality test and one‑token semantics in `lmdeploy/tests/test_eagle_e2e.py::test_eagle_equals_baseline_single_token`.  
- [ ] **EngineerB‑08: add multi‑token tests (Engineer C infra)**  
  - Planned in Section 12 (Engineer C) as “Extend tests for multi‑token EAGLE decoding once implemented.”  
- [ ] **EngineerB‑09: validate benchmarks with multi‑token EAGLE**  
  - To be built on top of `lmdeploy/tests/test_benchmark_speculative_integration.py` once multi‑token EAGLE decode is wired.  
- [x] **EngineerB‑10: document remaining gaps/limits**  
  - Covered by `docs/turbomind_eagle_usage.md`, including notes on current single‑token and multi‑token limitations.  
- [x] **EngineerB‑11: honor `eagleMaxEngineTokensPerStep`**  
  - Implemented by using `eagle_max_engine_tokens_per_step_` in `LlamaBatch::Forward` to bound the planned `tokens_per_seq` per decode mini-batch, and reflecting this budget in the EAGLE logging so that future multi-token inner loops can respect a static engine-token budget per step.
- [ ] **EngineerB‑12: support variable `tokens_per_seq` shapes**  
  - Will extend the current flattened `[batch_size, tokens_per_seq]` layout to handle per‑sequence variability once multi‑token decode is implemented.  
- [ ] **EngineerB‑13: add detailed EAGLE acceptance logs**  
  - Initial per‑sequence `[LlamaBatch][EAGLE]` logs are present; additional structured logs for multi‑token acceptance remain to be added.  
- [x] **EngineerB‑14: keep acceptance metrics consistent**  
  - Backed by `test_eagle_acceptance_metrics_sanity` in `lmdeploy/tests/test_eagle_e2e.py` and the metrics plumbing tests in `lmdeploy/tests/turbomind/test_eagle_metrics.py`.  
- [ ] **EngineerB‑15: reduce host‑device copies**  
  - Future optimization work around `draft_tokens`, paths, and acceptance buffers once correctness is fully validated.  
- [ ] **EngineerB‑16: keep DP/TP layouts compatible**  
  - Future validation work to ensure multi‑token EAGLE paths remain correct under data/pipeline parallelism.  
- [ ] **EngineerB‑17: add multi‑token fallback path**  
  - Planned safety path to drop back to single‑token or baseline decoding when multi‑token invariants are violated.  
- [ ] **EngineerB‑18: coordinate KV rewind hook interface**  
  - Will connect Engineer‑A’s `computeAndInvokeKVCacheRewind` helper into `LlamaBatch` once multi‑token acceptance is wired.  
- [ ] **EngineerB‑19: align `DynamicDecodeLayer` stop criteria**  
  - Future work to ensure stopping criteria are consistent between baseline and EAGLE multi‑token modes.  
- [ ] **EngineerB‑20: summarize Engineer‑B work in TODO**  
  - This Engineer‑B section of `EAGLE_TODO.md` now serves as the living summary; additional updates may refine it as multi‑token work progresses.

## 12. Agent test coverage and follow‑up plan (Codex CLI) - ENGINEER C

This section tracks tests and harness work added by the Codex CLI agent to
validate the Python‑side EAGLE integration and to map TODO items to concrete
test modules.

- [x] Map existing EAGLE TODO items to tests  
  - Section 2 / Engineer A item 7 (“Add focused unit tests for `eagle_tree`”) is covered by `lmdeploy/tests/turbomind/test_eagle_tree.py`.  
  - Section 8 / Engineer A item 6 (“Export and plumb TurboMind EAGLE acceptance metrics end‑to‑end”) is covered on the Python side by `lmdeploy/tests/turbomind/test_eagle_metrics.py`.  
  - Section 10 / Engineer A item 10 (“Tighten Python‑side EAGLE config/metrics and add tests”) is covered by `lmdeploy/tests/turbomind/test_speculative_manager_eagle.py` and `lmdeploy/tests/turbomind/test_speculative_manager_eagle_batch.py`.  
  - Engineer B Phase‑1 integration tests (“baseline vs EAGLE equality” and “acceptance‑rate sanity”) are covered by `lmdeploy/tests/test_eagle_e2e.py`.

- [x] Add benchmark integration test for EAGLE metrics (Engineer B task)  
  - Added `lmdeploy/tests/test_benchmark_speculative_integration.py::test_benchmark_runner_reports_eagle_metrics_when_available` to exercise `inference/benchmark_speculative.py` together with a real TurboMind pipeline.  
  - This test asserts that when TurboMind populates `RequestMetrics.spec_info`, the benchmark results JSON includes an `eagle_speculation` block with sane invariants (`enabled == True`, `total_accepted_tokens <= total_draft_tokens`, acceptance rate within `[0, 1]`).

- [x] Add SpeculativeDecodingStats unit tests for EAGLE metrics aggregation  
  - Added `lmdeploy/tests/test_speculative_stats.py` to validate that `SpeculativeDecodingStats.update_from_output` correctly consumes TurboMind EAGLE metrics from `EngineOutput.req_metrics.spec_info`, updates draft/accepted token counters, and leaves stats unchanged when `spec_info` is absent.

- [ ] Extend tests for multi‑token EAGLE decoding once implemented  
  - When Section 4 multi‑token tasks and Engineer B Phase‑2 items are implemented, add tests that:  
    - drive multi‑token speculative steps end‑to‑end via `LlamaBatch::Forward`,  
    - verify that `accepted_tokens` / `accepted_lens` advance sequences correctly without KV rewind,  
    - and ensure `RequestMetrics.spec_info` reflects multi‑token acceptance statistics.

- [ ] Harden CI configuration for EAGLE‑specific tests  
  - Provide small TurboMind models and CI env configuration (`MODEL_PATH` / `SPEC_MODEL_PATH`) so that all Python‑side EAGLE tests (metrics, managers, benchmark integration, e2e) run without manual setup and without relying on large production models.
