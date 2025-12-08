# TurboMind EAGLE Integration TODO

This file tracks all work needed to bring TurboMind’s EAGLE speculative decoding to full, production use. Items marked `[x]` are already implemented in this repo; items marked `[ ]` are pending.

Status Legend:
- ✅ Implemented + tested (production-ready for current scope)
- 🧪 Implemented (prototype; GPU/CI validation pending or limited in scope)
- ⏳ Planned (not implemented)

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

## 11. Implementation work split (coordination)

To avoid overlap between engineers working on TurboMind EAGLE, we track who owns which slices of the above TODOs.

### Module / KV / metrics / Python wrappers

The tasks below are the 10 concrete items for the module/KV/metrics/Python wrapper layer, all drawn from the TODOs above and chosen to avoid LlamaBatch / `LlamaV2_eagle` / sampling work owned by other parts of the system. Items 1–10 are now implemented; the next focus areas (Phase 2) are additional optimizations, metrics, and safety hardening in EagleModule/KV/Python wrappers that other engineers can rely on.

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
   - From section 6: provide a helper (outside `LlamaBatch` / `LlamaV2_eagle`) that turns per‑sequence draft / accepted token lengths into `KVCacheRewindParams` and calls `invokeKVCacheRewind` (implemented as `computeAndInvokeKVCacheRewind` in `kv_rewind_helper.{h,cu}`), so decode logic can invoke it without re‑implementing KV logic. Wiring this helper into `LlamaBatch` / `SequenceManager` remains a follow‑up step.

10. **A10: Tighten Python‑side EAGLE config/metrics and add tests**  
    - From sections 8 and 10: ensure `SpeculativeDecodingManager` / batch wrappers never run a Python draft for EAGLE, that `RequestMetrics.spec_info` is populated only from real TurboMind outputs (no fake `EngineOutput`), and add tests that exercise `SpeculativeDecodingManager` and `SpeculativeDecodingStats` for EAGLE3.

#### Phase 2 (next 10 tasks, module/KV/metrics)

Status: **[x] A10–A20 completed**

Module/KV/metrics plan code mapping (A10–A20):

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
      These tests intentionally avoid touching `LlamaBatch` / `LlamaV2_eagle` and will be filled in once the multi‑token loop is integrated.

#### Phase 3 (next 10 tasks, module/KV/metrics)

21. **A21: Bind EAGLE device kernels for acceptance/mask tests** – **🧪 prototype (GPU/CI validation pending)**  
    - Added lightweight C++/pybind11 bindings around the EAGLE CUDA kernels in `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}` (at least `acceptDraftTokens` and `invokePackAcceptedPaths`) and introduced GPU-backed tests under `lmdeploy/tests/turbomind/test_speculative_kernels.py` that compare device results against the existing Python reference implementations for acceptance and path packing.  
    - **Scope:** Not required for baseline decode; bindings are primarily for tests/benchmarks and are not a hard dependency for production EAGLE paths (core kernels are already used from C++).  
    - **CI:**  
      - Run `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestAcceptDraftTokensDevice` with CUDA and a built `_turbomind` extension.  
      - Run `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestPackAcceptedPathsDevice` with CUDA and `_turbomind`.  
      - Run `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestAcceptanceStatsDevice` with CUDA and `_turbomind`.  
    - **Note:** Feature is experimental; do not rely on the Python bindings for production decode paths until the above CI jobs are green.

22. **A22: Microbenchmark EAGLE acceptance and KV rewind kernels** – **🧪 prototype (coverage limited to small shapes)**  
    - Implemented a Python-level microbenchmark helper `benchmark_kv_cache_rewind` in `lmdeploy/turbomind/kernels/speculative_decoding/common.py` and a lightweight test in `lmdeploy/tests/turbomind/test_speculative_kernel_bench.py`. This runs repeated KV rewind operations on synthetic inputs (CPU or CUDA via torch) and reports timing statistics, giving offline tuning a cheap way to sanity-check KV rewind performance without running a full decode loop.  
    - Additional microbenchmarks for acceptance/mask kernels (`benchmark_accept_draft_tokens`, `benchmark_pack_accepted_paths`) have been added as well; these rely on `_turbomind` when running on CUDA and are validated by small sanity tests but still benefit from broader GPU/CI coverage on larger shapes.  
    - **CI:**  
      - Run `pytest lmdeploy/tests/turbomind/test_speculative_kernel_bench.py` with CUDA and `_turbomind` built, ensuring both KV rewind and acceptance/pack benchmarks run on GPU.  
      - Optionally extend CI to capture benchmark outputs (e.g. as artifacts) for basic perf regression checks.

23. **A23: Fuzz-style tests for `eagle_tree` invariants** – **✅ implemented + tested (CPU-only)**  
    - Extended `lmdeploy/tests/turbomind/test_eagle_tree.py` with fuzz-style tests (`test_random_choices_preserve_invariants` plus larger variants) that build random `choices` tables and draft token sequences and assert that `SpeculationTree::buildTreeWithChoices` / `extractPaths` always produce a flattened path array of length `num_paths * max_depth`, exercising the internal invariants under a variety of shapes.  
    - These tests are CPU-only and already run in standard CI; no GPU dependency here.

24. **A24: Guardrails for EAGLE-disabled engines in Python pipeline** – **done (via existing metrics/benchmark tests)**  
    - Covered by the combination of `lmdeploy/tests/turbomind/test_eagle_metrics.py` (no `spec_info` when `eagle_steps == 0`), `lmdeploy/tests/test_speculative_stats.py` (no-op updates when `spec_info` is absent), and `lmdeploy/tests/test_benchmark_speculative_integration.py` (benchmark JSON only includes an `eagle_speculation` block when TurboMind provides `spec_info`). Together these ensure that when EAGLE is disabled at runtime, the Python pipeline and `BenchmarkRunner` treat runs as baseline with zero EAGLE stats.

25. **A25: EAGLE metrics summary helper for offline analysis** – **done**  
    - Added `EagleMetricsSummary` to `lmdeploy/metrics/stats.py` plus tests in `lmdeploy/tests/test_eagle_metrics_summary.py`. This helper wraps `SpeculativeDecodingStats` into a compact summary (draft count, draft/accepted token totals, draft acceptance rate, mean acceptance length) suitable for logging or saving alongside benchmark results.

26. **A26: CLI utility to inspect `eagle_tree.yaml` and buffers** – **✅ implemented (offline-only tool)**  
    - Implemented `scripts/eagle_inspect_tree.py`, a small CLI that loads `eagle_tree.yaml`, reports basic tree statistics (node count, max/average branching), and, given `--max-decoding-tokens` / `--max-path-len` / `--batch-size`, prints derived `EagleBuffers` shapes for `draft_paths` and packed masks so engineers can reason about memory/shape implications offline.  
    - Later extended with rough KV-block usage estimates (`--block-size`) to help reason about speculative KV over-provisioning. This script is pure Python/CPU and does not depend on CUDA, but its assumptions should be revisited periodically as EAGLE buffer layouts evolve.

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

#### Future Phase (A31+ prototypes – GPU validation pending)

31. **A31: Tree-based device acceptance kernel (paths-level)** – **🧪 prototype (GPU/CI validation pending)**  
    - Added `invokeTreeAcceptByIdsWithPaths` in `lmdeploy/turbomind/kernels/speculative_decoding/tree_accept_kernels.{h,cu}` to evaluate SpeculationTree paths entirely on device:
      - For each sequence, walks all candidate paths in a batched `paths` tensor `[maxBatchSize, numPaths, maxPathLen]` using the same acceptance rule as the existing host-side logic in `LlamaV2_eagle.cc` (accept while `draft_id == target_id`, include the first mismatching target token).
      - Produces `best_path_ids` and `accepted_lens` per slot, and materializes accepted target tokens for the best path into `accepted_tokens`.  
    - This kernel is currently only used via test/benchmark bindings (not the production decode loop) and requires **GPU-backed unit tests in CI (including different batch sizes, `num_paths`, and `max_path_len`)** before being wired into `LlamaV2_eagle` or treated as production-ready.  
    - **Scope:** Not wired into `LlamaBatch` / `LlamaV2_eagle`; only reachable via `_turbomind` bindings and dedicated tests/benchmarks. Feature is experimental and should not be relied upon for production decode paths yet.  
    - **CI:**  
      - Build `_turbomind` with CUDA and run  
        `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestTreeAcceptTokensDevice`.

32. **A32: Pybind bindings and tests for tree-accept kernel** – **🧪 prototype (GPU/CI validation pending)**  
    - Exposed the tree-accept kernel to Python as `_turbomind.eagle_tree_accept_tokens` in `src/turbomind/python/bind.cpp`, accepting CUDA tensors for `draft_ids`, `target_ids`, `paths`, `best_path_ids`, `accepted_lengths`, `accepted_tokens`, and optional `batch_slots`.  
    - Added GPU-backed tests in `lmdeploy/tests/turbomind/test_speculative_kernels.py::TestTreeAcceptTokensDevice` that compare `_turbomind.eagle_tree_accept_tokens` against a Python reference mirroring the host-side acceptance logic. These tests are guarded with `torch.cuda.is_available()` and will be **skipped** on CPU-only environments; they must be exercised in a CUDA-enabled CI configuration (with a built `_turbomind` extension) before we mark A31/A32 as fully validated.  
    - **Scope:** Binding is not used in production EAGLE paths; it exists for A-scope validation and microbenchmarks only.  
    - **CI:**  
      - Same as A31: run `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestTreeAcceptTokensDevice` with CUDA and `_turbomind` built.

33. **A33: Prepare kernels for variable `tokens_per_seq` layouts** – **🧪 prototype (design scaffolding; validation pending)**  
    - The existing acceptance and packing kernels (`acceptDraftTokens`, `invokePackAcceptedPaths`, and the new tree-accept helper) are written against batched layouts with explicit `batch_slots` mappings and max-size parameters (`max_batch_size`, `max_draft_tokens`, `num_paths`, `max_path_len`), which makes them structurally compatible with future variable `tokens_per_seq` schemes (unused positions can be denoted via sentinel indices such as `-1`).  
    - No production call sites have been updated yet to drive truly per-sequence `tokens_per_seq` variation; additional work (and tests) will be needed to:
      - thread per-sequence token counts into the relevant helpers, and  
      - validate correctness under non-uniform layouts via new GPU-backed tests (once CI has a CUDA build).  
    - For now, these kernels should be considered **prototype-ready but not fully validated for variable per-sequence layouts**; downstream engineers should treat them as experimental until dedicated GPU tests (including multi-token end-to-end scenarios, TP/PP/DP configurations, and KV rewind integration) are added and passing in CI.  
    - **Scope:** No production call sites currently drive truly per-sequence `tokens_per_seq` variation; the design is in place, but wiring + tests are still TODO.  
    - **CI (future):**  
      - Add multi-token E2E tests in `lmdeploy/tests/turbomind/test_eagle_multi_token_future.py` once the multi-token loop is ready.  
      - Extend `test_benchmark_speculative_integration.py` to cover multi-token EAGLE runs and validate metrics/KV rewind invariants under variable `tokens_per_seq`.

34. **A34: Offline TurboMind pipeline examples for EAGLE** – **🧪 implemented (examples; manual GPU run required)**  
    - Updated `lmdeploy/examples/speculative_decoding_example.py` to use the high-level `lmdeploy.pipeline` TurboMind backend together with `TurbomindEngineConfig` and `SpeculativeConfig` for:
      - `draft_target` speculative decoding,
      - `eagle3` EAGLE speculative decoding (including aggregation of `SpeculativeDecodingStats` and `EagleMetricsSummary` from pipeline responses), and
      - `ngram` speculative decoding.  
    - The script now demonstrates an end-to-end offline EAGLE/EAGLE3 run (kernels → EagleModule → TurboMind pipeline → Python metrics) without touching `LlamaBatch` / `LlamaV2_eagle`, and makes `req_metrics.spec_info` usage explicit for EAGLE runs.  
    - **Scope:** Example-only; it assumes real TurboMind model artifacts at the placeholder paths and is intended for offline experimentation and debugging rather than CI.  
    - **CI / testing:**  
      - Syntax/packaging is covered by `python -m compileall lmdeploy/examples/speculative_decoding_example.py`.  
      - Full runs should be exercised on a CUDA-enabled machine with small TurboMind models by engineers or future GPU CI (no dedicated pytest currently).

35. **A35: Print EAGLE metrics in `benchmark_speculative.py` CLI** – **✅ implemented (leverages existing JSON tests)**  
    - Extended `inference/benchmark_speculative.py::BenchmarkRunner.run_test_scenario` to print `mean_acceptance_rate` and `mean_acceptance_length` whenever the `eagle_speculation` block is present in the benchmark results, so offline TurboMind benchmarks surface EAGLE acceptance behaviour directly in stdout alongside throughput/latency/memory.  
    - This keeps the JSON schema unchanged (the `eagle_speculation` block is still produced by `EagleMetricsSummary.to_dict()` plus an `enabled` flag) and simply makes EAGLE metrics more visible during manual benchmarking.  
    - **Scope:** Purely a reporting enhancement; no changes to engine wiring, kernels, or metrics computation.  
    - **CI / testing:**  
      - Structural/semantic checks for the `eagle_speculation` block remain covered by `lmdeploy/tests/test_benchmark_speculative_integration.py::test_benchmark_runner_reports_eagle_metrics_when_available` (run in CUDA-enabled CI with `MODEL_PATH` / `SPEC_MODEL_PATH` set).  
      - The new print path is best-effort and does not require additional automated tests.

36. **A36: Harden core EAGLE kernels for edge cases** – **🧪 prototype (GPU/CI validation pending)**  
    - Tightened the EAGLE acceptance, packing, tree-accept, and KV-rewind kernels in `lmdeploy/turbomind/kernels/speculative_decoding/common.{h,cu}` and `tree_accept_kernels.{h,cu}` to handle edge conditions more robustly:
      - `acceptDraftTokensKernel` now guards against out-of-range `batch_slots`, negative `best_path_id` (treated as "no best path"), and path indices outside `[0, max_draft_tokens)`, defaulting to zero accepted tokens for such slots instead of reading out of bounds.  
      - `packAcceptedPathsKernel` now validates `batch_slots` against `max_batch_size`, skips slots with invalid `best_path_id` or non-positive `accepted_len`, and clamps writes to the valid range implied by `accepted_lengths_cumsum`, treating negative path indices as terminators.  
      - `kvCacheRewindKernel` now uses `max_batch_size` to ignore out-of-range slots and early-returns when batch size, block size, or max_blocks_per_seq are nonsensical, avoiding stray writes to block_tables/kv_cache_blocks.  
      - A new GPU test in `lmdeploy/tests/turbomind/test_speculative_kernels.py::TestAcceptDraftTokensDevice::test_device_accept_handles_empty_paths_and_negative_best_path` validates that `best_path_id == -1` and all-`-1` paths result in zero accepted tokens and unchanged sequence lengths.  
    - **Scope:** Kernel behaviour remains backward compatible for valid inputs; changes only add guards for malformed or partially-initialized buffers commonly hit during experimentation.  
    - **CI / testing:**  
      - Run `pytest lmdeploy/tests/turbomind/test_speculative_kernels.py::TestAcceptDraftTokensDevice` and `::TestPackAcceptedPathsDevice` on CUDA with a built `_turbomind` extension to exercise the hardened kernels on device.  
      - Add future GPU tests for KV rewind edge cases once `computeAndInvokeKVCacheRewind` is fully wired into decode loops.

37. **A37: Expose EagleModule buffer / tree introspection helpers** – **✅ implemented (internal utility)**  
    - Extended `EagleModule` with additional read-only accessors used by A-scope tooling and potential future buffer sizing helpers:
      - `getHiddenUnits()` and `getVocabSize()` expose the draft model dimensions inferred from `config.yaml`.  
      - `getMaxTreeNodes()` reports `max_decoding_tokens * max_draft_path_len`, a conservative upper bound on per-step tree nodes for sizing SpeculationTree/EagleBuffers.  
    - Strengthened `EagleModule::forward` checks:
      - If `enabled_` is false, `forward` now logs a clear warning and acts as a pass-through on `hidden_states` instead of attempting a partial draft forward.  
      - Existing hidden-dim mismatch warnings remain, but now run under an explicit "module disabled / weights not initialized" guard for cleaner failure modes.  
    - **Scope:** No behaviour changes for correctly configured engines; the new accessors are used only by diagnostics and potential future tooling.  
    - **CI / testing:** Covered implicitly by existing EAGLE unit/integration tests that exercise `EagleModule::load`/`forward` via TurboMind; no additional tests required in this pass.

38. **A38: SpeculativeConfig/EAGLE runtime validation helper** – **✅ implemented (warning-only, offline use)**  
    - Added `validate_eagle_runtime_config(engine_config, spec_cfg)` to `lmdeploy/speculative_config.py`. This helper performs non-fatal runtime checks for EAGLE/EAGLE3 configurations:
      - Verifies that a non-None `SpeculativeConfig` with `method in {"eagle", "eagle3"}` is paired with an engine_config that has a `speculative_config`.  
      - When an engine-side speculative config object is available, it attempts to obtain a dict (via `to_turbomind_spec_dict` or `__dict__`) and calls `check_turbomind_spec_alignment` to warn on drifts between Python and engine settings.  
      - Emits a warning when `engine_config.enable_metrics` is False, since this would suppress EAGLE metrics on `req_metrics.spec_info`.  
    - **Scope:** Helper is opt-in and side-effect free with respect to decode logic; it is intended for offline inspection tools, examples, and debug scripts.  
    - **CI / testing:** Basic import/compile sanity is covered; callers should add targeted tests around warning behaviour as needed.

39. **A39: Offline TurboMind EAGLE inspect helper (Python)** – **🧪 implemented (requires GPU / real models)**  
    - Introduced `lmdeploy/turbomind/eagle_inspect.py::inspect_offline_eagle`, a small A-scope utility that:
      - Builds a TurboMind pipeline via `lmdeploy.pipeline` with a `TurbomindEngineConfig(speculative_config=SpeculativeConfig(method="eagle3", ...))`.  
      - Uses `SpeculativeDecodingStats` + `EagleMetricsSummary` to aggregate `req_metrics.spec_info` across responses.  
      - Prints the resulting `EagleMetricsSummary.to_dict()` and returns it as a dict.  
      - Calls `validate_eagle_runtime_config` to surface obvious misconfigurations without affecting engine behaviour.  
    - **Scope:** Purely an offline helper; it does not alter any decode paths or B-scope integration. Intended for engineers running small EAGLE-capable models on GPU.  
    - **CI / testing:**  
      - Syntax is checked via `python -m compileall lmdeploy/lmdeploy/turbomind/eagle_inspect.py`.  
      - Full runs should be exercised manually or in a GPU CI job configured with `MODEL_PATH` / `SPEC_MODEL_PATH`; no automated pytest currently drives this helper.

40. **A40: EAGLE tree/CLI what-if enhancements** – **✅ implemented (offline-only tooling)**  
    - Extended `scripts/eagle_inspect_tree.py` to provide richer guidance for tuning `SpeculativeConfig` against a given `eagle_tree.yaml`:
      - Computes and prints the observed maximum path depth from `SpeculationTree.getPathsFlat()` and the maximum number of non-leaf nodes per level via `getNonLeafNodes(level)`, alongside existing branching stats.  
      - Adds `--num-spec-tokens` and `--what-if` flags. In what-if mode, the script reports recommended structural fields for EAGLE/EAGLE3:
        - `num_speculative_tokens` (from `--num-spec-tokens`),  
        - recommended `max_path_len` (observed depth),  
        - recommended `max_non_leaves_per_layer` (max non-leaf count per level), and
        - the current `--max-decoding-tokens` treated as `SpeculativeConfig.max_decoding_tokens`.  
      - Emits a warning when `num_speculative_tokens` exceeds the observed tree depth, indicating that the tree cannot represent that many speculative tokens per step.  
      - Keeps and documents the existing rough KV-block usage estimates based on `--block-size`, now clearly labelled as speculative-only and approximate.  
    - **Scope:** Script remains CPU-only and does not depend on TurboMind; it is intended for offline reasoning about tree/config/KV trade-offs.  
    - **CI / testing:**  
      - Syntax sanity via `python -m compileall scripts/eagle_inspect_tree.py`.  
      - Behaviour is validated manually by engineers using sample `eagle_tree.yaml` files and SpeculativeConfig-like parameters.

### LlamaBatch / LlamaV2_eagle / sampling / integration tests

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

Phase 2 – multi-token EAGLE integration (single-GPU offline path implemented, multi-GPU/CI still TODO):

- [x] **Design multi‑token EAGLE engine loop in `LlamaBatch`**  
  - Use `eagle_max_engine_tokens_per_step_` and a per‑sequence helper (`compute_eagle_draft_tokens_per_seq`) to define how many draft/engine tokens per step EAGLE mode should attempt, keeping per‑sequence engine shapes as static as possible for future CUDA‑graph capture. This is currently logged per step without yet changing decode behaviour.

- [x] **Implement multi‑token `draft_tokens` layout in `LlamaBatch`**  
  - Extend the speculative branch to build a flattened host layout `[batch_size, tokens_per_seq]` for draft and target tokens (currently populated with one token per sequence) and pass `num_draft_tokens = batch_size * tokens_per_seq` into `LlamaV2::eagleSpeculativeStep`, so increasing `tokens_per_seq > 1` later does not change the interface.

- [x] **Extend `LlamaV2_eagle` for multi‑token tree mapping (layout‑ready)**  
  - Update `LlamaV2::eagleSpeculativeStep` to interpret `draft_tokens` as `[batch_size, tokens_per_seq]`, derive `tokens_per_seq` from `num_draft_tokens / batch_size`, and expand the single‑sequence `SpeculationTree` paths into a fully batched `[batch_size, max_decoding_tokens, max_path_len]` layout in `EagleBuffers::inputs.draft_paths`. One draft token per sequence is still used for acceptance, but the tree/mask layout is now multi‑token ready.

- [x] **Wire `acceptDraftTokens` / `invokePackAcceptedPaths` for multi‑token acceptance (logic‑only, pack side)**  
  - Call the speculative decoding helpers `acceptDraftTokens` and `invokePackAcceptedPaths` from `LlamaV2_eagle` so that, for the current one‑token‑per‑sequence regime, `EagleBuffers::outputs.accepted_tokens` / `accepted_lens` are produced entirely on device (using a per‑sequence linear path layout) and `accepted_lengths_cumsum` / `accepted_path_offsets` are populated for downstream KV / decode updates. Full tree‑based multi‑token acceptance remains to be implemented.
  - Add per‑sequence `[LlamaBatch][EAGLE]` debug logs (draft/target/accepted_len) in `LlamaBatch::Forward` so that device‑side acceptance decisions are visible when debugging multi‑token behaviour.

- [x] **Advance sequences by accepted tokens in `LlamaBatch` (tp=1 offline)**  
  - Implemented in `LlamaBatch::advanceSequencesByEagleAcceptance`: for each active slot, `accepted_lens` / `accepted_tokens` from `LlamaV2::eagleSpeculativeStep` are validated against the committed `DynamicDecode` token and then used to append extra tokens on the `[S, B]` time axis (`token_ids_buf_`) and bump `sequence_lengths_`. A per-slot kill helper (`disableEagleMultitokenForSlot`) enforces invariants (first token match, no EOS in extras, bounds, time-axis consistency), and `g.step` is advanced by the maximum extra accepted length so downstream code sees the extended sequence.

- [x] **Wire `computeAndInvokeKVCacheRewind` into real KV/block tables (tp=1 offline)**  
  - Implemented in `LlamaBatch::runEagleKVRewind`: per-slot draft/accepted lengths (driven by `eagle_planned_tokens_per_seq_` and `accepted_lens`) are converted into rewind lengths, host block tables are built from `Sequence::blocks`, and `computeAndInvokeKVCacheRewind` is invoked with real `block_tables`, `batch_slots`, and `kv_cache_blocks`. After the helper returns, tail blocks are removed from `Sequence::blocks`/`block_unique_ids`, the corresponding blocks are unlocked in `SequenceManager` (via `UnlockBlocks`), and `seq->cache_len` is decremented accordingly. Request metrics (`eagle_total_rewound_tokens`, `eagle_rewind_steps`) are updated in lockstep.

- [x] **Support per‑sequence `tokens_per_seq` and runtime fallbacks (tp=1 offline)**  
  - `eagle_planned_tokens_per_seq_` carries a per-slot plan, initially derived from `compute_eagle_draft_tokens_per_seq` and then clamped by remaining `max_new_tokens` and per-slot kill. At commit time, extra tokens are further capped by the per-request `max_new_tokens` budget; any violation of invariants (length > planned, out-of-range indices, EOS in extras, finished slot with multi-token acceptance, time-axis mismatch) triggers `disableEagleMultitokenForSlot` for that slot, permanently reverting it to single-token EAGLE semantics.

- [x] **Make multi-token EAGLE3 config-driven (no env/per-request mode switches)**  
  - Multi-token enablement is controlled purely by `SpeculativeConfig` and topology: `speculative_config.method in {"eagle","eagle3"}`, `num_speculative_tokens > 1`, and `tp == 1`. On the C++ side, this flows through `EngineParam.spec_max_decoding_draft_tokens` into `LlamaV2::eagle_max_engine_tokens_per_step_`, and `LlamaBatch::compute_eagle_draft_tokens_per_seq` uses that budget plus per-slot clamping to choose `tokens_per_seq`. The old env flags (`LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL`, `LMDEPLOY_EAGLE_FORCE_SINGLE_TOKEN`, `LMDEPLOY_EAGLE_DISABLE_MULTI_TOKEN`) and the per-request `disable_eagle_multitoken` flag have been removed from the runtime path. Debug/metrics verbosity now comes from `SpeculativeConfig.eagle_debug` / `eagle_metrics_debug`.

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

Plan code mapping (EngineerB‑01…EngineerB‑20):

- [x] **B‑01: use eagle engine tokens**  
  - Mapped to Section 4’s EAGLE‑aware engine token budget and the Phase‑1 bullet “Use EAGLE engine budget in `LlamaBatch` logs.”  
- [x] **B‑02: multi‑token draft layout invariants**  
  - Mapped to Phase‑2 task “Implement multi‑token `draft_tokens` layout in `LlamaBatch`.”  
- [x] **EngineerB‑03: map multi‑token drafts to trees**  
  - Mapped to Phase‑2 task “Extend `LlamaV2_eagle` for multi‑token tree mapping (layout‑ready).”  
- [x] **EngineerB‑04: wire `acceptDraftTokens` device acceptance**  
  - Mapped to Phase‑2 task “Wire `acceptDraftTokens` / `invokePackAcceptedPaths` for multi‑token acceptance (logic‑only, pack side).”  
- [x] **B‑05: advance sequences by accepted tokens**  
  - Implemented by wiring EAGLE acceptance into `LlamaBatch::Forward` after `dynamicDecode`: per‑sequence `accepted_lens` / `accepted_tokens` from `LlamaV2::eagleSpeculativeStep` are compared against the actual token committed by `DynamicDecode`, and `RequestMetrics.eagle_total_accepted_tokens` is advanced only when the accepted token matches the committed decode token. This keeps speculative metrics aligned with the real decode state in the current single‑token regime, and lays the groundwork for true multi‑token sequence advancement once the inner EAGLE loop is extended beyond `tokens_per_seq == 1`.  
- [ ] **B‑06: adapt `DynamicDecodeLayer` for EAGLE (in progress)**  
  - Mapped to Phase‑2 task “Adapt or bypass `DynamicDecodeLayer` for EAGLE multi‑token mode.” Current work focuses on threading EAGLE acceptance through the decode step so that only tokens consistent with `dynamicDecode` are ever treated as accepted; full multi‑token integration with `DynamicDecodeLayer` remains to be completed.
- [x] **B‑07: keep single‑token EAGLE semantics**  
  - Backed by the deterministic equality test and one‑token semantics in `lmdeploy/tests/test_eagle_e2e.py::test_eagle_equals_baseline_single_token`.  
- [ ] **B‑08: add multi‑token tests (test infra)**  
  - Planned in Section 12 as “Extend tests for multi‑token EAGLE decoding once implemented.”  
- [ ] **B‑09: validate benchmarks with multi‑token EAGLE**  
  - To be built on top of `lmdeploy/tests/test_benchmark_speculative_integration.py` once multi‑token EAGLE decode is wired.  
- [x] **B‑10: document remaining gaps/limits**  
  - Covered by `docs/turbomind_eagle_usage.md`, including notes on current single‑token and multi‑token limitations.  
- [x] **B‑11: honor `eagleMaxEngineTokensPerStep`**  
  - Implemented by using `eagle_max_engine_tokens_per_step_` in `LlamaBatch::Forward` to bound the planned `tokens_per_seq` per decode mini-batch, and reflecting this budget in the EAGLE logging so that future multi-token inner loops can respect a static engine-token budget per step.
- [x] **B‑12: support variable `tokens_per_seq` shapes (tp=1 offline)**  
  - The flattened `[batch_size, tokens_per_seq]` layout is now driven by a per-sequence plan in `eagle_planned_tokens_per_seq_`, which is clamped per-slot by `max_new_tokens` and per-slot gating. All per-slot bounds (acceptance lengths, KV draft lengths, rewinds) are computed using this per-sequence view rather than a single global scalar, while the engine-facing shapes remain static for CUDA efficiency.
- [x] **EngineerB‑13: add detailed EAGLE acceptance logs (single-GPU)**  
  - Per-sequence `[LlamaBatch][EAGLE]` debug logs cover draft/target tokens, accepted lengths, planned tokens per seq, rewind lengths, and step-level totals. Additional end-of-run summaries will be added as part of future production hardening, but current logs are sufficient to debug single-GPU multi-token behaviour.
- [x] **EngineerB‑14: keep acceptance metrics consistent**  
  - Backed by `test_eagle_acceptance_metrics_sanity` in `lmdeploy/tests/test_eagle_e2e.py` and the metrics plumbing tests in `lmdeploy/tests/turbomind/test_eagle_metrics.py`.  
- [ ] **EngineerB‑15: reduce host‑device copies**  
  - Future optimization work around `draft_tokens`, paths, and acceptance buffers once correctness is fully validated.  
- [ ] **EngineerB‑16: keep DP/TP layouts compatible**  
  - Future validation work to ensure multi‑token EAGLE paths remain correct under data/pipeline parallelism; currently `LlamaBatch::isEagleMultiTokenStepEnabled` hard-disables multi-token when `tp_size_ != 1`, so multi-token EAGLE is single-GPU-only.
- [x] **EngineerB‑17: add multi‑token fallback path (per-slot, tp=1)**  
  - A per-slot kill helper (`disableEagleMultitokenForSlot`) is invoked on any hard invariant violation (length mismatches, EOS in extras, finished slot with multi-token acceptance, KV geometry issues, time-axis inconsistencies). Once killed, a slot permanently falls back to single-token EAGLE for the remainder of the request. A higher-level per-request/batch fallback policy can be added in future phases if needed.
- [x] **EngineerB‑18: coordinate KV rewind hook interface (tp=1 offline)**  
  - Engineer‑A’s `computeAndInvokeKVCacheRewind` helper is fully integrated into `LlamaBatch::runEagleKVRewind`, using `SequenceManager`'s block tables and block pointers as described above.
- [ ] **EngineerB‑19: align `DynamicDecodeLayer` stop criteria (future)**  
  - Remaining future work is to formally validate EOS/stop/max_new_tokens equivalence between baseline and multi-token EAGLE3 using dedicated E2E tests and CI runs. Current code enforces first-token equality, prohibits EOS in extra accepted tokens, and caps extra tokens by `max_new_tokens`, but a full DynamicDecode-aware equivalence suite remains to be added.
-- [x] **Summary of Phase‑1 work (single‑GPU EAGLE3)**  
  - This section of `EAGLE_TODO.md` reflects the current status: single-GPU offline multi-token EAGLE3 is implemented and usable; future phases will focus on multi-GPU enablement, CI-backed E2E equivalence tests, and long-tail robustness improvements.

#### Phase 1 summary (single-GPU offline EAGLE3)

For `tp=1` TurboMind engines configured with `SpeculativeConfig(method="eagle3", num_speculative_tokens>1)`:

- Multi-token EAGLE3 is fully wired into `LlamaBatch::Forward` and `LlamaV2_eagle`:
  - Draft planning, multi-step draft/target layouts, device-side acceptance, multi-token sequence advancement, KV rewind, per-slot kill, and metrics are all active and driven by `SpeculativeConfig`/`EngineParam`.
- Mode selection is config-driven and explicit:
  - Single-token vs multi-token EAGLE is controlled by `num_speculative_tokens` and `tp == 1` only; env flags and per-request mode switches have been removed from the runtime path.
- Offline Python tooling makes behaviour easy to sanity-check:
  - `validate_eagle_runtime_config` enforces a sensible single-GPU multi-token configuration (method, tp, structural limits, session length, metrics).
  - `inspect_offline_eagle` and `eagle3_multitoken_smoke` build a TurboMind pipeline with multi-token EAGLE3 enabled and report both text outputs and speculative metrics (`mean_acceptance_length`, `mean_acceptance_rate`, draft/accepted/rewound tokens).

Future B-scope work (Phase 2+) will focus on:

- Enabling and validating multi-token EAGLE3 under TP/PP/DP.
- Adding CI-backed multi-token E2E tests (including `test_eagle_multi_token_future.py`) that exercise EOS/stop/max_new_tokens equivalence.
- Further performance tuning (host–device copies, layout optimizations) once correctness is fully validated.

## 12. Agent test coverage and follow‑up plan (Codex CLI)

This section tracks tests and harness work added by the Codex CLI agent to
validate the Python‑side EAGLE integration and to map TODO items to concrete
test modules.

- [x] Map existing EAGLE TODO items to tests  
  - Section 2 item 7 (“Add focused unit tests for `eagle_tree`”) is covered by `lmdeploy/tests/turbomind/test_eagle_tree.py`.  
  - Section 8 item 6 (“Export and plumb TurboMind EAGLE acceptance metrics end‑to‑end”) is covered on the Python side by `lmdeploy/tests/turbomind/test_eagle_metrics.py`.  
  - Section 10 item 10 (“Tighten Python‑side EAGLE config/metrics and add tests”) is covered by `lmdeploy/tests/turbomind/test_speculative_manager_eagle.py` and `lmdeploy/tests/turbomind/test_speculative_manager_eagle_batch.py`.  
  - Phase‑1 integration tests (“baseline vs EAGLE equality” and “acceptance‑rate sanity”) are covered by `lmdeploy/tests/test_eagle_e2e.py`.

- [x] Add benchmark integration test for EAGLE metrics  
  - Added `lmdeploy/tests/test_benchmark_speculative_integration.py::test_benchmark_runner_reports_eagle_metrics_when_available` to exercise `inference/benchmark_speculative.py` together with a real TurboMind pipeline.  
  - This test asserts that when TurboMind populates `RequestMetrics.spec_info`, the benchmark results JSON includes an `eagle_speculation` block with sane invariants (`enabled == True`, `total_accepted_tokens <= total_draft_tokens`, acceptance rate within `[0, 1]`).

- [x] Add SpeculativeDecodingStats unit tests for EAGLE metrics aggregation  
  - Added `lmdeploy/tests/test_speculative_stats.py` to validate that `SpeculativeDecodingStats.update_from_output` correctly consumes TurboMind EAGLE metrics from `EngineOutput.req_metrics.spec_info`, updates draft/accepted token counters, and leaves stats unchanged when `spec_info` is absent.

- [ ] Extend tests for multi‑token EAGLE decoding once implemented  
  - When Section 4 multi-token tasks and later multi-token integration work are implemented, add tests that:  
    - drive multi-token speculative steps end‑to‑end via `LlamaBatch::Forward`,  
    - verify that `accepted_tokens` / `accepted_lens` advance sequences correctly without KV rewind,  
    - and ensure `RequestMetrics.spec_info` reflects multi-token acceptance statistics.

- [ ] Harden CI configuration for EAGLE‑specific tests  
  - Provide small TurboMind models and CI env configuration (`MODEL_PATH` / `SPEC_MODEL_PATH`) so that all Python‑side EAGLE tests (metrics, managers, benchmark integration, e2e) run without manual setup and without relying on large production models.

## 13. TurboMind target‑tree decode for EAGLE3 – production plan

This section is the source of truth for the remaining *unimplemented* work needed to make TurboMind’s EAGLE3 behave like TensorRT‑LLM’s
`eagleDecodeDraftTokens + eagleDecodingLayer` on GPT‑OSS‑120B. No scaffolding or demo stubs are allowed here: each item must be implemented
with production correctness, shape/dtype safety, and CI coverage.

### 13.1 Base TurboMind decode & KV integration

- [ ] **A‑T1: Define target‑tree decode boundary in TurboMind**
  - Decide the public C++ entry point that will perform target‑tree decode for a single generation step, e.g.:
    - Current prototype: `void LlamaV2::targetTreeDecode(int batch_size, const int* d_sequence_lengths);`
  - Constraints:
    - Must **not** change existing baseline decode semantics when EAGLE is disabled.
    - Must run after draft tree build / mask generation and before `eagleSpeculativeStep`.
  - Current wiring:
    - `LlamaV2_eagle::eagleSpeculativeStep` calls `targetTreeDecode(batch_size, /*d_sequence_lengths=*/nullptr)` after tree build and mask
      generation when `EngineParam.enable_eagle_target_tree == true`.
    - `LlamaBatch::Forward` no longer calls `targetTreeDecode` directly; it still fabricates `target_tokens` on host and passes them into
      `eagleSpeculativeStep`.

- [ ] **A‑T2: Clarify KV/cache invariants for tree decode**
  - Document the expected behaviour of the KV cache during target‑tree decode:
    - Prefix KV must be reused from the main decode path.
    - Tree decode must not corrupt or leak KV blocks for sequences that continue after this step.
  - In `LlamaBatch::runEagleKVRewind` and `SequenceManager`, define a clear contract:
    - What KV state is valid **before** calling target‑tree decode.
    - What KV state is allowed to change **after** acceptance and KV rewind.

- [ ] **A‑T3: Choose execution granularity for tree decode**
  - Decide between:
    - Per‑tree‑depth decode (decode all nodes at a given depth together).
    - Flattened “all nodes at once” decode with a packed attention mask.
  - This choice drives:
    - How many tokens (`num_tree_tokens`) are passed into the base model per step.
    - How attention masks / position ids are constructed.
  - Document the chosen strategy in this file and in a short in‑repo design note (e.g. `docs/turbomind_eagle_target_tree.md`).

- [ ] **A‑T4: Define input/output tensor layouts for tree decode**
  - For the chosen execution strategy, fix and document:
    - Layout of `target_tree_input_ids` (SB vs BS).
    - Layout of `target_tree_position_ids`.
    - Layout of any tree‑specific attention masks (packed vs boolean).
    - Mapping from a flat “tree token index” to `(slot, token_idx)` in EAGLE’s node space.
  - These layouts must be stable and shared across kernel and test implementations.

- [x] **A‑T5: Gate target‑tree decode with runtime guards**
  - `EngineParam.enable_eagle_target_tree` is plumbed from the Triton YAML `speculative_config.enable_target_tree` flag in `LlamaTritonModel.cc`. `LlamaV2::isTargetTreeDecodeEnabled()` exposes this to `LlamaBatch`, and `targetTreeDecode` is only invoked when EAGLE is enabled, buffers are allocated, and `enable_eagle_target_tree == true`. When the flag is false, TurboMind continues to rely on the existing single‑step target logits path and host‑fabricated `target_tokens`.

### 13.2 EAGLE3 target‑tree CUDA path & integration

- [x] **B‑T1: Implement `PrepareGenTargetTreeInputs` kernel and API**
  - New files: `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/target_tree_decode.{h,cu}`.
  - Public API (host‑callable):
    - `void invokePrepareGenTargetTreeInputs(const PrepareGenTargetTreeParams& params);`
  - Responsibilities:
    - Take `EagleBuffers::inputs.draft_paths` `[max_batch_size, max_decoding_tokens, max_path_len]` and `batch_slots` `[batch_size]`.
    - For each active slot:
      - Identify the tree nodes to decode this step (v1 can use all non‑root nodes or all leaf nodes).
      - Build a flattened `output_ids` buffer `[num_tree_tokens]` containing the **draft** token IDs in decode order.
      - Build `position_ids` `[num_tree_tokens]` consistent with TurboMind’s rope strategy (base sequence length + depth).
      - Build `hidden_indices` `[num_tree_tokens]` mapping each flat index → `(slot, token_idx)` in EAGLE node space.
    - Fill `spec_gen_lengths`, `next_sequence_lengths`, `next_context_lengths` `[batch_size]` to describe the speculative pass size.

  **Status (Engineer B – implemented / staging only):**

  - `PrepareGenTargetTreeParams` is implemented in `target_tree_decode.h` with:
    - `draft_paths`, `batch_slots`, `draft_tokens`,
    - `base_sequence_lengths`, `base_context_lengths`,
    - `output_ids`, `position_ids`, `hidden_indices`,
    - `spec_gen_lengths`, `next_sequence_lengths`, `next_context_lengths`,
    - `batch_size`, `max_batch_size`, `max_decoding_tokens`, `max_path_len`, `stream`.
  - `invokePrepareGenTargetTreeInputs` and `prepareGenTargetTreeInputsKernel` in `target_tree_decode.cu`:
    - For each active slot, walk `draft_paths` and emit up to `max_decoding_tokens` non‑root nodes into `output_ids` / `position_ids` / `hidden_indices`.
    - When the optional metadata buffers are non‑null:
      - `spec_gen_lengths[local_idx]` is set to the number of emitted tree tokens.
      - `next_sequence_lengths[local_idx]` is set to `base_seq_len + emitted`.
      - `next_context_lengths[local_idx]` is set to `base_ctx_len` (tree decode does not extend the context window).
  - `LlamaV2::targetTreeDecode` wires this kernel to `EagleBuffers`:
    - Inputs: `inputs.draft_paths`, `inputs.draft_tokens`, and optional `d_sequence_lengths`.
    - Outputs: `inputs.eagle_net_input_ids`, `inputs.eagle_net_position_ids`, `inputs.eagle_net_hidden_indices`,
      plus per‑slot lengths `inputs.eagle_net_gen_lens`, `inputs.eagle_net_seq_lens`, `inputs.eagle_net_ctx_lens`.
  - These staged buffers are used only as inputs for a future base‑model target‑tree decode pass; acceptance still consumes
    host‑fabricated `target_tokens`.

- [ ] **B‑T2: Add target‑tree decode wrapper in `LlamaV2`**
  - Implement `LlamaV2::targetTreeDecode` to:
    - Call `invokePrepareGenTargetTreeInputs` with:
      - `base_input_ids = token_ids_buf_`,
      - `base_seq_lengths = sequence_lengths_`,
      - `base_context_lengths = context_length_buf_` (or equivalent),
      - `draft_paths = eagle_buffers.inputs.draft_paths`,
      - `batch_slots = [0..batch_size-1]`.
    - Run the base model (same kernels as regular decode) on `output_ids` / `position_ids` to produce logits for each tree token.
    - Reduce logits to top‑1 target IDs per node and write them into:
      - `EagleBuffers::inputs.target_tokens[slot * max_decoding_tokens + token_idx]`.
  - **Current status (Engineer B – ⏳ pending decode):**
    - `LlamaV2::targetTreeDecode(int batch_size, const int* d_sequence_lengths)` exists and is called from
      `LlamaV2_eagle::eagleSpeculativeStep` after tree build + mask generation when `enable_eagle_target_tree == true`.
    - Today it only launches `invokePrepareGenTargetTreeInputs` to fill the `eagle_net_*` staging buffers listed in B‑T1.
    - No base‑model decode is yet run over these tree tokens, and `EagleBuffers::inputs.target_tokens` is **not** written on device.
    - `eagleSpeculativeStep` still relies on host‑fabricated `target_tokens` from `LlamaBatch::Forward`, mirrored into
      `EagleBuffers::inputs.target_tokens` before calling `invokeTreeAcceptByIdsWithPaths`.
  - Constraints:
    - No new `cudaMalloc/cudaFree` in the decode hot path; use pre‑allocated buffers in `EagleBuffers` or model scratch.
    - No changes to baseline decode outputs when EAGLE is disabled.

- [ ] **B‑T3: Integrate target‑tree masks with TurboMind attention**
  - Use existing `EagleBuffers::inputs.packed_masks` (from `invokeGetPackedMaskFromPath`) to enforce tree‑structured attention:
    - Tree nodes must attend only to allowed ancestors and siblings as in TRT‑LLM’s EAGLE.
  - Decide whether to:
    - Extend existing attention kernels to accept an optional packed tree mask, or
    - Introduce a small wrapper that applies the mask at the logits or context level.
  - Update `LlamaV2` / `unified_attention_layer` to consume these masks only in the target‑tree decode path.

- [ ] **B‑T4: Wire `invokeTreeAcceptByIdsWithPaths` to new `target_tokens`**
  - Ensure that `LlamaV2::eagleSpeculativeStep` is called **after** `targetTreeDecode` and that:
    - `EagleBuffers::inputs.target_tokens` hold the per‑node target IDs computed by the base model.
    - `invokeTreeAcceptByIdsWithPaths` sees:
      - `draft_ids = inputs.draft_tokens`,
      - `target_ids = inputs.target_tokens`,
      - `paths = inputs.draft_paths` (with correct node indices),
      - `batch_slots` mapping local batch → slot.
  - Remove any remaining dependencies on host‑fabricated `target_tokens` from `LlamaBatch::Forward`.

- [ ] **B‑T5: Ensure BF16/MXFP4 correctness for target‑tree logits**
  - Verify (by inspection and unit tests) that:
    - All target‑tree decode activations are computed in BF16 (matching GPT‑OSS compute).
    - Logit reductions (argmax) use FP32 accumulation or equivalent for numerical stability.
  - Guard against shape/dtype mismatches:
    - Add checks around `EagleBuffers::inputs.target_tokens` and the logits buffer to ensure `dtype == int32` for IDs and consistent float dtype for logits.
    - On any mismatch, log a clear `[LlamaV2][EAGLE][fallback]` message and disable target‑tree decode for that engine.

### 13.3 Validation, metrics, and benchmarks

- Status: core tree‑aware EAGLE3 metrics and benchmark plumbing are implemented; C‑T1/C‑T2/C‑T3/C‑T4/C‑T5 below remain focused on **tests**, **end‑to‑end validation**, and **docs**.

- [x] **C‑3: Implement EAGLE3 acceptance metrics over tree nodes**
  - Extended `RequestMetrics` (`src/turbomind/utils/metrics.h`) with tree‑specific counters:
    - `eagle_tree_draft_tokens`, `eagle_tree_target_tokens`, `eagle_tree_accepted_tokens`, and exposed them to Python via `_turbomind.RequestMetrics` (`src/turbomind/python/bind.cpp`).
  - In `LlamaBatch::updateEagleMetricsAndKVLengths` (`src/turbomind/models/llama/LlamaBatch.cc`), when `enable_eagle_target_tree` is set for the engine, these counters are updated per step alongside the existing EAGLE totals, so we can distinguish tree‑decode behaviour from baseline speculative decode.
  - `lmdeploy/turbomind/turbomind.py::_get_metrics` now forwards these fields into a nested `spec_info["tree_decode"]` block (`num_tree_draft_tokens`, `num_tree_target_tokens`, `num_tree_accepted_tokens`) when non‑zero, keeping the top‑level schema stable for existing consumers.

- [x] **C‑4: Hook target‑tree decode into benchmark tooling**
  - `inference/benchmark_speculative.py::BenchmarkRunner` accepts an `enable_target_tree` flag and passes it through to `SpeculativeConfig(enable_target_tree=...)` when constructing TurboMind pipelines, so scenarios can explicitly exercise the target‑tree path.
  - `BenchmarkRunner.run_benchmark` still aggregates core EAGLE stats via `SpeculativeDecodingStats` / `EagleMetricsSummary`, but the resulting `eagle_speculation` JSON block now includes:
    - `target_tree_enabled: bool` mirroring the runner configuration, and
    - an optional nested `tree_decode` summary when tree metrics are present (contributed by `EagleMetricsSummary.to_dict()`).
  - `BenchmarkRunner.run_test_scenario` prints a short “EAGLE target‑tree” line (tree acceptance rate + total accepted tree tokens) when such metrics are available, so offline runs surface tree effectiveness directly in stdout without breaking existing JSON consumers.

- [x] **C‑5: Add string‑level alignment debugging in Python**
  - Introduced `lmdeploy/turbomind/debug_eagle.py` with:
    - `format_eagle_alignment(...)` – given `(draft_ids_path, target_ids_path, accepted_tokens, tokenizer, baseline_text)`, returns a human‑readable multi‑line string showing ids and decoded text for each sequence, plus an optional baseline `DynamicDecode` text.
    - `print_eagle_alignment(...)` – convenience wrapper that prints the formatted alignment for quick inspection in notebooks or loggers.
  - These helpers are engine‑agnostic and operate purely on recorded token ids and a tokenizer, making them suitable for offline analysis of recorded EAGLE tree runs without requiring additional C++ hooks.

- [ ] **C‑T1: Unit tests for `invokePrepareGenTargetTreeInputs`**
  - Add tests in `lmdeploy/tests/turbomind/test_target_tree_decode.py` to validate that:
    - For simple synthetic trees and fixed `draft_paths`, the kernel produces:
      - The expected `output_ids` sequence.
      - Correct `hidden_indices` mapping back to `(slot, token_idx)`.
    - `spec_gen_lengths` and `next_sequence_lengths` are consistent with the number of nodes selected per batch.
  - Cover corner cases:
    - Single‑path trees, bushy trees, empty/non‑leaf‑only levels.

- [ ] **C‑T2: Device‑level tests for target‑tree decode logits → target_ids**
  - Add tests in `lmdeploy/tests/turbomind/test_target_tree_logits.py` that:
    - Use a tiny TurboMind model (or a synthetic “identity logits” head) to:
      - Run `LlamaV2::targetTreeDecode` on a known tree.
      - Check that `EagleBuffers::inputs.target_tokens` match expected IDs for each node.
  - These tests must run on CUDA in CI (with a small model to keep runtime reasonable).

- [ ] **C‑T3: End‑to‑end acceptance behaviour tests**
  - Extend `lmdeploy/tests/test_eagle_e2e.py` (or a new file) with:
    - A scenario where:
      - Draft model proposals are partially correct along a tree.
      - Target model logits are forced (via a test model) to agree or disagree at specific nodes.
    - Assertions that:
      - `accepted_tokens` / `accepted_lens` from TurboMind match the expected longest prefix rule.
      - `updateEagleMetricsAndKVLengths` reports correct acceptance lengths and KV rewind lengths.

- [ ] **C‑T4: Benchmark and metrics validation for GPT‑OSS‑120B + EAGLE3**
  - Add a dedicated benchmark script (or extend `benchmark_speculative.py`) to:
    - Run GPT‑OSS‑120B + `gpt‑oss‑120b‑Eagle3` with target‑tree decode enabled.
    - Log:
      - Mean accepted tokens per step,
      - Draft/accepted/rewound token totals,
      - Latency per token and throughput vs baseline.
  - Define a minimal acceptance threshold (e.g. `mean_accepted_tokens > 1.1` on standard prompts) for considering target‑tree decode “effective.”
  - Wire this into CI as an optional “smoke benchmark” job that runs on a smaller model but verifies end‑to‑end wiring.

- [ ] **C‑T5: Documentation and runbook updates**
  - Update `docs/turbomind_eagle_usage.md` and/or a new `docs/turbomind_eagle_target_tree.md` section with:
    - A description of the target‑tree decode path, including diagrams of:
      - Base model forward,
      - Tree masks,
      - Target IDs per node,
      - Acceptance and KV rewind.
    - A troubleshooting guide for:
      - Low acceptance rates under target‑tree decode,
      - Shape/dtype assertion failures,
      - Regressions in baseline decode when EAGLE is enabled.
