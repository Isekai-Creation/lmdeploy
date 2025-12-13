# EAGLE3 – Optimizations, Parity, and Long‑Context Plan

This file tracks the **optimization + parity work** for EAGLE3 speculative
decoding in TurboMind (LMDeploy) on GPT‑OSS‑120B, beyond the stability /
geometry work already captured in `EAGLE_TODO_FINAL`.

It assumes:

- Main model: `models/gpt-oss-120b`
- Draft model (HF / TRT‑aligned): `models/gpt-oss-120b-eagle3`
- TurboMind draft conversion: `models/gpt-oss-120b-eagle3-tm-draft`

> Status legend:
> - `[x]` implemented / validated in this repo on the current sm120 box
> - `[~]` partially implemented or validated, more tuning/coverage needed
> - `[ ]` not implemented / not validated yet

---

## 0. Reality Check – Where We Actually Are

**Weights and references**

- `[x]` We do have GPT‑OSS‑120B and GPT‑OSS‑120B‑Eagle3 weights on this box:
  - `models/gpt-oss-120b` – main GPT‑OSS‑120B model in HF layout.
  - `models/gpt-oss-120b-eagle3` – Eagle3 draft checkpoint in HF layout.
  - `models/gpt-oss-120b-eagle3-tm-draft` – TurboMind‑converted draft dir
    (config.yaml + BF16 weights) used by `_turbomind`.
- `[~]` What we do **not** have wired here is a **TRT‑LLM engine build** and
  TRT‑LLM’s own test/bench harnesses (e.g. `test_eagle3.py`,
  `test_kv_cache_reuse.py`) running side‑by‑side on this GPU.
  - That means we can do:
    - TM vs HF / “reference PyTorch” comparisons,
    - but not a full TM vs TRT‑LLM numerical diff on this machine.

**Stability**

- `[x]` TurboMind offline pipeline stability is verified with prefix caching
  enabled and `TM_CACHE_MAX_ENTRY_COUNT=0.75`:
  - `results/20251212_215840`: 8K single baseline + `num_spec_tokens ∈ {2,3,4,5}`.
  - `results/20251212_220131`: 32K single baseline + `num_spec_tokens ∈ {2,3,4,5}`.
  - `results/20251212_220420`: 16K batch4 baseline + `num_spec_tokens ∈ {2,3,4,5}`.
  - No CUDA errors, no Python tracebacks; BlockTrie “clamping cached prefix”
    warnings remain non‑fatal.
- `[x]` Harmony parsing failures no longer crash offline pipelines:
  - With `LMDEPLOY_DISABLE_HARMONY=1`: Harmony parsing is skipped.
  - With Harmony enabled: parser failures are caught and we fall back to raw /
    generic detokenization (`spec_suite_baseline_075_harmony_enabled.log`).

**Numeric parity vs TRT‑LLM**

- `[~]` Geometry and converter are aligned with GPT‑OSS‑120B‑Eagle3:
  - `draft_hidden=2880`, `q_size=4096`, `qkv_in_dim=5760`, `fc_in_dim=8640`,
    `capture_layers=[1,17,32]`.
  - `EagleModule::load` accepts the converted draft and runs Eagle3 draft
    attention + FFN + LM head in draft space.
- `[ ]` We still do **not** have 1:1 numeric parity vs TRT‑LLM:
  - No local TRT‑LLM reference run with the same weights.
  - `eagle3_compare.py` has not been tuned against TRT‑LLM outputs to the
    point where logits cosines / top‑k overlap match NVIDIA’s tolerances.

**Performance parity vs TRT‑LLM**

- `[x]` We have real benchmark numbers for TurboMind offline:
  - Latest sweep with prefix caching enabled, `TM_CACHE_MAX_ENTRY_COUNT=0.75`
    (`WARMUP_RUNS=0`, `MEASUREMENT_RUNS=1`):
    - 8K single (`results/20251212_215840`): baseline 119 tok/s; spec 2 slower,
      spec 3 slightly slower, spec 4/5 faster (126 / 170 tok/s).
    - 32K single (`results/20251212_220131`): baseline 117 tok/s; spec
      `num_spec_tokens ∈ {2,3,4,5}` all slower (≈30–55 tok/s).
    - 16K batch4 (`results/20251212_220420`): baseline 393 tok/s; spec
      `num_spec_tokens ∈ {2,3,4,5}` all slower (≈76–147 tok/s).
- `[ ]` We are **not** at TRT‑LLM kernel/perf parity:
  - Large BF16 GEMMs on sm120 still hit:
    - `No feasible kernel found for sm120_bf16_bf16_bf16_ttt_fff_3x34560x2880_1`
      and `...5x34560x2880_1`.
  - They fall back to naive BF16 / cuBLAS, which is correct but slow.
  - TRT‑LLM uses tuned CUTLASS/GMMA kernels; they are ahead on raw kernels.

**SpecPV**

- `[ ]` SpecPV is wired in the code (flags, KV cache), but **not enabled** in
  the current benchmark runs:
  - `enable_specpv=False` in `SpeculativeConfig`.
  - `LlamaV2::isSpecPVEnabled()` always returns false in these runs.
- All stability/perf work so far is **full‑KV EAGLE3 only**; no EAGLE3+SpecPV
  validation has been done yet for 16K/32K.

---

## 1. sm90 / sm120 Kernel Optimizations (GEMM)

Goal: stop hitting “No feasible kernel” for hot GEMMs on sm90/sm120, so that
EAGLE3 speculative steps are **cheap enough** that 2–3 draft tokens already
beat baseline, not only 4–5.

### 1.1 Inventory hot shapes

- `[x]` From current logs on this sm120 box (EAGLE3 single‑context 16K/32K):
  - `sm120_bf16_bf16_bf16_ttt_fff_3x34560x2880_1`
  - `sm120_bf16_bf16_bf16_ttt_fff_5x34560x2880_1`
- `[ ]` Still to do:
  - Collect a short list of all distinct `(M, K, N)` shapes seen for:
    - Draft LM‑head GEMMs,
    - Draft mid‑layer FC GEMMs,
    - Any large attention Q/K/V projections on the draft path.

### 1.2 Add fused sm90/sm120 BF16 kernels

- `[x]` Implement dense BF16 GMMA kernels and registrations in:
  - `src/turbomind/kernels/gemm/arch/config_sm80_s16816.h` via
    `Config_F16_dense` for FP16/BF16 dense row‑major GEMMs.
  - `src/turbomind/kernels/gemm/kernel/sm90_16816_16.cu` for `Sm90` BF16
    dense kernels, covering the EAGLE3 draft/FFN/LM‑head layouts where
    `pack_a/pack_b/pack_u/pack_v` are all zero.
- Effects on this sm120 box:
  - “No feasible kernel found for sm120_bf16_bf16_bf16_ttt_fff_*x34560x2880_1”
    is no longer seen in current 8K/16K/32K EAGLE3 runs.
  - 32K single, `num_spec_tokens=3` improved from “often slower than
    baseline” in early logs to **>1×** baseline in the dense‑GMMA
    configuration before runtime tuning.

### 1.3 Tuning and validation

- `[~]` Use the existing GEMM tuner + env (e.g. `TM_GEMM_TUNE`) to:
  - Generate dispatch cache entries for sm120 on representative EAGLE3
    workloads (e.g. 32K single, `num_spec_tokens ∈ {3,5}`).
  - Compare fused kernels vs cuBLAS for accuracy in BF16/FP16.
  - Current status:
    - A first tuning pass with `TM_GEMM_TUNE` + `TM_GEMM_EXPORT` during
      32K single runs produced a dispatch cache where:
      - 32K single, `num_spec_tokens=5` improved to ≈1.0× baseline.
      - 32K single, `num_spec_tokens=3` regressed vs the dense‑GMMA default.
    - This cache is **not** yet locked in; further tuning is required to
      favor the 3‑token sweet spot.
- `[ ]` Add a small GEMM harness or unittest that:
  - Runs the key shapes (3×34560×2880, 5×34560×2880, etc),
  - Confirms:
    - No “No feasible kernel” logs,
    - Relative error vs cuBLAS is within BF16/FP16 expectations.

### 1.4 Re‑benchmark after kernels

- `[~]` Re‑run:
  - `SCENARIOS=baseline,single,large-context ./run_spec_suite.sh`
  - For each context (8K/16K/32K) and `num_spec_tokens ∈ {2,3,4,5}`.
- Current observations after dense BF16 GMMA + initial tuning + Eagle3
  gemm2 wiring on this sm120 box (prefix caching 0.75, Harmony disabled):
  - 32K single (`results/20251213_023905`):
    - Baseline: throughput_tokens_per_sec.mean ≈ **115.6 tok/s**.
    - Speculative:
      - `num_spec_tokens=2`: ≈ **30.9 tok/s**, mean_accept_len ≈ 2.33.
      - `num_spec_tokens=3`: ≈ **111.6 tok/s**, mean_accept_len ≈ 3.22
        (now ~0.97× baseline, improved from ~0.93× before gemm2 wiring).
      - `num_spec_tokens=4`: ≈ **56.7 tok/s**, mean_accept_len ≈ 4.57.
      - `num_spec_tokens=5`: ≈ **11.3 tok/s**, mean_accept_len ≈ 3.83.
  - 16K batch4 (`results/20251213_024340`):
    - Baseline_Batch4_Context16K:
      - throughput_tokens_per_sec.mean ≈ **333.3 tok/s**.
    - Speculative_Batch4_Context16K:
      - `num_spec_tokens=2`: ≈ **107.9 tok/s**, mean_accept_len ≈ 2.0.
      - `num_spec_tokens=3`: ≈ **165.2 tok/s**, mean_accept_len ≈ 3.08.
      - `num_spec_tokens=4`: ≈ **169.1 tok/s**, mean_accept_len ≈ 2.13.
      - `num_spec_tokens=5`: ≈ **181.1 tok/s**, mean_accept_len ≈ 2.33.
    - EAGLE3 remains <1× baseline on 16K batch4; acceptance is healthy, so
      the remaining gap is dominated by draft compute + SDPA + tail/pipeline
      structure rather than pure GEMM throughput.
- Targets remain:
  - 2–3‑token EAGLE3 should at least match or beat baseline at 8K/16K.
  - 2–3‑token EAGLE3 at 32K should reach **2–3×** baseline once kernels
    and pipeline overhead are fully tuned.

---

## 2. Long‑Context Tail Stability (32K+5 and beyond)

**Current status**

- `[x]` The 32K + 5‑token tail crash is fixed:
  - Failing pointer was `eagle_kv_block_tables_`.
  - The raw `core::Copy` was copying `host_elems > dev_elems` bytes into a
    pooled buffer that had been shrunk.
  - Fix: clamp `n_elems = min(host_elems, dev_elems)` before calling
    `core::Copy`, with a one‑time debug log when clamping occurs.
  - `Speculative_Single_Context32K_5tokens` now completes and produces a
    valid JSON with `eagle_speculation` metrics.

**Remaining long‑context tasks**

- `[x]` Clamp all EAGLE host→device copies to device buffer size:
  - `LlamaBatch.cc`: pooled `core::Copy` callsites already clamp.
  - `LlamaV2.cc`: clamp `cudaMemcpyAsync` into `EagleBuffers` inputs/outputs
    (draft tokens, target tokens, paths, accepted lens cumsum, prev lens).
- `[ ]` Run additional stress scenarios:
  - 32K and 64K single‑context with:
    - `num_spec_tokens ∈ {2,3,4,5}`,
    - `TM_CACHE_MAX_ENTRY_COUNT ∈ {0.75, 0.9}`.
  - Confirm:
    - No `cudaErrorInvalidValue` / `cudaErrorIllegalAddress`.
    - BlockTrie “clamping cached prefix” warnings remain non‑fatal.

---

## 3. Numeric Parity vs TRT‑LLM (Stagewise)

Goal: use **the weights in `models/`** plus a TRT‑LLM reference run to tune
TurboMind until stagewise numerics and logits match within target tolerances.

### 3.1 Reference setup

- `[ ]` On a machine with TRT‑LLM:
  - Use `models/gpt-oss-120b-eagle3` as the draft checkpoint for both:
    - TRT‑LLM build (EAGLE3 engines),
    - LMDeploy/TurboMind converter (already done here).
  - Build TRT‑LLM EAGLE3 engines for GPT‑OSS‑120B with:
    - `--enable-eagle`, `--eagle-draft-length`, etc, per NVIDIA docs.

### 3.2 Stagewise comparison harness

- `[~]` `LM/lmdeploy/tests/turbomind/eagle3_compare.py` exists and calls
  `_turbomind.eagle3_forward_debug` to get:
  - `fc_out`, `attn_out`, `ffn_out`, `pre_head_hidden`, `logits`.
- `[ ]` Extend the harness so it can:
  - Load reference tensors from:
    - TRT‑LLM Python EAGLE3 path, or
    - A PyTorch script using the same HF weights and math as TRT‑LLM.
  - Compute:
    - `mean_abs_diff`, `max_abs_diff`,
    - cosine similarity,
    - top‑k overlap for logits.

### 3.3 Tuning to match TRT‑LLM

- `[ ]` For each stage, tune until:
  - `cosine(logits) ≳ 0.98` (or TRT‑LLM’s own tolerance),
  - top‑k overlap is acceptable.
- Potential knobs:
  - RoPE scaling (YARN, base, factors, offsets),
  - SDPA masking / head layout,
  - FFN ordering / activation,
  - LM‑head routing (draft_hidden vs base_hidden, padding).

---

## 4. TurboMind vs TRT‑LLM Bench Matrix

Goal: fair comparison of E2E throughput/latency and EAGLE acceptance between
TurboMind offline and TRT‑LLM, using the same weights / prompts.

### 4.1 TurboMind side (this repo)

- `[x]` `run_spec_suite.sh` + `benchmark_speculative.py` already produce:
  - Scenarios: `baseline`, `single`, `batch`, `large-context`, `stress`.
  - Metrics: tokens/s, ms/token, memory, `eagle_speculation` summary.
- `[x]` Ensure `eagle_speculation` is written for all speculative scenarios:
  - Include:
    - `mean_acceptance_rate`,
    - `mean_acceptance_length`,
    - `total_accepted_tokens`,
    - fraction of steps with `accepted_len ≥ 2`.

### 4.2 TRT‑LLM side

- `[ ]` Build EAGLE3 engines for GPT‑OSS‑120B and run their spec bench
  (or a minimal offline client) for:
  - 8K / 16K / 32K contexts,
  - `num_spec_tokens ∈ {2,3,4,5}`,
  - comparable prompts / temperatures.
- `[ ]` Capture:
  - throughput, latency, memory,
  - EAGLE acceptance metrics if exposed (or compute from logs).

### 4.3 Apples‑to‑apples comparison

- `[ ]` For each scenario/context:
  - Compare:
    - TurboMind baseline vs TurboMind EAGLE3 vs TRT‑LLM EAGLE3.
  - Determine:
    - Whether TurboMind EAGLE3 is within a few percent of TRT‑LLM,
    - Whether TurboMind offline ever surpasses TRT‑LLM’s serve stack on
      your workloads (end‑to‑end latency).

---

## 5. SpecPV on Top of EAGLE3

Goal: once full‑KV EAGLE3 is tuned and stable, enable SpecPV (partial KV)
and validate that long‑context behavior is correct and faster / cheaper.

### 5.1 Enable SpecPV (config)

- `[ ]` In LMDeploy:
  - Expose `enable_specpv` and `specpv_*` (sink, retrieval, window) in the
    config used by `benchmark_speculative.py`.
  - For GPT‑OSS‑120B‑Eagle3, start with conservative defaults and, if
    possible, match TRT‑LLM’s recommended values.

### 5.2 Validate correctness and stability

- `[ ]` For contexts 16K and 32K, `num_spec_tokens ∈ {3,5}`:
  - Compare:
    - EAGLE3 full‑KV vs EAGLE3+SpecPV token streams,
    - acceptance metrics,
    - any new CUDA errors.
  - SpecPV should:
    - Preserve token streams modulo small numerical differences,
    - Not introduce new illegal memory errors or tail bugs.

### 5.3 Benchmark SpecPV vs full‑KV

- `[ ]` For each scenario:
  - Baseline (no spec),
  - EAGLE3 full‑KV,
  - EAGLE3+SpecPV.
- Compare:
  - throughput,
  - memory usage,
  - mean acceptance, `≥2` tokens fraction.

### 5.4 Document trade‑offs

- `[ ]` After measurement:
  - Document when SpecPV should be on/off for GPT‑OSS‑120B‑Eagle3:
    - Default settings for 16K/32K,
    - Any prompts or batch sizes where SpecPV clearly wins or loses.

---

## 6. Final Parity & “Surpass” Checklist

- `[ ]` **Numeric parity** vs TRT‑LLM:
  - EAGLE3 attention/FFN/LM‑head numerics match TRT‑LLM (via
    `eagle3_compare.py` + TRT reference).
- `[ ]` **Acceptance parity**:
  - Mean acceptance length / rate and `≥2` token fraction for key prompts
    match TRT‑LLM within agreed tolerances.
- `[ ]` **Long‑context behavior**:
  - 32K / 64K (full‑KV and SpecPV) run stably with EAGLE3; no tail crashes;
    KV reuse and block manager behavior match expectations.
- `[ ]` **Kernel parity**:
  - No “No feasible kernel” logs for hot shapes on sm90/sm120; fused BF16
    kernels exist and are used.
- `[ ]` **Perf comparison**:
  - TurboMind EAGLE3 throughput on 8K/16K/32K within a few percent of
    TRT‑LLM’s EAGLE3 on the same hardware (or better, if the offline path
    + optimizations allow it).

Once these boxes are checked, **EAGLE_TODO_FINAL** and this
`EAGLE_TODO_OPTMIZATIONS.md` can be updated to:

- mark EAGLE3 as **parity‑complete** vs TRT‑LLM for GPT‑OSS‑120B, and
- call out any deliberate design deviations or extra TurboMind‑specific
  optimizations (e.g., better offline batching, SpecPV behavior).
