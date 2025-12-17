# EAGLE3 TurboMind TODOs (2–3×, 100% readiness)

This file tracks the concrete engineering work required for TurboMind EAGLE3 to reliably hit:

* **32K single, spec-3: ≥2–3× baseline**
* **16K batch4, spec-3: ≥1× baseline**

All items are scoped to `LM/lmdeploy` and are **non-optional** for “100% achieved”.

> **Rule:** No `SCENARIOS=large-context` full runs until micro gates are green **3 consecutive times**.

---

## -1. Reference mapping vs TensorRT-LLM / SGLang / EasyDeL

This table is the ground truth for what TurboMind is missing relative to the reference EAGLE3 implementations. Every row must either be
implemented or explicitly marked “not needed” for our workloads.

| Area                           | TensorRT-LLM / SGLang / EasyDeL                           | TurboMind (LMDeploy)                                | Status / Action                                  |
|--------------------------------|-----------------------------------------------------------|-----------------------------------------------------|--------------------------------------------------|
| Tree / paths representation    | TRT: `draftTokenTreeKernels.cu` builds compact tree;     | `build_eagle3_kv_spans_kernel`, GPU paths, packed   | Partially mapped; no linked-list tree. See 1.x   |
|                                | SGLang: `eagle_utils.cu` `build_tree_efficient` with     | masks in `eagle3_attention_kernels.cu`.            | for active-slot aware tree usage.               |
|                                | `retrive_index/next_token/next_sibling`.                 |                                                     |                                                  |
| Acceptance / verify            | TRT: `eagleDecodingKernels.cu` `AcceptDraftTokens*`      | `tree_accept_kernels.cu`, Contract A enforced,      | Contract semantics aligned; perf still lacking.  |
|                                | kernels run fully on GPU; SpecDec PV support.            | but no SpecPV and no FR-spec.                      |                                                  |
| Base context prep (multi-q)    | TRT: `prepareCtxEagleNetInputsKernel` builds multi-q     | `DynamicDecodeLayer::ForwardMultiStep` uses        | **Missing:** GPU-native multi-q context prep.    |
|                                | contexts on device for EAGLE base verify/commit.         | 1-token base + tail patch; no multi-q base path.   | See section 5.                                   |
| Draft attention FMHA           | TRT/FlashInfer: `FmhaKernelType::SpecDecodingGeneration` | `eagle3_fmha_multi_cta_kernel1` + reduction,       | Functional but not tuned to TRT-class. See 4.x.  |
|                                | kernels, tile scheduler, GMMA/TMA tuned per arch.        | tile skipping, per-token spans, sm120 path.        |                                                  |
| Draft FFN / LM-head            | TRT/EasyDeL: tuned CUTLASS kernels, often “thin” draft   | Full-width FFN + LM-head using gemm2; small-M      | **Missing:** tuned small-M kernels and           |
|                                | (smaller draft hidden or vocab) to cut FLOPs.            | shapes logged but not fully tuned or slimmed.      | optional draft-vocab LM-head. See 2.x & 3.x.     |
| Active-slot scheduling         | TRT/SGLang: work scheduled only for non-finished slots;  | All slots still enter draft path; finished slots   | **Missing:** active-slot compaction. See 1.x.    |
|                                | finished slots do not pay draft cost.                    | pay cost and dilute acceptance.                    |                                                  |
| Host vs device glue            | TRT/SGLang: context prep, KV updates, tree, accept,      | Many pieces GPU-native; remaining host glue (e.g.  | Partially mapped; need PERF_MODE host glue       |
|                                | commit entirely on GPU; host only orchestrates.          | per-step metrics, a few syncs) not yet proven      | trimming + NVTX proof. See 0.1 & 6.x.            |
| SpecPV                         | TRT/EasyDeL: SpecPV kernels for KV reuse;                | `SpecPV/` and `SPECPV_TODO.md` exist but are       | Blocked until full-KV gates (0.3.x) are green.   |
|                                | base EAGLE path already ≥2×.                             | intentionally disabled while full-KV < targets.    |                                                  |

## 0. Invariants and gates (definition of “green”)

### 0.1 PERF_MODE invariants (must hold on every perf/micro run)

* [ ] **(0.1.1)** No GEMM fallback for any `EAGLE3_*` tag (fatal abort).
* [ ] **(0.1.2)** No untuned (“generic”) GEMM kernel used for `EAGLE3_*` in PERF_MODE (tuned-only dispatch enforced).
* [ ] **(0.1.3)** No non-terminal multi-token disable (`disabled_multitoken_other == 0`) (fatal abort).
* [ ] **(0.1.4)** No KV span / packed-mask integrity failures (SPAN_DEBUG hard-fail available).
* [ ] **(0.1.5)** No 1D `output_ids` layout inference in `DynamicDecodeLayer::ForwardMultiStep` in PERF_MODE (fatal abort).
* [ ] **(0.1.6)** No debug D2H copies or `cudaStreamSynchronize` in decode loop unless explicitly requested by `*_DEBUG` env.
* [ ] **(0.1.7)** PERF_MODE uses deterministic decode (greedy, no sampling drift) for baseline+spec.

### 0.2 Micro perf gates (must pass before any full large-context suite)

* [ ] **(0.2.1)** 16K batch4 micro run (PERF_MODE, `LMDEPLOY_EAGLE_MICRO_STEPS=128`):

  * [ ] `throughput(spec-3) ≥ 1.00× throughput(baseline)`
  * [ ] `mean_accept_len(spec-3) ≥ 2.2`
  * [ ] All PERF_MODE invariants hold.
* [ ] **(0.2.2)** 32K single micro run (PERF_MODE, `LMDEPLOY_EAGLE_MICRO_STEPS=512`):

  * [ ] `throughput(spec-3) ≥ 2.00× throughput(baseline)`
  * [ ] `mean_accept_len(spec-3) ≥ 3.0`
  * [ ] All PERF_MODE invariants hold.

### 0.3 Full perf gates (define “100% achieved”)

* [ ] **(0.3.1)** 32K single full run, PERF_MODE:

  * [ ] `throughput(spec-3) ≥ 2.0× throughput(baseline)`
* [ ] **(0.3.2)** 16K batch4 full run, PERF_MODE:

  * [ ] `throughput(spec-3) ≥ 1.0× throughput(baseline)`

---

## 1. Fundamental design fix #1: Active-slot compaction (batch4 critical)

**Why:** TRT-LLM / sglang do not pay draft work for finished slots. TurboMind must do the same or batch4 will never hit ≥1×.

**Goal:** only active slots enter EAGLE drafter, tree/masks, and LM-head; dead slots do zero draft compute and do not dilute acceptance.

**Files:**

* `src/turbomind/models/llama/LlamaBatch.cc`
* `src/turbomind/models/llama/LlamaV2.cc`
* `src/turbomind/models/llama/EagleBuffers.{h,cc}`
* `src/turbomind/models/llama/EagleModule.cc`
* `src/turbomind/models/llama/EagleDraftLayer.cu`
* `src/turbomind/models/llama/eagle3_attention_layer.cc`
* `src/turbomind/kernels/speculative_decoding/common.{h,cu}`

### TODOs

* [ ] **(1.1)** Define “active slot” and compute active list on device

  * [ ] Implement kernel `build_active_slots` producing:

    * `active_slots[0..active_count-1]` (int32)
    * `active_inverse[slot] = idx or -1`
    * `active_count` (int32)
  * [ ] Active iff `!finished && seq_len < limit && !eos_terminated`.
  * [ ] Provide debug counter: `active_count` per step (log under `LMDEPLOY_EAGLE_ACTIVE_DEBUG=1`).

* [ ] **(1.2)** Compact drafter inputs

  * [ ] Gather `last_token_hidden` and `captured_hidden` into `[active_count, …]`.
  * [ ] Gather all per-slot arrays used by EAGLE3:

    * `kv_lens_runtime`, `runtime_offsets`, `tree_offsets` (if used),
    * any gating params (PERF_MODE = none),
    * any slot-specific metadata the FMHA path reads.
  * [ ] Assert no finished slot appears in compact buffers (debug only).

* [ ] **(1.3)** Run EagleModule/EagleDraftLayer on compact batch only

  * [ ] Eagle draft forward must accept compact batch size and produce:

    * compact draft logits/tokens and hidden outputs only for active slots.
  * [ ] Ensure FMHA `token_num` = `active_count * tokens_per_seq` not full batch.

* [ ] **(1.4)** Compact tree/masks/acceptance

  * [ ] Modify speculative decode kernels to accept `active_slots` (preferred).
  * [ ] Or intermediate: run accept kernels on compact batch and scatter results.
  * [ ] Dead slots must output:

    * accepted_len=0 (or 1 only if baseline root required),
    * draft tokens = -1,
    * packed mask = 0.

* [ ] **(1.5)** Scatter outputs back to full slots

  * [ ] Scatter `accepted_lens`, `accepted_tokens`, `draft_tokens`, `target_tokens` back into full slot arrays using `active_inverse`.
  * [ ] Ensure DynamicDecode tail commit uses full slot buffers but only active slots have extras.

* [ ] **(1.6)** Exit criteria for compaction

  * [ ] In 16K batch4 micro:

    * As soon as 1–3 slots finish, `EAGLE3_*` GEMM M dimension drops proportionally.
    * FMHA tile stats show fewer tokens/heads executed (proportional to active slots).
    * Throughput improves vs pre-compaction.

---

## 2. Fundamental design fix #2: Draft-vocab LM-head (TRT draft vocab / sglang FR-spec)

**Why:** Full-vocab LM-head is a dominant cost in the draft path. TRT-LLM and sglang reduce LM-head cost structurally.

**Goal:** draft predicts in a smaller vocab (e.g., top-32K), then maps to full vocab IDs before acceptance/commit.

**Files:**

* `src/turbomind/models/llama/EagleModule.{h,cc}`
* `src/turbomind/models/llama/EagleDraftLayer.cu`
* `src/turbomind/models/llama/EagleBuffers.{h,cc}`
* Python converter/export path (`lmdeploy/turbomind/eagle_draft_converter.py`, or wherever draft weights are exported)
* `lmdeploy/pytorch/spec_decode/proposers/eagle3.py` (reference for mapping semantics)

### TODOs

* [ ] **(2.1)** Introduce C++ draft vocab mode

  * [ ] Add `draft_vocab_size` and `draft_vocab_ids` / `draft_id_to_target_id` support in `EagleModule`.
  * [ ] Store `d_draft_to_target[draft_vocab_size]` on device.

* [ ] **(2.2)** Export/convert draft LM-head for draft vocab

  * [ ] Modify converter to export:

    * `lm_head_draft: [hidden, draft_vocab_size]`
    * `draft_id_to_target_id: [draft_vocab_size]`
  * [ ] Ensure config.yaml carries `draft_vocab_size` and mapping info.

* [ ] **(2.3)** Draft logits computed in small vocab

  * [ ] In `EagleModule::forward`, LM-head GEMM must be `[M, hidden] x [hidden, draft_vocab_size]`.
  * [ ] Log shape as `EAGLE3_LM_HEAD_SMALLVOC` (must show N = draft_vocab_size).

* [ ] **(2.4)** Map draft ids → full vocab ids on device before acceptance

  * [ ] After argmax/topk, apply `full_id = d_draft_to_target[draft_id]`.
  * [ ] Ensure all downstream buffers (`draft_tokens`) are **full vocab IDs**.

* [ ] **(2.5)** Optional FR-spec tooling (sglang-style)

  * [ ] Add tool to build `draft_vocab_ids` from frequency list.
  * [ ] Add tool to gather LM-head columns to draft LM-head weight.

* [ ] **(2.6)** Exit criteria for draft vocab

  * [ ] `EAGLE3_LM_HEAD` GEMM N drops from full vocab → draft_vocab_size.
  * [ ] 32K single spec-3 micro throughput improves materially without collapsing acceptance.

---

## 3. GEMM tuning: make tuned-only real (not a trap)

**Goal:** all EAGLE3 draft GEMMs (FFN, FC, LM-head) are tuned for small-M shapes and always hit tuned kernels in PERF_MODE.

**Files:**

* `src/turbomind/utils/eagle_debug.h`
* `src/turbomind/kernels/gemm/gemm.cu`
* `tools/tune_eagle3_gemm_sm120.py`
* `tools/run_eagle3_gemm_tune_sm120.sh` (create if needed)

### TODOs

* [ ] **(3.1)** Ensure GEMM shape export completeness

  * [ ] PERF_MODE micro runs produce `build/eagle3_gemm_shapes_sm120.json`.
  * [ ] JSON must include all EAGLE3 tags:

    * `EAGLE3_FFN_*`, `EAGLE3_FC`, `EAGLE3_LM_HEAD(_SMALLVOC)`.

* [ ] **(3.2)** Create one-command tune+export pipeline

  * [ ] Script reads JSON, selects top N shapes by count, runs tuner/export.
  * [ ] Writes tuned cache artifact for sm120.

* [ ] **(3.3)** Load tuned cache at runtime + log once

  * [ ] `[EAGLE3][GEMM_TUNED] loaded entries=... device=sm120`.

* [ ] **(3.4)** PERF_MODE enforcement

  * [ ] In PERF_MODE, any `EAGLE3_*` GEMM cache miss is fatal.
  * [ ] Add counters: tuned_hits/total_eagle3_gemms (must be ~100%).

* [ ] **(3.5)** Exit criteria

  * [ ] PERF_MODE micro runs no longer abort.
  * [ ] Tuned hit rate ~100% for `EAGLE3_*`.
  * [ ] Batch4 spec-3 throughput increases after tuning.

---

## 4. FMHA tuning + defaults (sm120)

**Goal:** tuned tile geometry (KV_TILE/MAX_TILES/BLOCK/HEADS_PER_CTA) chosen from data, not guesswork.

**Files:**

* `src/turbomind/models/llama/eagle3_attention_layer.cc`
* `src/turbomind/models/llama/eagle3_attention_kernels.cu`
* `tools/sweep_fmha_sm120.py`

### TODOs

* [ ] **(4.1)** Fully wire FMHA knobs

  * [ ] `TM_EAGLE3_FMHA_KV_TILE`
  * [ ] `TM_EAGLE3_FMHA_MAX_TILES`
  * [ ] `TM_EAGLE3_FMHA_BLOCK`
  * [ ] `TM_EAGLE3_FMHA_HEADS_PER_CTA`
  * [ ] `TM_EAGLE3_FMHA_QTOKENS_PER_CTA` (optional)

* [ ] **(4.2)** Tile stats summary (opt-in only)

  * [ ] Reset counters once per run.
  * [ ] Log `[EAGLE3][FMHA_TILE_STATS] total=.. span_empty=.. mask_empty=.. executed=..`.
  * [ ] No D2H copies unless stats env enabled.

* [ ] **(4.3)** Micro sweep driver outputs JSON

  * [ ] Sweep grid over KV_TILE/MAX_TILES/BLOCK/HEADS_PER_CTA.
  * [ ] Run:

    * 32K single micro spec-3
    * 16K batch4 micro spec-3
  * [ ] Record throughput + accept_len + tile stats.
  * [ ] Write `build/fmha_sweep_sm120.json`.

* [ ] **(4.4)** Bake defaults from sweep

  * [ ] Default config for `batch==1`.
  * [ ] Default config for `batch>=4 && kv_len>=8192`.
  * [ ] Document chosen defaults with sweep JSON reference.

* [ ] **(4.5)** Exit criteria

  * [ ] Default (no env overrides) achieves within a few % of best sweep result.

---

## 5. PERF_MODE must be “quiet”: eliminate remaining host overhead

**Goal:** PERF_MODE loop is GPU-driven; CPU does not stall or do per-step heavy work.

**Files:**

* `src/turbomind/models/llama/LlamaV2.cc`
* `src/turbomind/models/llama/LlamaBatch.cc`
* `src/turbomind/layers/DynamicDecodeLayer.cc`

### TODOs

* [ ] **(5.1)** Gate all debug paths OFF by default

  * [ ] `ALIGN_DEBUG`, `ACCEPT_DEBUG`, `SPAN_DEBUG`, `FMHA_AB` must be opt-in.
  * [ ] Zero unconditional D2H copies in PERF_MODE path.

* [ ] **(5.2)** Remove per-step synchronizations

  * [ ] No `cudaStreamSynchronize` inside decode loop in PERF_MODE.
  * [ ] Batch metrics copies must be batched and minimal.

* [ ] **(5.3)** Add NVTX ranges (required for proof)

  * [ ] `BASE_DECODE_STEP`
  * [ ] `EAGLE_DRAFT_ATTENTION_FMHA`
  * [ ] `EAGLE_DRAFT_FFN`
  * [ ] `EAGLE_DRAFT_LM_HEAD`
  * [ ] `EAGLE_TREE_MASKS`
  * [ ] `EAGLE_ACCEPT`
  * [ ] `EAGLE_TAIL_COMMIT`

* [ ] **(5.4)** Exit criteria

  * [ ] Nsight Systems shows GPU dominates, CPU not stalling per step.
  * [ ] PERF_MODE logs stay minimal.

---

## 6. Batch4-specific acceptance dilution fixes (must match TRT/sglang behavior)

**Goal:** don’t let finished slots or dead work dilute acceptance; only active slots participate (reinforced by section 1).

### TODOs

* [ ] **(6.1)** Ensure finished-slot draft work is zero (compaction or hard masking)
* [ ] **(6.2)** Ensure “finished slot” multi-token disable triggers only on true EOS/limit
* [ ] **(6.3)** Add one-time summary log in PERF_MODE:

  * `disabled_finished`, `active_count avg/min/max`, `effective_accept_per_active_slot`

**Exit criteria:** batch4 mean_accept_len rises and no longer collapses due to dead slots.

---

## 7. Gates: micro first, then full suites

**Files:**

* `tools/check_perf_gates.py`
* `benchmark_speculative.py`
* `run_spec_suite.sh`

### TODOs

* [ ] **(7.1)** Make micro gates first-class command

  * [ ] Run baseline+spec-3 micro 32K single
  * [ ] Run baseline+spec-3 micro 16K batch4
  * [ ] Verify via `check_perf_gates.py`

* [ ] **(7.2)** Enforce gate order

  * [ ] `run_spec_suite.sh` refuses full large-context until micro gates pass 3 times.

* [ ] **(7.3)** Full gates

  * [ ] Run full 32K single PERF_MODE and validate ≥2×
  * [ ] Run full 16K batch4 PERF_MODE and validate ≥1×

---

## 8. Optional (blocked): SpecPV plan only after full-KV gates

* [ ] **(8.1)** Design SpecPV enablement once full-KV gates met
* [ ] **(8.2)** Implement behind feature flag (no-op until validated)
* [ ] **(8.3)** SpecPV sweeps targeting 4–6×

---

# Summary of what’s “fundamental”

If we still plateau at ~1.5× after tuning FMHA+GEMM, the **only remaining fundamental** levers are:

* **(A)** active-slot compaction (must be done; TRT/sglang do this)
* **(B)** truncated draft vocab LM-head / FR-spec (must be done for 120B-scale vocab costs)
* **(C)** tuned-only GEMMs + FMHA tuning (must be done, but cannot compensate if A/B are missing)

Once (1) and (2) are implemented + (3)/(4) tuned, 2–3×/≥1× becomes realistic without “magic”.
