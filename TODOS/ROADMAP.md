# LMDeploy / TurboMind Roadmap – GPT‑OSS‑120B → MLA/DSA + EAGLE3 + SpecPV

This roadmap tracks medium/long‑term work to:

- Convert `GPT‑OSS‑120B` under `models/` to a **DeepSeek‑style MLA + DSA** model via `TransMLA/`.
- Integrate the converted model into **LMDeploy/TurboMind** while keeping **EAGLE3 multi‑token speculative decoding** and **SpecPV** working.
- Incrementally improve performance and parity vs **TensorRT‑LLM** without regressing stability.

It does not attempt to describe TurboMind’s generic HF→TM format conversion or its impact on model quality; the focus here is the MLA/DSA path.

In parallel, TurboMind itself must remain a **clean, fast baseline**:

- The original non‑speculative TurboMind decode path for GPT‑OSS‑120B (int8/unquantized KV) should stay close to the historical 8da9 behaviour in both
  geometry and performance.
- New features (EAGLE3, NVFP4 KV, DriftEngine, MLA, sparse attention) are allowed to be more complex as long as:
  - They are strictly gated by config/flags.
  - They do not permanently degrade the TurboMind‑only baseline when disabled.
  - There is always an easy way to re‑run a “legacy baseline” for regression checks.

---

## Phase 1 – EAGLE3 multi‑token stability (baseline architecture)

Goal: ensure the existing GPT‑OSS‑120B + EAGLE3 pipeline is stable across all practical scenarios before layering MLA/DSA and SpecPV.

1. Stabilize single‑context EAGLE3 runs
   - [x] Single‑context 8K / 16K / 32K, `num_spec_tokens ∈ {2,3,4,5}` run without CUDA errors.
   - [x] 32K + 5 draft tokens tail crash fixed (per‑site host→device copy clamping).
   - [x] EAGLE progress markers (5–100%) and `eagle_speculation` JSON blocks present for all passing scenarios.
2. Stabilize batch EAGLE3 runs
   - [x] Batch=4, context 16K, 3‑token speculative decode runs to completion.
   - [ ] Extend to higher batch sizes / contexts as memory allows; document OOM as capacity limits, not EAGLE3 bugs.
3. Eliminate unintended shallow/fallback paths for GPT‑OSS‑120B
   - [x] EagleModule accepts GPT‑OSS‑120B‑Eagle3 draft (geometry aligned with converter/tests).
   - [x] Prefix caching bug in `LlamaBatch::AllocateBuffer` fixed.
   - [ ] For “supported” configs (e.g. single‑context 8K/16K/32K, 3 or 5 tokens), ensure logs are free of:
     - `Eagle3DraftLayer][fallback] ... shallow attention`
     - `EAGLE3][Attention][fallback] ... pass-through`
     - `UnifiedDecoder][EAGLE3][fallback] ffn_layer_ is null`
   - [ ] Confirm Eagle3 attention + FFN + LM‑head are fully active (no permanent shallow path) in those configs.
4. Harden pooled‑buffer safety and tail behaviour
   - [x] All EAGLE‑related `core::Copy` sites that hit pooled buffers now clamp to `min(host_len, dev_len)`.
   - [x] 32K + 5‑token tail crash (invalid argument in `core::Copy`) fixed via targeted clamping at the offending call.
   - [ ] Periodically re‑run spec suite when changing EAGLE/TurboMind tail logic to ensure no regressions.
5. Preserve and expose a fast TurboMind‑only baseline
   - [x] Restore pooled CUDA allocator behaviour (`CudaMemPoolAllocator`) so baseline decode no longer pays the cost of raw `cudaMalloc`/`cudaFree` and
         matches the 8da9 TurboMind allocator semantics.
   - [x] Introduce `LMDEPLOY_TURBOMIND_BASELINE_FASTPATH` env flag. When set, and when no `speculative_config` is attached to
         `TurbomindEngineConfig`, TurboMind:
         - Disables the metrics system for that engine instance (`enable_metrics=False`).
         - Leaves all numerics, KV sizing, and kernel selection unchanged.
         - Provides a low‑overhead, reproducible “fast baseline” mode for non‑speculative int8/FP16/BF16 KV, suitable for comparing main vs 8da9 and for
           guarding against future regressions.
   - [ ] Optionally add deeper C++‑level fast paths (e.g., skipping speculative‑only buffer setup and tree‑mask plumbing) guarded by the same flag, as long
         as they remain numerically identical to the default baseline.

Stability details and lower‑level EAGLE3 TODOs live primarily in:

- `EAGLE_TODO_FINAL`
- `EAGLE_TODO_OPTMIZATIONS.md`
- `SPECPV_TODO.md`

---

## Phase 2 – GPT‑OSS‑120B → MLA + DSA conversion (TransMLA)

Goal: produce a **GPT‑OSS‑120B‑MLA‑DSA** checkpoint under `models/` using `TransMLA/`, then finetune to recover quality.

5. Fix MLA/DSA target configuration for GPT‑OSS‑120B (no stubs)
   - [ ] Inspect `models/gpt-oss-120b/config.json` and record:
     - `hidden_size`, `num_attention_heads`, any `num_key_value_heads` (GQA), `num_hidden_layers`.
     - RoPE/YARN parameters (base, scaling, offset) so TransMLA’s partial‑RoPE step can be configured correctly.
   - [ ] Based on TransMLA’s README and DeepSeek MLA recommendations, pick *concrete* MLA hyperparameters for 120B:
     - `qk_mqa_dim` (e.g. 64 as suggested for FlashMLA/H100) and corresponding `collapse = head_dim // qk_mqa_dim`.
     - `kv_lora_rank` (e.g. 512 initially; adjust later if memory or quality require it).
     - Optional `q_lora_rank` if query low‑rank is enabled (or explicitly set to `None` if we keep Q dense).
   - [ ] Decide which decoder layers are converted:
     - Initial assumption: convert **all** self‑attention layers of GPT‑OSS‑120B to MLA (consistent with TransMLA examples).
     - If we later decide to keep some layers dense (e.g. first/last), document that explicitly.
   - [ ] For DSA (DeepSeek Sparse Attention), define an initial target sparsity pattern for GPT‑OSS‑120B‑MLA:
     - Local window size (e.g. 128/256), global token stride, and any block‑sparse structure suitable for 8K/16K/32K.
     - Which layers use DSA vs dense MLA.

6. Design and implement the GPT‑OSS‑120B → MLA conversion pipeline with TransMLA
   - [ ] Set up a dedicated TransMLA env (can be the existing `TransMLA/` clone):
     - Ensure it can load `models/gpt-oss-120b` via `AutoModelForCausalLM` with `trust_remote_code=True`.
     - Confirm `model.config.model_type` is one of the supported types (`llama`, `qwen2`, `mistral`, `mimo`) or note any required adapter.
   - [ ] Define the **exact** conversion command (no placeholder values), e.g.:
     - `python TransMLA/transmla/converter.py \`
       `  --model-path models/gpt-oss-120b \`
       `  --save-path models/gpt-oss-120b-mla \`
       `  --dtype bf16 \`
       `  --device cuda:0 \`
       `  --cal-dataset wikitext2 \`
       `  --cal-nsamples 128 \`
       `  --cal-max-seqlen 256 \`
       `  --cal-batch-size 8 \`
       `  --ppl-eval-batch-size 4 \`
       `  --freqfold auto \`
       `  --collapse auto \`
       `  --qk-mqa-dim <chosen_qk_mqa_dim> \`
       `  --q-lora-rank <chosen_q_lora_rank_or_None> \`
       `  --kv-lora-rank <chosen_kv_lora_rank> \`
       `  --deepseek-style`
     - Replace `<…>` with the actual numbers chosen in step 5; remove `--deepseek-style` if GPT‑OSS‑120B is not compatible with DeepSeek‑style configs.
   - [ ] Run the converter end‑to‑end once and check:
     - Converted weights and tokenizer are written under `models/gpt-oss-120b-mla`.
     - The modified `config.json` declares the DeepSeek MLA structure expected by Transformers/vLLM.
     - Basic HF `AutoModelForCausalLM.from_pretrained("models/gpt-oss-120b-mla", trust_remote_code=True)` works and can generate text.
   - [ ] Measure pre‑MLA vs post‑MLA perplexity on the calibration dataset (TransMLA already prints this):
     - Record `Original ppl`, `Partial RoPE ppl`, and `MLA ppl` for GPT‑OSS‑120B.
     - If the MLA ppl blow‑up is unacceptable, iterate on `qk_mqa_dim`, `kv_lora_rank`, or freqfold/collapse choices before moving on.

7. Add DSA (DeepSeek Sparse Attention) structure on top of GPT‑OSS‑120B‑MLA
   - [ ] From DeepSeek / FlashMLA / TRT‑LLM docs, extract a *concrete* DSA pattern that works well for 8K/16K/32K:
     - E.g., block‑sparse banded attention with additional global tokens, or DeepSeek’s documented sparse schema.
   - [ ] Decide how to encode that pattern for GPT‑OSS‑120B‑MLA:
     - Via config fields (e.g. `sparse_attention` settings) if supported by the DeepSeek/MLA modeling code.
     - Or via auxiliary metadata files (masks/indices) consumed by the runtime kernels.
   - [ ] Modify the converted MLA config and/or attention modules so that:
     - They implement the chosen DSA pattern without breaking RoPE/YARN or MLA KV semantics.
     - They remain loadable by both Transformers and any runtime (vLLM / TurboMind) that will later consume this model.
   - [ ] Save this as a distinct model dir, e.g. `models/gpt-oss-120b-mla-dsa`, to preserve the pure MLA checkpoint as a fallback.

8. Plan and implement post‑conversion finetuning (quality recovery)
   - [ ] Define *concrete* finetune objectives:
     - Base LM loss on a mixture of web/corpus data similar to the original GPT‑OSS‑120B training distribution.
     - Optional task‑oriented SFT or R1‑style distillation, if desired for downstream quality.
   - [ ] Decide parameter subsets:
     - Start with finetuning only MLA/DSA‑specific parameters (low‑rank adapters, sparse attention parameters, possibly norms).
     - Escalate to full attention‑block finetune only if quality recovery plateaus.
   - [ ] Specify training setup for 120B:
     - Global batch size, sequence length_schedule (covering 8K–32K), optimizer and LR schedule, number of steps/epochs.
     - Checkpointing and evaluation cadence (PPL on held‑out sets, a small battery of tasks).
   - [ ] Define acceptance criteria before we declare GPT‑OSS‑120B‑MLA‑DSA “good enough” to integrate into TurboMind:
     - Perplexity deltas vs original GPT‑OSS‑120B on key corpora.
     - Task metric thresholds (or “no worse than X% relative” on a chosen benchmark suite).
   - [ ] Once finetune completes and passes thresholds, mark `models/gpt-oss-120b-mla-dsa` as the **canonical MLA/DSA checkpoint** for downstream TurboMind / EAGLE3 / SpecPV work.

Progress, decisions, and open issues for this phase should be mirrored into:

- `MLA_SpecPV_EAGLE_TASKS.md`

---

## Phase 3 – Integrate GPT‑OSS‑120B‑MLA‑DSA into TurboMind (EAGLE3 + SpecPV‑safe)

Goal: make GPT‑OSS‑120B‑MLA‑DSA a first‑class TurboMind model, with KV layout, attention, RoPE/YARN, and speculative paths correctly wired.

8. Define MLA KV cache and attention semantics in TurboMind
   - [ ] Specify how MLA latent state (C) and any low‑rank K/V factors are represented in TurboMind’s KV cache abstraction.
   - [ ] Decide whether KV rewind and prefix caching operate directly on latent representations or on reconstructed K/V slices.
   - [ ] Align MLA attention masks (including DSA sparsity) with TurboMind’s masking APIs and long‑context support (8K/16K/32K).
9. Ensure EAGLE3 invariants hold under MLA/DSA
   - [ ] Decide whether:
     - Draft and target both run MLA/DSA, or
     - Only target runs MLA/DSA and draft remains dense (less likely, but possible).
   - [ ] Ensure:
     - Tokenization, RoPE/YARN, and attention layout are identical between draft and target.
     - KV reuse / tree acceptance semantics remain correct with MLA KV storage.
   - [ ] Extend EagleModule / Eagle3DraftLayer to understand MLA/DSA geometry for GPT‑OSS‑120B‑MLA‑DSA.
10. SpecPV compatibility with MLA/DSA
   - [ ] Define how SpecPV partial‑KV storage interacts with MLA latent state and DSA:
     - What gets cached (latent, reconstructed K/V, or both).
     - How windows/sinks map onto sparse patterns.
   - [ ] Update SpecPV kernels and metadata to respect MLA/DSA layouts without changing high‑level APIs.
   - [ ] Validate:
     - EAGLE3 + SpecPV runs at 16K/32K with MLA/DSA without new stability issues.

---

## Phase 4 – Kernel/performance integration (MLA/DSA + external kernels)

Goal: once MLA/DSA GPT‑OSS‑120B is integrated and stable, incrementally replace hot kernels with better implementations without breaking behaviour.

11. Inventory hot GEMMs and attention kernels for MLA/DSA GPT‑OSS‑120B
   - [ ] Identify Q/K/V GEMMs, FFN GEMMs, and LM‑head GEMMs that dominate runtime for MLA/DSA on sm90/sm120.
   - [ ] Identify attention kernel patterns (dense, MLA, DSA) that can benefit from specialized kernels.
12. Map external kernel sources to TurboMind abstractions
   - [ ] From `DeepGEMM/`, identify GEMM/attention kernels suitable for MLA/DSA (BF16/FP8, GMMA).
   - [ ] From `flashinfer/` and `FlashMLA/`, identify kernels for MLA/DSA and long‑context attention (dense/sparse).
   - [ ] From `TensorRT-LLM/` and `sglang/`, identify MLA/Sparse MLA and EAGLE3‑compatible kernels or algorithms.
   - [ ] Design how these plug into:
     - TurboMind’s GEMM registry,
     - TurboMind’s attention layers,
     - Without regressing the existing non‑MLA/EAGLE3 paths.
13. Incremental perf rollout
   - [ ] Introduce new kernels behind feature flags or arch checks (sm90/sm120) with correctness tests vs cuBLAS/current path.
   - [ ] Re‑run spec suite (8K/16K/32K, single and batch) on GPT‑OSS‑120B‑MLA‑DSA + EAGLE3(+SpecPV) to confirm stability and measure speedups.

Kernel‑level design details and concrete tasks should be tracked in:

- `EAGLE_TODO_OPTMIZATIONS.md`
- `EAGLE_OPTIMIZATION_TABLES.md`

---

## Phase 4.1 – EAGLE3‑aware FMHA and tree kernels (best‑effort design)

Goal: go beyond generic TurboMind attention and build a **first‑class EAGLE3 FMHA pipeline**, leveraging patterns from TensorRT‑LLM, FlashInfer, SGLang, and DeepGEMM to make full‑KV EAGLE3 genuinely fast (2–3× vs baseline) before enabling SpecPV.

17. GPU‑native EAGLE3 tree representation
   - [ ] Replace scalar `runtime_offsets` / `tree_offsets` / `successor_*` adjacency scans with a **GPU tree builder**, inspired by SGLang’s `build_tree_efficient`:
     - Produce a linked‑list representation per step: `retrive_index`, `retrive_next_token`, `retrive_next_sibling`.
     - Optionally emit a **bit‑packed tree mask** (per token) suitable for feeding directly into FMHA kernels.
   - [ ] Ensure the tree builder:
     - Operates purely on device buffers (no per‑step host allocations),
     - Is aware of `num_spec_tokens`, `depth`, and long‑context constraints (8K/16K/32K).

18. Tree‑aware FMHA runner for EAGLE3 (FlashInfer / TRT‑LLM style)
   - [ ] Introduce a dedicated **EAGLE3 FMHA runner** (separate from generic `AttentionUniversal`):
     - Mirrors TensorRT‑LLM’s `TllmGenFmhaKernel` design:
       - Pre‑registers a bank of FMHA kernels with meta (head dims, tile sizes, mask types, kernel type, scheduler, multi‑CTA KV mode).
       - Selects the best kernel at runtime from a `RunnerParams`‑like struct that encodes **spec‑dec generation** mode (EAGLE3) and long‑context geometry.
     - Integrates **custom mask preparation** à la TRT‑LLM’s `prepareCustomMask`:
       - Maps our bit‑packed tree mask (from step 17) into the mask format expected by the FMHA kernels.
   - [ ] Start with BF16/FP16 decode‑phase kernels for:
     - `head_dim=64` (primary GPT‑OSS‑120B‑Eagle3 draft geometry),
     - `num_heads_q=64`, `num_heads_kv=8`,
     - `seq_len_q = num_spec_tokens`, `seq_len_kv` up to at least 32K.

19. Multi‑CTA KV and reduction for long‑context spec‑dec
   - [ ] Adopt FlashInfer’s multi‑CTA KV pattern for EAGLE3 spec‑dec:
     - Split KV into tiles (`tileSizeKv ∈ {64,128}`) and compute QK across multiple CTAs per KV span.
     - Use a separate **reduction kernel** (like FlashInfer’s `runFmhaReduction`) to combine partial results for each Q token.
   - [ ] Design kernel configs so that:
     - Each speculative token gets enough CTAs to saturate SMs at 16K/32K,
     - The tile scheduler understands EAGLE3’s tree window (only visits tiles that contain valid KV tokens).

20. QKV layout alignment with external FMHA kernels
   - [ ] Align EAGLE3 Q/K/V layout and strides with FlashInfer/TRT‑LLM trtllm‑FMHA expectations:
     - Use a standard `[batch, heads, seq, head_dim]` (or flattened equivalent) so we can:
       - Either directly call FlashInfer’s trtllm FMHA from TurboMind, or
       - Mirror its kernel meta and launch conventions in a TurboMind‑native implementation.
   - [ ] Update `Eagle3AttentionLayer` to:
     - Pack Q/K/V into this layout,
     - Fill an FMHA runner params struct (not just `AttentionParams`) with spec‑dec‑specific fields (kernel type, mask type, Eagle tree mode).

21. DeepGEMM integration for EAGLE3 FC and LM‑head
   - [ ] Use `DeepGEMM/` as the primary backend for hot EAGLE3 GEMMs (FC + LM‑head), especially the problematic BF16 shapes seen on sm120:
     - Route EAGLE3 FC and LM‑head matmuls through DeepGEMM’s autotuned dispatcher when shapes and dtypes are supported.
     - Fall back to TurboMind’s `gemm2` only when DeepGEMM does not have a tuned kernel.
   - [ ] Validate:
     - No “no feasible kernel” logs remain for EAGLE3 shapes,
     - Draft compute ceases to be the bottleneck once FMHA is optimized.

22. A/B and targets before SpecPV
   - [ ] Gate the full “best‑effort” EAGLE3 FMHA path behind a dedicated flag (e.g. `TM_ENABLE_EAGLE3_TRITON_FMHA`):
     - When off: use existing streaming SDPA + gemm2 (baseline behaviour).
     - When on: GPU tree build → custom mask → EAGLE3 FMHA runner (FlashInfer/TRT‑style) + DeepGEMM FC/LM‑head.
   - [ ] Re‑run `run_spec_suite.sh` on GPT‑OSS‑120B‑Eagle3 with this path enabled:
     - 8K/16K/32K single, 16K batch4; `num_spec_tokens ∈ {2,3,4,5}`.
     - Targets before enabling SpecPV:
       - 32K single, `num_spec_tokens=3` ≥ **2×** baseline throughput.
       - 16K batch4, `num_spec_tokens=3` ≥ **1×** baseline.
   - [ ] Only after these thresholds are met and numerics are stable do we proceed to SpecPV integration for EAGLE3.

Design discussions and kernel‑level tasks for this phase should be kept in:

- `EAGLE_TODO_OPTMIZATIONS.md` (EAGLE3‑specific),
- `EAGLE_OPTIMIZATION_TABLES.md` (measured configs and speedups),
- and cross‑referenced to external repos (`TensorRT-LLM/`, `flashinfer/`, `sglang/`, `DeepGEMM/`) where we mirror or reuse kernels.

---

## Phase 5 – Evaluation, parity, and documentation

Goal: evaluate GPT‑OSS‑120B‑MLA‑DSA + TurboMind + EAGLE3(+SpecPV) vs the baseline GPT‑OSS‑120B and vs TensorRT‑LLM.

14. Accuracy / quality evaluation
   - [ ] Define benchmark suites (PPL + downstream tasks) to compare:
     - Baseline GPT‑OSS‑120B (non‑MLA) vs GPT‑OSS‑120B‑MLA‑DSA (post‑finetune).
     - EAGLE3 vs EAGLE3+SpecPV on both models.
   - [ ] Use `eagle3_compare.py` (and TRT‑LLM equivalents) when official GPT‑OSS‑Eagle3 weights are available to assess stagewise differences.
15. Performance and acceptance metrics
   - [ ] For 8K/16K/32K, batch 1 / 4 / N:
     - Compare baseline vs EAGLE3 vs EAGLE3+SpecPV on:
       - Throughput (tok/s), latency (ms/tok), memory usage.
       - EAGLE metrics (mean_accept_rate, mean_accept_len, derived per‑step acceptance).
   - [ ] Run side‑by‑side comparisons vs TensorRT‑LLM’s EAGLE3 benchmarks where possible (matching context, batch, num_spec_tokens).
16. Summarize recommended configs and trade‑offs
   - [ ] In `EAGLE_TODO_OPTMIZATIONS.md` and `EAGLE_OPTIMIZATION_TABLES.md`, document:
     - Recommended configurations for GPT‑OSS‑120B‑MLA‑DSA under TurboMind (contexts, batch sizes, num_spec_tokens).
     - When to enable SpecPV (context lengths, memory budgets).
     - Any known deviations vs TensorRT‑LLM (tree structure, masks, acceptance rules) that are intentional.

This roadmap should stay high‑level and forward‑looking; detailed per‑milestone work items and low‑level design discussions belong in the more focused EAGLE/MLA/Spe
