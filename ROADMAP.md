# LMDeploy / TurboMind Roadmap – GPT‑OSS‑120B → MLA/DSA + EAGLE3 + SpecPV

This roadmap tracks medium/long‑term work to:

- Convert `GPT‑OSS‑120B` under `models/` to a **DeepSeek‑style MLA + DSA** model via `TransMLA/`.
- Integrate the converted model into **LMDeploy/TurboMind** while keeping **EAGLE3 multi‑token speculative decoding** and **SpecPV** working.
- Incrementally improve performance and parity vs **TensorRT‑LLM** without regressing stability.

It does not attempt to describe TurboMind’s generic HF→TM format conversion or its impact on model quality; the focus here is the MLA/DSA path.

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
