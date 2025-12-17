# GPT-OSS + EAGLE3 + SpecPV + MLA/DSA – Task & Strategy Doc

This document tracks the current work streams around:

- TurboMind EAGLE3 speculative decoding for GPT-OSS.
- SpecPV-style partial KV verification.
- MLA-lite KV compression and DSA-lite sparse attention retrofits.
- The proposed **Strategy B** finetune plan (retrofit + train only new heads).

It is intended as a living engineering plan, not a paper.

---

## 1. Current Work Streams & Tasks

### 1.1 EAGLE3 target-tree decode & draft layer

Code anchors:
- `lmdeploy/src/turbomind/models/llama/LlamaV2.cc`
- `lmdeploy/src/turbomind/models/llama/EagleModule.{h,cc}`
- `lmdeploy/src/turbomind/models/llama/EagleDraftLayer.{h,cc}`
- `lmdeploy/src/turbomind/models/llama/unified_decoder.{h,cc}`
- `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/*`

Status (from `EAGLE_TODO.md` / `EAGLE_TODO_COMPLETE` and code):

- Target-tree decode:
  - Scratch KV tree decode and packed tree masks are implemented and wired via `LlamaV2::runEagleTargetTreeDecode`.
  - Logits for tree tokens are produced into a dedicated FP32 buffer and converted to `target_tokens` via device argmax kernels (e.g. `target_tree_decode.cu`).
  - Acceptance in `dynamicDecodeWithSpecMulti` is driven by device-side `invokeTreeAcceptByIdsWithPaths` using real tree `target_tokens`.
- Eagle3 draft:
  - Draft inputs are captured from selected LLaMA layers (`UnifiedDecoder` multi-layer capture).
  - A lightweight Eagle3 draft block is available in `EagleModule` and a separate `Eagle3DraftLayerWeight` holder.
  - `UnifiedDecoder::ForwardDraft` routes via `Eagle3DraftLayer` and `postDecodeEmbedding` to produce draft logits in the main engine.
  - Bounded device top‑k over draft logits mirrors `EagleModule::forward_draft_tree` to fill `EagleBuffers::inputs.draft_tokens` / host mirrors.

Remaining / refinement tasks:

- Tighten Eagle3 draft numerics vs HF / TRT reference:
  - Use `eagle_forward_logits_debug` and `lmdeploy/tests/turbomind/eagle3_compare.py` to compare:
    - FC output, attention output, FFN output, pre-head hidden, logits.
  - Iterate residual placements / norms and capture-layer choices until stage-wise cosines and argmax/top‑k alignment are acceptable.
- Continue hardening multi-token EAGLE3 integration:
  - Ensure `eagle_max_engine_tokens_per_step_` and per-slot gating drive multi-token plans consistently.
  - Keep KV rewind, acceptance metrics, and dynamic decode stop criteria aligned with TensorRT‑LLM Eagle3 semantics.

### 1.2 SpecPV-style partial KV verification

Code anchors:
- `lmdeploy/src/turbomind/models/llama/specpv_kv_cache.{h,cc}`
- `lmdeploy/src/turbomind/models/llama/LlamaV2.cc` (SpecPV gating + update)
- `lmdeploy/SPECPV_TODO.md`

Status:

- `SpecPVCacheConfig`:
  - Encodes block geometry: `block_size`, `n_sink_blocks`, `n_retrieval_blocks`, `n_window_blocks`, `n_spec_tokens_buf`.
  - Provides derived sizes: `sink_size`, `retrieval_size`, `window_size`, `buffer_size`, `total_budget`.
- `PartialKVCache`:
  - Owns per-layer KV buffers and splits them into sink / retrieval / window / buffer segments.
  - Maintains:
    - `verified_lens_` per layer and `global_verified_len_`.
    - Block summaries (`key_summary_max_`, `key_summary_min_`) and counts.
    - Candidate-region bookkeeping (staged speculative tokens).
  - Implements:
    - `summary_key_states` (SpecPV Eq. (1)-style summaries).
    - `refresh_retrieval` (block scoring + retrieval/window selection).
    - `stage_candidates` / `promote_candidates` for speculative tokens.
    - `update` / `active_prefix` / `reset_buffer` for verified buffer management.
    - `update_after_acceptance` as a conservative safety hook (budget checks).
- `LlamaV2` SpecPV wiring:
  - Gating via `isSpecPVEnabled()` / `shouldUseSpecPV(seq_len)`.
  - `initSpecPVFromFullKV` seeds sink/retrieval/window from full KV.
  - `updateSpecPVAfterAcceptance`:
    - Uses `sequence_length` to decide when SpecPV is active.
    - Seeds from full KV on first long-context step.
    - Incrementally calls `PartialKVCache::update()` to append new tail tokens into the buffer and drives full-refresh cycles when buffer is nearly full or partial steps exceed a threshold.
    - Calls `update_after_acceptance` today for basic invariants (budget checks).

Remaining / refinement tasks:

- Finish end-to-end partial KV usage in target-tree decode:
  - Extend `LlamaV2::runEagleTargetTreeDecode` to select between full‑KV and partial‑KV modes based on `shouldUseSpecPV`.
  - In SpecPV mode, build the effective KV view from:
    - sink + retrieval + window + buffer slices in `PartialKVCache`.
    - Append tree tokens into the buffer via `update` or staged candidates, then run the decode over that partial KV.
- Strengthen SpecPV bookkeeping:
  - Wire `update_after_acceptance` to meaningful per-slot advances rather than a constant advance of 1:
    - Use committed tail lengths from `DynamicDecodeLayer` / `ForcedTailContext`.
    - On each multi-token step, call `update_after_acceptance(slot, committed_lengths[slot], new_seq_len)` for each layer.
  - Ensure full steps (full-refresh) reset `verified_lens_`, candidates, and `global_verified_len_` correctly.
- Add more GPU-backed tests for SpecPV:
  - Invariants on `verified_lens_` / `global_verified_len_`.
  - Partial vs full KV consistency checks on synthetic KV inputs.

### 1.3 MLA-lite KV compression (approximate MLA)

Goal:

- Add MLA-lite **latent KV** to GPT‑OSS / TurboMind so that:
  - The KV cache stores a smaller latent vector `c_t` per token (`d_c << 2 * d_k_total`).
  - K/V are reconstructed from `c_t` on demand for attention.
  - The path works in **prefill** and **decode**, with or without EAGLE/speculative decoding, and can be toggled by config.

#### 1.3.1 Config & weights

- Config:
  - Extend model config (YAML / JSON) with:
    - `attention.use_mla_kv: bool`
    - `attention.mla_latent_dim: int` (d_c, e.g. 256/512).
  - In C++ (TurboMind):
    - Add fields to `AttentionParam` / equivalent:
      - `bool use_mla_kv{false};`
      - `int  mla_latent_dim{0};`
- Weights (`LlamaAttentionWeight`):
  - Existing:
    - `qkv` / `q_proj`, `kv_a_proj` / `kv_b_proj` etc. (GPT‑OSS attention).
  - MLA-lite additions per layer:
    - `bool   use_mla_kv{false};`
    - `int    mla_latent_dim{0};`
    - `Tensor mla_W_c;      // [d_model, d_c]`
    - `Tensor mla_W_k_lat;  // [d_c, n_kv_heads * d_head]`
    - `Tensor mla_W_v_lat;  // [d_c, n_kv_heads * d_head]`
  - Loader:
    - If MLA tensors (e.g. `model.layers.{L}.self_attn.mla_W_*`) are present, load them and set `use_mla_kv=true`, `mla_latent_dim=mla_W_c.shape(1)`.
    - Keep original K/V projections around for fallback and A/B.
- Checkpoint surgery (Phase‑1 script):
  - `scripts/gptoss_mla_kv_factor.py`:
    - Reads GPT‑OSS HF shards and `model.safetensors.index.json`.
    - For each layer, loads `self_attn.k_proj.weight` / `v_proj.weight` (W_k, W_v) and performs low-rank SVD on `[W_k | W_v]`:
      - Produces `W_c`, `W_k_lat`, `W_v_lat` with rank `d_c`.
    - Saves MLA-lite tensors into a standalone safetensors file:
      - `model.layers.{L}.self_attn.mla_W_c`
      - `model.layers.{L}.self_attn.mla_W_k_lat`
      - `model.layers.{L}.self_attn.mla_W_v_lat`.

#### 1.3.2 KV-cache layout (with latent C)

- Extend KV cache to store latent C per layer:
  - Existing:
    - `K_cache[layer, batch, n_kv_heads, max_seq, d_head]`
    - `V_cache[layer, batch, n_kv_heads, max_seq, d_head]`
  - New latent cache:
    - `C_cache[layer, batch, max_seq, d_c]`
- KV view helpers:
  - `LayerKvView` / equivalent should expose:
    - Pointers/slices for `k`, `v`, and `c`.
    - Logical sequence length per layer (`current_len`).
  - KV management (rewind/copy/gather) must treat `c` exactly like K/V:
    - `rewind(new_len)` updates logical length for K, V, and C.
    - `copy_prefix(src, dst, len)` copies C segments along with K/V.

#### 1.3.3 LlamaAttention forward paths

Assume a `UnifiedAttentionLayer`/`LlamaAttention` style interface with separate **prefill** and **decode** flows.

- Prefill (B×T segment):
  - Projection:
    - Always compute Q from hidden states:
      - `Q = H @ W_q`.
    - If `use_mla_kv == false`:
      - Compute `K = H @ W_k`, `V = H @ W_v` as today.
    - If `use_mla_kv == true`:
      - Compute latent:
        - `C = H @ W_c` → `[B, T, d_c]`.
      - Reconstruct K/V for this segment:
        - `K = C @ W_k_lat`, `V = C @ W_v_lat`.
  - RoPE:
    - Apply existing RoPE kernels to Q and K as usual (per-head reshape).
  - KV caching:
    - Baseline path:
      - Store K/V into KV cache for positions `[0..T-1]`.
    - MLA-lite path:
      - Store C into `C_cache` for positions `[0..T-1]`.
      - Optionally store K/V too for debugging during rollout.
  - Attention:
    - For prefill, use the dense K/V computed from this H (no change to attention kernels).

- Decode (single step t):
  - Hidden input for step t: `h_t` (`[B, d_model]`).
  - If `use_mla_kv == false`:
    - Same as today: compute `q_t`, `k_t`, `v_t`, store `k_t/v_t` in caches, attend over full K/V history.
  - If `use_mla_kv == true`:
    - Projections:
      - `q_t = h_t @ W_q`, `c_t = h_t @ W_c`.
    - Cache:
      - Store `c_t` into `C_cache` at position `t`.
    - Reconstruct history:
      - Fetch `C_hist = C_cache[:, :t+1, :]`.
      - Compute:
        - `K_hist = C_hist @ W_k_lat`, `V_hist = C_hist @ W_v_lat`.
      - Apply RoPE to `q_t` and `K_hist` (per-position, per-head).
      - Reshape to `[B, n_kv_heads, t+1, d_head]`.
    - Attention:
      - Call existing decode attention kernel with `q_t`, `K_hist`, `V_hist`.
    - Complexity:
      - This keeps O(T²) attention but adds GEMMs for `C_hist @ W_k_lat` / `C_hist @ W_v_lat`. Phase‑1 focuses on correctness + memory layout, not compute wins.

#### 1.3.4 GEMM / CUDA placement

- Reuse existing GEMM wrappers and cuBLAS/cuBLASLt paths used for Q/K/V projections:
  - Prefill:
    - `H_flat = [B*T, d_model]`, `W_c: [d_model, d_c]`, `W_k_lat/W_v_lat` same pattern.
  - Decode:
    - `h_t: [B, d_model]` for `q_t`, `c_t`.
    - `C_hist_flat = [B*(t+1), d_c]` for reconstructing K/V.
- No changes to RoPE kernels or core attention CUDA templates are needed for Phase‑1; they see dense Q/K/V as before.

#### 1.3.5 Speculative decoding / EAGLE interaction

- KV rewinds:
  - Wherever `SequenceManager` / KV helpers truncate or rewind KV (`current_len` changes), ensure `C_cache` uses the same logical length as K/V.
  - No P2P copies are required beyond what already exists for KV; just include `c` in prefix copy operations.
- Branching / tree decode:
  - Any KV branch, duplication, or gather (e.g. for EAGLE trees or beam search) must copy `c` segments alongside K/V.
  - Provide a `copy_prefix` helper that copies K, V, and C consistently.
- Draft vs target:
  - Draft (EAGLE) model may remain non‑MLA; MLA-lite applies to the **target** GPT‑OSS decoder.
  - Easiest integration:
    - After each accepted prefix (prompt + accepted draft tokens), rebuild target KV via a prefill pass:
      - This fills `C_cache` and ensures subsequent decode runs on MLA-lite KV.
  - More advanced synchronized KV (sharing KV between draft/target) is a later optimization; Phase‑1 only requires target-side MLA-lite correctness.

#### 1.3.6 Validation tasks

- Sanity and A/B:
  - With `rank = d_k_total + d_v_total` (full‑rank), verify:
    - `||W_k - W_c @ W_k_lat||` and `||W_v - W_c @ W_v_lat||` are near machine epsilon.
    - Prefill+decode logits match baseline within numeric noise on small prompts.
  - With reduced ranks (`d_c = 768, 512, 256`), log:
    - Relative reconstruction error norms for K/V.
    - Perplexity deltas on small validation sets.
  - Measure KV memory footprint:
    - Confirm KV cache reduces from `2 * d_k_total` floats/token/layer to `d_c` floats/token/layer in MLA mode.


### 1.4 DSA-lite sparse attention (DeepSeek-style indexing)

Goal:

- Introduce a **learned sparse mask** for long-context attention, inspired by DSA:
  - At each step, select a limited set of past positions `S(t)` (e.g. k ≈ 1024–2048).
  - Restrict attention to `S(t)` instead of all tokens.
  - Use a small learned indexer, trained to approximate full attention behavior.

Conceptual design:

- Per_layer indexer:
  - Add `W_indexer`: hidden_dim → d_idx (e.g. 256).
  - Compute `z_i = h_i W_indexer` for past tokens and `z_t = h_t W_indexer` for the current token.
  - Score: `score(t, s) = f(z_t, z_s)` (e.g. ReLU dot product).
  - Select top‑k positions `S(t)` by score; form a sparse mask that only allows attention over `S(t)` (plus any mandatory tokens such as sink / BOS).
- Integration:
  - Implement DSA-lite masks in the TurboMind attention backend as an optional path:
    - For short contexts: keep full attention.
    - For long contexts (e.g. > 8k–16k): apply the DSA-lite mask.
  - Combine naturally with SpecPV:
    - SpecPV reduces the **sequence length** seen by verification;
    - DSA-lite further sparsifies attention within that partial KV.

Near-term tasks:

- Implement an indexer API in C++ / CUDA:
  - Given `h_i`, produce `z_i` and a per-query top‑k index set.
  - Pack `S(t)` into a GPU-friendly sparse mask (e.g. packed mask or index list).
- Wire into attention kernels:
  - Add a mask mode that restricts the QK computations to `S(t)` per token.
  - Start with a host-side prototype: compute `S(t)` on CPU for debugging, then move to device-side kernels.
- Gating and safety:
  - Gated by context length and configuration.
  - Fall back to full attention if the indexer misbehaves (e.g. fails invariants, or acceptance/quality metrics drop beyond thresholds).

---

## 2. Strategy B – Retrofit + Finetune (MLA-lite + DSA-lite)

This section details **Strategy B**: retrofit MLA-lite KV compression and DSA-lite sparse attention onto GPT‑OSS‑120B and finetune only the new parameters.

### 2.1 What gets trained (and what stays frozen)

- GPT‑OSS‑120B backbone:
  - ~117B total parameters.
  - 36 layers, MoE with ~5.1B active parameters per token.
  - **Frozen** in Strategy B.
- New parameters (trained):
  - MLA-lite KV compression:
    - Per layer:
      - `W_c`: 2880 → d_c (e.g. d_c ≈ 512).
      - `W_k_latent`: d_c → 64 * 64.
      - `W_v_latent`: d_c → 64 * 64.
    - Roughly 5–6M parameters per layer → ~200M total.
  - DSA-lite indexer:
    - Per layer:
      - `W_indexer`: 2880 → d_idx (e.g. 256).
      - Additional small head-wise weights & thresholds.
    - Roughly 1M / layer → ~36M total.
  - Optional:
    - Tiny LoRA blocks on selected attention / output heads if needed to repair edge cases.
- Total new trainable parameters:
  - On the order of **~250M parameters** (plus any small LoRA), with **all GPT‑OSS weights frozen**.

### 2.2 Precision & memory (NVIDIA vs TPU)

- Training precision:
  - Train in **BF16 (or FP8)**.
  - GPT‑OSS MXFP4 weights are used for inference; training uses a higher‑precision representation and then re‑quantizes if needed.
- Weight memory (BF16):
  - 117B frozen weights × 2 bytes ≈ 234 GB.
  - ~250M new weights × 2 bytes ≈ 0.5 GB.
  - With ZeRO‑style sharding and tensor/expert parallelism:
    - **8×H100 80 GB** is a comfortable baseline (weights + activations + optimizer for new params).
    - On TPUs (v5p / v6e), think on the order of **16–32 chips** for similar headroom.
- Optimizer state:
  - Only allocated for new params (≈250M):
    - Roughly 2–3 GB total across all devices for Adam (weights + m + v).
  - Gradients for new params are similarly lightweight.
- Activations:
  - Dominated by the frozen GPT‑OSS backbone (5.1B active params per token).
  - With activation checkpointing and reasonable sequence/batch sizes, 8×H100 80 GB is sufficient.

### 2.3 Data requirements (tokens)

Since Strategy B is **distillation / finetune**, not pretraining, token budgets are modest compared to full training:

- Tier 1 – Minimal MLA-lite only (no DSA at first):
  - Goal: 2× KV compression, minimal quality loss.
  - Data: ~0.5–1 B tokens.
  - Objective: distill student logits/hidden states to match the original GPT‑OSS backbone running full KV.
- Tier 2 – MLA-lite + initial DSA-lite:
  - Goal: 2–4× KV compression, DSA-lite that sparsifies long-range attention while preserving quality.
  - Data: ~2–10 B tokens.
  - Training:
    - First phase: distill MLA-lite compression on 1–5 B tokens (logits/hidden MSE or KL).
    - Second phase: train DSA indexer using either:
      - Teacher attention distributions (if accessible), or
      - Teacher hidden/logits after full attention, comparing against sparse-attention student.
- Tier 3 – Aggressive MLA/DSA tuned for long-context:
  - Goal: 4–8× KV compression, strong DSA-lite on contexts up to 128k.
  - Data: ~10–20 B tokens, including:
    - Long-context sequences (tens of k tokens).
    - Task-specific data (code, math, tool traces, CoT) to keep reasoning behavior strong.

### 2.4 Training time & cost (order-of-magnitude)

Reference point:

- NVIDIA reports GPT‑OSS‑120B pretraining consumed roughly **2.1M H100 GPU-hours** for ~100 T tokens (order of magnitude), i.e. ~5×10⁷ tokens per H100-hour under highly optimized training.

For Strategy B, assuming conservative throughput (≈30–50 M tokens / H100-hour), ballpark GPU-hours:

- **1 B tokens** → ~20–35 H100-hours.
- **5 B tokens** → ~100–170 H100-hours.
- **10 B tokens** → ~200–350 H100-hours.
- **20 B tokens** → ~400–700 H100-hours.

Wall-clock examples:

- On **8×H100 80 GB**:
  - 5 B tokens (~150 GPU-hours) → ≈19 hours of training.
- On **32×H100**:
  - 10 B tokens (~250 GPU-hours) → ≈8 hours.

At a notional **$2 / H100-hour**:

- 1 B tokens → ≈$60 in raw GPU time.
- 5 B tokens → ≈$300.
- 10 B tokens → ≈$600.
- 20 B tokens → ≈$1.2k.

These are **core training costs only**; real project cost is dominated by engineering, experimentation, and evaluation.

### 2.5 Losses & training objectives

Suggested loss stack for Strategy B:

- MLA-lite KV compression:
  - Distillation from the frozen GPT‑OSS teacher:
    - KL divergence between student and teacher logits.
    - Optional MSE between selected hidden layers (e.g. last hidden, mid-layer features).
  - Optionally add regularizers encouraging stable behavior at long context (e.g. small L2 change in logits when compressing KV).
- DSA-lite indexer:
  - If teacher attention weights are accessible:
    - Cross-entropy loss between teacher attention map and a sparse approximation induced by the indexer.
  - Otherwise:
    - MSE / KL between:
      - Teacher hidden/logits (full attention),
      - Student hidden/logits (sparse attention with indexer).
- Multi-stage schedule:
  1. Train MLA-lite only (full attention; no DSA) until student matches teacher well.
  2. Freeze MLA-lite, enable DSA-lite sparse masks, train the indexer (and possibly small LoRA) to recover teacher performance.
  3. Optionally fine-tune both MLA-lite and indexer jointly in a final polishing phase.

### 2.6 Deployment implications

- On NVIDIA GPUs:
  - Inference using TensorRT‑LLM / TurboMind can:
    - Keep MXFP4 weights for GPT‑OSS backbone.
    - Use MLA-lite latent KV to shrink KV cache and bandwidth.
    - Use DSA-lite masks in attention kernels for long-context steps.
  - EAGLE3 + SpecPV stack:
    - EAGLE3 reduces decode steps (multi-token acceptance).
    - SpecPV reduces KV positions for target-tree verification.
    - MLA-lite reduces KV footprint per position.
    - DSA-lite further sparsifies attention within those positions.
- On TPUs:
  - Training in BF16 and serving in BF16 is straightforward.
  - For GPU deployment, export and quantize to MXFP4 as a separate step if desired.

---

## 3. Next Steps (Suggested)

1. **Lock in MLA-lite geometry**:
   - Choose `d_c` and where MLA-lite is applied (all layers vs last N layers).
   - Prototype SVD-based initialization on a subset of layers.
2. **Add engine/config flags and stubs**:
   - Introduce config options for:
     - `enable_mla_kv_compression`, `mla_latent_dim`, etc.
     - `enable_dsa_indexer`, `dsa_top_k`, and context thresholds.
   - Stub out MLA-lite KV and DSA-lite mask paths in TurboMind / UnifiedAttentionLayer.
3. **Prototype no-training MLA-lite**:
   - Implement latent KV storage + on-the-fly reconstruction with SVD init.
   - Measure:
     - KV memory reduction.
     - Throughput at long context (with and without SpecPV).
     - Basic perplexity / accuracy deltas.
4. **Plan Strategy B finetune runs**:
   - Decide token budget (e.g. 2–5 B for an initial serious run).
   - Prepare a distillation dataset (mixture of web/code/math, with long-context coverage).
   - Set up training scripts for:
     - MLA-lite only (phase 1).
     - DSA-lite indexer (phase 2).
5. **Integrate with EAGLE3 + SpecPV**:
   - Ensure MLA-lite and DSA-lite paths are compatible with:
     - EAGLE3 draft/target-tree decode layouts.
     - SpecPV partial KV buffer and active-prefix views.
   - Add metrics and debug logging to compare:
     - Base vs EAGLE3 vs EAGLE3+SpecPV vs EAGLE3+SpecPV+MLA/DSA.

This doc should stay in sync with `EAGLE_TODO.md`, `EAGLE_TODO_COMPLETE`, and `SPECPV_TODO.md` as the implementation progresses. 

---

## 4. HF LlamaMLA past_key_values layout (inspect_mla_past_kv.py)

To ground TurboMind’s MLA cache design, we added `LM/lmdeploy/inspect_mla_past_kv.py` and ran it against `models/gpt-oss-120b-mla-dsa`.

Key findings:

- Config:
  - `architectures = ["LlamaMLAForCausalLM"]`
  - `kv_lora_rank = 512`
  - `qk_rope_head_dim = 64`, `qk_nope_head_dim = 64`, `qk_head_dim = 128`
  - `v_head_dim = 64`
  - `num_hidden_layers = 36`
- `past_key_values`:
  - For both the initial forward (prefill) and a second step with `past_key_values` fed back in:
    - `past_key_values` is a list of length 36.
    - Each entry is a 2-tuple, but both elements are `None`:
      - `Layer XX: type=tuple, n_parts=2, part0=None, part1=None` for all layers.
- Interpretation:
  - This LlamaMLA variant uses the new `Cache` abstraction in `transformers.cache_utils`.
  - The real K/V (or latent C) tensors are stored inside the `Cache` object; the tuple returned in `past_key_values` only carries placeholders.
  - There is no explicit dense or latent KV tensor layout exposed through the HF API for us to mirror.

Implications for TurboMind MLA:

- We must define our own MLA KV/latent cache tensors:
  - e.g. a `C_cache` of width `kv_lora_rank` plus derived dense K/V views as needed.
  - KV rewind / prefix-copy helpers need to operate on this latent cache, not on HF-style `past_key_values`.
- We should *not* try to treat HF `past_key_values` as a ground-truth tensor layout; instead, we take MLAAttention’s math (latent + up-projections + RoPE split) as the source of truth and design TurboMind’s cache and rewind semantics accordingly.

## HF MLP / FFN structure vs checkpoint (Gate A)

- `inspect_llama_mla_modules.py` on `models/gpt-oss-120b-mla-dsa` shows:
  - Runtime MLP module type is `transformers.models.llama.modeling_llama.LlamaMLP` (dense FFN).
  - For layer 0 and layer 18:
    - `gate_proj.weight`, `up_proj.weight`, and `down_proj.weight` all exist with shape `(2880, 2880)`.
- On-disk checkpoint inspection across all `model-*.safetensors` shards shows:
  - No keys matching `model.layers.*.mlp.gate_proj.weight`, `model.layers.*.mlp.up_proj.weight`, or `model.layers.*.mlp.down_proj.weight`.
  - Many keys of the form `model.layers.*.mlp.experts.*` and `model.layers.*.mlp.router.*` are present.
- Conclusion (Gate A):
  - The HF model we instantiate has a **dense LlamaMLP FFN** (gate/up/down), but the **safetensors checkpoint only contains MoE-style `mlp.experts.*` and `mlp.router.*` weights**.
  - The dense MLP weights used at runtime are newly initialised in memory, not present as `mlp.gate_proj/up_proj/down_proj` tensors in the checkpoint files.
  - TurboMind’s current FFN reader, which expects on-disk `mlp.gate_proj/up_proj/down_proj` tensors, therefore sees missing keys and fails during export.
