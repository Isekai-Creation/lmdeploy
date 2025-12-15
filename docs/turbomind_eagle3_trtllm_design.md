# TurboMind Eagle‑3 vs TensorRT‑LLM – Design Notes

This document captures the working spec for matching TurboMind’s
EAGLE‑3 path to TensorRT‑LLM on **GPT‑OSS‑120B‑Eagle3**. It is the
reference for Phase 0 / 1 items in `EAGLE_TODO_FINAL`.

The focus here is:

- how TensorRT‑LLM wires Eagle‑3 into the base model,
- what tensor shapes we must reproduce stage‑by‑stage,
- how those concepts map to TurboMind (`EagleModule`, `UnifiedDecoder`,
  `Eagle3DraftLayer`, KV/cache layout).

Where code snippets are implied, they are taken from:

- TensorRT‑LLM:
  - `tensorrt_llm/models/eagle/model.py`
  - `tensorrt_llm/_torch/speculative/eagle3.py`
  - `tensorrt_llm/_torch/models/modeling_speculative.py`
- EAGLE repo:
  - `EAGLE/eagle/model/cnets.py`

The goal is to stay descriptive and avoid introducing new behaviour
that diverges from these sources.

---

## 1. Eagle‑3 Inference Loop (TensorRT‑LLM, GPT‑OSS‑120B)

This section describes a *single step* of Eagle‑3 speculative decoding
in TensorRT‑LLM, in the **one‑model** setup used for
GPT‑OSS‑120B‑Eagle3.

### 1.1 Target model forward and hidden capture

- The base model is a LLaMA‑family decoder (GPT‑OSS‑120B) wrapped in
  TensorRT‑LLM’s `DecoderModelForCausalLM`.
- During a normal forward pass, TensorRT‑LLM drives it with an
  `AttentionMetadata` instance that carries:
  - per‑request `num_seqs`, `num_contexts`, `num_generations`,
  - context/generation lengths, KV cache pointers, etc.
- For Eagle‑3 **one‑model**:
  - `Eagle3OneModelSpecMetadata` (in `_torch/speculative/eagle3.py`)
    holds a `hidden_states` buffer and a set of `layers_to_capture`.
  - `layers_to_capture` defaults to:
    - `(1, num_layers // 2 - 1, num_layers - 4)` when `num_layers > 5`,
    - otherwise, a single or few layers, depending on depth.
  - On each captured layer, after the layer’s RMSNorm + residual,
    `maybe_capture_hidden_states`:
    - writes the **last token** hidden states for each sequence into
      `hidden_states[:, i * hidden_size : (i + 1) * hidden_size]`,
      where `i` indexes the capture slot.
  - The resulting layout after a target step is:
    - `hidden_states`: `[max_tokens, hidden_size * num_capture_layers]`.

For the two‑model Eagle‑3 path (`Eagle3ResourceManager`) the capture
logic is similar, but the buffer is shared across target and draft
models. TurboMind’s `UnifiedDecoder` follows the **one‑model** capture
pattern.

### 1.2 Eagle‑3 FC, draft model, and logits

- In TensorRT‑LLM the Eagle‑3 draft is represented by
  `Eagle3DraftModel` / `Eagle3ForCausalLM`
  (`_torch/models/modeling_speculative.py`).
- Draft hidden dimensionalities:
  - `config.pretrained_config.hidden_size` – base model hidden,
  - `spec_config.num_capture_layers` – number of captured layers,
  - `hidden_size_in`:
    - `target_hidden_size` if present in the pretrained config,
    - otherwise `hidden_size`.
- Pre‑FC:
  - `Eagle3DraftModel.fc` is constructed when
    `num_capture_layers > 1` as:
    - `Linear(hidden_size_in * num_capture_layers, hidden_size)`.
  - `Eagle3ForCausalLM.apply_eagle3_fc`:
    - takes concatenated captured hidden states
      `[tokens, hidden_size_in * num_capture_layers]`,
    - if the last dimension is not already `hidden_size`, applies
      `fc` to reduce it to `[tokens, hidden_size]`.
- Draft model forward:
  - `Eagle3ForCausalLM.forward`:
    - calls `apply_eagle3_fc(spec_metadata.get_hidden_states())`,
    - feeds the result as `hidden_states` into `Eagle3DraftModel`.
  - Inside `Eagle3DraftModel.forward`:
    - token embeddings are looked up via `embed_tokens` (possibly
      shared with the target model),
    - mid‑layers are `Eagle3DecoderLayer` blocks that perform
      RMSNorm + attention + FFN with Eagle‑3 geometry,
    - a final RMSNorm (`self.norm`) returns both
      **pre‑head** and **post‑norm** hidden states.
  - The LM head:
    - either reuses the target LM head, or an Eagle‑specific head,
      depending on the checkpoint mapping,
    - produces `logits` for the draft tokens.

### 1.3 Tree construction, sampling, and acceptance

- Eagle‑3 in TensorRT‑LLM uses a **static tree** per step:
  - the tree structure (fan‑out and depth) is encoded by
    `SpecTreeManager` and the Eagle decoding config
    (`EagleDecodingConfig`),
  - `Eagle3OneModelWorker` and related samplers manage:
    - how many draft tokens to generate (`max_draft_len`),
    - per‑layer expansion, scores, and parents.
- For the one‑model path:
  - after obtaining draft logits, `Eagle3OneModelWorker`:
    - samples draft tokens greedily (or via Top‑k / multinomial,
      depending on config),
    - builds a tree of candidate continuations for each request,
    - compares draft tokens against target logits over a window of
      positions to compute `accepted_tokens` and
      `num_accepted_tokens`.
- KV/cache behaviour:
  - acceptance indices are used to:
    - rewrite the **next** draft input IDs,
    - update tree masks and packed masks for the next step,
    - compute KV rewind lengths (number of speculatively decoded
      tokens that were rejected and must be dropped from KV).
  - KV layout and rewind semantics are wired through
    `KVCacheParams` and `AttentionMetadata` and are exercised by
    tests like `test_kv_cache_reuse.py`.

For TurboMind, `EagleModule` owns the tree structure and KV rewind
helpers; `LlamaV2::dynamicDecodeWithSpecMulti` coordinates tree decode,
acceptance, and calls into `DynamicDecodeLayer::ForwardMultiStep` to
commit accepted tails.

---

## 2. Eagle‑3 Draft Tensor Shapes (GPT‑OSS‑120B‑Eagle3)

This section summarises the tensors we need to match at each stage of
the Eagle‑3 draft path, based on:

- the HF **draft checkpoint** (`nvidia/gpt-oss-120b-Eagle3`),
- EAGLE’s PyTorch reference (`eagle/model/cnets.py`),
- TensorRT‑LLM’s Eagle‑3 draft model.

### 2.1 HF / EAGLE midlayer (Eagle‑3 draft)

From the HF Eagle‑3 midlayer checkpoint and EAGLE code:

- FC over captured hidden:
  - `fc.weight`:
    - shape `[hidden, 3 * hidden]` for GPT‑OSS‑120B‑Eagle3,
    - for this checkpoint: `[2880, 8640]`.
- Midlayer attention projections (non‑LLaMA geometry):
  - `midlayer.self_attn.q_proj.weight`:
    - `[q_out, q_in]` = `[4096, 2880]`,
  - `midlayer.self_attn.k_proj.weight`:
    - `[kv_out, q_in]` = `[ 512, 2880]`,
  - `midlayer.self_attn.v_proj.weight`:
    - `[kv_out, q_in]` = `[ 512, 2880]`,
  - `midlayer.self_attn.o_proj.weight`:
    - `[q_in, q_out]` = `[2880, 4096]`.
- In EAGLE’s `LlamaDecoderLayeremb` (`cnets.py`):
  - `self.self_attn` is a `LlamaAttention` whose projections are:
    - `q_proj: Linear(2 * hidden_size, num_heads * head_dim)`,
    - `k_proj: Linear(2 * hidden_size, num_kv_heads * head_dim)`,
    - `v_proj: Linear(2 * hidden_size, num_kv_heads * head_dim)`,
    - `o_proj: Linear(num_heads * head_dim, hidden_size)`.
  - Input to attention is:
    - `hidden_states = cat([input_emb_norm, hidden_norm], dim=-1)`,
    - shape `[batch, seq, 2 * hidden_size]`.
- From the HF weights above we therefore know:
  - `q_in` = `2 * hidden_size` = `2880`,
  - `q_out` = `num_heads * head_dim` = `4096`,
  - `kv_out` = `num_kv_heads * head_dim` = `512`.

TurboMind records these as:

- `eagle_q_size`   = `q_out`   (4096),
- `eagle_kv_size`  = `kv_out`  (512),
- `eagle_qkv_in_dim` = `q_in`  (2880),
- `eagle_fc_in_dim`  = `3 * hidden` (8640).

These values come directly from `_infer_eagle3_geometry` in
`lmdeploy/turbomind/eagle_draft_converter.py` and are persisted into
`config.yaml`.

### 2.2 TensorRT‑LLM Eagle‑3 draft model

In `_torch/models/modeling_speculative.py`:

- `Eagle3DraftModel`:
  - has `hidden_size` = `config.hidden_size` (Eagle‑3 draft hidden),
  - has `hidden_size_in`:
    - `config.target_hidden_size` when provided (target model hidden),
    - otherwise `config.hidden_size`.
  - when `spec_config.num_capture_layers > 1`:
    - defines `fc: Linear(hidden_size_in * num_capture_layers,
      hidden_size)`.
- `Eagle3ForCausalLM`:
  - exposes an `apply_eagle3_fc` helper:
    - input: `[tokens, hidden_size_in * num_capture_layers]`,
    - output: `[tokens, hidden_size]` via `fc` when needed.

The midlayer attention inside `Eagle3DecoderLayer` uses the same
Eagle‑3 geometry as the HF midlayer described above; TurboMind’s
`Eagle3AttentionWeight` mirrors the raw HF layout and defers head
partitioning to future CUDA kernels.

### 2.3 Stage‑wise tensor summary (draft path)

For GPT‑OSS‑120B‑Eagle3, one Eagle‑3 step involves:

1. **Capture (target model, TurboMind `UnifiedDecoder`):**
   - `eagle_capture_hidden`: `[batch, hidden * num_capture_layers]`,
   - capture slots: `(1, L//2 - 1, L-4)` for deep models, matching
     `Eagle3OneModelSpecMetadata`.
2. **Pre‑FC (TurboMind `EagleModule`):**
   - `captured_hidden_states`:
     - passed from `LlamaV2::runEagle3DraftTreeDecode` into
       `EagleModule::forward_draft_tree`,
     - same shape as `eagle_capture_hidden`.
3. **Eagle‑3 FC:**
   - weights: `eagle_fc.weight` with shape `[3 * hidden, hidden]`,
   - TurboMind stores it as `[eagle_fc_in_dim, hidden_units]`,
     where `eagle_fc_in_dim = eagle_fc_in_dim_` from config.
   - output: `fc_out` (`debug_fc_out` in `EagleModule`):
     - `[batch, hidden_units]`.
4. **Pre‑attention RMSNorm:**
   - TurboMind normalises `fc_out` with `hidden_norm` into
     `attn_input` (`debug_attn_input`):
     - `[batch, hidden_units]`.
5. **Eagle‑3 attention (to be implemented in TurboMind):**
   - inputs:
     - `attn_input`: `[batch, hidden_units]`,
     - native Eagle‑3 projections:
       - `q_proj`: `[eagle_q_size, eagle_qkv_in_dim]`,
       - `k_proj`: `[eagle_kv_size, eagle_qkv_in_dim]`,
       - `v_proj`: `[eagle_kv_size, eagle_qkv_in_dim]`,
       - `o_proj`: `[eagle_qkv_in_dim, eagle_q_size]`.
   - expected outputs:
     - `attn_out`: `[batch, hidden_units]` (after `o_proj`).
6. **FFN and output norm:**
   - TurboMind’s `Eagle3DraftLayer` uses:
     - fused gate/up projection (`mlp_gate_up`),
     - down projection (`mlp_down`),
     - `post_attn_norm` / `output_norm`.
   - Stage tensors:
     - `debug_attn_out`: `[batch, hidden_units]`,
     - `debug_ffn_out`: `[batch, hidden_units]`,
     - `debug_pre_head_hidden`: `[batch, hidden_units]`.
7. **LM head:**
   - weights: `weights_.lm_head` / `lm_head_weight_`:
     - `[hidden_units, vocab_size]`,
   - logits: `[batch, vocab_size]` (TurboMind `debug_logits`),
     to be matched against Eagle‑3 / TensorRT‑LLM logits.

TurboMind’s Edison‑style debug helpers (`eagle_forward_debug`) are
designed to expose exactly these stage tensors for
`tests/turbomind/eagle3_compare.py`.

---

## 3. Mapping TensorRT‑LLM ↔ TurboMind Components

This section aligns key abstractions between TensorRT‑LLM’s Eagle‑3
implementation and TurboMind’s current structure.

### 3.1 Spec metadata and hidden capture

- **TensorRT‑LLM**
  - `Eagle3OneModelSpecMetadata`:
    - owns `hidden_states` buffer
      `[max_tokens, hidden_size * num_capture_layers]`,
    - decides `layers_to_capture` and manages capture indices,
    - provides `get_hidden_states()` to draft models.
  - `Eagle3ResourceManager` (two‑model path):
    - similar buffer but shared between target and draft engines,
    - extra bookkeeping for slot IDs and total draft tokens.
- **TurboMind**
  - `UnifiedDecoder`:
    - has `eagle_capture_layers_` and `eagle_capture_hidden`:
      - captures last‑token hidden states after each layer’s
        final RMSNorm, matching `layers_to_capture`,
      - layout `[batch, hidden_units * num_capture_layers]`.
    - passes `eagle_capture_hidden` to `LlamaV2` /
      `EagleModule::forward_draft_tree` as `captured_hidden_states`.
  - This mirrors `Eagle3OneModelSpecMetadata.get_hidden_states()`
    feeding into `Eagle3ForCausalLM.apply_eagle3_fc`.

### 3.2 Draft FC and attention stack

- **TensorRT‑LLM**
  - `Eagle3DraftModel.fc`:
    - reduces concatenated captured hidden to draft hidden.
  - `Eagle3DecoderLayer`:
    - performs Eagle‑3 attention + FFN with RoPE and KV semantics,
      using the HF `midlayer.*` weights.
- **TurboMind**
  - `EagleModule`:
    - owns `eagle_fc.weight` (full 3×hidden FC),
    - runs a single FC + RMSNorm to produce `attn_input_scratch_`,
    - currently uses a shallow attention path built on
      `attn_qkv`/`attn_o` for a *single‑position* attention block.
  - `Eagle3DraftLayer`:
    - wraps norms, attention backend, and FFN:
      - `UnifiedAttentionLayer` path for LLaMA‑compatible geometry,
      - future `Eagle3AttentionLayer` path for native Eagle‑3 geometry,
      - shallow QKV fallback otherwise.
  - `Eagle3AttentionWeight`:
    - holds native `eagle_q_proj/k_proj/v_proj/o_proj` tensors and
      geometry (`q_out`, `kv_out`, `q_in`), matching HF.

The outstanding work is to implement a real Eagle‑3 attention backend
(`Eagle3AttentionLayer::Forward`) that uses these native weights and
integrates with TurboMind’s KV/cache layout.

### 3.3 Tree decode, acceptance, and KV

- **TensorRT‑LLM**
  - `Eagle3OneModelWorker`:
    - orchestrates draft sampling (`draft_decoder`),
      acceptance (`sample_and_accept_draft_tokens`),
      and preparation of next‑step inputs.
  - Plugins in `tensorrt_llm/models/eagle/model.py`:
    - `eagle_prepare_drafter_inputs_plugin`,
    - `eagle_draft_decoder_plugin`,
    - `eagle_sample_and_accept_draft_plugin`,
    - handle on‑device tree construction, acceptance, and hand‑off
      between target and draft engines.
  - KV/cache rewind:
    - encoded via `KVCacheParams` and tested in
      `test_kv_cache_reuse.py`.
- **TurboMind**
  - `EagleModule::forward_draft_tree`:
    - owns the draft tree building on device,
    - feeds target logits and draft logits into an acceptance kernel,
    - writes accepted tokens and paths into `EagleBuffers`.
  - `LlamaV2::dynamicDecodeWithSpecMulti`:
    - calls `runEagle3DraftTreeDecode` (draft tree) and
      `runEagleTargetTreeDecode` (target verification),
    - manages packed masks and forced tails for
      `DynamicDecodeLayer::ForwardMultiStep`.
  - `kv_rewind_helper.cu` and `DynamicDecodeLayer`:
    - implement KV over‑provisioning and rewind (per‑sequence),
      driven by accepted lengths.

Functionally, TurboMind already mirrors TensorRT‑LLM’s high‑level
Eagle‑3 flow. The main remaining gaps are:

- real Eagle‑3 attention kernels,
- stage‑wise numeric validation vs HF/TRT,
- tuning KV semantics for long runs and SpecPV.

These items are tracked in `EAGLE_TODO_FINAL` (Phases 2–5).

---

## 4. Perf gates and tuning tooling

TurboMind wires Eagle‑3 perf validation and tuning through a small set
of CLI tools and environment flags:

- `benchmark_speculative.py`
  - Drives baseline and Eagle‑3 speculative runs for:
    - 32K single (`--scenario single`)
    - 16K batch4 large‑context (`--scenario large-context`)
  - Respects `LMDEPLOY_EAGLE_PERF_MODE=1` and
    `LMDEPLOY_EAGLE_MICRO_STEPS` so micro runs can exercise long‑context
    behaviour without full 16K/32K sweeps.
- `tools/check_perf_gates.py`
  - Consumes JSON outputs from `benchmark_speculative.py` and enforces
    the micro perf gates:
    - 32K single micro: spec‑3 ≥ 2.0× baseline and
      `mean_acceptance_length ≥ 3.0`.
    - 16K batch4 micro: spec‑3 ≥ 1.0× baseline and
      `mean_acceptance_length ≥ 2.2`.
- `run_spec_suite.sh`
  - Entry point for the full speculative suite. When `SCENARIOS` contains
    `large-context`, it automatically:
    - Runs the 32K single + 16K batch4 micro gates in PERF_MODE
      (3 consecutive passes by default).
    - Uses `LMDEPLOY_EAGLE_MICRO_STEPS_SINGLE` / `_BATCH4` to control
      micro step counts (defaults 512 / 128).
    - Refuses to run full large‑context scenarios if any micro gate fails.
- `tools/sweep_fmha_sm120.py`
  - Sweeps `TM_EAGLE3_FMHA_{KV_TILE,MAX_TILES,BLOCK,HEADS_PER_CTA}` over
    32K single + 16K batch4 micro runs and writes
    `build/fmha_sweep_sm120.json` with throughput, acceptance, and tile
    stats, which is used to choose near‑optimal FMHA defaults.
- `tools/tune_eagle3_gemm_sm120.py`
  - Reads `build/eagle3_gemm_shapes_sm120.json` (emitted when
    `LMDEPLOY_EAGLE_GEMM_SHAPE_LOG=1` and `LMDEPLOY_EAGLE_PERF_MODE=1`
    are enabled), aggregates GEMM shapes by tag
    (`EAGLE3_FFN_*`, `EAGLE3_FC`, `EAGLE3_LM_HEAD(_SMALLVOC)`) and
    prepares inputs for the gemm2 tuner/export pipeline.

In PERF_MODE, GEMM dispatch treats any `EAGLE3_*` tag as tuned‑only:
cache misses for Eagle‑3 draft FFN/FC/LM‑head shapes abort instead of
falling back to generic kernels or cuBLAS, ensuring that 32K single and
16K batch4 perf runs track the tuned configuration derived from the FMHA
sweep and GEMM tuning pipeline.
