# TurboMind EAGLE3 Target‑Tree Decode – KV/Cache Contract

This document specifies how TurboMind’s KV/cache state must behave when we add a **target‑tree decode** path for EAGLE3. The goal is to let
the base model verify EAGLE tree nodes without breaking baseline decode or corrupting KV.

This is a design contract for engine code (`LlamaBatch`, `LlamaV2`, `SequenceManager`, KV helpers). CUDA kernels and higher‑level wiring
must conform to this contract.

---

## 1. Context and scope

- Target model: TurboMind Llama (e.g. GPT‑OSS‑120B) using standard KV cache and `SequenceManager`.
- Draft model: EagleNet / EAGLE3 head integrated via `EagleModule`.
- EAGLE tree:
  - Structure lives in `EagleBuffers::inputs.draft_paths`
    (`[max_batch_size, max_decoding_tokens, max_path_len]`).
  - Per‑node draft ids live in `EagleBuffers::inputs.draft_tokens`
    (`[max_batch_size, max_decoding_tokens]`).
- Target‑tree decode is enabled when:
  - `EngineParam.enable_speculative_decoding == true`,
  - `EngineParam.spec_method in {"eagle", "eagle3"}`,
  - `EngineParam.enable_eagle_target_tree == true`,
  - EAGLE module/buffers are initialized (`LlamaV2::isEagleEnabled()`).

This document does **not** change baseline decode; when `enable_eagle_target_tree == false` the engine must behave exactly as today.

---

## 2. High‑level KV/cache invariants

For each active sequence (slot):

1. **Prefix KV is authoritative**
   - Before target‑tree decode, KV represents:
     - The prompt (prefill) and all tokens accepted and committed by `dynamicDecode` up to the current step.
   - No EAGLE draft token is committed to KV until it has been:
     - Verified by the target model, **and**
     - Accepted by the EAGLE acceptance logic, **and**
     - Reflected in the decode state (`token_ids_buf_`, `sequence_lengths_`).

2. **Tree decode must be KV‑safe**
   - Any forward pass used to score tree nodes must **not**:
     - Leak KV blocks (untracked new blocks left “in use” for dead branches),
     - Corrupt existing prefix KV for continuing sequences.
   - Acceptable strategies:
     - Use **scratch KV** for tree tokens and discard it after acceptance, **or**
     - Reuse existing KV blocks but fully rewind them for rejected branches before they become visible to `SequenceManager`.

3. **KV rewind is the only permanent KV mutation for EAGLE**
   - After a speculative step, the only allowed KV change stemming from EAGLE is:
     - A **rewind** of KV tail blocks corresponding to rejected draft tokens.
   - This is already implemented for per‑step tails via:
     - `computeAndInvokeKVCacheRewind` (kv_rewind_helper),
     - `LlamaBatch::runEagleKVRewind` (uses `Sequence::blocks`, `Sequence::cache_len`).
   - Target‑tree decode must integrate with this mechanism; it must not introduce additional KV mutation paths.

4. **Baseline invariants must hold when EAGLE is disabled**
   - When `enable_speculative_decoding == false` or `enable_eagle_target_tree == false`:
     - KV semantics must be identical to existing TurboMind behaviour,
     - No EAGLE kernels or KV helpers should be consulted.

---

## 3. Pre‑ and post‑conditions around target‑tree decode

We describe the KV contract relative to the main decode loop (`LlamaBatch::Forward`) and EAGLE helpers.

### 3.1 Pre‑conditions (before calling `LlamaV2::targetTreeDecode`)

For each slot `slot` in `[0, active_size)`:

- `SequenceManager` state:
  - `Sequence::blocks` contains the KV block IDs for the **entire** prefix:
    - prompt + all previously accepted tokens.
  - `Sequence::cache_len` equals the number of tokens represented in those blocks.
  - No EAGLE draft nodes from the current step are represented in `blocks` or `cache_len`.

- Decode buffers:
  - `token_ids_buf_` contains token ids up to `g.step` (time index) consistent with `sequence_lengths_`.
  - `sequence_lengths_[slot]` is the effective length of the sequence in `token_ids_buf_` and `Sequence::tokens`.

- EAGLE buffers:
  - `EagleBuffers::inputs.draft_tokens` holds the draft proposals for the current step:
    - Layout: `[max_batch_size, max_decoding_tokens]`.
  - `EagleBuffers::inputs.draft_paths` and `inputs.packed_masks` describe the EAGLE tree for this step:
    - Populated by `LlamaV2::eagleSpeculativeStep` / tree build.

- KV in memory:
  - May contain additional blocks beyond those referenced by `Sequence::blocks` (free/unused blocks in the manager).
  - These free blocks may be used for scratch KV during target‑tree decode but must be returned to a neutral state (or treated as free) before the next non‑speculative step.

### 3.2 Call sequence

For EAGLE‑enabled, target‑tree‑enabled steps, the intended call order is:

1. `LlamaBatch::Forward` computes `decoder_features` and runs `dynamicDecode` for the current step.
2. EAGLE draft branch:
   - `LlamaV2::eagleDraftForward` → draft logits.
   - Host builds `draft_tokens` and (temporarily) host `target_tokens`.
3. **Target‑tree input prep (current implementation, inside `eagleSpeculativeStep`):**
   - `LlamaV2_eagle::eagleSpeculativeStep` builds the tree and packed masks, then, when
     `EngineParam.enable_eagle_target_tree == true`, calls:
     - `LlamaV2::targetTreeDecode(batch_size, /*d_sequence_lengths=*/nullptr)`:
       - Calls `invokePrepareGenTargetTreeInputs`, which:
         - Reads `EagleBuffers::inputs.draft_paths` and `inputs.draft_tokens`,
         - Uses a base sequence length of `0` for now (since `d_sequence_lengths == nullptr`),
         - Writes:
           - `eagle_net_input_ids` (tree tokens),
           - `eagle_net_position_ids` (per‑token positions),
           - `eagle_net_hidden_indices` (flat index → `(slot, token_idx)`),
           - `eagle_net_gen_lens` / `eagle_net_seq_lens` / `eagle_net_ctx_lens` (per‑slot metadata when provided).
       - **KV is not touched** at this stage.
4. **Target‑tree base decode (future work):**
   - A dedicated decode pass runs the base model on `eagle_net_input_ids` / `eagle_net_position_ids`, using:
     - Prefix KV from `SequenceManager`,
     - Scratch KV for tree tokens.
   - Produces per‑node target logits → per‑node `target_ids` written into `EagleBuffers::inputs.target_tokens`.
5. EAGLE acceptance + KV rewind:
   - `LlamaV2::eagleSpeculativeStep` (continuing after the optional target‑tree input prep):
     - Uses `inputs.draft_tokens`, **host‑fabricated** `inputs.target_tokens` (mirrored from `LlamaBatch::Forward`), `inputs.draft_paths`
       and `invokeTreeAcceptByIdsWithPaths` to compute accepted tokens.
     - Packs accepted paths via `invokePackAcceptedPaths`.
   - `LlamaBatch::updateEagleMetricsAndKVLengths` / `runEagleKVRewind`:
     - Advance sequences by accepted tokens,
     - Rewind KV for rejected ones via `computeAndInvokeKVCacheRewind`.

### 3.3 Post‑conditions (after acceptance + KV rewind)

After EAGLE acceptance and KV rewind complete for a step:

- For each slot:
  - `Sequence::tokens` and `token_ids_buf_` reflect the updated sequence, including EAGLE‑accepted tokens.
  - `Sequence::blocks` and `Sequence::cache_len` reflect:
    - The prefix up to the last **accepted** token,
    - No KV state for rejected tree tokens.
  - Any scratch KV used for tree decode has been:
    - Either fully rewound (blocks marked free), or
    - Fully ignored (never attached to `Sequence::blocks`).

- Global invariants:
  - Next call to `LlamaBatch::Forward` sees a consistent decode state exactly as if the accepted tokens had been generated by baseline decode, with no dangling KV for rejected branches.

---

## 4. Target‑tree KV usage patterns

Two concrete patterns are acceptable for the future base decode over tree tokens. Both must obey the pre/post‑conditions above.

### 4.1 Scratch KV for tree tokens (preferred for clarity)

1. Allocate a scratch KV region (blocks or buffers) **separate** from `SequenceManager`’s live block tables.
2. When running the base model on tree tokens:
   - Reuse prefix KV from `SequenceManager` as read‑only.
   - Write all new K/V for tree tokens into scratch KV.
3. After acceptance:
   - Discard scratch KV contents (no need to copy anything into main KV).
   - Use `runEagleMultiTokenAdvance` + `runEagleKVRewind` to update `Sequence::tokens` / `Sequence::blocks` / `cache_len` based solely on accepted tokens and existing prefix KV.

Properties:

- Very low risk of corrupting live KV.
- Allows experimentation with tree decode without touching `SequenceManager`’s allocation logic.

### 4.2 Shared KV with strict rewind discipline

If scratch KV is not feasible, a shared KV scheme must:

1. Reserve a range of blocks per sequence for speculative use.
2. During tree decode:
   - Reuse prefix blocks as read‑only,
   - Write tree tokens into the reserved speculative blocks.
3. After acceptance/rewind:
   - For accepted tokens that become part of the committed prefix:
     - Optionally keep the corresponding blocks (if they align with forward stepping),
     - Or copy the accepted portion into new canonical blocks.
   - For rejected tokens:
     - Fully rewind speculative blocks via `invokeKVCacheRewind` (or a dedicated helper),
     - Remove them from any block tables so they are indistinguishable from “never used”.

This approach is more complex and must be carefully verified with targeted tests. The contract here is that **no speculative block survives** unless it is explicitly adopted as part of the committed prefix and reflected in `Sequence::blocks`.

---

## 5. Failure modes and safe fallbacks

Implementations of target‑tree decode must adhere to the following safety rules:

1. **On any KV mismatch or decode error:**
   - Log a clear warning:
     - e.g. `[LlamaV2][EAGLE][fallback] target-tree decode unavailable; reverting to single-step target logits`
   - Set `enable_eagle_target_tree = false` for the engine (or disable EAGLE for that run).
   - Fall back to the existing host‑fabricated `target_tokens` path for acceptance.

2. **No partial KV adoption**
   - It is never acceptable to:
     - Partially adopt tree KV (some nodes) without correctly updating `Sequence::blocks` / `cache_len`,
     - Leave speculative KV blocks in a state where later code could misinterpret them as valid prefix KV.

3. **CI and validation**
   - Any change that uses KV in target‑tree decode must be accompanied by:
     - Unit tests for:
       - Correct rewind lengths,
       - No block leaks under repeated speculative steps.
     - E2E tests that:
       - Compare baseline vs EAGLE sequences,
       - Check `Sequence::blocks` and `cache_len` for consistency after long runs.

---

## 6. Summary for implementers

- `LlamaV2::targetTreeDecode` + `invokePrepareGenTargetTreeInputs` define the **input** view of the EAGLE tree:
  - Flattened tree tokens,
  - Positions derived from `sequence_lengths_`,
  - Mapping back to `(slot, token_idx)` via `hidden_indices`.
- The **base model tree decode** must:
  - Use prefix KV as read‑only,
  - Place tree KV either in scratch or strictly rewound speculative blocks,
  - Produce per‑node target ids to feed into `EagleBuffers::inputs.target_tokens`.
- KV behaviour is purely additive to existing rewind logic:
  - The only permanent KV changes for EAGLE are **rewinds** of rejected tokens and the normal commit of accepted tokens into the prefix via `advanceSequencesByEagleAcceptance` + `runEagleKVRewind`.

All future CUDA / C++ changes for target‑tree decode must be implemented and reviewed against this contract.
