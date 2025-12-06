Here’s a draft README you can hand directly to **Engineer B**.

---

# Engineer B – Multi-Token EAGLE Implementation Guide

**Scope:**
This document defines what you (Engineer B) own for **multi-token EAGLE** in `lmdeploy`, how it should behave (using TensorRT-LLM EAGLE-3 as the mental model), and the concrete tasks + acceptance criteria before anything is marked **DONE** in `EAGLE_TODO.md`.

You own **integration**, not the low-level EAGLE kernels:

* ✅ You **do own**: `LlamaBatch`, `LlamaV2_eagle`, `LlamaV2`, `DynamicDecodeLayer`, `SequenceManager`/KV cache integration, TP/PP/DP correctness, E2E tests & benchmarks.
* ❌ You **do not own**: `EagleModule`, speculative kernels, KV rewind kernels, SpeculativeConfig core, etc. (these are Engineer A’s territory).

Your job is to wire all of that into a **correct, production-safe multi-token EAGLE decode loop**.

---

## 1. Reference Mental Model: What “Correct” Multi-Token EAGLE Looks Like

Use TensorRT-LLM EAGLE-3 as the reference behaviour:

1. **Draft head & tree**

   * A small “EAGLE head” (in our case, EagleModule + tree) proposes up to `draft_len` tokens per sequence (possibly in a tree).
   * For each active sequence `b` at a step:

     * `draft_len[b]` tokens are proposed.
     * The target model (DynamicDecode) verifies them in one pass.

2. **Longest prefix acceptance**

   * For each sequence `b`, EAGLE returns:

     * `accepted_tokens[b][0 … len_b-1]`
     * `accepted_len[b] >= 1`
   * **Invariant:**
     `accepted_tokens[b][0]` is exactly what the base model would have produced as the next token (i.e., matches `DynamicDecode`’s sampled token).
   * The longest matching prefix is committed, and the rest are discarded:

     * `draft_len[b] = number of speculative tokens proposed`
     * `accepted_len[b] = number of tokens committed`
     * `rewind_len[b] = draft_len[b] - accepted_len[b]` (tokens to discard from KV).

3. **Committing multiple tokens**

   * One **engine step** can advance a sequence by **N tokens** (`accepted_len[b]`), not just 1.
   * For each sequence `b`:

     * Append `accepted_tokens[b][0 … accepted_len[b]-1]` to `token_ids_buf_` and output.
     * Increment `sequence_lengths_[b] += accepted_len[b]`.
   * The notion of “step” changes:

     * A decode step = one EAGLE **verify cycle**, not “one token”.

4. **KV cache rewind**

   * The model tentatively processes `draft_len[b]` tokens for verification.
   * After acceptance:

     * Only the first `accepted_len[b]` tokens are kept in KV.
     * The extra `rewind_len[b]` tokens are **logically removed** from per-sequence KV block tables.
   * Implementation:

     * Compute per-sequence `rewind_len[b]`.
     * Pass that to a KV rewind kernel (`KVCacheRewindParams` → `invokeKVCacheRewind`) that marks KV blocks as free and/or zeroes them.
   * Metrics (per request):

     * `eagle_total_draft_tokens += sum(draft_len[b])`
     * `eagle_total_accepted_tokens += sum(accepted_len[b])`
     * `eagle_total_rewound_tokens += sum(rewind_len[b])`
     * `eagle_steps += 1` and `eagle_rewind_steps += 1` when any rewind happens.

5. **Output equivalence**

   * Despite speculative steps, the final token sequence must be exactly what vanilla one-token decoding would produce.
   * Multi-token EAGLE is a **pure speedup**, not a change in output distribution.

Your integration in `lmdeploy` must preserve these invariants.

---

## 2. Files and Ownership

You will primarily work in:

* `src/turbomind/models/llama/LlamaBatch.cc/.h`
* `src/turbomind/models/llama/LlamaV2.cc/.h`
* `src/turbomind/models/llama/LlamaV2_eagle.cc` (or equivalent EAGLE integration file)
* `src/turbomind/models/llama/SequenceManager.*` & KV/block table handling
* `src/turbomind/kernels/speculative_decoding/kv_rewind_helper.{h,cu}` (call-site design; core kernel is A-scope)
* `src/turbomind/utils/eagle_debug.h` (env flags/gating)
* Tests:

  * `tests/turbomind/test_eagle_multi_token_future.py` (your main playground)
  * `tests/test_benchmark_speculative_integration.py`
  * Any additional `tests/turbomind/test_eagle_e2e*.py`

Always keep **`EAGLE_TODO.md`** updated with your status; don’t check a B-item as DONE until:

* Code is implemented, **and**
* At least one test exercises it.

---

## 3. Current State (High-Level)

Already in place (you can rely on these):

* **Single-token EAGLE** path is working and covered by tests.
* `compute_eagle_draft_tokens_per_seq` now returns multi-token budgets (no final clamp to 1).
* Eagle tree + kernels + KV rewind kernel + metrics plumbing exist (Engineer A).
* You’ve already added:

  * Env kill-switches:
    `LMDEPLOY_EAGLE_FORCE_SINGLE_TOKEN`, `LMDEPLOY_EAGLE_DISABLE_MULTI_TOKEN`.
  * An **experimental** multi-token helper:
    `LlamaBatch::advanceSequencesByEagleAcceptance(...)`, gated by `LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL` and `tp_size_ == 1`.

BUT:

* KV rewind is not wired.
* DynamicDecode semantics remain single-token.
* Multi-GPU behaviour for multi-token is not guaranteed.
* There is **no E2E multi-token test** yet.

Everything in this doc is about taking that partial work to a **full, test-backed** multi-token implementation.

---

## 4. Task Breakdown for Engineer B

### Task B1 – Finalize Multi-Token Sequence Advancement in `LlamaBatch`

**Goal:**
For one decode step, use EAGLE’s `accepted_len[b]` and `accepted_tokens[b]` to advance each sequence by multiple tokens, while preserving correctness.

**Where to work:**

* `LlamaBatch.cc`

  * `LlamaBatch::Forward(...)` – EAGLE branch, after DynamicDecode.
  * `LlamaBatch::advanceSequencesByEagleAcceptance(...)` – your helper.

**What to ensure:**

1. **Invariants enforced:**

   * For each active sequence `b`:

     * If `accepted_len[b] == 0` → treat as bug; log and fallback (single-token).
     * If `accepted_len[b] >= 1`:

       * Assert/check that `accepted_tokens[b][0] == dynamic_token[b]` (the token produced by DynamicDecode for this step).
       * If mismatch → log `[LlamaBatch][EAGLE][fallback]` and **skip multi-token for that sequence**.

2. **Token buffer updates:**

   * `token_ids_buf_` rows correspond to time steps, columns to sequences.
   * At step `g.step` (after committing the DynamicDecode token):

     * Let `base_step = g.step - 1` (index of current token row).
     * For each sequence `b` with `accepted_len[b] > 1`:

       * Write `accepted_tokens[b][1..accepted_len[b]-1]` into rows `[base_step+1 ... base_step+accepted_len[b]-1]`, column `b`.
   * Ensure you **never write beyond**:

     * Buffer shape (`session_len_ * 2` or equivalent bound).
     * `active_size` in the sequence dimension.

3. **Sequence length updates:**

   * Maintain a host copy of `sequence_lengths_` per step **only when needed**:

     * Copy `sequence_lengths_` device → host once.
     * For each `b`, increment `sequence_lengths_[b] += (accepted_len[b] - 1)` (extra tokens).
     * Copy host → device once.
   * Avoid per-token host↔device round trips.

4. **Step counter semantics:**

   * Concept: at each engine step, we commit **at least 1** token and possibly `max_extra = max_b(accepted_len[b] - 1)` extra tokens.
   * Model consistent behaviour:

     * `g.step` should increase by `1 + max_extra` for the entire batch (because the “time axis” has advanced that far).
   * Make sure any other logic (e.g., `invokeGatherOutput`) that uses `g.step` continues to see a consistent time index.

5. **Gating & fallbacks:**

   * Multi-token path must be **fully disabled** unless:

     * `LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL=1`
     * `tp_size_ == 1`
     * `g.partial == 0`
   * On any per-step or per-sequence invariant violation:

     * Log readable message.
     * Don’t crash; just continue in single-token semantics for that sequence/step.

**Acceptance criteria:**

* You can instrument logs (behind `LMDEPLOY_EAGLE_METRICS_DEBUG`) to show lines like:

  ```text
  [LlamaBatch][EAGLE_METRICS] step=5 batch=4 draft_tokens=12 accepted_tokens=8
  ```

* Manual sanity: when `LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL=1` and a tiny model is used, you see steps where some sequences advance by 2–3 tokens (visible in debug output).

* Existing single-token tests remain green (multi-token flag off).

---

### Task B2 – Integrate KV Rewind via `computeAndInvokeKVCacheRewind`

**Goal:**
Align KV cache state with multi-token acceptance: per sequence, free tail KV blocks for draft tokens that were not accepted.

**Where to work:**

* `LlamaBatch.cc`:

  * Same EAGLE branch that already knows `eagle_tokens_per_seq`, `accepted_len[b]`.
* `kv_rewind_helper.{h,cu}` (A-scope physically, but you must call it).

**What to do:**

1. **Compute per-sequence `draft_len` and `accepted_len`:**

   * `draft_len[b]` = `eagle_tokens_per_seq` (or per-sequence tokens_per_seq once you add it).
   * `accepted_len[b]` = `eagle_accepted_lens[b]`.
   * **Sanity:** `1 <= accepted_len[b] <= draft_len[b]`.
   * `rewind_len[b] = draft_len[b] - accepted_len[b]` (can be 0).

2. **Map to KV blocks:**

   * Identify:

     * `block_size` (tokens per KV block).
     * `max_blocks_per_seq`.
     * Per-slot block tables used by `SequenceManager`.
   * For each **global slot** corresponding to sequence `b`:

     * Compute `blocks_to_free = ceil(rewind_len[b] / block_size)`.
     * That is what `KVCacheRewindParams` will encode.

3. **Call KV rewind helper:**

   * Build a small `EagleKVRewindConfig` (or equivalent) with:

     * `rewind_lengths` array (size `max_batch_size`).
     * `batch_slots` mapping (from local batch index → global slot).
     * `block_tables` tensor pointer.
     * `kv_cache_blocks` pointer (optional for tests/metrics).
   * Call `computeAndInvokeKVCacheRewind(...)` (A-scope function) from LlamaBatch after acceptance.

4. **Metrics:**

   * Update `RequestMetrics` fields:

     * `eagle_total_rewound_tokens += sum_b(rewind_len[b])`.
     * `eagle_rewind_steps += 1` for the step if any `rewind_len[b] > 0`.
   * These already flow through Python `_get_metrics` → `SpeculativeDecodingStats`.

**Acceptance criteria:**

* In a debug run (with EAGLE enabled) you can see:

  * Rewind happening (via `LMDEPLOY_EAGLE_KV_DEBUG` logs).
  * `num_rewound_tokens` and `rewind_steps` non-zero in metrics.
* Tests (see B7) confirm:

  * `eagle_total_draft_tokens = accepted + rewound` over the run.
  * KV tables reflect correct tail block freeing for simple synthetic cases.

---

### Task B3 – Per-Sequence `tokens_per_seq[b]` Handling

**Goal:**
Support variable draft budgets per sequence while preserving static shapes and simple memory layouts.

**What this means:**

* Today, `compute_eagle_draft_tokens_per_seq` returns a **single** `eagle_tokens_per_seq` integer.
* In a more advanced design, you may want per-sequence budgets (e.g., because some sequences are nearly finished, or have lower remaining tokens).

**Your job (for now):**

* Define a clear plan for:

  * `planned_tokens_per_seq[b]` array (per sequence).
  * How to pack/unpack between:

    * Tree layout (paths),
    * `draft_ids`, `target_ids`,
    * and acceptance outputs.
* Implement the **minimal** necessary abstraction so future per-sequence budgets won’t break your current code.

**Acceptance criteria (for now):**

* There is a clear, commented code path and data structure where `planned_tokens_per_seq` would plug in.
* All current code works with a single shared `eagle_tokens_per_seq`, but it’s obvious where per-sequence extension goes.

---

### Task B4 – Integrate Multi-Token Semantics with `DynamicDecodeLayer`

**Goal:**
Ensure DynamicDecode’s notion of sequence length, finished status, stop criteria, and `max_new_tokens` semantics are correct in presence of multi-token steps.

**Where:**

* `DynamicDecodeLayer` and its call sites in `LlamaBatch::Forward`.

**Key requirements:**

1. **Stop criteria / EOS handling:**

   * If a sequence hits EOS within an accepted sequence of tokens:

     * That sequence must be marked finished.
     * No extra tokens beyond EOS should ever be committed.
   * `accepted_len[b]` must be truncated appropriately if the EAGLE path proposes tokens past EOS.

2. **`max_new_tokens` and `sequence_lengths_`:**

   * If a user requested `max_new_tokens = N`, total tokens produced (per sequence) must respect this bound even when we commit >1 per step.
   * You must ensure:

     * `sequence_lengths_[b]` + `accepted_len[b]` does not exceed `seq_limit_len_[b]` (or equivalent).
     * If it would, adjust `accepted_len[b]` downwards (and thus `rewind_len[b]` upwards).

3. **Finished sequences in a multi-token step:**

   * A sequence may finish mid-chunk (due to EOS or `max_new_tokens`).
   * After that sequence is finished:

     * It must not participate in further decoding steps.
     * KV rewind and metrics should treat it as frozen.

**Acceptance criteria:**

* In tests (B7), we can construct a tiny scenario where:

  * EAGLE proposes 3 tokens, but the 2nd is EOS:

    * You accept 2 tokens, drop the 3rd.
    * Sequence is marked finished, and no more tokens are generated.

---

### Task B5 – Multi-GPU (TP/PP/DP) Behaviour & Fallback

**Goal:**
Declare and enforce the supported parallelism modes for multi-token EAGLE.

**Steps:**

1. **Explicitly support (initially) only TP-size 1:**

   * You already gate on `tp_size_ == 1` in the experimental helper.
   * Make this explicit in logs and docs (B8).

2. **Future: support TP > 1**

   * Design so that:

     * EAGLE acceptance results are consistent across ranks.
     * `accepted_len` and `rewind_len` are broadcast to all TP ranks.
     * KV rewind is applied consistently on all ranks.

3. **Fallbacks:**

   * If unsupported configuration (e.g., `tp_size_ > 1` for now):

     * Log a clear warning.
     * Force single-token EAGLE for that request (or engine).

**Acceptance criteria:**

* In code, there is exactly one place where we decide “multi-token allowed or not” based on `tp_size_`, `pp_size`, `dp_size`.
* In logs, unsupported modes show an explicit message and continue safely in single-token mode.

---

### Task B6 – Runtime Fallbacks & Kill-Switches

**Goal:**
Ensure we never crash or corrupt state due to multi-token bugs; we just fall back cleanly to single-token.

**What to implement:**

* For each of these situations:

  * Acceptance invariants violated (first token mismatch, lengths inconsistent).
  * KV rewind params invalid (negative lengths, index issues).
  * Unsupported parallelism mode.
* Behaviour must be:

  * Log one clear warning with `[LlamaBatch][EAGLE][fallback]` or `[LlamaV2][EAGLE]`.
  * Disable multi-token EAGLE for that **request** or **engine** (e.g., a per-request flag or global engine flag).
  * Continue execution in single-token mode without breaking metrics.

**Acceptance criteria:**

* Tests (B7) deliberately trigger a mismatch / bad input and confirm:

  * Run doesn’t crash.
  * Output is still valid.
  * Multi-token stops on that request, but other requests/steps still run.

---

### Task B7 – Multi-Token E2E Tests & Benchmarks

**Goal:**
You may not mark multi-token EAGLE as done until there are tests that actually hit the path.

**Where:**

* `tests/turbomind/test_eagle_multi_token_future.py`
* `tests/test_benchmark_speculative_integration.py` (or a new multi-token variant)

**Minimum test suite:**

1. **Functional unit/integration test (CUDA-gated):**

   * Require:

     * CUDA available.
     * `_turbomind` built.
     * Small EAGLE-capable model via `MODEL_PATH` / `SPEC_MODEL_PATH`.
   * Set:

     * `LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL=1`.
   * Assert:

     * Some steps report `num_accepted_tokens > num_draft_tokens / something` OR at least `accepted_len[b] > 1` for some `b`.
     * Final number of tokens generated is consistent with `sum(accepted_len)` over steps.
     * Metrics invariants: `num_accepted_tokens <= num_draft_tokens`.

2. **KV rewind correctness test:**

   * Use a very small synthetic run:

     * Known draft and accepted pattern (e.g., draft_len=4, accepted_len=2).
   * After decode:

     * Inspect KV block tables or metrics:

       * `eagle_total_rewound_tokens == sum(draft_len - accepted_len)`.
       * Block table tail entries are freed/zeroed as expected (A-scope helpers can aid here).

3. **Benchmark integration:**

   * Extend benchmark test to:

     * Run with multi-token EAGLE and experimental flag on.
     * Verify `eagle_speculation` block in JSON includes:

       * Non-zero `total_accepted_tokens`.
       * `mean_acceptance_rate > 1 / num_spec_tokens` if multi-token is working.

**Acceptance criteria:**

* Running:

  ```bash
  pytest tests/turbomind/test_eagle_multi_token_future.py \
         tests/test_benchmark_speculative_integration.py
  ```

  shows:

  * tests either **skip cleanly** when CUDA/paths are missing,
  * OR **pass** when environment is provided.

---

### Task B8 – Docs: Update `turbomind_eagle_usage.md`

**Goal:**
Make sure users and ops understand what multi-token EAGLE does, when it’s supported, and how to disable it.

**Additions to the doc:**

1. **New env flag:**

   ```text
   - `LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL=1` – enable experimental
     multi-token EAGLE in TurboMind. Currently supported only for
     single-GPU (tp_size=1) setups. When enabled, the engine may accept
     multiple tokens per speculative step; see the limitations below.
   ```

2. **Limitations section:**

   * Single-GPU only for now.
   * May fall back to single-token per request if invariants fail.
   * KV rewind & metrics semantics (briefly):

     * We track draft, accepted, rewound tokens per request.
     * Output remains bit-equivalent to vanilla decoding.

3. **Debugging section:**

   * Show how to use:

     ```text
     LMDEPLOY_EAGLE_DEBUG=1
     LMDEPLOY_EAGLE_KV_DEBUG=1
     LMDEPLOY_EAGLE_METRICS_DEBUG=1
     LMDEPLOY_EAGLE_FORCE_SINGLE_TOKEN=1
     LMDEPLOY_EAGLE_DISABLE_MULTI_TOKEN=1
     LMDEPLOY_EAGLE_MULTI_TOKEN_EXPERIMENTAL=1
     ```

   * Explain what to look for in logs to confirm multi-token is active (e.g., metrics logs with `accepted_tokens > draft_batch_size`).

**Acceptance criteria:**

* Docs mention all relevant envs and clearly mark multi-token as experimental, single-GPU only.
* There is at least one short “cookbook” example of running a small multi-token benchmark.

---

## 5. How to Use This Document Day-to-Day

1. Keep **`EAGLE_TODO.md` open** alongside this doc.
2. For each B-item you touch:

   * Identify which task (B1–B8) it maps to.
   * Implement code in the files listed.
   * Add/update tests.
   * Only then mark the B-item as **done** in `EAGLE_TODO.md`.
3. Always run, at minimum:

   ```bash
   pytest tests/turbomind/test_eagle_metrics.py \
          tests/test_speculative_stats.py \
          tests/test_benchmark_speculative_integration.py
   ```

   and your new multi-token tests before pushing.

If you treat this doc as your **integration spec**, and `EAGLE_TODO.md` as your status board, we’ll end up with a **real**, test-backed multi-token EAGLE implementation instead of a permanently “experimental” path.
