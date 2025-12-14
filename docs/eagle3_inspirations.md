# Eagle‑3 Reference Patterns (TRT‑LLM / SGLang / EasyDeL)

This note records which external kernels we treat as design references for TurboMind EAGLE‑3, what we mirror as patterns (not code), and where we intentionally diverge.

## TensorRT‑LLM

**Files inspected**

- `cpp/tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.cu`
  - `assembleTargetLogitsOffsets`
  - `prepareCtxEagleNetInputsKernel` + `invokePrepareCtxEagleNetInputs`
  - `buildLeafMask`, `getNonLeafEndingSubtree`, `prepareGenEagleNetInputsKernel`
- `cpp/tensorrt_llm/kernels/speculativeDecoding/draftTokenTreeKernels.cu`
  - `extractRealDraftTokensKernel` + `invokeExtractRealDraftTokens`
- `cpp/include/tensorrt_llm/runtime/eagleBuffers.h`
  - `EagleBuffers::Inputs` / `EngineOutputs`

**Patterns we mirror**

- End‑to‑end GPU pipeline for EAGLE:
  - Context prep (`prepareCtxEagleNetInputsKernel`) builds multi‑q EagleNet inputs, sequence lengths, and hidden‑state indices entirely on device.
  - Generation prep (`prepareGenEagleNetInputsKernel`) uses per‑batch block scans/reductions to derive per‑request lengths, offsets, and packed masks without host loops.
- Tree / mask construction:
  - `buildLeafMask` + `getNonLeafEndingSubtree` operate on flattened `paths` to derive non‑leaf tokens and packed draft‑draft masks using shared memory histograms.
  - All indices are expressed in flattened `[batch, path, level]` space, avoiding pointer‑chasing structures.
- Draft token materialization:
  - `extractRealDraftTokensKernel` treats “tokens to expand this layer” as a compact list and writes back only the needed subset into a per‑request buffer.
- Buffer organization:
  - `EagleBuffers::Inputs`/`EngineOutputs` clearly separate max‑batch runtime buffers (`[maxBatchSize, …]`) from engine inputs (`[numSequences, …]`), and keep all per‑step Eagle state grouped behind a single struct.

**Deliberate LMDeploy differences**

- We keep TurboMind’s existing `EagleBuffers` / `EagleModule` layout, but:
  - EAGLE‑3 context prep and generation prep will be moved to GPU kernels (multi‑q aware) following the `prepareCtxEagleNetInputsKernel` pattern instead of host‑side loops.
  - Active‑slot compaction is explicit: kernels will consume `[active_count, …]` tensors and an `active_slots` indirection instead of always looping over full `batchSize`.
- Tree / masks:
  - We reuse TurboMind’s target‑tree representation but align semantics with TRT‑LLM’s “flattened paths + per‑level scans” model rather than introducing a new linked structure.
- We do not copy any TRT‑LLM code or internal types; we only mirror the kernel–buffer contracts (what each kernel consumes/produces and how shapes scale with batch/paths/levels).

## SGLang

**File inspected**

- `sgl-kernel/csrc/speculative/eagle_utils.cu`
  - `build_tree_efficient` / `build_tree_efficient_partial_packed`
  - `VerifyTreeGreedy` + `verify_tree_greedy`

**Patterns we mirror**

- Tree as arrays + linked lists:
  - Tree structure is encoded via `parent_list`, `selected_index`, and linked‑list style arrays `retrive_index`, `retrive_next_token`, `retrive_next_sibling`.
  - Tree masks support multiple modes (`FULL_MASK`, `QLEN_ONLY`, bit‑packed) without changing the logical traversal.
- Greedy verification:
  - `VerifyTreeGreedy` walks `retrive_next_token` / `retrive_next_sibling` starting from the last accepted node, compares against `target_predict`, and emits `accept_index` + `accept_token_num` per batch.
  - Acceptance is defined over the tree, not over a flat list of tokens.

**Deliberate LMDeploy differences**

- TurboMind keeps its existing “target tree” buffers but:
  - Acceptance kernels will adopt the same logical traversal: walk successors via per‑node `next_token` / `next_sibling` indices and accumulate accepted positions.
  - Packed tree masks will be optional, controlled via envs, to avoid unconditional D2H traffic in PERF_MODE.
- We align our acceptance semantics with SGLang’s greedy verifier (tree‑aware, per‑request) but keep the implementation in native CUDA/C++ instead of PyTorch extension style.

## EasyDeL / eSurge

**Files inspected**

- `easydel/inference/esurge/core/manager.py`
  - `CacheManager`, especially `allocate_slots` and `get_computed_pages`
- `easydel/inference/esurge/scheduler/scheduler.py`
  - `Scheduler.__init__` (`use_eagle`, `num_spec_tokens`, `num_lookahead_tokens`)
  - `Scheduler.schedule` integration of `num_lookahead_tokens` into `allocate_slots`

**Patterns we mirror**

- KV/page allocator aware of speculative lookahead:
  - `num_lookahead_tokens` is derived from speculative config (EAGLE) and threaded into `CacheManager.allocate_slots`.
  - The allocator always reserves `num_new_tokens + num_lookahead_tokens` capacity, so speculative KV is first‑class in paging decisions.
- Single source of truth for “spec depth”:
  - Scheduler stores `num_spec_tokens` / `num_lookahead_tokens` once and uses them consistently for:
    - KV allocation,
    - scheduled speculative tokens bookkeeping (`scheduled_spec_decode_tokens`),
    - prefix cache hit computation.

**Deliberate LMDeploy differences**

- TurboMind’s `LlamaBatch` / `DynamicDecodeLayer` will:
  - Treat “planned speculative tokens” as part of the per‑step KV length planning, mirroring the `num_lookahead_tokens` pattern.
  - Keep existing KV management (no new page allocator) but make lookahead explicit in the per‑slot KV span metadata and EAGLE3 FMHA path.

## Summary: How this informs our implementation

- Active‑slot compaction:
  - We follow TRT‑LLM and EasyDeL in never paying per‑step compute for finished slots by driving all drafter/tree/LM‑head kernels over compacted `[active_count, …]` tensors, with explicit indirections back to `[batch_size, …]`.
- Tree / acceptance:
  - We treat SGLang’s `build_tree_efficient` + `VerifyTreeGreedy` as the semantic reference for Eagle‑3 tree topology and greedy acceptance, but implement it in TurboMind’s existing tree buffers and CUDA style.
- Context prep and KV planning:
  - We move base‑context preparation for Eagle‑3 to GPU kernels shaped like `prepareCtxEagleNetInputsKernel` and align KV span planning with EasyDeL’s `num_lookahead_tokens`‑aware allocator, while preserving TurboMind’s buffer and API surface.

