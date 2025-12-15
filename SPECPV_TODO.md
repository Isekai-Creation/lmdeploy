# SpecPV v1 – TurboMind EAGLE3 partial-KV summary (implementation notes)

This file was previously used as a long-form RFC and has since been overwritten by benchmark logs. The authoritative, structured SpecPV
checklist now lives in `lmdeploy/EAGLE_TODO_COMPLETE` Section 5. This stub summarises the key v1 behaviours and open items for Engineer B.

## Implemented in v1

- PartialKVCache core:
  - `SpecPVCacheConfig` and `PartialKVCache` (`src/turbomind/models/llama/specpv_kv_cache.{h,cc}`) allocate per-layer KV buffers
    `[max_batch, num_kv_heads, total_budget, head_dim]` and expose `sink`, `retrieval`, `window`, and `buffer` segments for both K and V.
  - `summary_key_states` and `refresh_retrieval`:
    - Flatten full-prefix KV to float32 via `LlamaV2::flattenPrefixKVForLayer` (fp16/bf16 → float32).
    - Compute Kmax/Kmin summaries (host and CUDA paths) and select retrieval/window tokens, writing K/V into the partial cache.
  - `update` / `reset_buffer`:
    - Append newly verified tail tokens into the buffer and maintain `verified_lens_` / `global_verified_len_`.

- LlamaV2 integration:
  - `SpecPVCacheConfig` / `PartialKVCache` members and gating in `LlamaV2`:
    - Constructed when `EngineParam.enable_specpv && spec_method == "eagle3" && enable_eagle_target_tree`.
    - Strict geometry check, including a hard guard `specpv_n_spec_tokens_buf >= eagle_max_engine_tokens_per_step_`.
  - Seeding from full KV:
    - `LlamaV2::initSpecPVFromFullKV` flattens full-prefix KV via `flattenPrefixKVForLayer`, seeds sink/retrieval/window slices, and resets
      the buffer.
  - Partial/full switching:
    - `shouldUseSpecPV(seq_len)` gates partial mode based on context length and partial-KV budget.
    - `updateSpecPVAfterAcceptance`:
      - Seeds SpecPV on first use or after full-refresh.
      - Incrementally appends newly committed tail tokens via `PartialKVCache::update(...)`.
      - Drives a full-refresh when `specpv_partial_steps_` exceeds `specpv_full_refresh_steps` or buffer headroom is low.

- Tree decode integration:
  - `LlamaV2::runEagleTargetTreeDecode` SpecPV branch:
    - Computes a partial prefix length from SpecPV budget and `PartialKVCache::global_verified_len()`.
    - Builds prefix+tree block geometry from this partial prefix.
    - Fills scratch prefix blocks from `PartialKVCache::active_prefix` using the same block layout as full-KV, then runs
      `UnifiedDecoder::Forward` with tree masks unchanged.

- Dtype coverage:
  - SpecPV seeding:
    - `flattenPrefixKVForLayer` supports fp16 and bf16 KV caches and converts to float32 for SpecPV summaries and retrieval.
  - SpecPV tree decode:
    - SpecPV tree decode is supported for fp16 and bf16 TurboMind KV:
      - fp16: scratch KV prefix blocks are populated via `fill_layer_half` (float32 → half) in `runEagleTargetTreeDecode`.
      - bf16: scratch KV prefix blocks are populated via `fill_layer_bf16` (float32 → nv_bfloat16) when BF16 is enabled.
    - Quantized KV (int8/int4) is explicitly unsupported in SpecPV v1:
      - Any quantized KV config disables SpecPV via `[LlamaV2][SpecPV][fallback]` and uses the full-KV EAGLE3 path for tree decode.

## v1 behaviour vs future work

- Buffer semantics in v1:
  - The SpecPV buffer (`n_spec_tokens_buf`) holds **accepted tail tokens only**, appended after EAGLE acceptance via
    `updateSpecPVAfterAcceptance` + `PartialKVCache::update`.
  - In-flight tree candidates for the current step remain in the tree scratch KV and are not yet staged inside PartialKVCache.

- Open item – S4.1 (candidate staging):
  - Future work (S4.1 in `EAGLE_TODO_COMPLETE`) is to:
    - Stage tree candidate K/V in the SpecPV buffer during `runEagleTargetTreeDecode` while they are still unverified.
    - After acceptance, promote accepted candidates to the verified region and drop the rejected ones from the partial view.
  - This will require:
    - Exposing per-layer tree K/V from the decoder in a safe way.
    - Extending PartialKVCache with a clear “candidate vs verified” split inside the buffer.
  - Given the complexity and risk, S4.1 is intentionally left for a later iteration once the current v1 SpecPV path is validated on GPUs.

For the full, structured checklist (S1.x–S5.x), refer to `lmdeploy/EAGLE_TODO_COMPLETE` Section 5 (SpecPV). This stub is meant only as a
high‑level reminder of what v1 supports and what remains open.***
