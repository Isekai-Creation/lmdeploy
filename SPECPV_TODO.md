# SpecPV-style Partial Verification for TurboMind EAGLE3 – RFC / TODO

## 0. Scope and Goals

- **Scope**: Integrate a SpecPV-style *partial KV verification* path into the TurboMind + EAGLE3 pipeline, alongside the existing target-tree decode, without regressing baseline non-EAGLE and non-partial paths.
- **Primary goals**:
  - Reduce verification cost for **long-context** EAGLE3 speculative decoding by:
    - Maintaining a **partial KV cache** for the target model (sink + retrieval + window + speculative buffer) instead of always using full KV.
    - Running **partial verification** most of the time and **periodic full verification** to refresh and correct accumulated errors.
  - Reuse the existing **EAGLE3 draft** and **target-tree decode** work:
    - Draft remains EAGLE3-style (no change to the draft model itself).
    - Target-tree decode continues to produce per-node `target_tokens` on device; partial/full KV governs the *verification* cost, not the acceptance semantics.
  - Keep **dynamic decode, EOS/stop, max_new_tokens** semantics identical to the current “full KV + EAGLE3” pipeline.

- **Non-goals (for this RFC)**:
  - No training changes (assume draft head is already EAGLE3 / GPT-OSS-Eagle3).
  - No changes to TensorRT-LLM; this is TurboMind-only.
  - No high-level benchmark harness work (we can extend `benchmark_speculative.py` later as a follow-up).

---

## 1. High-level Design: How SpecPV Maps to TurboMind

### 1.1 SpecPV concepts

From `SpecPV/PAPER.md` (§3.2 *Partial Verification*, §3.3 *Rectified with Full Verification*) and `SpecPV/specpv/kv/kv_cache.py`:

- **Partial KV cache** (per layer, per batch):
  - Split into four contiguous segments in **token order**:
    - **Sink** (`CacheConfig.n_sink_blocks` in `kv_cache.py`): earliest tokens (few blocks) kept always (attention “anchor”).
    - **Retrieval** (`CacheConfig.n_retrieval_blocks` + `PartialKVCache.refresh_retrieval`): dynamically selected blocks with high similarity to current query states, chosen via block summaries (`Kmax`, `Kmin` as in Eq. (1) and (2) of the paper).
    - **Window** (`CacheConfig.n_window_blocks`): the last `N` tokens (recency window) copied directly from recent full KV.
    - **Spec buffer** (`CacheConfig.n_spec_tokens_buf` + `PartialKVCache.update`): candidate / partially verified tokens, to be corrected / trimmed after verification.
  - The entire partial cache is treated as a **single contiguous KV segment** during attention.

- **Partial vs full verification**:
  - For short context: verification uses **full KV**; no partial cache needed.
  - Once context length exceeds `partial_kv_budget`:
    - One **full verification** initializes/refreshes the partial KV segments.
    - Subsequent steps use **partial verification**:
      - Attention only sees sink + retrieval + window + spec buffer.
      - Buffer holds tokens subject to correction/removal based on verification outcome.
    - Periodic full verification “resets” errors and recomputes partial KV from scratch.

### 1.2 TurboMind integration surface

TurboMind has:

- **Draft**: EAGLE3 draft head + `EagleModule` (already wired; conceptually similar to SpecPV’s `DraftAdapter` in `specpv/speculate/draft.py`).
- **Verification**:
  - **Target-tree decode** with scratch KV (`runEagleTargetTreeDecode`) + packed tree masks to compute per-node `target_tokens` on device.
  - Baseline single-token DynamicDecode path with full KV and no partial caching.
- **EAGLE acceptance**:
  - `invokeTreeAcceptByIdsWithPaths` uses device `draft_tokens` / `target_tokens` to decide accepted length per path; ID-equality semantics.

We will:

- Introduce a **SpecPV-style partial KV cache** for the *target model*, separate from the existing TurboMind KV manager, but reusing its block geometry where feasible.
- Add a **SpecPV mode** that:
  - For short contexts, behaves like today: full KV verification.
  - For long contexts:
    - Maintains a **partial KV view** (sink + retrieval + window + spec buffer).
    - Uses this view in the target-tree decode pass (or, for smaller configs, in a streaming “tree verify” pass).
    - Periodically calls into a full KV pass to recompute both full and partial KV caches.

### 1.3 Reference map to SpecPV implementation

This section anchors the TurboMind TODOs to concrete SpecPV code and paper passages.

- **Partial KV data structure**:
  - Paper: §3.2 *Partial Verification* (definition of sink tokens, retrieval tokens, local window tokens, and buffer; Fig. 2(b)).
  - Code: `SpecPV/specpv/kv/kv_cache.py`:
    - `CacheConfig` (fields: `block_size`, `n_sink_blocks`, `n_retrieval_blocks`, `n_window_blocks`, `n_spec_tokens_buf`, `total_budget`).
    - `PartialKVCache.__init__`:
      - Allocates `key_cache[layer]["all"]` / `value_cache[layer]["all"]` with shape `[max_batch_size, num_kv_heads, total_budget, head_dim]`.
      - Defines slices `"sink"`, `"retrieval"`, `"window"`, `"buffer"` exactly in that order.

- **Block summaries and retrieval**:
  - Paper: Eq. (1)–(3) in §3.2:
    - `S_i = (K_max^i, K_min^i)` and block scores `s_{i,j} = max(q_j K_max^T, q_j K_min^T)`, reduced to `s_i` via `f` (`max` / `mean` / last).
  - Code: `PartialKVCache.summary_key_states` and `PartialKVCache.refresh_retrieval` in `kv_cache.py`:
    - `summary_key_states` incrementally fills `key_states_summary[layer]["max"/"min"]` per block.
    - `refresh_retrieval`:
      - Computes similarity scores against summaries.
      - Picks top-`CacheConfig.n_retrieval_blocks` blocks.
      - Gathers them into `"retrieval"` slice and fills `"window"` with the most recent tokens.

- **Partial vs full verification switch**:
  - Paper: Fig. 2(a–c) and §3.2/§3.3 (full verification for short context; partial verification once context exceeds budget; periodic full verification to refresh).
  - Code:
    - `SpecPV/specpv/speculate/utils.py::should_partial_verify`:
      - Uses `partial_past_key_values.enabled`, `retrieval_initialized`, and `get_seq_length() + total_tokens + 1 <= total_budget`.
    - `tree_decoding` in `utils.py`:
      - Chooses between:
        - Full verification: uses `full_past_key_values` when partial not active.
        - Partial verification: uses `partial_past_key_values` alone when `should_partial_verify` is true.
      - Handles missing full-verified tokens (`missing_lens`) by prepending them to `tree_candidates`.

- **KV refresh and partial buffer maintenance**:
  - Paper: §3.3 *Rectified with Full Verification* (periodic full verification to eliminate accumulated errors and refresh partial KV cache).
  - Code:
    - `SpecPV/specpv/speculate/utils.py::update_inference_inputs`:
      - After a verification step, rewrites full KV so accepted tokens are contiguous and drops invalid segments.
      - Rolls back `PartialKVCache`:
        - Uses `verified_lens` and `global_verified_lens`.
        - Rewrites `"buffer"` slice, zeroing out unused parts.
    - `PartialKVCache.init_key_values` / `PartialKVCache.reset`:
      - Initialize partial cache from full KV once, then reuse across steps.

- **Self-speculative loop with EAGLE3 draft**:
  - Paper: §3.1 *Self-Speculative Decoding Framework* (draft reuses target features; verification produces the features needed for next step).
  - Code:
    - `SpecPV/specpv/speculate/speculator.py::spec_generate`:
      - Prefill via `chunked_prefilling` (produces full KV and EAGLE3 tree draft).
      - Loop:
        - Optionally initialize partial KV when `input_ids.shape[1] > partial_past_key_values.cache_config.total_budget` and `SpecConfig.enable_partial_kv`.
        - Call `tree_decoding` (partial or full verify).
        - Use `evaluate_posterior` to compute accept length and `update_inference_inputs` to:
          - Roll KV (full + partial).
          - Run new tree draft on accepted hidden states.

The TurboMind TODOs below are designed to have **one-to-one correspondences** to these SpecPV mechanisms, but adapted to our existing C++/CUDA + TurboMind-specific abstractions.

---

## 2. Feature Set and Expected Gains

### 2.1 Features

1. **Partial KV cache for target verification**
   - Store a compressed view of the long context in a four-part layout (sink / retrieval / window / spec buffer).
   - Allow verification (target-tree decode) to attend only to this partial cache when context is long.

2. **Block-level retrieval and summaries**
   - Maintain `Kmax` / `Kmin` per block and layer.
   - Select the most relevant blocks per verification step based on query states from candidate tokens (and/or recently verified tokens).

3. **Spec buffer for partially verified tokens**
   - Keep candidate + partially verified tokens in a buffer segment until verification decides which tokens remain.
   - After verification and acceptance:
     - Move accepted tokens into the “trusted” part (sink/retrieval/window).
     - Drop invalid tokens from the buffer.

4. **Full verification + partial refresh**
   - Periodically run a **full KV** verification pass:
     - Rebuild partial KV segments from full KV.
     - Reset the spec buffer and retrieval selection.

5. **Configurable budgets and thresholds**
   - Expose a `SpecPVConfig` for:
     - Block size.
     - Sink / retrieval / window / buffer budgets.
     - Frequency and triggers for full verification (e.g., buffer overflow, max partial steps).

### 2.2 End-to-end workflow / pipeline (target-tree + SpecPV)

The intended TurboMind + SpecPV pipeline mirrors the SpecPV loop in `speculator.spec_generate` + `utils.tree_decoding` / `update_inference_inputs`:

1. **Prefill stage (existing)**:
   - TurboMind:
     - Run base model with full KV to process the prompt (and possibly initial tokens).
     - EAGLE3 draft head prefilled as we already do.
   - SpecPV reference:
     - `chunked_prefilling` in `SpecPV/specpv/speculate/utils.py`:
       - Runs the base model with `full_past_key_values`.
       - Runs EAGLE3 draft (`ea_layer.tree_draft`) over concatenated hidden states.

2. **Initial full verification (short context or first long step)**:
   - TurboMind:
     - For early steps or when `sequence_len <= partial_kv_threshold`:
       - Use existing `runEagleTargetTreeDecode` with scratch KV built from full KV.
     - When `sequence_len` exceeds `partial_kv_threshold` for the first time:
       - Still run one full-KV target-tree decode step.
       - After acceptance, call `initSpecPVFromFullKV` to seed `PartialKVCache` from full KV.
   - SpecPV reference:
     - Paper Fig. 2(c) and §3.3: full verification once to initialize partial KV cache as context grows.

3. **Partial verification loop (long context)**:
   - For subsequent decode steps when `shouldUseSpecPV` is true:
     1. Build the **tree** and draft tokens as today (EAGLE3 draft).
     2. Decide partial vs full:
        - `LlamaV2::shouldUseSpecPV(sequence_len)` mirrors `should_partial_verify` in `SpecPV/specpv/speculate/utils.py`.
     3. Run target-tree decode:
        - Use `PartialKVCache` view (sink + retrieval + window + buffer) as base KV.
        - Append tree tokens’ KV into the buffer slice (S4.1).
     4. Compute logits + `target_tokens`:
        - Same as our existing tree logits → target IDs kernel.
     5. Acceptance:
        - Use `invokeTreeAcceptByIdsWithPaths` as today (ID-equality semantics).
     6. KV updates:
        - After `advanceSequencesByEagleAcceptance`, update:
          - Full KV (already handled by TurboMind KV / rewind helpers).
          - Partial KV:
            - Update `verified_lens` / `global_verified_lens` and buffer contents, analogous to `update_inference_inputs` in SpecPV.

4. **Periodic full verification and partial refresh**:
   - When buffer or budget thresholds are hit (S4.3):
     - Run a full-KV `runEagleTargetTreeDecode` step.
     - Rebuild partial KV from full KV with `initSpecPVFromFullKV`.
     - Zero buffer and reset `verified_lens`.
   - SpecPV reference:
     - §3.3: partial verification periodically rectified by full verification; buffer reset and retrieval set recomputed.

5. **Termination**:
   - EOS/stop/max_new_tokens remain governed by DynamicDecode + EAGLE acceptance (unchanged logic).
   - SpecPV only changes which KV entries are visible during verification, not which tokens are accepted.

### 2.3 Usage APIs & configuration examples

This subsection specifies how SpecPV should be exposed to users (YAML / Python) and which internal APIs are expected to be present, with concrete code sketches.

#### 2.3.1 Triton YAML / EngineParam usage

- Extend Triton `speculative_config` to include SpecPV fields, similar to `enable_target_tree` in `LlamaTritonModel.cc`:

```yaml
speculative_config:
  enable_speculative_decoding: true
  method: eagle3

  # existing EAGLE flags
  enable_target_tree: true

  # new SpecPV flags
  enable_specpv: true
  specpv_block_size: 16            # tokens per block
  specpv_n_sink_blocks: 2          # sink budget
  specpv_n_retrieval_blocks: 256   # retrieval budget
  specpv_n_window_blocks: 8        # local window
  specpv_n_spec_tokens_buf: 128    # speculative buffer for tree tokens
  specpv_partial_threshold: 4096   # activate SpecPV beyond this context length
  specpv_full_refresh_interval: 32 # max partial steps before forced full verify
```

- Parse into `EngineParam` in `LlamaTritonModel.cc`:

```cpp
// in LlamaTritonModel::init() or similar
auto& spec_reader = config_["speculative_config"];

engine_param_.enable_specpv              = spec_reader["enable_specpv"].as<bool>(false);
engine_param_.specpv_block_size          = spec_reader["specpv_block_size"].as<int>(16);
engine_param_.specpv_n_sink_blocks       = spec_reader["specpv_n_sink_blocks"].as<int>(2);
engine_param_.specpv_n_retrieval_blocks  = spec_reader["specpv_n_retrieval_blocks"].as<int>(256);
engine_param_.specpv_n_window_blocks     = spec_reader["specpv_n_window_blocks"].as<int>(8);
engine_param_.specpv_n_spec_tokens_buf   = spec_reader["specpv_n_spec_tokens_buf"].as<int>(128);
engine_param_.specpv_partial_threshold   = spec_reader["specpv_partial_threshold"].as<int>(4096);
engine_param_.specpv_full_refresh_steps  = spec_reader["specpv_full_refresh_interval"].as<int>(32);
```

- Extend `EngineParam` (`llama_params.h`) accordingly:

```cpp
struct EngineParam {
    // ... existing fields ...

    bool enable_eagle_target_tree{false};

    // SpecPV partial KV config (EAGLE3-only)
    bool enable_specpv{false};
    int  specpv_block_size{16};
    int  specpv_n_sink_blocks{2};
    int  specpv_n_retrieval_blocks{256};
    int  specpv_n_window_blocks{8};
    int  specpv_n_spec_tokens_buf{128};
    int  specpv_partial_threshold{4096};
    int  specpv_full_refresh_steps{32};
};
```

These correspond directly to `CacheConfig` fields and the partial/full switch logic in SpecPV (`kv_cache.CacheConfig`, `speculator.SpecConfig`, and `utils.should_partial_verify`).

#### 2.3.2 Python LMDeploy API usage

Expose SpecPV via LMDeploy’s Python configuration, similar to `SpeculativeConfig` today:

```python
from lmdeploy import SpeculativeConfig, TurbomindEngineConfig

spec_cfg = SpeculativeConfig(
    enable_speculative_decoding=True,
    method='eagle3',
    enable_eagle_target_tree=True,

    # new SpecPV options
    enable_specpv=True,
    specpv_block_size=16,
    specpv_n_sink_blocks=2,
    specpv_n_retrieval_blocks=256,
    specpv_n_window_blocks=8,
    specpv_n_spec_tokens_buf=128,
    specpv_partial_threshold=4096,
    specpv_full_refresh_interval=32,
)

engine_cfg = TurbomindEngineConfig(
    # ... other engine options ...
    speculative_config=spec_cfg,
)
```

Internally, these fields should be serialized into the Triton YAML (or engine JSON) and mapped onto `EngineParam` as shown above.

#### 2.3.3 Internal C++ API surfaces

The following C++ APIs are expected to exist and be used from `LlamaBatch` / `LlamaV2_eagle`:

```cpp
// Partial KV cache config and object
struct SpecPVCacheConfig {
    int block_size;
    int n_sink_blocks;
    int n_retrieval_blocks;
    int n_window_blocks;
    int n_spec_tokens_buf;

    int sink_size() const      { return n_sink_blocks * block_size; }
    int retrieval_size() const { return n_retrieval_blocks * block_size; }
    int window_size() const    { return n_window_blocks * block_size; }
    int total_budget() const {
        return sink_size() + retrieval_size() + window_size() + n_spec_tokens_buf;
    }
};

class PartialKVCache {
public:
    PartialKVCache(const SpecPVCacheConfig& cfg,
                   int max_batch_size,
                   int num_kv_heads,
                   int head_dim);

    // block summaries (Eq. (1) in the paper)
    void summary_key_states(int layer_idx,
                            const Tensor& key_states,  // [B,H,L,D]
                            int seq_len);

    // retrieval refresh (Eq. (2),(3))
    void refresh_retrieval(int layer_idx,
                           const Tensor& query_states,  // [B,H,Q,D]
                           const Tensor& key_states,    // [B,H,L,D]
                           const Tensor& value_states,  // [B,H,L,D]
                           int seq_len);

    // append new (partially verified) tokens into buffer
    std::pair<Tensor, Tensor> update(int layer_idx,
                                     const Tensor& new_keys,   // [B,H,l,D]
                                     const Tensor& new_values);// [B,H,l,D]

    int  get_seq_length(int layer_idx = 0) const;
    void reset();

    bool enabled{false};
    bool retrieval_initialized{false};
    int  global_verified_lens{0};
    // ... per-layer state ...
};

// in LlamaV2
bool LlamaV2::isSpecPVEnabled() const {
    return engine_param_.enable_specpv && specpv_supported_;
}

bool LlamaV2::shouldUseSpecPV(int seq_len) const {
    if (!isSpecPVEnabled() || !specpv_kv_cache_) {
        return false;
    }
    if (seq_len <= engine_param_.specpv_partial_threshold) {
        return false;
    }
    const int current = specpv_kv_cache_->get_seq_length();
    const int max_tokens = specpv_cache_config_.total_budget();
    return current + eagle_max_engine_tokens_per_step_ + 1 <= max_tokens;
}
```

These sketches mirror SpecPV’s Python classes:
- `PartialKVCache` ↔ `SpecPV/specpv/kv/kv_cache.py::PartialKVCache`.
- `isSpecPVEnabled` / `shouldUseSpecPV` ↔ `specpv/speculate/utils.py::should_partial_verify`.

#### 2.3.4 Reference implementations from SpecPV (translated)

Below are condensed, C++-style translations of key SpecPV functions to clarify the expected behaviour TurboMind must implement.

1. **Block summaries (`summary_key_states`)**  
   Based on `PartialKVCache.summary_key_states` in `SpecPV/specpv/kv/kv_cache.py` and Eq. (1) in the paper:

```cpp
// Pseudo-code; actual implementation will be CUDA kernels
void PartialKVCache::summary_key_states(int layer_idx,
                                        const Tensor& key_states, // [B,H,L,D]
                                        int seq_len)
{
    const int block = cfg_.block_size;
    const int sink  = cfg_.sink_size();

    // number of blocks beyond sink tokens
    const int expected_blocks = std::max(0, (seq_len - sink) / block);
    int& existing_blocks      = summary_block_count_[layer_idx];

    if (expected_blocks <= existing_blocks) {
        return;
    }

    for (int b = existing_blocks; b < expected_blocks; ++b) {
        const int start = sink + b * block;
        const int end   = start + block;
        // slice: key_states[..., start:end, :]
        Tensor ks_block = key_states.slice(2, start, end);  // [B,H,block,D]

        // reduce over token dimension to get max/min summaries
        Tensor kmax = ks_block.max(/*dim=*/2); // [B,H,D]
        Tensor kmin = ks_block.min(/*dim=*/2); // [B,H,D]

        // write into summary buffers at index b
        key_summary_max_[layer_idx].index_put_({Slice(), Slice(), b}, kmax);
        key_summary_min_[layer_idx].index_put_({Slice(), Slice(), b}, kmin);
    }

    existing_blocks = expected_blocks;
}
```

2. **Retrieval refresh (`refresh_retrieval`)**  
   Based on `PartialKVCache.refresh_retrieval` in `SpecPV/specpv/kv/kv_cache.py` and Eq. (2),(3):

```cpp
void PartialKVCache::refresh_retrieval(int layer_idx,
                                       const Tensor& query_states, // [B,H,Q,D]
                                       const Tensor& key_states,   // [B,H,L,D]
                                       const Tensor& value_states, // [B,H,L,D]
                                       int seq_len)
{
    const int num_blocks = summary_block_count_[layer_idx];
    if (num_blocks == 0) {
        return;
    }

    // key_summary_max/min: [B,H,num_blocks,D]
    Tensor kmax = key_summary_max_[layer_idx];
    Tensor kmin = key_summary_min_[layer_idx];

    // scores: s_{i,j} = max(q_j·Kmax_i^T, q_j·Kmin_i^T)
    Tensor sim_max = matmul(query_states, kmax.transpose(-1, -2)); // [B,H,Q,N]
    Tensor sim_min = matmul(query_states, kmin.transpose(-1, -2)); // [B,H,Q,N]
    Tensor scores  = max(sim_max, sim_min);                        // [B,H,Q,N]

    // reduce over Q (queries) using max, as in SpecPV
    scores = scores.max(/*dim=*/2);                                // [B,H,N]

    const int topk_blocks = std::min(cfg_.n_retrieval_blocks, num_blocks);
    auto [top_vals, top_idx] = topk(scores, topk_blocks, /*dim=*/2); // [B,H,K]

    // Convert block indices -> token indices
    // token_idx = block_idx * block_size + offset
    Tensor offsets = arange(0, cfg_.block_size);                   // [block_size]
    Tensor token_indices = top_idx.unsqueeze(-1) * cfg_.block_size + offsets;
    // token_indices: [B,H,K,block_size] -> [B,H,K*block_size]
    token_indices = token_indices.reshape({B, H, topk_blocks * cfg_.block_size});

    // gather keys/values at token_indices into retrieval slice
    Tensor idx_exp = token_indices.unsqueeze(-1).expand_as(/*[B,H,R,D]*/);
    Tensor retri_k = take_along_dim(key_states, idx_exp, /*dim=*/2);
    Tensor retri_v = take_along_dim(value_states, idx_exp, /*dim=*/2);

    key_cache_[layer_idx]["retrieval"].copy_(retri_k);
    value_cache_[layer_idx]["retrieval"].copy_(retri_v);

    // fill window with most recent tokens
    const int win_size = cfg_.window_size();
    const int win_start = seq_len - win_size;
    const int win_end   = seq_len;

    Tensor win_k = key_states.slice(2, win_start, win_end);
    Tensor win_v = value_states.slice(2, win_start, win_end);

    key_cache_[layer_idx]["window"].copy_(win_k);
    value_cache_[layer_idx]["window"].copy_(win_v);

    verified_lens_[layer_idx] = 0;
}
```

3. **Partial vs full verify decision (`shouldUseSpecPV`)**  
   Based on `should_partial_verify` in `SpecPV/specpv/speculate/utils.py`:

```cpp
bool LlamaV2::shouldUseSpecPV(int seq_len) const
{
    if (!isSpecPVEnabled() || !specpv_kv_cache_) {
        return false;
    }

    // context must be beyond threshold
    if (seq_len <= engine_param_.specpv_partial_threshold) {
        return false;
    }

    if (!specpv_kv_cache_->enabled || !specpv_kv_cache_->retrieval_initialized) {
        return false;
    }

    const int current = specpv_kv_cache_->get_seq_length();
    const int budget  = specpv_cache_config_.total_budget();
    const int needed  = eagle_max_engine_tokens_per_step_ + 1; // draft tokens + sample

    return (current + needed) <= budget;
}
```

These snippets are not drop-in code, but they define the exact semantics and data flow TurboMind should implement to match SpecPV’s partial KV behaviour while integrating with EAGLE3 target-tree decode. 

### 2.2 Expected gains

Compared to current TurboMind EAGLE3 + target-tree decode:

- **Lower verification cost at long context**:
  - Tree decode / verify passes see a much smaller KV segment (partial KV) instead of the full context.
  - For 32–64K contexts, this reduces attention flops and memory traffic per verification step.

- **Preserved semantics**:
  - ID-based acceptance semantics remain unchanged: we still require `draft_id == target_id` per node.
  - EOS/stop/max_new_tokens semantics remain aligned with baseline (DynamicDecode + EAGLE acceptance).
  - Partial KV only affects *where* the target attends, not *what* we accept; periodic full verification corrects drift.

- **Better utilization of EAGLE3 draft**:
  - When base verification is cheaper, we can afford larger trees / more speculative tokens while staying within latency budgets.

---

## 3. Building Blocks and TODOs (Code-focused)

Below, tasks are grouped into phases but can be implemented incrementally. All items are **code-first** (no tests/docs required in this RFC).

### 3.1 Phase 1 – Partial KV cache core

**Goal**: Introduce a partial KV cache for the target model, modeled after `SpecPV/specpv/kv/kv_cache.py`, but adapted to TurboMind’s KV and block structures.

#### 3.1.1 Core types and configuration

- [x] **S1.1 – Define SpecPV cache config**
  - File(s): `lmdeploy/src/turbomind/models/llama/LlamaV2.h` (or a new `specpv_kv_cache.h`).
  - Add a `SpecPVCacheConfig` struct with:
    - `block_size` (tokens per block; align with TurboMind KV manager block size).
    - `n_sink_blocks`, `n_retrieval_blocks`, `n_window_blocks`, `n_spec_tokens_buf`.
    - Derived sizes: `sink_size`, `retrieval_size`, `window_size`, `total_budget`.
  - Hook it into engine params:
    - Add `enable_specpv`, `specpv_block_size`, `specpv_retrieval_blocks`, etc., to engine/JSON config.
    - Default: `enable_specpv = false` (no behaviour change by default).
   - Fact mapping:
     - Mirrors `CacheConfig` in `SpecPV/specpv/kv/kv_cache.py`, including the `total_budget` logic.
   - Implementation status:
     - `EngineParam` exposes `enable_specpv` and the `specpv_*` fields in `llama_params.h`; LMDeploy’s `SpeculativeConfig` and Triton backend
       (`LlamaTritonModel`) already parse and forward these into the TurboMind engine config.

- [x] **S1.2 – PartialKVCache class (TurboMind version)**
  - File(s): new `lmdeploy/src/turbomind/models/llama/specpv_kv_cache.{h,cc}`.
  - Implement a `PartialKVCache` (C++/CUDA side) analogous to SpecPV’s `PartialKVCache`:
    - Per-layer KV storage: one big tensor `[max_batch_size, num_kv_heads, total_budget, head_dim]`.
    - Split into slices:
      - `sink`, `retrieval`, `window`, `buffer`.
    - Track:
      - `retrieval_initialized` (bool),
      - `enabled` (bool),
      - `verified_lens[layer]` (number of tokens stored in buffer),
      - `global_verified_lens` (across layers),
      - `key_states_summary[layer].{max,min}` as `[B, H, max_blocks, D]`.
    - Behaviour must match:
      - `PartialKVCache.__init__`, `summary_key_states`, `refresh_retrieval`, `update`, `get_seq_length`, `reset` in `SpecPV/specpv/kv/kv_cache.py`.
    - Expose methods:
      - `summary_key_states(layer_idx, key_states, seq_len)`,
      - `refresh_retrieval(layer_idx, query_states, key_states, value_states, seq_len)`,
      - `update(layer_idx, new_key_states, new_value_states) -> (key_view, value_view)`,
      - `add_to_sink(layer_idx, key_states, value_states)` (initial copy).
      - `get_seq_length(layer_idx)`, `reset()`.
   - Implementation status:
     - `SpecPVCacheConfig` and `PartialKVCache` are implemented in `specpv_kv_cache.{h,cc}`:
       - Per-layer KV buffers `[max_batch, num_kv_heads, total_budget, head_dim]` are allocated for keys and values.
       - Segment views (`sink`, `retrieval`, `window`, `buffer`) exist for both K and V.
       - `summary_key_states`, `refresh_retrieval`, `update`, and `reset_buffer` are implemented in a float32 host-backed version; they mirror
         the SpecPV Python behaviour but currently run on CPU (future work may move them to CUDA kernels).

#### 3.1.2 Initialization and wiring to LlamaV2

- [x] **S1.3 – Allocate partial KV in LlamaV2**
  - File(s): `LlamaV2.h`, `LlamaV2.cc`.
  - Add members:
    - `std::unique_ptr<PartialKVCache> specpv_kv_cache_;`
    - Flags: `bool specpv_enabled_`, `bool specpv_supported_;`.
    - In `LlamaV2` ctor:
      - If `engine_param_.enable_specpv` and EAGLE3 target-tree is supported:
        - Build `SpecPVCacheConfig` from engine params and TurboMind block geometry.
        - Allocate `specpv_kv_cache_` (float32 KV view) and set `specpv_supported_ = true` when allocation/geometry succeed.
        - On mismatch or allocation failure, log `[LlamaV2][SpecPV][fallback] ...` and disable SpecPV for the engine.
    - Fact mapping:
      - Matches the constructor-time guards we already use in `LlamaV2` for `tree_hidden_states_` / `tree_logits_buffer_`, and SpecPV’s runtime checks before enabling partial KV.
    - Implementation status:
      - Implemented in `LlamaV2` ctor: SpecPV geometry is validated against `cache_block_seq_len`, head count, and total budget, with hard
        `[SpecPV][fallback]` logging and disabling on any mismatch.

- [x] **S1.4 – Initialize partial KV from full KV**
  - File(s): `LlamaV2.cc` or `LlamaBatch.cc`, depending on where we manage KV reuse.
  - Add a helper:
    - `void LlamaV2::initSpecPVFromFullKV(const SequenceManager& seq_mgr, cudaStream_t stream);`
  - Behaviour:
    - For each active sequence / layer:
      - Read prefix full KV blocks (similar to how `runEagleTargetTreeDecode` builds scratch KV).
      - Copy them into `specpv_kv_cache_->sink` and/or `retrieval`/`window` segments as the initial partial KV.
    - Set `specpv_kv_cache_->enabled = true`, `retrieval_initialized = true`, `global_verified_lens = current_sequence_len`.
    - Fact mapping:
      - Equivalent to `PartialKVCache.init_key_values(full_past_key_values)` in `SpecPV/specpv/kv/kv_cache.py`, which seeds sink and other segments from full KV.
    - Implementation status:
      - Implemented as `LlamaV2::initSpecPVFromFullKV`, which flattens the full-prefix KV per layer via `flattenPrefixKVForLayer` and calls
        `PartialKVCache::summary_key_states` and `refresh_retrieval` to populate sink/retrieval/window, then resets the speculative buffer
        and marks `specpv_retrieval_initialized_ = true`.

---

### 3.2 Phase 2 – Partial vs full verification gating

**Goal**: Decide when to use partial KV vs full KV for verification, in the context of EAGLE3 target-tree decode.

#### 3.2.1 Step-level gating logic

- [x] **S2.1 – Implement `shouldUseSpecPV`**
  - File(s): `LlamaV2.cc`.
  - Add a method:
    - `bool LlamaV2::shouldUseSpecPV(int sequence_len) const;`
  - Logic (inspired by `should_partial_verify` in SpecPV):
    - Preconditions:
      - `specpv_enabled_ && specpv_supported_`.
      - `specpv_kv_cache_ != nullptr`.
    - Conditions:
      - `sequence_len > specpv_config.partial_kv_threshold` (context long enough).
      - `specpv_kv_cache_->enabled` or can be lazily `initSpecPVFromFullKV(...)`.
      - `specpv_kv_cache_->get_seq_length() + max_spec_tokens <= total_budget`.
    - Fact mapping:
      - Mirrors `should_partial_verify` in `SpecPV/specpv/speculate/utils.py`, which checks `enabled`, `retrieval_initialized`, and `get_seq_length() + total_tokens + 1 <= total_budget`.
    - Implementation status:
      - Implemented as `LlamaV2::shouldUseSpecPV(int seq_len) const`, which gates on `engine_param_.enable_specpv`, `specpv_supported_`,
        a live `specpv_kv_cache_`, `specpv_partial_threshold`, and the partial-KV budget relative to `eagleMaxEngineTokensPerStep()`.

#### 3.2.2 Integration with `runEagleTargetTreeDecode`

- [x] **S2.2 – Add SpecPV branch inside `runEagleTargetTreeDecode`**
  - File(s): `LlamaV2.cc`.
  - Today, `runEagleTargetTreeDecode`:
    - Builds scratch KV + packed masks.
    - Calls `UnifiedDecoder::Forward` over tree tokens, using full “prefix + tree” KV.
  - Extend it to:
    - If `shouldUseSpecPV(sequence_len)`:
      - Use **partial KV** as the base context:
        - Build a scratch KV **on top of `specpv_kv_cache_` view**, or
        - Directly feed a `spec_packed_mask`-compatible KV layout that only spans sink/retrieval/window/buffer tokens.
      - Ensure tree tokens attend only to this partial segment.
    - Else:
      - Keep using the existing full-KV scratch path (no SpecPV involvement).
  - Fact mapping:
    - Conceptually identical to `tree_decoding` in `SpecPV/specpv/speculate/utils.py`, which chooses between full KV (`full_past_key_values`) and partial KV (`partial_past_key_values`) based on `should_partial_verify`.
  - Implementation status:
    - `LlamaV2::runEagleTargetTreeDecode` now has a SpecPV branch that:
      - Derives a partial prefix length from the SpecPV budget and `PartialKVCache::global_verified_len()`,
      - Builds prefix+tree block geometry from this partial prefix,
      - Populates scratch KV blocks for prefix from `PartialKVCache::active_prefix(...)` instead of `Sequence::blocks`,
      - Leaves tree masks, logits→target_ids, and acceptance unchanged, and falls back to the full-KV path on any SpecPV invariant failure.

- [x] **S2.3 – Maintain `global_verified_lens` and per-layer `verified_lens` in tree decode**
  - On each EAGLE step where target-tree decode runs:
    - When **full-KV verify** is used for that step:
      - Update `specpv_kv_cache_->global_verified_lens` to equal the fully verified prefix length for the sequence.
      - This mirrors `partial_past_key_values.global_verified_lens = input_ids.shape[1]` in `SpecPV/specpv/speculate/utils.py::tree_decoding`.
    - When **partial-KV verify** is used:
      - Use EAGLE acceptance results (after `advanceSequencesByEagleAcceptance`) to:
        - Append newly verified tokens into the buffer via `PartialKVCache::update`.
        - Update per-layer `verified_lens[layer]` to reflect how many tokens in the buffer are now fully verified and visible to attention.
      - This is analogous to SpecPV’s `update_inference_inputs` logic, which both:
        - Rewrites full KV to make accepted tokens contiguous, and
        - Updates `PartialKVCache.verified_lens` while keeping `global_verified_lens` as the length of the fully trusted prefix.
  - Fact mapping:
    - Mirrors:
      - `partial_past_key_values.global_verified_lens = input_ids.shape[1]` in `tree_decoding`.
      - Per-layer `verified_lens` increments and buffer rewrites in `update_inference_inputs` and `PartialKVCache.update` in SpecPV.
  - Implementation status:
    - TurboMind tracks the fully verified prefix length via `LlamaV2::specpv_full_prefix_len_` and uses `PartialKVCache::verified_lens_` /
      `global_verified_len_` to represent the number of tail tokens stored in the SpecPV buffer. `updateSpecPVAfterAcceptance` computes the
      committed `max_len` after EAGLE acceptance and:
      - On first / full-refresh steps, calls `initSpecPVFromFullKV(max_len, ...)` to seed sink/retrieval/window and reset buffer lengths.
      - On subsequent steps, flattens the full KV prefix to `[B,H,L,D]` via `flattenPrefixKVForLayer`, slices the tail tokens beyond
        `specpv_full_prefix_len_`, and calls `PartialKVCache::update(layer, new_k, new_v)` per layer to append them into the buffer.
    - `runEagleTargetTreeDecode` uses `PartialKVCache::global_verified_len()` together with the configured sink/retrieval/window sizes to
      bound the effective prefix length in SpecPV mode. This differs slightly from the SpecPV paper’s naming (where `global_verified_lens`
      is the full prefix length), but achieves the same effect with an explicit `specpv_full_prefix_len_` field.

---

### 3.3 Phase 3 – Block summaries and retrieval refresh

**Goal**: Implement SpecPV’s block summary and retrieval selection, and integrate it with TurboMind attention.

#### 3.3.1 Block summaries (`Kmax`, `Kmin`)

- [ ] **S3.1 – Implement summary kernels**
  - File(s): `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/specpv_kv_kernels.cu` (new) or `eagle_kernels.cu` if suitable.
  - Implement CUDA kernels:
    - `summary_key_states`:
      - Inputs: `key_states[B, H, L, D]`, `seq_len`, `block_size`.
      - Outputs per block `i`:
        - `Kmax[i] = max(key_states[..., block_i_tokens, :])`
        - `Kmin[i] = min(key_states[..., block_i_tokens, :])`
      - Store as `[B, H, max_blocks, D]` for each layer.
  - Wire kernels into `PartialKVCache::summary_key_states`.
  - Fact mapping:
    - Directly corresponds to `PartialKVCache.summary_key_states` in `SpecPV/specpv/kv/kv_cache.py`, which maintains per-block max/min summaries.
  - Implementation status:
    - A first implementation exists in `PartialKVCache::summary_key_states` (host-side float32), which computes per-block Kmax/Kmin
      summaries and stores them in CPU tensors. A dedicated CUDA kernel file (`specpv_kv_kernels.cu`) remains future work for performance.

#### 3.3.2 Retrieval refresh

- [ ] **S3.2 – Implement `refresh_retrieval`**
  - File(s): same kernel file as S3.1 + `specpv_kv_cache.cc`.
  - Logic (blend of SpecPV and TurboMind abstractions):
    - Given:
      - `query_states[B, H, Q, D]` (queries from last verified tokens / tree candidates).
      - `Kmax`, `Kmin`.
    - Compute scores:
      - `s_{i,j} = max(q_j * Kmax_i^T, q_j * Kmin_i^T)`.
      - Reduce across queries to `s_i` by `max` or `mean`.
    - Choose top-`K` blocks (`n_retrieval_blocks`) by `s_i` per batch/head.
    - Convert block indices to token indices:
      - `indices = block_index * block_size + [0..block_size-1]`.
    - Gather keys/values from full KV (or from `sink + window`) into the `retrieval` segment of partial KV.
    - Fill `window` segment with the most recent tokens from full KV (last `window_size` tokens).
  - Fact mapping:
    - Matches `PartialKVCache.refresh_retrieval` in `SpecPV/specpv/kv/kv_cache.py`, which:
      - Computes per-block scores using `key_states_summary`.
      - Selects top blocks and fills `"retrieval"` and `"window"` slices accordingly.
  - Implementation status:
    - Implemented in `PartialKVCache::refresh_retrieval` as a host-based float32 path that scores blocks using Kmax/Kmin summaries, selects
      top retrieval blocks, and fills retrieval/window K/V slices. A future CUDA implementation can replace the current CPU loops.

---

### 3.4 Phase 4 – Spec buffer and periodic full verification

**Goal**: Implement the spec buffer for partially verified tokens and the logic for periodic full verification to “refresh” partial KV.

#### 3.4.1 Spec buffer usage

- [ ] **S4.1 – Append candidate tokens to buffer during verification**
  - During target-tree decode / verification:
    - When using SpecPV, candidate tokens’ KV (for tree nodes) are written into the **buffer** slice of partial KV (analogous to SpecPV’s `update`).
    - After acceptance:
      - Accepted tokens move from buffer into “trusted” region (implicit via `verified_lens` and retrieval/window updates).
      - Rejected tokens are dropped (just overwritten / masked).
  - Fact mapping:
    - Follows `PartialKVCache.update` in `SpecPV/specpv/kv/kv_cache.py`, which writes new tokens into the `"buffer"` slice and updates `verified_lens` when they become fully verified via `update_inference_inputs`.
  - Implementation status:
    - TurboMind’s current SpecPV v1 uses the buffer exclusively for **accepted tail tokens**, not for in-flight candidate tree tokens:
      - After each EAGLE step where `shouldUseSpecPV(max_len)` is true and SpecPV has been seeded, `updateSpecPVAfterAcceptance` flattens
        the full KV prefix, slices the newly committed tail (`max_len - specpv_full_prefix_len_`), and calls
        `PartialKVCache::update(layer, new_k, new_v)` per layer to append into the buffer.
      - Tree candidate tokens for the current step are still stored in the usual tree scratch KV and are not yet staged in the SpecPV
        buffer during verification.
    - Moving candidate tree tokens into the buffer during verification (and only keeping accepted ones) remains future work.

- [ ] **S4.2 – Bind buffer capacity to EAGLE tree size**
  - Ensure:
    - `n_spec_tokens_buf >= eagle_max_engine_tokens_per_step_ + safety_margin`.
  - If EAGLE config + SpecPV config violate this:
    - Log `[LlamaV2][SpecPV][fallback] buffer too small for EAGLE tree; disabling SpecPV`.
    - Disable SpecPV for the lifetime of the engine (`specpv_supported_ = false`).
  - Fact mapping:
    - Inspired by SpecPV’s `CacheConfig.n_spec_tokens_buf` and the way `initialize_past_key_values` sets it to `partial_spec_tokens + draft_model.total_tokens + 1`.
  - Implementation status:
    - TurboMind enforces buffer-related safety indirectly:
      - `LlamaV2::shouldUseSpecPV` uses `specpv_cache_config_.total_budget()` and the per-step `eagleMaxEngineTokensPerStep()` budget to
        gate whether SpecPV may be used at a given context length.
      - `updateSpecPVAfterAcceptance` triggers a full-refresh when `specpv_kv_cache_->global_verified_len()` plus the next step’s token
        budget would overflow the configured `n_spec_tokens_buf`.
    - A stricter pre-flight check that explicitly verifies `n_spec_tokens_buf >= eagle_max_engine_tokens_per_step_ + margin` before
      enabling SpecPV remains TODO.

#### 3.4.2 Periodic full verification and refresh

- [x] **S4.3 – Full verification refresh trigger**
  - Add criteria:
    - `specpv_kv_cache_->verified_lens[layer]` exceeds some fraction of retrieval/window budgets.
    - Buffer tokens + new candidate tokens would overflow `n_spec_tokens_buf`.
    - Or user-configured `max_partial_steps_before_full_verify`.
  - When triggered:
    - Run a **full KV verification** step for the EAGLE tree (use existing full-KV `runEagleTargetTreeDecode` path).
    - After that full step:
      - Reinitialize partial KV via `initSpecPVFromFullKV`.
      - Reset `verified_lens` and buffer.
  - Fact mapping:
    - Implements the “periodic full verification” described in §3.3 of the paper and realized in practice by the interplay between full KV calls and `PartialKVCache.reset` / `init_key_values` in SpecPV.
  - Implementation status:
    - Implemented via `LlamaV2::updateSpecPVAfterAcceptance`:
      - Maintains a per-engine `specpv_partial_steps_` counter incremented on each step where SpecPV is active.
      - Computes a `buffer_close_to_full` condition from `PartialKVCache::global_verified_len()`, `SpecPVCacheConfig::buffer_size()`, and
        `eagleMaxEngineTokensPerStep()`.
      - When either the step counter exceeds `specpv_full_refresh_steps` or the buffer is close to full, logs a `[SpecPV] full-refresh`
        message, calls `specpv_kv_cache_->reset()`, clears `specpv_retrieval_initialized_`, resets `specpv_partial_steps_` and
        `specpv_full_prefix_len_`, thereby forcing the next step to run full-KV tree decode before reseeding SpecPV from the refreshed
        prefix.

---

### 3.5 Phase 5 – Config, gating, and safety

**Goal**: Ensure SpecPV can be turned on/off cleanly and never breaks baseline behaviour.

#### 3.5.1 Engine params and gating

- [x] **S5.1 – Engine param parsing**
  - File(s): engine param parser (where `enable_eagle_target_tree` is parsed).
  - Add fields:
    - `enable_specpv` (bool),
    - `specpv_block_size`, `specpv_n_sink_blocks`, `specpv_n_retrieval_blocks`, `specpv_n_window_blocks`, `specpv_n_spec_tokens_buf`,
    - `specpv_partial_threshold`, `specpv_full_refresh_interval` (optional).
  - Implementation status:
    - Engine params and Triton config parsing for SpecPV are implemented: `EngineParam` exposes `enable_specpv` and the `specpv_*` fields in
      `llama_params.h`, and `LlamaTritonModel` reads them from `speculative_config` YAML. LMDeploy’s `SpeculativeConfig` mirrors these
      fields and passes them into `TurbomindEngineConfig.speculative_config`.

- [x] **S5.2 – Guarded usage**
  - Ensure every SpecPV entry point checks:
    - `enable_specpv && specpv_supported_`.
  - If any invariant fails at runtime (dtype/layout mismatch, insufficient buffer, incompatible vocab, etc.):
    - Log `[LlamaV2][SpecPV][fallback] ...`.
    - Set `specpv_supported_ = false`.
    - Reset / free `specpv_kv_cache_`.
    - Continue with **baseline target-tree decode** or full non-SpecPV EAGLE pipeline.
  - Implementation status:
    - All SpecPV entry points in `LlamaV2` (constructor, `flattenPrefixKVForLayer`, `initSpecPVFromFullKV`, `updateSpecPVAfterAcceptance`,
      and `runEagleTargetTreeDecode`) gate on `engine_param_.enable_specpv` and `specpv_supported_`, and on any invariant failure they log a
      `[LlamaV2][SpecPV][fallback]` message, clear `specpv_kv_cache_`, reset SpecPV state, and revert to the full-KV EAGLE3 path.

#### 3.5.2 Dtype and geometry invariants

- [x] **S5.3 – Dtype checks (BF16 / FP32 / MXFP4)**
  - Check:
    - Flattening the full KV prefix for SpecPV seeding currently supports fp16/bf16 caches; quantized KV (int8/int4) is not used in SpecPV mode.
    - The partial KV cache itself uses a float32 view internally for K/V and summaries; tree decode still uses fp16 KV blocks for attention
      and keeps logits FP32 as in the full-KV path.
  - On mismatch:
    - Same fallback as above, no change to baseline.
  - Implementation status:
    - `flattenPrefixKVForLayer` enforces fp16/bf16 KV and logs a `[SpecPV][fallback]` message and disables SpecPV on any dtype/geometry/
      allocation mismatch. `runEagleTargetTreeDecode`’s SpecPV branch additionally checks that the scratch KV dtype is unquantized fp16 and
      falls back to full-KV EAGLE3 otherwise.

---

## 4. Longer-term / Optional TODOs

These are follow-ups once the core SpecPV mechanics are in place.

- [ ] **L1 – SpecPV-aware benchmarking**
  - Extend `benchmark_speculative.py` / LMDeploy EAGLE benchmark scripts:
    - Add flags to enable SpecPV, set partial KV budgets, and full refresh intervals.
    - Record:
      - Context length vs speedup vs draft accept length.
      - Partial KV budget sweeps (512 – 8K tokens).

- [ ] **L2 – Per-request adaptive budgets**
  - Allow different SpecPV cache budgets per request:
    - E.g. smaller budgets for very long contexts, larger for “medium” contexts.
  - Implement simple heuristics based on `prompt_len` and `max_new_tokens`.

- [ ] **L3 – Shared design with non-EAGLE speculative paths**
  - Once stable, expose SpecPV partial KV for other draft modes (e.g. Medusa/MTP or plain self-speculation) in TurboMind, not only EAGLE3.

---

## 5. Summary

- SpecPV gives us a concrete blueprint for **partial KV verification**: sink + retrieval + window + buffer, block-wise summaries, partial vs full verify, and periodic refresh.
- This RFC lays out **code-level building blocks** to add a SpecPV-style path into TurboMind’s EAGLE3 target-tree decode:
  - Partial KV cache implementation,
  - Gating logic for when to use it,
  - Block summary + retrieval kernels,
  - Spec buffer and full verification refresh,
  - Robust gating and fallbacks.
- Implementation should preserve all current semantics when `enable_specpv` is off or falls back, while unlocking better verification efficiency for long-context EAGLE3 when it is on.
