# TurboMind EAGLE Usage and Debugging

This document summarizes how to enable TurboMind EAGLE in LMDeploy,
what metrics and helper tools are available, and how to debug common
failure modes using the new logging, NVTX ranges, and Python wrappers.

## Enabling TurboMind EAGLE

- Configure a draft model and EAGLE3 speculative decoding via
  `SpeculativeConfig`:

  - `method="eagle3"`
  - `model`: path to the draft (EagleNet) model
  - `num_speculative_tokens`: number of draft tokens per step
  - Optional structural parameters:
    - `max_path_len`
    - `max_decoding_tokens`
    - `max_non_leaves_per_layer`

- Pass the `SpeculativeConfig` into `TurbomindEngineConfig`:

  ```python
  from lmdeploy import TurbomindEngineConfig
  from lmdeploy.speculative_config import SpeculativeConfig

  spec_cfg = SpeculativeConfig(
      method="eagle3",
      model="/path/to/draft-model",
      num_speculative_tokens=4,
  )

  engine_cfg = TurbomindEngineConfig(
      tp=1,
      session_len=32768,
      max_batch_size=8,
      speculative_config=spec_cfg,
  )
  ```

- On the C++ side, TurboMind reads this configuration into
  `EngineParam::speculative_config`. The helper
  `SpeculativeConfig.to_turbomind_spec_dict()` produces a dict that
  mirrors the keys used by `EngineParam` so alignment can be checked
  offline via `check_turbomind_spec_alignment`.

## Enabling EAGLE3 **multi-token** on TurboMind (single GPU, offline)

Multi-token behaviour is controlled **only** by the speculative config and topology:

- `SpeculativeConfig.method in {"eagle", "eagle3"}`.
- `SpeculativeConfig.num_speculative_tokens > 1` → multi-token EAGLE3.
- `tp == 1` → multi-token enabled; for `tp != 1` TurboMind falls back to single-token EAGLE internally.
- A valid draft model configured via `SpeculativeConfig.model`.

No `LMDEPLOY_EAGLE_*` flags and no per-request `disable_eagle_multitoken` field are required or supported for mode selection.

Example (Python offline pipeline):

```python
from lmdeploy import pipeline as lm_pipeline
from lmdeploy.messages import GenerationConfig, TurbomindEngineConfig
from lmdeploy.speculative_config import SpeculativeConfig, validate_eagle_runtime_config

spec_cfg = SpeculativeConfig(
    method="eagle3",
    model="/path/to/draft_model",     # EagleNet draft
    num_speculative_tokens=4,         # >1 => multi-token
    eagle_debug=False,
    eagle_metrics_debug=True,
)

engine_cfg = TurbomindEngineConfig(
    tp=1,                             # multi-token currently tp=1 only
    session_len=256,                  # e.g. >= 4 * max_new_tokens
    speculative_config=spec_cfg,
    enable_metrics=True,
)

# Fail fast on misconfig (tp != 1, bad max_decoding_tokens, etc.)
validate_eagle_runtime_config(engine_cfg, spec_cfg)

pipe = lm_pipeline(
    model_path="/path/to/turbomind_model",
    backend_config=engine_cfg,
    speculative_config=spec_cfg,
)

gen_cfg = GenerationConfig(
    max_new_tokens=64,
)

out = pipe(["Hello, world"], gen_config=gen_cfg)[0]
print(out.text)

# EAGLE metrics from TurboMind
print(out.req_metrics.spec_info)
```

Mode selection is now purely config-driven:

- **EAGLE disabled**: no `speculative_config` on the engine.
- **Single-token EAGLE**: `method in {"eagle","eagle3"}`, `num_speculative_tokens == 1`.
- **Multi-token EAGLE3**: `method in {"eagle","eagle3"}`, `num_speculative_tokens > 1`, and `tp == 1`.

If `tp != 1`, multi-token is automatically disabled inside TurboMind and the engine behaves like single-token EAGLE or baseline.

## Offline EagleModule, KV, and Metrics Behaviour

The EAGLE integration in TurboMind is structured so that the main
components can be exercised independently of the full decode loop:

- **EagleModule**:

  - Loads draft model weights from the directory pointed to by
    `SpeculativeConfig.model`.
  - Uses a shallow draft network (RMSNorm + attention + FC + LM head)
    wired via `LlamaLinear` to produce draft logits for the current
    target hidden states.
  - Is constructed and managed by `LlamaV2` and is only considered
    enabled when `EagleModule::load` succeeds.

- **EAGLE KV over-provisioning**:

  - When EAGLE is enabled, TurboMind over-provisions the KV cache
    based on `spec_max_decoding_tokens` and `spec_max_draft_path_len`
    so that speculative tokens can be held without starving normal
    decode state.
  - The additional KV capacity is reflected in the effective
    `cache_max_block_count` used when instantiating `SequenceManager`.

- **KV rewind helper**:

  - `computeAndInvokeKVCacheRewind` (implemented in
    `lmdeploy/lmdeploy/turbomind/kernels/speculative_decoding/kv_rewind_helper.cu`)
    takes per-sequence draft/accepted token counts and computes
    rewind lengths for the KV cache.
  - It then calls `invokeKVCacheRewind` with a `KVCacheRewindParams`
    struct so the kernel can mark tail KV blocks as logically free.
  - This helper is deliberately decoupled from `LlamaBatch` so it can
    be reused from multiple integration points and unit tested via
    the Python wrapper in
    `lmdeploy/turbomind/kernels/speculative_decoding/common.py`.

- **EAGLE metrics**:

  - TurboMind accumulates per-request speculative metrics in
    `RequestMetrics`:
    - `eagle_total_draft_tokens`
    - `eagle_total_accepted_tokens`
    - `eagle_steps`
    - `eagle_total_rewound_tokens`
    - `eagle_rewind_steps`
  - These are exposed to Python via `_turbomind.RequestMetrics` and
    attached to `EngineOutput.req_metrics.spec_info` in
    `lmdeploy/turbomind/turbomind.py::_get_metrics` whenever
    `eagle_steps > 0`.
  - `SpeculativeDecodingStats` and `EagleMetricsSummary` in
    `lmdeploy/metrics/stats.py` provide a convenient way to aggregate
    these metrics across many requests for offline analysis.

## EAGLE Failure Modes and Debugging Runbook

This section describes common issues when enabling TurboMind EAGLE and
how to debug them.

### 1. Draft model load failures

**Symptoms**

- Logs contain `[EAGLE][EagleModule::load]` warnings or errors.
- EAGLE appears to be silently disabled and runs fall back to baseline.

**Checklist**

- Verify that `config.yaml` in the draft model directory includes:
  - `model_config.hidden_units`
  - `model_config.vocab_size`
  - `model_config.head_num`
  - `model_config.size_per_head`
  - `model_config.inter_size`
- Ensure the consistency constraint holds:
  `hidden_units == head_num * size_per_head`.
- Confirm that all required weight files exist and have the expected
  size. On mismatch, `EagleModule::load` will log a clear message and
  disable EAGLE for the engine.
- Check that the draft model path configured in
  `TurbomindEngineConfig.speculative_config.model` matches the on-disk
  directory you expect.

### 2. EAGLE disabled at runtime

**Symptoms**

- `req_metrics.spec_info` is absent (or `None`) on `EngineOutput`.
- Benchmark JSON has no `eagle_speculation` block.
- Generation behaviour matches baseline exactly.

**Checklist**

- Confirm `SpeculativeConfig.method` is `"eagle"` or `"eagle3"`.
- Use `SpeculativeConfig.to_turbomind_spec_dict()` together with
  `check_turbomind_spec_alignment` to ensure the Python-side
  configuration matches the engine `speculative_config` block.
- Make sure a draft model path is provided and that
  `EagleModule::load` succeeded (look for `[EAGLE]` logs at engine
  startup).
- Verify that `RequestMetrics.eagle_steps` is greater than zero for
  speculative runs; `_get_metrics` only populates `spec_info` when
  real EAGLE steps were executed.

### 3. Unexpected KV rewind behaviour

**Symptoms**

- Generated text appears truncated or repeated after speculative steps.
- GPU memory usage spikes unexpectedly over time.

**Checklist**

- Enable detailed KV rewind logs:
  - Set `LMDEPLOY_EAGLE_KV_DEBUG=1` (or `LMDEPLOY_EAGLE_DEBUG=1`) in
    the environment before starting the engine.
  - Inspect `[EAGLE][KVRewind]` lines to confirm that rewind lengths
    and slot indices match expectations.
- Use the Python helper
  `lmdeploy.turbomind.kernels.speculative_decoding.common.invoke_kv_cache_rewind`
  together with `benchmark_kv_cache_rewind(...)` to sanity-check the
  kernel behaviour on synthetic data.
- Ensure that `EagleKVRewindConfig` (block size, max batch size,
  max blocks per sequence) matches the KV cache configuration used by
  the engine. Mismatches here can cause incorrect block reuse.

### 4. Profiling EAGLE hot paths

When profiling with Nsight Systems or Nsight Compute, EAGLE-specific
workloads are annotated via NVTX ranges so they can be distinguished
from the rest of the decode loop:

- Ensure the build enables NVTX support.
- Relevant ranges include:
  - `EagleModule::forward` – the draft network forward path.
  - `EAGLE::KVCacheRewind` – the host-side KV rewind helper +
    kernel.
- You can further enable detailed logging via:
  - `LMDEPLOY_EAGLE_KV_DEBUG=1` – KV rewind-specific traces from the C++ helper and kernel.

### Debug / metrics flags (config-driven)

Most EAGLE debug/metrics verbosity is controlled via `SpeculativeConfig`, not environment variables:

- `eagle_debug: bool` – enables additional C++ EAGLE debug traces (e.g. draft/accept logs).
- `eagle_metrics_debug: bool` – enables extra per-step metrics logs in both C++ (`[LlamaBatch][EAGLE_METRICS]`) and Python (`[EAGLE][Metrics]`).

These flags flow through `SpeculativeConfig` → `EngineParam` and into the TurboMind backend. The only remaining env knob is `LMDEPLOY_EAGLE_KV_DEBUG` for KV-specific debugging.

## Sanity checking that EAGLE3 multi-token is actually active

To quickly verify that EAGLE3 multi-token is wired correctly on a single GPU, you can use the smoke helper:

```python
from lmdeploy.turbomind.eagle_inspect import eagle3_multitoken_smoke

eagle3_multitoken_smoke(
    model_path="/path/to/turbomind_model",
    spec_model_path="/path/to/draft_model",
    prompt="Hello, world",
    num_spec_tokens=4,
    max_new_tokens=32,
)
```

This will:

- Build a TurboMind pipeline with `method="eagle3"`, `num_speculative_tokens=4`, `tp=1`.
- Run a single prompt.
- Print:
  - The generated text.
  - `req_metrics.spec_info` from TurboMind.
  - A small “sanity” summary, including:
    - `num_drafts > 0?`
    - `multi_token_effect? mean_acceptance_length > 1`

If `mean_acceptance_length > 1`, then multi-token EAGLE3 is actually committing extra tokens (not just running in single-token mode).

`inspect_offline_eagle(...)` also prints a summary line with:

- `num_spec_tokens` (from `SpeculativeConfig.num_speculative_tokens`)
- `mean_acceptance_length`
- `mean_acceptance_rate`

For multi-token to be effective, you should see `mean_acceptance_length > 1.0` for realistic prompts/models.

## Using EAGLE Metrics in Benchmarks

- The benchmark helper `inference/benchmark_speculative.py` aggregates
  TurboMind EAGLE metrics across runs via `SpeculativeDecodingStats`
  and `EagleMetricsSummary`.

- When speculative decoding is enabled and the engine reports
  `spec_info` from TurboMind, the JSON results include an
  `eagle_speculation` block:

  - `enabled`: `true`
  - `num_drafts`: number of speculative drafts observed
  - `total_draft_tokens`: total draft tokens proposed
  - `total_accepted_tokens`: total tokens accepted from drafts
  - `mean_acceptance_rate`: `total_accepted_tokens / total_draft_tokens`
  - `mean_acceptance_length`: average accepted length per draft

- When EAGLE is disabled or the engine does not report speculative
  metrics, this block is omitted so downstream tooling can treat the
  run as a pure baseline.

## Kernel Microbenchmarks and CLI Helpers

For low-level performance investigation and shape sanity checks, a few
lightweight tools are available:

- `lmdeploy.turbomind.kernels.speculative_decoding.common`:

  - `benchmark_kv_cache_rewind(...)` – runs the KV rewind helper on
    synthetic inputs (CPU or CUDA via torch) and reports timing stats.
  - `benchmark_accept_draft_tokens(...)` – exercises the acceptance
    kernel (`eagle_accept_draft_tokens`) on synthetic drafts/targets
    and reports average latency and accept operations per second.
  - `benchmark_pack_accepted_paths(...)` – benchmarks the path-packing
    kernel (`eagle_pack_accepted_paths`) for a given batch size,
    number of paths, and path length.

- `scripts/eagle_inspect_tree.py`:

  - Loads an `eagle_tree.yaml` via `--tree`.
  - Prints basic tree statistics (node count, branching factor).
  - Reports approximate `EagleBuffers`-style shapes for `draft_paths`
    and packed attention masks given `--batch-size`,
    `--max-decoding-tokens`, and `--max-path-len`.

## Optional EAGLE Tools (offline diagnostics)

The following tools are optional extras for engineers tuning EAGLE and
do not affect normal decode behaviour:

- `scripts/eagle_inspect_tree.py --what-if`:

  - In addition to the basic stats above, `--what-if` mode derives
    recommended structural parameters for `SpeculativeConfig` from the
    tree:
      - observed maximum path depth (for `max_path_len`),
      - maximum non-leaf nodes per layer (for `max_non_leaves_per_layer`),
      - and rough speculative KV usage based on `--block-size`.
  - Example:

    ```bash
    python scripts/eagle_inspect_tree.py \
      --tree /path/to/eagle_tree.yaml \
      --max-decoding-tokens 16 \
      --max-path-len 8 \
      --batch-size 4 \
      --block-size 16 \
      --num-spec-tokens 4 \
      --what-if
    ```

- `lmdeploy.turbomind.eagle_inspect.inspect_offline_eagle`:

  - Builds a TurboMind pipeline with an EAGLE3 `SpeculativeConfig`,
    runs a small set of prompts, and prints an `EagleMetricsSummary`
    derived from `req_metrics.spec_info`.
  - Intended for GPU-enabled environments with small TurboMind/EAGLE
    models.
  - Example:

    ```python
    from lmdeploy.turbomind.eagle_inspect import inspect_offline_eagle

    summary = inspect_offline_eagle(
        model_path="/path/to/target/turbomind_model",
        spec_model_path="/path/to/eagle/draft_model",
        num_spec_tokens=4,
        max_new_tokens=64,
    )
    print(summary)
    ```

These helpers stay strictly within Engineer-A scope (kernels,
EagleModule, metrics, and offline tooling) and are safe to use without
modifying LlamaBatch, LlamaV2_eagle, or DynamicDecodeLayer.
