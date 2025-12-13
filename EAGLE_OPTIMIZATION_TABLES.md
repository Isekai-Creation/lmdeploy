# EAGLE3 Optimization Tables – Current Measured Performance

This file records **actual benchmark results** from this repo on the
current sm120 machine, using:

- Main model: `models/gpt-oss-120b`
- Draft model: `models/gpt-oss-120b-eagle3`
- Draft TM dir: `models/gpt-oss-120b-eagle3-tm-draft`
- TurboMind offline pipeline via `run_spec_suite.sh` /
  `benchmark_speculative.py`
- **SpecPV disabled** in all runs below (`enable_specpv=False`).

Values are taken directly from the JSON files under `results/` and
`results/test_metrics/` as of this snapshot.

All throughput figures are mean `throughput_tokens_per_sec.mean`.
All latency figures are mean `latency_ms_per_token.mean`.

---

## 1. Single‑Context, Batch=1 (8K / 32K)

### 1.1 Context 8K, batch_size = 1

Source: `results/test_metrics/baseline_single_context8k.json` and
`results/test_metrics/speculative_single_context8k.json`

| JSON file                                | Scenario name              | Context | Spec? | num_spec_tokens | Throughput (tok/s) | Latency (ms/tok) | EAGLE mean_accept_rate | EAGLE mean_accept_len |
|------------------------------------------|----------------------------|---------|-------|-----------------|--------------------|------------------|------------------------|-----------------------|
| `baseline_single_context8k.json`         | Baseline_Single_Context8K  | 8192    | No    | 0               | 119.74             | 8.35             | –                      | –                     |
| `speculative_single_context8k.json`      | Speculative_Single_Context8K | 8192  | Yes   | 3               | 136.45             | 7.33             | 0.628                  | 2426.0                |

Notes:

- At 8K, 3‑token EAGLE3 **beats** baseline (≈1.14× throughput, lower latency).
- SpecPV is off; gains are purely from multi‑token full‑KV EAGLE3.

### 1.2 Context 32K, batch_size = 1 (older vs newer runs)

We have **two sets** of 32K single‑context results:

1. Older runs in `results/test_metrics/`
2. Newer runs in `results/20251212_185539/` after fixing the 32K+5 tail bug
   and other EAGLE3 issues.

#### 1.2.1 Older 32K runs (test_metrics)

Source: `results/test_metrics/baseline_single_context32k.json`,
`results/test_metrics/speculative_single_context32k_3tokens.json`

| JSON file                                   | Scenario name                   | Context | Spec? | num_spec_tokens | Throughput (tok/s) | Latency (ms/tok) | EAGLE mean_accept_rate | EAGLE mean_accept_len |
|---------------------------------------------|---------------------------------|---------|-------|-----------------|--------------------|------------------|------------------------|-----------------------|
| `baseline_single_context32k.json`           | Baseline_Single_Context32K      | 32768   | No    | 0               | 118.22             | 8.46             | –                      | –                     |
| `speculative_single_context32k_3tokens.json`| Speculative_Single_Context32K_3tokens | 32768 | Yes   | 3               | 26.76              | 37.38            | 1.000                  | 4.0                   |

This older 32K 3‑token result is **much slower than baseline** and reflects
pre‑fix behaviour (before tail crash fixes and other EAGLE3 updates).

#### 1.2.2 Newer 32K runs (post‑tail‑fix)

Source: `results/20251212_185539/baseline_single_context32k.json`,
`results/20251212_185539/speculative_single_context32k_3tokens.json`,
`results/20251212_185539/speculative_single_context32k_5tokens.json`

| JSON file                                               | Scenario name                        | Context | Spec? | num_spec_tokens | Throughput (tok/s) | Latency (ms/tok) | EAGLE mean_accept_rate | EAGLE mean_accept_len |
|---------------------------------------------------------|--------------------------------------|---------|-------|-----------------|--------------------|------------------|------------------------|-----------------------|
| `20251212_185539/baseline_single_context32k.json`       | Baseline_Single_Context32K           | 32768   | No    | 0               | 113.95             | 8.78             | –                      | –                     |
| `20251212_185539/speculative_single_context32k_3tokens.json` | Speculative_Single_Context32K_3tokens | 32768 | Yes   | 3               | 65.56              | 15.25            | 0.740                  | 1085.0                |
| `20251212_185539/speculative_single_context32k_5tokens.json` | Speculative_Single_Context32K_5tokens | 32768 | Yes   | 5               | 164.25             | 6.09             | 0.615                  | 121.0                 |

Observations:

- Baseline vs 3‑token EAGLE3 (newer run):
  - 3‑token EAGLE3 is still **slower** than baseline at 32K
    (≈65.6 tok/s vs ≈114 tok/s).
  - On this build, with current BF16 GEMM fallbacks, the fixed cost of
    the draft/tree/KV path is too high for only 3 proposed tokens at
    32K context. Treat this configuration as **functionally correct but
    not performance‑optimal**.
- Baseline vs 5‑token EAGLE3:
  - 5‑token EAGLE3 is **faster** than baseline:
    ≈164 tok/s vs ≈114 tok/s.
  - At 5 tokens, the speculative overhead is amortised enough to give a net
    speedup even with BF16 GEMM fallbacks.

---

## 2. Batch=4, Context 16K

Source: `results/test_metrics/baseline_batch4_context16k.json`,
`results/test_metrics/speculative_batch4_context16k_3tokens.json`

SpecPV is disabled here as well.

| JSON file                                   | Scenario name                     | Batch | Context | Spec? | num_spec_tokens | Throughput (tok/s) | Latency (ms/tok) | EAGLE mean_accept_rate | EAGLE mean_accept_len |
|---------------------------------------------|-----------------------------------|-------|---------|-------|-----------------|--------------------|------------------|------------------------|-----------------------|
| `baseline_batch4_context16k.json`           | Baseline_Batch4_Context16K        | 4     | 16384   | No    | 0               | 450.10             | 2.22             | –                      | –                     |
| `speculative_batch4_context16k_3tokens.json`| Speculative_Batch4_Context16K_3tokens | 4   | 16384   | Yes   | 3               | 68.26              | 14.65            | 0.736                  | 3017.63               |

This batch 16K run is currently **much slower** with 3‑token EAGLE3 than
baseline. It is a clear indicator that:

- The per‑step speculative overhead (extra GEMMs + tree + KV logic) is large
  compared to baseline at this context and batch size, and
- Without sm90/sm120 fused BF16 kernels, the multi‑token benefits are not
  realised for `num_spec_tokens=3` in this workload.

---

## 3. SpecPV Dimension (Current)

Across all the runs captured above:

- `use_speculation=True` only controls **EAGLE3**; SpecPV is off.
- There are currently **no** JSONs in this tree that represent:
  - EAGLE3 + SpecPV enabled, or
  - baseline vs EAGLE3 vs EAGLE3+SpecPV for the same scenario.

Once SpecPV is enabled and validated, this file should be extended with
additional rows capturing:

- `specpv_enabled` (bool) and any relevant SpecPV params (sink, retrieval,
  window),
- Throughput / latency / memory for:
  - Baseline (no spec),
  - EAGLE3 full‑KV,
  - EAGLE3 + SpecPV,
- Acceptance metrics for each of those variants, at least for:
  - Single context: 8K / 16K / 32K, batch 1,
  - Batch context: 16K / 32K, batch 4 or more.

---

## 4. How to Use These Tables

- To track **optimizations over time**:
  - Re‑run `run_spec_suite.sh` after each major kernel / EAGLE3 / SpecPV
    change.
  - Copy the new JSON summaries into updated rows here, so regressions and
    improvements are visible at a glance.
- To reason about **where to focus next**:
  - 8K single context already shows gains for 3‑token EAGLE3.
  - 32K single context only shows a win at 5 tokens; 3 tokens is still
    slower than baseline.
  - 16K batch‑4 with 3 tokens is significantly slower than baseline.
  - These points line up with the remaining work called out in
    `EAGLE_TODO_OPTMIZATIONS.md`:
    - add sm90/sm120 fused BF16 kernels for the hot GEMMs,
    - tune EAGLE3 gating (`num_spec_tokens`, thresholds, alpha, temperature),
    - and then layer SpecPV on top for long contexts.
