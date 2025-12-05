# EAGLE CI / Test Configuration (Engineer C)

This document captures how to wire up TurboMind EAGLE tests in CI so they
run automatically with small, EAGLE-capable models.

## Environment variables

The following environment variables must be set for EAGLE end-to-end tests:

- `MODEL_PATH` – path to a TurboMind base model directory (e.g. a small GPT-OSS or Llama variant converted to TurboMind).
- `SPEC_MODEL_PATH` – path to the matching EAGLE draft model directory. If not set explicitly, tests may fall back to `MODEL_PATH`, but full EAGLE coverage requires a dedicated draft model.

These variables are consumed by:

- `lmdeploy/tests/test_benchmark_speculative_integration.py`
- `lmdeploy/tests/test_eagle_e2e.py`
- any future multi-token EAGLE tests under `lmdeploy/tests`.

## Recommended CI layout

In CI, mount or download small TurboMind engines to a fixed location, for example:

- `/models/gpt-oss-small` – base TurboMind engine directory.
- `/models/gpt-oss-small-eagle3` – matching EagleNet draft engine directory.

Then export:

```bash
export MODEL_PATH=/models/gpt-oss-small
export SPEC_MODEL_PATH=/models/gpt-oss-small-eagle3
```

before running the EAGLE tests.

## Running EAGLE tests in CI

With `MODEL_PATH` / `SPEC_MODEL_PATH` set and a CUDA device available, CI
can run the EAGLE-related tests via:

```bash
pytest lmdeploy/tests/test_benchmark_speculative_integration.py \
       lmdeploy/tests/test_eagle_e2e.py \
       lmdeploy/tests/turbomind/test_eagle_metrics.py \
       lmdeploy/tests/turbomind/test_speculative_manager_eagle.py \
       lmdeploy/tests/turbomind/test_speculative_manager_eagle_batch.py
```

Additional multi-token EAGLE tests should be added to this list once the
multi-token decode loop is implemented in `LlamaBatch` / `LlamaV2_eagle`.

