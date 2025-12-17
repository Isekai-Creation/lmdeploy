#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper to run EAGLE3 micro perf gates before any
# large-context PERF_MODE benchmarking. This script is intended to be
# called from CI or local perf harnesses.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_DIR="${ROOT_DIR}/results_eagle3_micro"
mkdir -p "${MICRO_DIR}"

# Paths to models (adjust if needed or override via env)
MODEL_PATH="${MODEL_PATH:-/workspace/aimo/models/gpt-oss-120b}"

echo "[run_spec_suite] Environment:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  MICRO_DIR: ${MICRO_DIR}"

read -r -a BASELINE_CONTEXTS <<< "${SPEC_SUITE_CONTEXTS:-8192 16384 32768}"
read -r -a BASELINE_BATCHES <<< "${SPEC_SUITE_BATCH_SIZES:-1 4 8}"
MAX_NEW_ARGS=()
if [[ -n "${SPEC_SUITE_MAX_NEW_TOKENS:-}" ]]; then
  MAX_NEW_ARGS+=(--max-new-tokens "${SPEC_SUITE_MAX_NEW_TOKENS}")
fi

echo "  BASELINE_CONTEXTS: ${BASELINE_CONTEXTS[*]}"
echo "  BASELINE_BATCHES: ${BASELINE_BATCHES[*]}"
if [[ ${#MAX_NEW_ARGS[@]} -gt 0 ]]; then
  echo "  MAX_NEW_TOKENS: ${SPEC_SUITE_MAX_NEW_TOKENS}"
fi

# Run baseline-only drift benchmarks (non-speculative)
echo ""
echo "[run_spec_suite] Running DriftEngine baseline scenarios..."
python_args=(
  "${ROOT_DIR}/benchmark_speculative.py"
  --model-path "${MODEL_PATH}"
  --output-dir "${MICRO_DIR}"
  --scenario baseline
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}"
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"
)
python_args+=(--baseline-contexts)
python_args+=("${BASELINE_CONTEXTS[@]}")
python_args+=(--baseline-batch-sizes)
python_args+=("${BASELINE_BATCHES[@]}")
if [[ ${#MAX_NEW_ARGS[@]} -gt 0 ]]; then
  python_args+=("${MAX_NEW_ARGS[@]}")
fi
python3 "${python_args[@]}"

echo ""
echo "[run_spec_suite] Benchmarks completed. Results are in ${MICRO_DIR}"
