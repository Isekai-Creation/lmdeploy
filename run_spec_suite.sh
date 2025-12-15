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
SPEC_MODEL_PATH="${SPEC_MODEL_PATH:-/workspace/aimo/models/gpt-oss-120b-eagle3}"

echo "[run_spec_suite] Environment:"
echo "  MODEL_PATH: ${MODEL_PATH}"
echo "  SPEC_MODEL_PATH: ${SPEC_MODEL_PATH}"
echo "  MICRO_DIR: ${MICRO_DIR}"

# 1. Run Baseline Scenarios (8K Context)
# This tests standard generation without speculation hooks active
echo ""
echo "[run_spec_suite] Running Baseline (8K Context)..."
python3 benchmark_speculative.py \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${MICRO_DIR}" \
  --scenario baseline \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

# 2. Run Single-Batch 32K Scenarios (Baseline + Spec)
# This uses EAGLE3 and tests the core speculative decoding loop
echo ""
echo "[run_spec_suite] Running Single 32K Scenarios (Baseline + Spec)..."
python3 benchmark_speculative.py \
  --model-path "${MODEL_PATH}" \
  --spec-model-path "${SPEC_MODEL_PATH}" \
  --output-dir "${MICRO_DIR}" \
  --scenario single \
  --warmup-runs "${SPEC_SUITE_WARMUP_RUNS:-1}" \
  --measurement-runs "${SPEC_SUITE_MEASUREMENT_RUNS:-1}"

echo ""
echo "[run_spec_suite] Benchmarks completed. Results are in ${MICRO_DIR}"

