#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper to run EAGLE3 micro perf gates before any
# large-context PERF_MODE benchmarking. This script is intended to be
# called from CI or local perf harnesses.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MICRO_DIR="${ROOT_DIR}/results_eagle3_micro"
mkdir -p "${MICRO_DIR}"

BASELINE_SINGLE="${MICRO_DIR}/single32k_baseline.json"
SPEC_SINGLE="${MICRO_DIR}/single32k_spec.json"
BASELINE_B4="${MICRO_DIR}/batch4_16k_baseline.json"
SPEC_B4="${MICRO_DIR}/batch4_16k_spec.json"

echo "[run_spec_suite] Running micro gates for EAGLE3..."

python3 benchmark_speculative.py \
  --scenario single32k \
  --mode baseline \
  --output "${BASELINE_SINGLE}"

python3 benchmark_speculative.py \
  --scenario single32k \
  --mode eagle3_spec \
  --output "${SPEC_SINGLE}"

python3 tools/check_perf_gates.py \
  --mode single32k \
  --baseline-json "${BASELINE_SINGLE}" \
  --spec-json "${SPEC_SINGLE}"

python3 benchmark_speculative.py \
  --scenario batch4_16k \
  --mode baseline \
  --output "${BASELINE_B4}"

python3 benchmark_speculative.py \
  --scenario batch4_16k \
  --mode eagle3_spec \
  --output "${SPEC_B4}"

python3 tools/check_perf_gates.py \
  --mode batch4_16k \
  --baseline-json "${BASELINE_B4}" \
  --spec-json "${SPEC_B4}"

echo "[run_spec_suite] All EAGLE3 micro perf gates PASSED."

