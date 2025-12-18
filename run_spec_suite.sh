#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_DIR="${ROOT_DIR}/results_eagle3_micro"
mkdir -p "${MICRO_DIR}"

# Default to HF repo ID for GPT-OSS-20B. MODEL_PATH can be overridden
# to an explicit local directory, in which case no HF download is run.
MODEL_PATH="${MODEL_PATH:-https://huggingface.co/openai/gpt-oss-20b}"

# Pin HF cache/root to the models directory so that both the HF CLI and
# Transformers resolve GPT-OSS artifacts consistently on this machine.
HF_ROOT="/workspace/aimo/models"
export HF_HOME="${HF_HOME:-${HF_ROOT}}"

resolve_hf_model_path() {
  local path="$1"

  # Strip HF URL prefixes down to repo IDs.
  if [[ "$path" == https://huggingface.co/* ]]; then
    path="${path#https://huggingface.co/}"
  elif [[ "$path" == hf://* ]]; then
    path="${path#hf://}"
  fi

  # If it is already a local directory, use as-is.
  if [ -d "$path" ]; then
    printf '%s\n' "$path"
    return
  fi

  # Treat non-existent paths that contain '/' as HF repo IDs.
  if [[ "$path" == */* ]]; then
    local name="${path##*/}"
    local local_dir="${HF_ROOT}/${name}"

    if [ ! -d "$local_dir" ] || [ ! -f "${local_dir}/config.json" ]; then
      mkdir -p "$local_dir"
      HF_HOME="$HF_ROOT" hf download "$path" \
        --local-dir "$local_dir"
    fi

    printf '%s\n' "$local_dir"
    return
  fi

  # Fallback: treat as literal local path.
  printf '%s\n' "$path"
}

MODEL_PATH="$(resolve_hf_model_path "$MODEL_PATH")"

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
