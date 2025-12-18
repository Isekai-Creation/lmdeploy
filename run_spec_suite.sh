#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MICRO_DIR="${ROOT_DIR}/results_eagle3_micro"
mkdir -p "${MICRO_DIR}"

# Default to HF repo ID for GPT-OSS-20B. MODEL_PATH can be overridden
# either to another HF repo ID/URL or to an explicit local directory.
HF_DEFAULT_REPO="openai/gpt-oss-20b"
MODEL_PATH="${MODEL_PATH:-${HF_DEFAULT_REPO}}"

# Pin HF cache/root to the models directory so that both the HF CLI and
# Transformers resolve GPT-OSS artifacts consistently on this machine.
HF_ROOT="/workspace/aimo/models"
export HF_HOME="${HF_HOME:-${HF_ROOT}}"

# Auto-select TM_CACHE_MAX_ENTRY_COUNT for GPT-OSS runs based on
# current device memory usage when the user has not set it
# explicitly. Policy:
#   - If GPU memory used <= ~10GB (GPU mostly idle): use 0.75.
#   - If GPU memory used is around or below ~45GB: use 0.50.
#   - If GPU memory used > ~45GB: treat GPU as occupied and exit so
#     we do not oversubscribe KV cache.
choose_tm_cache_fraction() {
  # Respect explicit user choice if already set.
  if [[ -n "${TM_CACHE_MAX_ENTRY_COUNT:-}" ]]; then
    return
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    TM_CACHE_MAX_ENTRY_COUNT="0.75"
    export TM_CACHE_MAX_ENTRY_COUNT
    return
  fi

  local line total used used_gb
  line="$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -n1)"
  total="$(echo "$line" | awk -F',' '{print $1}' | xargs)"
  used="$(echo "$line" | awk -F',' '{print $2}' | xargs)"
  used_gb=$(( used / 1024 ))

  if (( used_gb <= 10 )); then
    TM_CACHE_MAX_ENTRY_COUNT="0.75"
  elif (( used_gb <= 45 )); then
    TM_CACHE_MAX_ENTRY_COUNT="0.50"
  else
    if [[ "${TM_DRIFT_IGNORE_GPU_OCCUPIED:-0}" != "1" ]]; then
      echo "[run_spec_suite] GPU memory appears occupied (used=${used_gb}GB > 45GB); refusing to auto-run GPT-OSS benchmarks." >&2
      echo "[run_spec_suite] Free GPU memory or export TM_CACHE_MAX_ENTRY_COUNT manually to override this guard," >&2
      echo "[run_spec_suite] or set TM_DRIFT_IGNORE_GPU_OCCUPIED=1 to bypass this guard at your own risk." >&2
      exit 1
    fi
    echo "[run_spec_suite] GPU memory appears occupied (used=${used_gb}GB > 45GB); proceeding due to TM_DRIFT_IGNORE_GPU_OCCUPIED=1 override." >&2
    # When heavily occupied, fall back to a conservative cache fraction.
    TM_CACHE_MAX_ENTRY_COUNT="0.50"
  fi

  export TM_CACHE_MAX_ENTRY_COUNT
}

# Auto-select KV cache fraction if not explicitly set.
choose_tm_cache_fraction

# Enable DriftEngine KV canary checks by default for drift spec runs so
# that any KV pointer/layout mismatch is caught before attention
# kernels run. Users can override/disable by explicitly setting
# TM_DRIFT_KV_CANARY.
if [[ -z "${TM_DRIFT_KV_CANARY:-}" ]]; then
  export TM_DRIFT_KV_CANARY=1
fi

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
    local repo="$path"
    local name="${repo##*/}"
    local local_dir="${HF_ROOT}/${name}"

    if [ ! -d "$local_dir" ] || [ ! -f "${local_dir}/config.json" ]; then
      mkdir -p "$local_dir"
      if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "[run_spec_suite] huggingface-cli not found; install via 'pip install --upgrade huggingface_hub'" >&2
        exit 1
      fi
      HF_HOME="$HF_ROOT" huggingface-cli download "$repo" \
        --local-dir "$local_dir" \
        --local-dir-use-symlinks False
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
