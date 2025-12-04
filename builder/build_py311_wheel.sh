#!/usr/bin/env bash
#
# Build a TurboMind-enabled lmdeploy wheel using Python 3.11.
# - Non-interactive (no prompts)
# - Tries to install python3.11 via apt if not already present
# - Builds a local venv, installs deps, and runs `python -m build --wheel`
#
# Usage (from lmdeploy repo root):
#   ./builder/build_py311_wheel.sh
#
# Optional environment variables:
#   PYTHON_BIN   - path to python3.11 (default: auto-detect / install)
#   VENV_DIR     - venv directory name (default: .venv_py311)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-.venv_py311}"

detect_or_install_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      echo "ERROR: PYTHON_BIN='${PYTHON_BIN}' not found in PATH" >&2
      exit 1
    fi
    return
  fi

  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.11)"
    export PYTHON_BIN
    return
  fi

  # Try to install python3.11 via apt (non-interactive, quiet)
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
    PYTHON_BIN="$(command -v python3.11)"
    export PYTHON_BIN
    return
  fi

  echo "ERROR: python3.11 not found and apt-get is unavailable." >&2
  echo "Please install Python 3.11 and set PYTHON_BIN before running this script." >&2
  exit 1
}

echo "[build_py311_wheel] Using lmdeploy source at: ${ROOT_DIR}"

detect_or_install_python
echo "[build_py311_wheel] Using Python: ${PYTHON_BIN}"

echo "[build_py311_wheel] Creating venv: ${VENV_DIR}"
rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "[build_py311_wheel] Upgrading pip + build deps (quiet)..."
"${VENV_PY}" -m pip install --upgrade --quiet pip
"${VENV_PIP}" install --quiet build wheel cmake_build_extension

echo "[build_py311_wheel] Installing lmdeploy build/runtime deps for CUDA..."
"${VENV_PIP}" install --quiet -r requirements/build.txt -r requirements/runtime_cuda.txt

echo "[build_py311_wheel] Building lmdeploy wheel with TurboMind..."
LMDEPLOY_TARGET_DEVICE=cuda "${VENV_PY}" -m build --wheel --quiet

echo "[build_py311_wheel] Done. Wheels are in: ${ROOT_DIR}/dist"

