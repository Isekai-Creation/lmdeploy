#!/usr/bin/env bash
set -euo pipefail

###############################################
# Config – tweak these if needed
###############################################

# CUDA toolkit to use
CUDA_HOME_DEFAULT="/usr/local/cuda-12.8"

# Python venv name for building lmdeploy + turbomind
VENV_NAME="venv_turbomind_build"

# CMake CUDA arch; if empty, auto-detect from nvidia-smi
TM_ARCH="${TM_ARCH:-}"

###############################################
# 0. Sanity checks
###############################################

if [[ ! -d "${CUDA_HOME_DEFAULT}" ]]; then
  echo "ERROR: Expected CUDA 12.8 toolkit at ${CUDA_HOME_DEFAULT} but it was not found."
  echo "       Install CUDA 12.8 first or adjust CUDA_HOME_DEFAULT in this script."
  exit 1
fi

if [[ ! -f "pyproject.toml" ]] || [[ ! -d "lmdeploy" ]]; then
  echo "ERROR: This script must be run from the root of the lmdeploy repo."
  echo "       (I expected pyproject.toml and the lmdeploy/ package directory here.)"
  exit 1
fi

###############################################
# 1. Environment for CUDA 12.8
###############################################

export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo ">>> Using CUDA_HOME=${CUDA_HOME}"
nvcc --version || { echo "ERROR: nvcc not found in PATH"; exit 1; }

###############################################
# 2. Detect GPU arch if TM_ARCH not set
###############################################

if [[ -z "${TM_ARCH}" ]]; then
  if command -v nvidia-smi &>/dev/null; then
    # e.g. '8.0' -> '80', '8.9' -> '89', '9.0' -> '90'
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ' | tr -d '.')
    if [[ -n "${CAP}" ]]; then
      TM_ARCH="${CAP}"
    else
      echo "WARN: Could not detect compute_cap; defaulting TM_ARCH=80"
      TM_ARCH="80"
    fi
  else
    echo "WARN: nvidia-smi not found; defaulting TM_ARCH=80"
    TM_ARCH="80"
  fi
fi

echo ">>> Building TurboMind for CMAKE_CUDA_ARCHITECTURES=${TM_ARCH}"

###############################################
# 3. Create / activate Python venv
###############################################

if [[ ! -d "${VENV_NAME}" ]]; then
  echo ">>> Creating venv: ${VENV_NAME}"
  python3 -m venv "${VENV_NAME}"
fi

echo ">>> Activating venv: ${VENV_NAME}"
# shellcheck source=/dev/null
source "${VENV_NAME}/bin/activate"

python -V
pip install --upgrade pip

###############################################
# 4. Install build dependencies
###############################################

echo ">>> Installing Python build dependencies..."
pip install -U 'cmake>=3.26' ninja build wheel

# CUDA-specific requirements (includes torch, etc.)
if [[ -f "requirements/requirements_cuda.txt" ]]; then
  pip install -r requirements/requirements_cuda.txt
else
  echo "WARN: requirements/requirements_cuda.txt not found; skipping."
fi

###############################################
# 5. Configure & build TurboMind with CMake
###############################################

BUILD_DIR="${PWD}/build_turbomind"
INSTALL_PREFIX="${PWD}/lmdeploy/lib"

echo ">>> Cleaning previous build dir: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

echo ">>> Configuring CMake project for TurboMind..."
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX:PATH="${INSTALL_PREFIX}" \
  -DPython3_ROOT_DIR="$(python -c 'import sys, pathlib; print(pathlib.Path(sys.executable).resolve().parent.parent)')" \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCALL_FROM_SETUP_PY:BOOL=ON \
  -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DBUILD_PY_FFI=ON \
  -DBUILD_MULTI_GPU=ON \
  -DUSE_NVTX=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${TM_ARCH}"

echo ">>> Building TurboMind (this may take a while)..."
cmake --build "${BUILD_DIR}" --config RelWithDebInfo

echo ">>> Installing TurboMind into ${INSTALL_PREFIX}..."
cmake --install "${BUILD_DIR}" --config RelWithDebInfo

###############################################
# 6. Quick import check
###############################################

echo ">>> Verifying lmdeploy + _turbomind import..."
python - << 'EOF'
import importlib
import lmdeploy
print("lmdeploy version:", lmdeploy.__version__)

try:
    import lmdeploy.turbomind as tm
    print("lmdeploy.turbomind imported OK")
except Exception as e:
    print("ERROR importing lmdeploy.turbomind:", e)
    raise

try:
    _tm = importlib.import_module("_turbomind")
    print("_turbomind C++ extension imported OK:", _tm)
except Exception as e:
    print("ERROR importing _turbomind:", e)
    raise
EOF

###############################################
# 7. Build wheel for Kaggle / distribution
###############################################

echo ">>> Building wheel..."
mkdir -p dist
pip wheel . -w dist

echo ">>> Done. Wheel(s) built in:"
realpath dist
ls -lh dist
