#!/usr/bin/env bash
set -euo pipefail

###############################################
# ASan-ENABLED Build for TurboMind
# This script builds with AddressSanitizer to
# detect buffer overflows and heap corruption.
###############################################

# CUDA toolkit to use
CUDA_HOME_DEFAULT="/usr/local/cuda-12.8"

# CMake CUDA arch; if empty, auto-detect from nvidia-smi
TM_ARCH="${TM_ARCH:-90}"

###############################################
# Sanity checks
###############################################

if [[ ! -d "${CUDA_HOME_DEFAULT}" ]]; then
  echo "ERROR: Expected CUDA 12.8 toolkit at ${CUDA_HOME_DEFAULT}"
  exit 1
fi

if [[ ! -f "pyproject.toml" ]] || [[ ! -d "lmdeploy" ]]; then
  echo "ERROR: Run from lmdeploy repo root."
  exit 1
fi

###############################################
# Environment
###############################################

export CUDA_HOME="${CUDA_HOME_DEFAULT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo ">>> Using CUDA_HOME=${CUDA_HOME}"
echo ">>> Building with AddressSanitizer enabled"
echo ">>> Target arch: ${TM_ARCH}"

nvcc --version || { echo "ERROR: nvcc not found"; exit 1; }

###############################################
# Configure & build with ASan
###############################################

BUILD_DIR="${PWD}/build_asan"
INSTALL_PREFIX="${PWD}/lmdeploy/lib"

echo ">>> Cleaning ASan build dir: ${BUILD_DIR}"
rm -rf "${BUILD_DIR}"

# ASan flags for C++ and CUDA host code
ASAN_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"

echo ">>> Configuring CMake with ASan flags..."
cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX:PATH="${INSTALL_PREFIX}" \
  -DPython3_ROOT_DIR="$(python3 -c 'import sys, pathlib; print(pathlib.Path(sys.executable).resolve().parent.parent)')" \
  -DPYTHON_EXECUTABLE="$(which python3)" \
  -DCALL_FROM_SETUP_PY:BOOL=ON \
  -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DBUILD_PY_FFI=ON \
  -DBUILD_MULTI_GPU=ON \
  -DUSE_NVTX=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${TM_ARCH}" \
  -DCMAKE_CXX_FLAGS="${ASAN_FLAGS}" \
  -DCMAKE_C_FLAGS="${ASAN_FLAGS}" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=address" \
  -DCMAKE_CUDA_FLAGS="-Xcompiler '${ASAN_FLAGS}' -Xlinker -fsanitize=address"

echo ">>> Building TurboMind with ASan (this will take a while)..."
cmake --build "${BUILD_DIR}" --config Debug -j$(nproc)

echo ">>> Installing ASan build into ${INSTALL_PREFIX}..."
cmake --install "${BUILD_DIR}" --config Debug

echo ">>> ASan build complete!"
echo ""
echo "To run with ASan, use:"
echo "  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.8 python3 your_script.py"
echo ""
echo "Or set:"
echo "  export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:halt_on_error=1"
