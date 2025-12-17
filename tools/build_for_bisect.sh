#!/usr/bin/env bash
set -e

# Assume we are in lmdeploy root or valid dir. 
# git bisect runs from the top level of the working tree.
# The user's cwd is /workspace/aimo/LM/lmdeploy.
cd /workspace/aimo/LM/lmdeploy

# Force usage of system python/conda python if available
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PATH}"

BUILD_DIR="build_turbomind"
INSTALL_PREFIX="lmdeploy/lib"

# Clean previous build
rm -rf "$BUILD_DIR"

# Configure TurboMind
# Prefer Ada/Hopper (sm89/sm90a) for broad compatibility. Blackwell
# (sm120) is only available when the CUDA toolkit supports it; using
# 120 unconditionally causes "Unsupported gpu architecture 'compute_120'"
# on CUDA 12.x toolchains.
echo "Configuring TurboMind build..."
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX:PATH="${INSTALL_PREFIX}" \
  -DPython3_ROOT_DIR="$(python -c 'import sys, pathlib; print(pathlib.Path(sys.executable).resolve().parent.parent)')" \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DCALL_FROM_SETUP_PY:BOOL=ON \
  -DBUILD_SHARED_LIBS:BOOL=OFF \
  -DBUILD_PY_FFI=ON \
  -DBUILD_MULTI_GPU=ON \
  -DUSE_NVTX=ON \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"

# Build
echo "Building TurboMind..."
cmake --build "$BUILD_DIR" --config RelWithDebInfo -j 32 

# Install
echo "Installing TurboMind..."
cmake --install "$BUILD_DIR" --config RelWithDebInfo
