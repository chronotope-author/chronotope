#!/usr/bin/env bash
# Second stage: the Python binding (_edgellm_runtime) for experimental.server (the OpenAI-compatible server the
# chronotope Python studies can point at). Reconfigures the existing build dir without changing its platform settings.
set -u
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
cd ~/TensorRT-Edge-LLM || exit 1
PYBIND_DIR=$(~/edgellm-venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())') || { log "pybind11 missing"; exit 1; }
log "configuring bindings (pybind11 at $PYBIND_DIR)"
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON -Dpybind11_DIR="$PYBIND_DIR" -DPython3_EXECUTABLE="$HOME/edgellm-venv/bin/python" 2>&1 | tail -5
log "building _edgellm_runtime"
cmake --build build --target _edgellm_runtime -j16 2>&1 | grep -E "error|Built target|\[100%\]" | tail -10
rc=${PIPESTATUS[0]}
log "rc=$rc"
find build -name "_edgellm_runtime*.so" | head -3
log "done"
