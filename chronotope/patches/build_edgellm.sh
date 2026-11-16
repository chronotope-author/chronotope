#!/usr/bin/env bash
# TensorRT Edge-LLM 0.10.0 C++ runtime for DGX Spark (GB10, CUDA 13.0), without root:
# TensorRT 10.16.1 (+cuda13.2, the release-tested version) is taken from the CUDA apt repo's debs and
# extracted into ~/trt-prefix; Edge-LLM is configured with -DTRT_PACKAGE_DIR pointing there.
set -u
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
mkdir -p ~/trt-debs/trt10-full ~/trt-prefix
cd ~/trt-debs/trt10-full
V=10.16.1.11-1+cuda13.2
if [ ! -f "libnvinfer-headers-dev_${V}_arm64.deb" ]; then
  log "downloading TensorRT ${V} debs"
  apt-get download libnvinfer10=$V libnvinfer-dev=$V libnvinfer-headers-dev=$V libnvinfer-plugin10=$V \
      libnvinfer-plugin-dev=$V libnvinfer-headers-plugin-dev=$V libnvonnxparsers10=$V libnvonnxparsers-dev=$V \
      libnvinfer-lean10=$V libnvinfer-dispatch10=$V libnvinfer-bin=$V || { log "download failed"; exit 1; }
fi
ls -la *.deb | awk '{print $5, $9}'
for d in *.deb; do dpkg-deb -x "$d" ~/trt-prefix || { log "extract failed: $d"; exit 1; }; done
log "extracted:"; ls ~/trt-prefix/usr/include/aarch64-linux-gnu/NvInfer.h ~/trt-prefix/usr/lib/aarch64-linux-gnu/libnvinfer.so* | head -4
cd ~/TensorRT-Edge-LLM
# chronotope/patches/edgellm-0.10.0-device-image-source.patch: the vision runner accepts a DEVICE-resident frame (cudaMemcpyDefault)
if ! grep -q "cudaMemcpyDefault" cpp/kernels/preprocessKernels/imageUtilKernels.cu; then
  git apply ~/chronotope/chronotope/patches/edgellm-0.10.0-device-image-source.patch && log "applied the chronotope device-image patch" || { log "patch failed"; exit 1; }
fi
mkdir -p build && cd build
log "configuring"
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR="$HOME/trt-prefix/usr" \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=gb10 -DCUDA_CTK_VERSION=13.0 -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_CUDA_FLAGS=-Xcompiler=-march=native \
  -DENABLE_CUTE_DSL=ALL 2>&1 | tail -25
log "building (make -j16)"
make -j16 2>&1 | grep -E "error|Error|warning: .*deprecated|Built target|\[100%\]" | tail -40
rc=${PIPESTATUS[0]}
log "make rc=$rc"
ls -la examples/llm/llm_build examples/llm/llm_inference examples/multimodal/visual_build 2>&1 | awk '{print $5, $9}'
log "done"
