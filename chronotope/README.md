# Native Chronotope controller

The `chronotope/` directory contains the C++/CUDA control implementation. The source defines the observation and control-state records, on-device binding and commitment logic, capture and action interfaces, and multiple planner backends.

## Build

A CUDA-capable NVIDIA GPU and compatible toolkit are required for native tests. CMake is configured for C++20 and CUDA. Additional TensorRT dependencies are needed for the in-process VLM backend.

```bash
cmake -S chronotope -B chronotope/build -DCMAKE_BUILD_TYPE=Release
cmake --build chronotope/build -j
```

Available targets and optional backend switches are defined in `CMakeLists.txt`. The selected architecture and TensorRT installation must match the host machine.

## Running

A synthetic moving-target case can be run without external capture or input hardware:

```bash
chronotope/build/chronotope --source synthetic:v=40 --planner detector \
  --deliberation-ms 500 --policy late_bind --hid fake --trace /tmp/chronotope-trace.jsonl
```

External camera and input-device modes require their respective drivers and permissions. Only use them in an environment in which actions are authorized.

## Planner backends and measurement

- `detector`: resident color-based control for controlled experiments.
- `vlm`: a planner reached through a local OpenAI-compatible endpoint. Image encoding and inference include host work.
- `trt`: in-process TensorRT Edge-LLM, subject to the configured runtime and its host synchronization.

These are different execution paths. Do not infer whole-system accelerator residency from the use of CUDA for binding or commitment. Inspect the runtime audit counters in `core.cu` and accompanying traces for the measured path. The Python integration and tests live under `python/`.
