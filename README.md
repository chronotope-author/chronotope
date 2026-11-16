# Chronotope

Source code for the paper *One Step, Four Clocks: How GUI Agents Fall Out of Step*. 

## Folders

- `src/chronotope/`: Python task definitions, Paused/Replay/Live harness, evaluators, controller implementations, and prompt construction.
- `chronotope/`: C++/CUDA controller, native build, detector, and planner backends.
- `tests/`, `studies/`, `specs/`, and `configs/`: source code and task specifications, including test contracts, experiment launchers, prompts, and example controller settings.
- `prompts/archived/`: **14 serialized prompt requests/messages** available in the original archive. All code-defined prompts are retained in source (notably `chronotope/src/agents3_prompt.h` and the relevant Python modules).

## Build and testing

From this directory, install the Python package and run a focused test set:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_stats.py tests/test_contracts.py tests/test_latest_frame.py tests/test_late_binding.py
```

The native controller uses CUDA, CMake, and optional TensorRT dependencies:

```bash
cmake -S chronotope -B chronotope/build -DCMAKE_BUILD_TYPE=Release
cmake --build chronotope/build -j
```



