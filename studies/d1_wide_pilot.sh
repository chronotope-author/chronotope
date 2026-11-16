#!/usr/bin/env bash
# closing_window with the deadline drawn log-uniform 1–64 s (covers the slowest agents): resident + in-process rows.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueD1W: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
run() { [ -d runs/d1/closing_window_wide/$1 ] && { say "$1 exists"; return; }; say "closing_window_wide / $1"; $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/closing_window_wide.jsonl --controller-json specs/d1/$1.json --name $1 --conditions live,replay,paused --allow-small --max-calls 400 --out runs/d1/closing_window_wide > runs/d1_closing_window_wide_$1.log 2>&1 || say "$1 exited $?"; }
for s in "$@"; do run $s; done
say "done ($*)"
