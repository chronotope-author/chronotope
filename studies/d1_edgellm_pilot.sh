#!/usr/bin/env bash
# D1 gym, the chronotope controller with the in-process TensorRT Edge-LLM planner (no vLLM). $1 = engines name under
# ~/tensorrt-edgellm-workspace (default Qwen3.5-0.8B); specs/d1/edgellm_<tag>_*.json must exist for it.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
TAG=${1:-qwen35_0p8b}
say "queueD1E: chronotope + Edge-LLM ($TAG) on the D1 gym (closing_window, moving_target, static_control)"
for fam in closing_window moving_target static_control; do
  case $fam in moving_target|static_control) SK=_balloon;; *) SK=;; esac   # the balloon skin needs the balloon prompt (5 / the UI-TARS balloon instruction)
  for spec in edgellm_${TAG}_late_bind edgellm_${TAG}_frozen edgellm_${TAG}_self_grounder edgellm_${TAG}_detector edgellm_${TAG}_ncc_then_detector; do
    [ -f specs/d1/$spec$SK.json ] || continue
    [ -d runs/d1/$fam/$spec ] && { say "queueD1E: $fam/$spec exists, skipped"; continue; }
    $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec$SK.json \
        --name $spec --conditions live,replay,paused --allow-small --max-calls 400 --out runs/d1/$fam > runs/d1_${fam}_${spec}.log 2>&1 \
        || say "queueD1E: $fam/$spec exited $?"
  done
  say "queueD1E: $fam done"
done
say "queueD1E done"
