#!/usr/bin/env bash
# D1 gym rows from the C++ agent PROCESS (chronotope --serve; K4b): the referee in Python, the agent and the model in one C++ process.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueD1C: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
TAG=${1:-qwen35_0p8b}
say "C++ agent process ($TAG) on the D1 gym"
for fam in closing_window moving_target static_control; do
  case $fam in moving_target|static_control) SK=_balloon;; *) SK=;; esac
  for spec in cpp_${TAG}_detector cpp_${TAG}_late_bind cpp_${TAG}_frozen; do
    [ -f specs/d1/$spec$SK.json ] || continue
    [ -d runs/d1/$fam/$spec ] && { say "$fam/$spec exists"; continue; }
    say "$fam / $spec"
    $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec$SK.json --name $spec \
        --conditions live,replay,paused --allow-small --max-calls 400 --out runs/d1/$fam > runs/d1_${fam}_${spec}.log 2>&1 || say "$fam/$spec exited $?"
  done
done
say "done ($TAG)"
