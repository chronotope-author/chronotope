#!/usr/bin/env bash
# UI-TARS-1.5-7B in-process rows, second attempt (2026-08-28b). The 2026-08-27 rows ran without a visual engine (the
# visual ONNX had float32 weights; TensorRT refused Half x Float). The ONNX under onnx/visual_fp16 has FP16 linear/conv
# weights and an FP16 output (the shape of the working 27B export). Waits for 'queueFARAb: done', builds the visual
# engine, discards the engine-less rows, then runs the 10-episode rows and the 100-episode rows.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueUITARSb: $*" | tee -a "$LOG"; }
WS=$HOME/tensorrt-edgellm-workspace; EL=$HOME/TensorRT-Edge-LLM; m=$WS/UI-TARS-1.5-7B
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
until grep -q "queueFARAb: done" "$LOG"; do sleep 120; done
while pgrep -f "studies/d1_distributional_gym.py" > /dev/null; do sleep 60; done
say "visual_build from onnx/visual_fp16"
$EL/build/examples/multimodal/visual_build --onnxDir "$m/onnx/visual_fp16" --engineDir "$m/engines" --minImageTokens 64 --maxImageTokens 1024 --maxImageTokensPerImage 1024 > "$m/visual_build_fp16.log" 2>&1; rc=$?; say "visual_build rc=$rc"
ls "$m/engines" | grep -qi visual || { say "no visual engine; rows skipped"; exit 1; }
for fam in closing_window moving_target static_control; do rm -rf runs/d1/$fam/edgellm_uitars7b_*; done   # engine-less 2026-08-27 rows
bash studies/d1_edgellm_pilot.sh uitars7b > runs/d1_edgellm_pilot_uitars_b.log 2>&1
for fam in closing_window moving_target; do
  case $fam in moving_target) SK=_balloon;; *) SK=;; esac
  for suf in detector late_bind frozen; do spec=edgellm_uitars7b_${suf}
    rm -rf runs/d1_100/$fam/$spec; say "$fam / $spec (100)"
    $D1 run --manifest runs/d1/manifests/${fam}_100.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay --allow-small --max-calls 20000 --out runs/d1_100/$fam > runs/d1_100_${fam}_${spec}.log 2>&1 || say "$fam/$spec (100) exited $?"
  done
done
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1 || say "master (100) failed"
say "done"
