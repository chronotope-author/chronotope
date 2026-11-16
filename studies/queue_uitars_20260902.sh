#!/usr/bin/env bash
# UI-TARS-1.5-7B in-process rows, third attempt (2026-09-02). The visual engine now exists: onnx/visual_fp16c is the
# fp16-weight export with 194 casts inserted by a type-repair pass (the original export traced with fp32 activations
# and elided every fp32 cast; TensorRT rejected Half x Float at the patch-embed conv and the rotary products). Built
# 2026-09-02 08:42 (visual_build_fp16c.log); a 10-episode static_control smoke cell (runs/d1_smoke) checked the
# engine answers with clicks on the target. Waits for queueFARAd done (anchored; the Fara continuation queue), then the 10-episode rows
# (studies/d1_edgellm_pilot.sh uitars7b) and the 100-episode rows on card/balloon.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueUITARSc: $*" | tee -a "$LOG"; }
m=$HOME/tensorrt-edgellm-workspace/UI-TARS-1.5-7B
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
until grep -q "\] queueFARAd: done$" "$LOG"; do sleep 120; done
while pgrep -f "studies/d1_distributional_gym.py" > /dev/null; do sleep 60; done
ls "$m/engines/visual" | grep -qi engine || { say "no visual engine; rows skipped"; exit 1; }
say "start"
bash studies/d1_edgellm_pilot.sh uitars7b > runs/d1_edgellm_pilot_uitars_c.log 2>&1
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
say "pilot rows done"
for fam in closing_window moving_target; do
  case $fam in moving_target) SK=_balloon;; *) SK=;; esac
  for suf in detector late_bind frozen; do spec=edgellm_uitars7b_${suf}
    [ -f runs/d1_100/$fam/$spec/outcomes.jsonl ] && [ "$(wc -l < runs/d1_100/$fam/$spec/outcomes.jsonl)" -ge 100 ] && { say "$fam/$spec (100) exists"; continue; }
    rm -rf runs/d1_100/$fam/$spec; say "$fam / $spec (100)"
    $D1 run --manifest runs/d1/manifests/${fam}_100.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay --allow-small --max-calls 20000 --out runs/d1_100/$fam > runs/d1_100_${fam}_${spec}.log 2>&1 || say "$fam/$spec (100) exited $?"
  done
done
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1 || say "master (100) failed"
say "done"
