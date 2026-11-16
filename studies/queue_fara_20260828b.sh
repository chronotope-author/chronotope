#!/usr/bin/env bash
# Fara1.5 rows, second attempt (2026-08-28b). The first (fara_rows_20260828.sh, 00:55) failed on every cell: the 10-episode
# runs with 'undefined symbol: chronotope_planner_trt_open' and the 100-episode runs without --allow-small. Waits for
# 'queue100b: done'. Ten episodes per cell on card/balloon/static (live,replay,paused), then 100 (live,replay).
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueFARAb: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
until grep -q "queue100b: done" "$LOG"; do sleep 120; done
while pgrep -f "studies/d1_distributional_gym.py" > /dev/null; do sleep 60; done
say "start"
for tag in fara9b fara4b; do
  for fam in closing_window moving_target static_control; do
    case $fam in moving_target|static_control) SK=_balloon;; *) SK=;; esac
    for suf in detector late_bind frozen self_grounder; do spec=edgellm_${tag}_${suf}
      [ -f runs/d1/$fam/$spec/outcomes.jsonl ] && { say "$fam/$spec exists"; continue; }
      rm -rf runs/d1/$fam/$spec; say "$fam / $spec (10)"
      $D1 run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay,paused --allow-small --max-calls 2000 --out runs/d1/$fam > runs/d1_${fam}_${spec}.log 2>&1 || say "$fam/$spec exited $?"
    done
  done
done
for tag in fara9b fara4b; do
  for fam in closing_window moving_target; do
    case $fam in moving_target) SK=_balloon;; *) SK=;; esac
    for suf in detector late_bind frozen; do spec=edgellm_${tag}_${suf}
      [ -f runs/d1_100/$fam/$spec/outcomes.jsonl ] && [ "$(wc -l < runs/d1_100/$fam/$spec/outcomes.jsonl)" -ge 100 ] && { say "$fam/$spec (100) exists"; continue; }
      rm -rf runs/d1_100/$fam/$spec; say "$fam / $spec (100)"
      $D1 run --manifest runs/d1/manifests/${fam}_100.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay --allow-small --max-calls 20000 --out runs/d1_100/$fam > runs/d1_100_${fam}_${spec}.log 2>&1 || say "$fam/$spec (100) exited $?"
    done
  done
done
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1 || say "master (100) failed"
say "done"
