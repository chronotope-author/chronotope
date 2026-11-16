#!/usr/bin/env bash
# Waits for the C++-agent rows (queueD1C) to finish, then renames chronotope -> chronotope in place, rebuilds, tests, regenerates the gym reports.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueK4C: $*" | tee -a "$LOG"; }
until grep -q "queueD1C: done" "$LOG"; do sleep 60; done
while pgrep -f "[d]1_distributional_gym.py run|[k]gua --serve" > /dev/null; do sleep 30; done
say "rename chronotope -> chronotope starting"
bash studies/k4c_rename.sh > runs/k4c_rename.log 2>&1 && say "rename + build + tests: $(tail -1 runs/k4c_rename.log | cut -c1-120)" || say "rename FAILED: $(tail -3 runs/k4c_rename.log | tr '\n' ' ' | cut -c1-200)"
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
for fam in closing_window moving_target static_control; do runs=$(for d in runs/d1/$fam/*/; do d=${d%/}; case $(basename $d) in payloads|*delib1000*|*delib5200*) continue;; esac; [ "$(wc -l < $d/outcomes.jsonl 2>/dev/null)" = "30" ] && echo -n "$d "; done); $PY studies/d1_distributional_gym.py report --family $fam --runs $runs --baseline chronotope_late_bind_delib0 --manifest runs/d1/manifests/$fam.jsonl --extra-law 1,64 --out results/D1_${fam}_pilot.md > runs/d1_report_$fam.log 2>&1 || say "report $fam failed"; done
$PY studies/d1_distributional_gym.py master --out results/D1_master.md > /dev/null 2>&1 || say "master failed"
say "done"
