#!/usr/bin/env bash
# R2 rerun (2026-08-25 night): the R5 R2 stage crashed at start (UnboundLocalError: local import of PayloadStore in
# r1_revealed_target.py main(), fixed). Waits for the 'queueR5 done' marker, then the study-process guard (which also
# excludes G1's g1_ run if it starts first), runs R2 + analyze. Gate on markers and study processes, NOT on queue bash
# pids (queue100's bash outlived its marker by 5 h in do_wait on its vLLM child). Markers "queueR2b ...".
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueR2b: waiting for 'queueR5 done'"
while ! grep -q "queueR5 done" "$LOG"; do sleep 15; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)_?[a-z_]*\.py" > /dev/null; do sleep 15; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
curl -s -m 3 "$BASE/models" | grep -q "\"$M\"" || { say "queueR2b FAIL: primary server not up"; exit 1; }
say "queueR2b: R2 fine window sweep (notification, Live only): windows 4.5,5.0,5.2,5.4,5.8,6.0 s x 100 seeds"
$PY studies/r1_revealed_target.py --model $M --base-url $BASE --no-thinking --skin notification --windows-s 4.5,5.0,5.2,5.4,5.8,6.0 --policies live \
    --seed-start 200 --n-seeds 100 --out runs/r2_window_sweep_100b --max-calls 900 > runs/r2_window_sweep_100b.log 2>&1 || say "queueR2b: R2 exited $?"
$PY studies/analyze_r2.py runs/r1_qwen38_notification_100 runs/r2_window_sweep_100b results/R2_window_sweep_100.md > /dev/null || say "queueR2b: R2 analyze exited $?"
say "queueR2b done"
