#!/usr/bin/env bash
# R2 rerun #2 (2026-08-26): the R2b stage failed its Paused gate (0/10) because the invocation omitted
# --prompt-version 4, which the cited R1 run (chain5.sh) used; the notification skin's default prompt is 2.
# Waits for the 'queueG1 done' marker (5-s polls so it precedes M3b), then the study-process guard, then R2 + analyze.
# Gate on markers and study processes, NOT on queue bash pids. Markers "queueR2c ...".
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueR2c: waiting for 'queueG1 done'"
while ! grep -q "queueG1 done" "$LOG"; do sleep 5; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)[a-z_]*\.py" > /dev/null; do sleep 5; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
curl -s -m 3 "$BASE/models" | grep -q "\"$M\"" || { say "queueR2c FAIL: primary server not up"; exit 1; }
say "queueR2c: R2 fine window sweep (notification, prompt 4, Live only): windows 4.5,5.0,5.2,5.4,5.8,6.0 s x 100 seeds"
$PY studies/r1_revealed_target.py --model $M --base-url $BASE --no-thinking --skin notification --prompt-version 4 \
    --windows-s 4.5,5.0,5.2,5.4,5.8,6.0 --policies live --seed-start 200 --n-seeds 100 \
    --out runs/r2_window_sweep_100c --max-calls 900 > runs/r2_window_sweep_100c.log 2>&1 || say "queueR2c: R2 exited $?"
$PY studies/analyze_r2.py runs/r1_qwen38_notification_100 runs/r2_window_sweep_100c results/R2_window_sweep_100.md > /dev/null || say "queueR2c: R2 analyze exited $?"
say "queueR2c done"
