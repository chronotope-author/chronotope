#!/usr/bin/env bash
# SUPERSEDED 2026-08-23 by chain8.sh (runs the same late_bind v=1,2 study, same --out dir). Do NOT re-slot or launch; nothing writes "chain6 done" anymore. — chronotope-79
# Fill the paper's TBD grid: late_bind at v in {1,2} (100 seeds) to complete speeds {0,1,2,4,8,16}.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while ! grep -q "chain6 done" runs/pipeline.log; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "chain7: C1 late_bind at v=1,2 (100 seeds)"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2 --policies late_bind --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1_qwen38_lb_v12_100 --max-calls 600 > runs/c1_qwen38_lb_v12_100.log 2>&1 || say "chain7 c1 exited $?"
say "chain7 done"
