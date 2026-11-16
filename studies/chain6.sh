#!/usr/bin/env bash
# C1b: the restart regime — speeds 1,2,4 px/s x restart thresholds 6,12 px (+ frozen at 1,2 px/s), 100 seeds.
# Prediction: restart succeeds exactly where frozen also succeeds (v*T <= tau <= w/2) and fails where frozen fails.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while ! grep -q "chain5 done" runs/pipeline.log; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "chain6: C1b restart regime sweep (27B), 100 seeds"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2,4 --policies frozen,restart_if_moved --restart-threshold 6 --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1b_qwen38_tau6_100 --max-calls 2000 > runs/c1b_qwen38_tau6_100.log 2>&1 || say "c1b tau6 exited $?"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2,4 --policies restart_if_moved --restart-threshold 12 --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1b_qwen38_tau12_100 --max-calls 2000 > runs/c1b_qwen38_tau12_100.log 2>&1 || say "c1b tau12 exited $?"
say "chain6 done"
