#!/usr/bin/env bash
# chain8 v2 (maintained by chronotope-83): after chain5, grant the wi010 slot (chronotope-a3's ~15-min
# P1-reschedule pilot), wait for it, then run C1b (restart regime) and late_bind v=1,2 — 100 seeds.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while ! grep -q "chain5 done" runs/pipeline.log; do sleep 120; done
grep -q "wi010 go" runs/pipeline.log || say "wi010 go"
say "chain8: holding for wi010 (chronotope-a3's slot; proceeds on 'wi010 done', or after 45 min ONLY once no wi010 process is on the GPU)"
for i in $(seq 1 45); do grep -q "wi010 done" runs/pipeline.log && break; sleep 60; done
# race guard (per chronotope-79): even after the 45-min cap, never start while a3's pilot is still running
while ! grep -q "wi010 done" runs/pipeline.log && pgrep -f "studies/p1_calendar\.py|studies/wi010_" > /dev/null; do sleep 60; done
grep -q "wi010 done" runs/pipeline.log || say "chain8: no 'wi010 done' after cap and no wi010 process running; proceeding"
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "chain8: C1b restart regime, tau=6 (v=1,2,4; frozen+restart; 100 seeds)"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2,4 --policies frozen,restart_if_moved --restart-threshold 6 --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1b_qwen38_tau6_100 --max-calls 2500 > runs/c1b_qwen38_tau6_100.log 2>&1 || say "c1b tau6 exited $?"
say "chain8: C1b restart regime, tau=12 (v=1,2,4; restart; 100 seeds)"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2,4 --policies restart_if_moved --restart-threshold 12 --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1b_qwen38_tau12_100 --max-calls 2500 > runs/c1b_qwen38_tau12_100.log 2>&1 || say "c1b tau12 exited $?"
say "chain8: C1 late_bind v=1,2 (100 seeds)"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --skip-calibration --velocities 1,2 --policies late_bind --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1_qwen38_lb_v12_100 --max-calls 700 > runs/c1_qwen38_lb_v12_100.log 2>&1 || say "chain8 lb exited $?"
say "chain8 done"
