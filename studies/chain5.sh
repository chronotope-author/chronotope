#!/usr/bin/env bash
# 100-seed principal cells (user rule 2026-08-23). Sequential on the GPU. Qwen3.8-27B served on :8000.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
BASE=http://127.0.0.1:8000/v1
PY="env PYTHONPATH=src .venv/bin/python"
M=Qwen/Qwen3.8-27B
for i in $(seq 1 120); do curl -s -m 3 "$BASE/models" | grep -q Qwen3.8-27B && break; sleep 10; done; say "chain5: server ready"
say "chain5: C1 balloon drift, 100 seeds (27B, thinking off)"
$PY studies/c1_balloon_drift.py --model $M --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 100 --calib-seed-start 2000 --n-calib 100 --out runs/c1_qwen38_nothink_100 --max-calls 3000 > runs/c1_qwen38_nothink_100.log 2>&1 || say "c1 exited $?"
say "chain5: C2 balloon jump, 100 seeds (27B)"
$PY studies/c2_balloon_jump.py --agent --model $M --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 100 --out runs/c2_qwen38_nothink_100 --max-calls 1500 > runs/c2_qwen38_nothink_100.log 2>&1 || say "c2 exited $?"
say "chain5: C3 balloon select, 100 seeds (27B) — waits for runs/C3_READY"
while [ ! -f runs/C3_READY ]; do sleep 30; done
$PY studies/c3_balloon_select.py --model $M --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 100 --calib-seed-start 2000 --n-calib 100 --out runs/c3_qwen38_nothink_100 --max-calls 2500 > runs/c3_qwen38_nothink_100.log 2>&1 || say "c3 exited $?"
say "chain5: P1 pending button, 100 seeds (27B)"
$PY studies/p1_pending_button.py --model $M --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 100 --out runs/p1_qwen38_nothink_100 --max-calls 8000 > runs/p1_qwen38_nothink_100.log 2>&1 || say "p1 exited $?"
say "chain5: R1 revealed target (notification, prompt 4), 100 seeds (27B)"
$PY studies/r1_revealed_target.py --model $M --base-url $BASE --no-thinking --skin notification --prompt-version 4 --out runs/r1_qwen38_notification_100 --max-calls 3000 --n-seeds 100 --seed-start 1000 > runs/r1_qwen38_notification_100.log 2>&1 || say "r1 exited $?"
say "chain5 done"
