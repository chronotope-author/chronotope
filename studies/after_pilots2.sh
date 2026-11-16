#!/usr/bin/env bash
# Continuation of after_pilots.sh (its test gate failed on a test assertion, fixed by hand): the A1 static-skin
# control pilot (Q3) and the paired animation-cost table. Detached; markers "after_pilots2 ...".
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f "studies/(c1_|c2_|c3_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 30; done
PY="env PYTHONPATH=src .venv/bin/python"; BASE=http://127.0.0.1:8000/v1; M=Qwen/Qwen3.8-27B
say "after_pilots2: A1 static-skin control pilot (Q3): gate + 5 windows x 10 seeds x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --skin player_static --out runs/a1_static_qwen38_nothink_pilot10 --max-calls 600 > runs/a1_static_qwen38_nothink_pilot10.log 2>&1 || say "after_pilots2: A1 static exited $?"
$PY studies/compare_prl_runs.py runs/a1_qwen38_nothink_pilot10 runs/a1_static_qwen38_nothink_pilot10 results/A1_animation_cost_pilot10.md --labels animated,static --cell-order w1,w2,w4,w8,w16 --pilot > /dev/null || say "after_pilots2: compare exited $?"
say "after_pilots2 done"
