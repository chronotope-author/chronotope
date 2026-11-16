#!/usr/bin/env bash
# CTRL queue (2026-08-26; cards V1-action-kind-validation, A8-preparation, O3-auto-diagnosis; branch build/v1-a8-o3).
# Waits for the 'queueM3b done' marker (60-s polls) and the study-process guard, checks that the primary server serves
# Qwen/Qwen3.8-27B, then runs the V1 model cells, the A8 model cells and O3 part 2, each followed by its analyze script;
# 100 seeds, --no-thinking, prompt versions explicit where a study takes one. Markers "queueCTRL ..." tee'd to
# runs/pipeline.log. Gate on markers and study processes, NOT on queue bash pids. Do not launch by hand before the
# branch is merged into ~/chronotope (paths are relative to ~/chronotope).
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueCTRL: waiting for M3d"
while ! grep -q "\] queueM3d done$" "$LOG"; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
curl -s -m 3 "$BASE/models" | grep -q "\"$M\"" || { say "queueCTRL FAIL: $M is not served at $BASE"; exit 1; }

say "queueCTRL: V1 model cells (player windows 1,8,16 s validate arm P/R/L; rotating code kind_change x 3; controls) x 100 seeds"
$PY studies/v1_kind_validation.py --model $M --base-url $BASE --no-thinking --seed-start 200 --n-seeds 100 \
    --out runs/v1_qwen38_nothink_100 --max-calls 3200 > runs/v1_qwen38_nothink_100.log 2>&1 || say "queueCTRL: V1 exited $?"
$PY studies/analyze_v1.py runs/v1_qwen38_nothink_100 results/V1_qwen38_nothink_100.md --recorded runs/a1_qwen38_nothink_100 \
    > /dev/null 2>&1 || say "queueCTRL: V1 analyze exited $?"
say "queueCTRL: V1 done"

say "queueCTRL: A8 model cells (fixed-layout notification w1,w2 prepared vs direct; seeded w8 prepared vs late_bind; E 8 s) x 100 seeds"
$PY studies/a8_preparation.py --model $M --base-url $BASE --no-thinking --seed-start 200 --n-seeds 100 \
    --out runs/a8_qwen38_nothink_100 --max-calls 900 > runs/a8_qwen38_nothink_100.log 2>&1 || say "queueCTRL: A8 exited $?"
$PY studies/analyze_a8.py results/A8_preparation_100.md --scripted results/A8_preparation_scripted_100.md \
    --model-run runs/a8_qwen38_nothink_100 --v1-run runs/v1_qwen38_nothink_100 > /dev/null 2>&1 || say "queueCTRL: A8 analyze exited $?"
say "queueCTRL: A8 done"

say "queueCTRL: O3 part 2 (reruns of the planned failing seeds under the selected and one wrong intervention)"
$PY studies/o3_auto_diagnosis.py --rerun --plan-json results/O3_plan.json --model $M --base-url $BASE --no-thinking \
    --out runs/o3_qwen38_nothink_100 --max-calls 3000 > runs/o3_qwen38_nothink_100.log 2>&1 || say "queueCTRL: O3 exited $?"
$PY studies/analyze_o3.py runs/o3_qwen38_nothink_100 results/O3_auto_diagnosis_100.md > /dev/null 2>&1 || say "queueCTRL: O3 analyze exited $?"
say "queueCTRL: O3 done"
say "queueCTRL done"
