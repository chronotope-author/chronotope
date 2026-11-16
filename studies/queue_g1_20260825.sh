#!/usr/bin/env bash
# G1 queue (2026-08-25): waits for 'queueR5 done', then runs the reflow-page study on the 27B server. Markers "queueG1 …".
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueG1: waiting for 'queueR5 done'"
while ! grep -q "queueR5 done" "$LOG" || pgrep -f '^bash studies/queue_r5_20260825.sh$' > /dev/null; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
curl -s -m 3 "$BASE/models" | grep -q "\"$M\"" || { say "queueG1 FAIL: 27B server not up"; exit 1; }
say "queueG1: G1 reflow page (browser rendering): gate + 5 cells x 100 seeds"
$PY studies/g1_reflow.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/g1_reflow_qwen38_nothink_100 --max-calls 1200 > runs/g1_reflow_qwen38_nothink_100.log 2>&1 || say "queueG1: G1 exited $?"
$PY studies/analyze_g1.py runs/g1_reflow_qwen38_nothink_100 results/G1_reflow_qwen38_nothink_100.md > /dev/null || say "queueG1: analyze exited $?"
say "queueG1 done"
