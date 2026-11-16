#!/usr/bin/env bash
# G2 queue (2026-08-26; cards/G2-live-page.md). Waits for the 'queueM3b done' marker and for no study process, checks that
# the primary 27B server is up (M3b restores it), then runs the live Chromium page cells at 100 seeds (step 1: frozen /
# late_bind / restart_if_moved with Replay; two-step: frozen / late_bind / chronotope) and the analysis. Markers "queueG2 ...".
# Gate on markers and study processes, NOT on queue bash pids.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueG2: waiting for M3d"
while ! grep -q "\] queueM3d done$" "$LOG"; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 60; done
MODEL=Qwen/Qwen3.8-27B; BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
for i in $(seq 1 60); do curl -s -m 3 "$BASE/models" | grep -q "\"$MODEL\"" && break; sleep 10; done
if ! curl -s -m 3 "$BASE/models" | grep -q "\"$MODEL\""; then say "queueG2 FAIL: $MODEL is not being served at $BASE"; exit 1; fi
TAG=g2_live_qwen38_nothink_100
say "queueG2: $TAG: step 1 (3 policies, Live + Replay) and two-step (3 policies) x 100 seeds, prompt 12/13, 15 Hz screenshots"
$PY studies/g2_live_page.py --model "$MODEL" --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 100 \
    --out "runs/$TAG" --max-calls 2400 > "runs/$TAG.log" 2>&1 || say "queueG2: $TAG exited $?"
$PY studies/analyze_g2.py "runs/$TAG" results/G2_live_page_100.md > /dev/null 2>&1 || say "queueG2: analyze exited $?"
say "queueG2 done"
