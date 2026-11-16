#!/usr/bin/env bash
# L1 queue (2026-08-26; cards/L1-normalized-law.md). Waits for the 'queueM3b done' marker, then for no study
# process, checks that the primary 27B is being served, runs the C1-size sweep (r in {6,12,24} px x v in
# {2,4,8,16,32} px/s x {frozen, late_bind}, 10 seeds, Live), then re-assembles the trial table and re-runs the
# L1 analysis (M3 rows included as they exist). Markers "queueL1 ..." in runs/pipeline.log.
# Gate on markers and study processes, NOT on queue bash pids (anchored pgrep; never kill by substring).
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueL1: waiting for M3d"
while ! grep -q "\] queueM3d done$" "$LOG"; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
MODEL=Qwen/Qwen3.8-27B
if ! curl -s -m 5 "$BASE/models" | grep -q "\"$MODEL\""; then
  say "queueL1 FAIL: $MODEL is not served at $BASE; not starting the sweep"; exit 1
fi
OUT=runs/c1_size_qwen38_nothink_10
if [ -d "$OUT" ]; then say "queueL1 FAIL: $OUT exists (traces are immutable; pick a fresh name)"; exit 1; fi
say "queueL1: C1-size sweep start ($MODEL, 3 x 5 x 2 cells x 10 seeds + gate)"
$PY studies/c1_size_sweep.py --model "$MODEL" --base-url $BASE --no-thinking --seed-start 1000 --n-seeds 10 \
    --out "$OUT" --max-calls 3400 > "$OUT.log" 2>&1 || say "queueL1: sweep exited $?"
say "queueL1: sweep done; assembling trials"
$PY studies/assemble_trials.py > runs/l1_assemble.log 2>&1 || say "queueL1: assemble_trials exited $?"
$PY studies/analyze_l1.py > runs/l1_analyze.log 2>&1 || say "queueL1: analyze_l1 exited $?"
say "queueL1 done"
