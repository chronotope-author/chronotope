#!/usr/bin/env bash
# P1 reschedule (WI-010) queue — owned by session chronotope-a3. Runs the 15-min gate+live pilot.
# Starts when EITHER "wi010 go" (peer agreement: before chain8) OR "chain8 done" (fallback: after)
# appears in the pipeline log, and never while another study holds the GPU.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while ! grep -q "wi010 go\|chain8 done" runs/pipeline.log; do sleep 60; done
# never start while another study process is on the GPU
while pgrep -f "studies/(c1_|c2_|c3_|p1_pending|r1_)" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "wi010: P1 reschedule gate (paused, 10 seeds) + live d8 none (10 seeds, pilot)"
$PY studies/p1_calendar.py --model $M --base-url $BASE --no-thinking --out runs/wi010_qwen38_nothink --max-calls 400 > runs/wi010_qwen38_nothink.log 2>&1 || say "wi010 exited $?"
say "wi010 done"
