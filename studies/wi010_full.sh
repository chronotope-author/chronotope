#!/usr/bin/env bash
# WI-010 full run (2026-08-24): calendar skin, gate 100 seeds + Live d8 x {none, harness_pending} x 100 seeds.
# Owner: this session. Never starts while another study is on the GPU (anchored pgrep).
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f '^.*python studies/(c1_|c2_|c3_|p1_|r1_|o1_)' > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "wi010_full (relaunch after asset bake): calendar gate (100 seeds) + live d8 none,harness_pending (100 seeds each)"
$PY studies/p1_calendar.py --model $M --base-url $BASE --no-thinking --memories none,harness_pending --gate-seeds 100 --seed-start 1000 --n-seeds 100 --out runs/wi010_qwen38_nothink_100 --max-calls 3000 > runs/wi010_qwen38_nothink_100.log 2>&1 || say "wi010_full exited $?"
say "wi010_full done"
