#!/usr/bin/env bash
# Agent pilots (2026-08-24), 10 seeds each, in order: A1 -> C-PRL -> A2 -> B1 -> O2. Owned by the review session.
# Conventions: fresh filename; detached launch (setsid nohup); never while another study is on the GPU;
# markers in runs/pipeline.log; src/chronotope is frozen while this runs.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f "studies/(c1_|c2_|c3_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
say "agent_pilots: A1 sponsor skip (agent) pilot: gate + 5 windows x 10 seeds x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --out runs/a1_qwen38_nothink_pilot10 --max-calls 600 > runs/a1_qwen38_nothink_pilot10.log 2>&1 || say "agent_pilots: A1 exited $?"
say "agent_pilots: C1-C3 P/R/L rows pilot: 5 cells x 10 seeds x P/R/L + diagnostic"
$PY studies/c_prl_rows.py --model $M --base-url $BASE --no-thinking --out runs/c_prl_qwen38_nothink_pilot10 --max-calls 600 > runs/c_prl_qwen38_nothink_pilot10.log 2>&1 || say "agent_pilots: C-PRL exited $?"
say "agent_pilots: A2 sponsor skip (pending) pilot: gate + 4 seeks x 3 arms x 10 seeds"
$PY studies/a2_sponsor_pending.py --model $M --base-url $BASE --no-thinking --out runs/a2_qwen38_nothink_pilot10 --max-calls 2500 > runs/a2_qwen38_nothink_pilot10.log 2>&1 || say "agent_pilots: A2 exited $?"
say "agent_pilots: B1 live trading (agent) pilot: gate + 4 regimes x 10 seeds x P/R/L"
$PY studies/b1_live_trading_agent.py --model $M --base-url $BASE --no-thinking --out runs/b1_qwen38_nothink_pilot10 --max-calls 500 > runs/b1_qwen38_nothink_pilot10.log 2>&1 || say "agent_pilots: B1 exited $?"
say "agent_pilots: O2 remedy matrix (agent) pilot: 3 columns x 4 rows x 10 seeds"
$PY studies/o2_remedy_matrix_agent.py --model $M --base-url $BASE --no-thinking --out runs/o2_qwen38_nothink_pilot10 --max-calls 400 > runs/o2_qwen38_nothink_pilot10.log 2>&1 || say "agent_pilots: O2 exited $?"
say "agent_pilots done"
