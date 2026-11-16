#!/usr/bin/env bash
# 100-seed agent runs (2026-08-24 plan; cards A1-agent, C-prl-rows, A2-pending, B1-agent, O2-agent). NOT launched
# automatically — the user launches: setsid nohup bash studies/agent_full.sh > runs/agent_full.log 2>&1 & disown
# Order = paper priority. ~16 GPU-hours. Markers "agent_full ..." in runs/pipeline.log. src/chronotope frozen meanwhile.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f "studies/(c1_|c2_|c3_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 60; done
PY="env PYTHONPATH=src .venv/bin/python"; BASE=http://127.0.0.1:8000/v1; M=Qwen/Qwen3.8-27B
say "agent_full: A1 sponsor skip (agent), animated player, 100 seeds x 5 windows x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --out runs/a1_qwen38_nothink_100 --max-calls 4000 --seed-start 200 --n-seeds 100 > runs/a1_qwen38_nothink_100.log 2>&1 || say "agent_full: A1 exited $?"
say "agent_full: C1-C3 P/R/L rows, 100 seeds x 5 cells x P/R/L (no diagnostic cell)"
$PY studies/c_prl_rows.py --model $M --base-url $BASE --no-thinking --out runs/c_prl_qwen38_nothink_100 --max-calls 3000 --seed-start 1000 --n-seeds 100 --no-diagnostic > runs/c_prl_qwen38_nothink_100.log 2>&1 || say "agent_full: C-PRL exited $?"
say "agent_full: A1 static player control, 100 seeds x 5 windows x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --skin player_static --out runs/a1_static_qwen38_nothink_100 --max-calls 4000 --seed-start 200 --n-seeds 100 > runs/a1_static_qwen38_nothink_100.log 2>&1 || say "agent_full: A1 static exited $?"
$PY studies/compare_prl_runs.py runs/a1_qwen38_nothink_100 runs/a1_static_qwen38_nothink_100 results/A1_animation_cost_100.md --labels animated,static --cell-order w1,w2,w4,w8,w16 > /dev/null || say "agent_full: compare exited $?"
say "agent_full: A2 sponsor skip (pending), 100 seeds x 4 seeks x {none, chronotope}"
$PY studies/a2_sponsor_pending.py --model $M --base-url $BASE --no-thinking --out runs/a2_qwen38_nothink_100 --max-calls 12000 --seed-start 400 --n-seeds 100 --arms none,chronotope > runs/a2_qwen38_nothink_100.log 2>&1 || say "agent_full: A2 exited $?"
say "agent_full: B1 live trading (agent), 100 seeds x 5 regimes (rnone, r2000, r6680, r6930s, r10000) x P/R/L"
$PY studies/b1_live_trading_agent.py --model $M --base-url $BASE --no-thinking --out runs/b1_qwen38_nothink_100 --max-calls 5000 --seed-start 200 --n-seeds 100 > runs/b1_qwen38_nothink_100.log 2>&1 || say "agent_full: B1 exited $?"
say "agent_full: O2 remedy matrix (agent), 100 seeds x 12 cells, closing window 6.2 s"
$PY studies/o2_remedy_matrix_agent.py --model $M --base-url $BASE --no-thinking --out runs/o2_qwen38_nothink_100 --max-calls 2000 --seed-start 1000 --n-seeds 100 --closing-window-s 6.2 > runs/o2_qwen38_nothink_100.log 2>&1 || say "agent_full: O2 exited $?"
say "agent_full done"
