#!/usr/bin/env bash
# 100-seed queue (2026-08-24 evening), owned by the paper-revision session. Order by claim dependence (CLAIMS.md §3):
# A1 animated -> A1 static -> C P/R/L rows -> UI-TARS C1 subset (server swap) -> B1 (full regimes) -> O2 (6.2-s window) -> A2.
# Conventions: fresh filename; detached (setsid nohup); markers "queue100 ..." in runs/pipeline.log; src/chronotope frozen.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f "studies/(c1_|c2_|c3_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH"
VLLM="$HOME/qwen-serve/.venv/bin/vllm"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {
  local rid="$1"; shift; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "queue100: starting vLLM for $rid"
  (cd "$HOME/qwen-serve" && nohup $SERVE_ENV "$VLLM" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 &)
  for i in $(seq 1 120); do curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queue100: server up: $rid"; return 0; }; sleep 10; done
  say "queue100 FAIL: $rid did not come up"; return 1
}

say "queue100: A1 sponsor skip (agent, animated player): gate + 5 windows x 100 seeds x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/a1_qwen38_nothink_100 --max-calls 5000 > runs/a1_qwen38_nothink_100.log 2>&1 || say "queue100: A1 exited $?"
$PY studies/analyze_prl.py runs/a1_qwen38_nothink_100 results/A1_qwen38_nothink_100.md --title "A1 sponsor skip (agent)" --cell-order w1,w2,w4,w8,w16 > /dev/null || say "queue100: A1 analyze exited $?"

say "queue100: A1 static player control: gate + 5 windows x 100 seeds x P/R/L"
$PY studies/a1_sponsor_skip_agent.py --model $M --base-url $BASE --no-thinking --skin player_static --n-seeds 100 --out runs/a1_static_qwen38_nothink_100 --max-calls 5000 > runs/a1_static_qwen38_nothink_100.log 2>&1 || say "queue100: A1 static exited $?"
$PY studies/analyze_prl.py runs/a1_static_qwen38_nothink_100 results/A1_static_qwen38_nothink_100.md --title "A1 sponsor skip (agent, static player)" --cell-order w1,w2,w4,w8,w16 > /dev/null || say "queue100: A1 static analyze exited $?"
$PY studies/compare_prl_runs.py runs/a1_qwen38_nothink_100 runs/a1_static_qwen38_nothink_100 results/A1_animation_cost_100.md --labels animated,static --cell-order w1,w2,w4,w8,w16 > /dev/null || say "queue100: compare exited $?"

say "queue100: C1-C3 P/R/L rows: 5 cells x 100 seeds x P/R/L + diagnostic"
$PY studies/c_prl_rows.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/c_prl_qwen38_nothink_100 --max-calls 3000 > runs/c_prl_qwen38_nothink_100.log 2>&1 || say "queue100: C-PRL exited $?"
$PY studies/analyze_prl.py runs/c_prl_qwen38_nothink_100 results/C_prl_qwen38_nothink_100.md --title "C1–C3 P/R/L rows (step-based agent)" --cell-order c1v4,c1v8,c1v16,c2j2,c3v8,c1v8lb --diagnostic c1v8lb > /dev/null || say "queue100: C-PRL analyze exited $?"

say "queue100: UI-TARS-1.5-7B native (prompt 7) C1 subset at 100 seeds: v in {0,8} x {frozen, late_bind}"
stop_server
if start_server ByteDance-Seed/UI-TARS-1.5-7B; then
  $PY studies/c1_balloon_drift.py --model ByteDance-Seed/UI-TARS-1.5-7B --base-url $BASE --no-thinking --prompt-version 7 --max-tokens 256 --velocities 0,8 --policies frozen,late_bind --no-controls --seed-start 1000 --n-seeds 100 --out runs/c1_uitars_native_100 --max-calls 3000 > runs/c1_uitars_native_100.log 2>&1 || say "queue100: UI-TARS C1 exited $?"
  $PY studies/analyze_c1.py runs/c1_uitars_native_100 results/C1_uitars_native_100.md > /dev/null || say "queue100: UI-TARS analyze exited $?"
fi
stop_server
start_server Qwen/Qwen3.8-27B --reasoning-parser qwen3 || { say "queue100 FAIL: 27B server did not return; stopping"; exit 1; }

say "queue100: B1 live trading (agent): gate + 5 regimes x 100 seeds x P/R/L"
$PY studies/b1_live_trading_agent.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/b1_qwen38_nothink_100 --max-calls 4500 > runs/b1_qwen38_nothink_100.log 2>&1 || say "queue100: B1 exited $?"
$PY studies/analyze_prl.py runs/b1_qwen38_nothink_100 results/B1_qwen38_nothink_100.md --title "B1 live trading (agent)" --cell-order rnone,r2000,r6680,r6930s,r10000 > /dev/null || say "queue100: B1 analyze exited $?"

say "queue100: O2 remedy matrix (agent): 3 columns x 4 rows x 100 seeds, closing window 6.2 s"
$PY studies/o2_remedy_matrix_agent.py --model $M --base-url $BASE --no-thinking --closing-window-s 6.2 --n-seeds 100 --out runs/o2_qwen38_nothink_100 --max-calls 2000 > runs/o2_qwen38_nothink_100.log 2>&1 || say "queue100: O2 exited $?"
$PY studies/analyze_o2.py runs/o2_qwen38_nothink_100 results/O2_qwen38_nothink_100.md > /dev/null || say "queue100: O2 analyze exited $?"

say "queue100: A2 sponsor skip (pending): gate + 4 seeks x 3 arms x 100 seeds"
$PY studies/a2_sponsor_pending.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/a2_qwen38_nothink_100 --max-calls 7000 > runs/a2_qwen38_nothink_100.log 2>&1 || say "queue100: A2 exited $?"
$PY studies/analyze_a2.py runs/a2_qwen38_nothink_100 results/A2_qwen38_nothink_100.md > /dev/null || say "queue100: A2 analyze exited $?"
say "queue100 done"
