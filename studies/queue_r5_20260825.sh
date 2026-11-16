#!/usr/bin/env bash
# Round-5 queue (2026-08-25), owned by the paper-revision session. Waits for queue100 to finish, then:
# R2 fine window sweep (Live only, 6 windows x 100 seeds) -> C4 rotating code (typed value; 7 cells x 100 seeds).
# Conventions: fresh filename; detached (setsid nohup); markers "queueR5 ..." in runs/pipeline.log; anchored pgrep.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueR5: waiting for 'queue100 done'"
while ! grep -q "queue100 done" "$LOG" || pgrep -f '^bash studies/queue100_20260824.sh$' > /dev/null; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)" > /dev/null; do sleep 60; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"; M=Qwen/Qwen3.8-27B
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH"
VLLM="$HOME/qwen-serve/.venv/bin/vllm"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
start_server() {
  local rid="$1"; shift; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "queueR5: starting vLLM for $rid"
  (cd "$HOME/qwen-serve" && nohup $SERVE_ENV "$VLLM" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 &)
  for i in $(seq 1 120); do curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queueR5: server up: $rid"; return 0; }; sleep 10; done
  say "queueR5 FAIL: $rid did not come up"; return 1
}
curl -s -m 3 "$BASE/models" | grep -q "\"$M\"" || start_server $M --reasoning-parser qwen3 || exit 1

say "queueR5: R2 fine window sweep (notification, Live only): windows 4.5,5.0,5.2,5.4,5.8,6.0 s x 100 seeds"
$PY studies/r1_revealed_target.py --model $M --base-url $BASE --no-thinking --skin notification --windows-s 4.5,5.0,5.2,5.4,5.8,6.0 --policies live \
    --seed-start 200 --n-seeds 100 --out runs/r2_window_sweep_100 --max-calls 900 > runs/r2_window_sweep_100.log 2>&1 || say "queueR5: R2 exited $?"
$PY studies/analyze_r2.py runs/r1_qwen38_notification_100 runs/r2_window_sweep_100 results/R2_window_sweep_100.md > /dev/null || say "queueR5: R2 analyze exited $?"

say "queueR5: C4 rotating code (typed value): gate + 7 cells x 100 seeds"
$PY studies/c4_rotating_code.py --model $M --base-url $BASE --no-thinking --n-seeds 100 --out runs/c4_qwen38_nothink_100 --max-calls 1500 > runs/c4_qwen38_nothink_100.log 2>&1 || say "queueR5: C4 exited $?"
$PY studies/analyze_c4.py runs/c4_qwen38_nothink_100 results/C4_qwen38_nothink_100.md > /dev/null || say "queueR5: C4 analyze exited $?"
say "queueR5 done"
