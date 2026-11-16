#!/usr/bin/env bash
# M4 (2026-08-26, cards/M4-cross-agent-prl.md phase 1): EvoCUA-8B and EvoCUA-32B (meituan, 20260105) on the notification
# rendering in their native S2 grammar (prompt 18 (no-wait)): Paused check + P/R/L at the fixed windows, then a Live-only sweep at
# windows = {0.5,0.75,1,1.25,1.5} x the agent's own median calibration B->D. Both are thinking models (the chat template
# opens <think> unconditionally): max_tokens 4096, temperature 0. Served one at a time with ~/gemma-serve (vllm 0.27.1).
# Waits for the marker 'queueM4b done', then for the GPU studies to be idle. Markers "queueM4c ..."; never a queue bash pid.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueM4c: waiting for M4b"
until grep -q "] queueM4b done$" "$LOG"; do sleep 120; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 60; done
GEMMA_VLLM="$HOME/gemma-serve/bin/vllm"; QWEN_VLLM="$HOME/qwen-serve/.venv/bin/vllm"
[ -x "$GEMMA_VLLM" ] && [ -x "$QWEN_VLLM" ] || { say "queueM4c FAIL: vllm binaries missing"; exit 1; }
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 vllm binary, $2 model id, rest: extra args
  local vllm="$1" rid="$2"; shift 2; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  local bindir; bindir=$(dirname "$vllm")
  say "queueM4c: starting vLLM ($vllm) for $rid"
  (cd "$HOME" && setsid -f env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$bindir:/usr/local/cuda/bin:$PATH" "$vllm" serve "$snap" \
      --served-model-name "$rid" --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 4 \
      --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 < /dev/null)
  sleep 20
  for i in $(seq 1 180); do
    curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queueM4c: server up: $rid"; return 0; }
    pgrep -f 'vllm serve' > /dev/null || { say "queueM4c FAIL: server process exited ($rid)"; return 1; }
    sleep 10
  done
  say "queueM4c FAIL: $rid did not come up"; return 1
}
run_agent() {  # $1 model id, $2 tag stem
  local rid="$1" stem="$2"
  local TAG="m4c_${stem}_notification_10"
  say "queueM4c: $TAG: gate + P/R/L 5 windows x 10 seeds (notification, prompt 18 (no-wait), thinking, 4096 tokens)"
  $PY studies/r1_revealed_target.py --model "$rid" --base-url $BASE --thinking --max-tokens 4096 --skin notification --prompt-version 18 \
      --seed-start 200 --n-seeds 10 --out "runs/$TAG" --max-calls 1800 > "runs/$TAG.log" 2>&1 || { say "queueM4c: $TAG exited $? (gate or budget)"; return 1; }
  $PY studies/analyze_r1.py "results/${TAG^^}.md" "runs/$TAG" > /dev/null 2>&1 || say "queueM4c: $TAG analyze exited $?"
  local TAG2="m4c_${stem}_ratio_10"
  say "queueM4c: $TAG2: Live at 0.5,0.75,1,1.25,1.5 x median calibration B->D"
  $PY studies/r1_revealed_target.py --model "$rid" --base-url $BASE --thinking --max-tokens 4096 --skin notification --prompt-version 18 \
      --seed-start 200 --n-seeds 10 --policies live --windows-ratio 0.5,0.75,1,1.25,1.5 --out "runs/$TAG2" --max-calls 600 > "runs/$TAG2.log" 2>&1 || say "queueM4c: $TAG2 exited $?"
  $PY studies/analyze_r2.py "runs/$TAG" "runs/$TAG2" "results/${TAG2^^}.md" > /dev/null 2>&1 || say "queueM4c: $TAG2 analyze exited $?"
}
stop_server
if start_server "$GEMMA_VLLM" meituan/EvoCUA-8B-20260105; then run_agent meituan/EvoCUA-8B-20260105 evocua8b; fi
stop_server
if start_server "$GEMMA_VLLM" meituan/EvoCUA-32B-20260105; then
  say "queueM4c: 32B y-diagnostic (10 seeds x 4 variants)"
  $PY studies/m4c_evocua_ydiag.py --model meituan/EvoCUA-32B-20260105 --base-url $BASE --out runs/m4c_evocua32b_ydiag.jsonl > runs/m4c_evocua32b_ydiag.log 2>&1 || say "queueM4c: ydiag exited $?"
  run_agent meituan/EvoCUA-32B-20260105 evocua32b
fi
stop_server
start_server "$QWEN_VLLM" Qwen/Qwen3.8-27B --reasoning-parser qwen3 || say "queueM4c: primary server not restored"
say "queueM4c done"
