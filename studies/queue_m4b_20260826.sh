#!/usr/bin/env bash
# M4 phase 3 (2026-08-26, cards/M4-cross-agent-prl.md U63): Agent S3 (gui-agents 0.3.x, as shipped: Worker + reflection +
# code agent) with a LOCAL planner and grounder so every clock is measured: planner EvoCUA-32B on :8000 (~/gemma-serve
# vllm), grounder UI-TARS-1.5-7B on :8001 (~/qwen-serve vllm), studies/agents3_service.py on :8100 (~/agents3-venv).
# The one exception to "one model served at a time": the agent IS the pair, and both servers are on the same GPU for
# the whole cell. If the 32B planner does not come up beside the grounder, EvoCUA-8B is the planner (marker says so).
# Paused check + P/R/L at the fixed windows, then Live at 0.5..1.5 x its own B->D. Waits for 'queueM4 done'.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueM4b: waiting for M4"
until grep -q "] queueM4 done$" "$LOG"; do sleep 120; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 60; done
GEMMA_VLLM="$HOME/gemma-serve/bin/vllm"; QWEN_VLLM="$HOME/qwen-serve/.venv/bin/vllm"; A3PY="$HOME/agents3-venv/bin/python"
[ -x "$GEMMA_VLLM" ] && [ -x "$QWEN_VLLM" ] && [ -x "$A3PY" ] || { say "queueM4b FAIL: binaries missing"; exit 1; }
PY="env PYTHONPATH=src .venv/bin/python"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_all() { pkill -f 'vllm serve' || true; pkill -f 'agents3_service.py' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 vllm binary, $2 model id, $3 port, $4 gpu fraction, rest: extra args
  local vllm="$1" rid="$2" port="$3" frac="$4"; shift 4; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  local bindir; bindir=$(dirname "$vllm")
  say "queueM4b: starting vLLM ($vllm) for $rid on :$port (gpu $frac)"
  (cd "$HOME" && setsid -f env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$bindir:/usr/local/cuda/bin:$PATH" "$vllm" serve "$snap" \
      --served-model-name "$rid" --host 127.0.0.1 --port "$port" --max-model-len 8192 --max-num-seqs 4 \
      --gpu-memory-utilization "$frac" --limit-mm-per-prompt '{"image":8}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_')_$port.log" 2>&1 < /dev/null)
  sleep 20
  for i in $(seq 1 180); do
    curl -s -m 3 "http://127.0.0.1:$port/v1/models" | grep -q "\"$rid\"" && { say "queueM4b: server up: $rid :$port"; return 0; }
    pgrep -f "vllm serve.*--port $port" > /dev/null || { say "queueM4b FAIL: server process exited ($rid)"; return 1; }
    sleep 10
  done
  say "queueM4b FAIL: $rid did not come up"; return 1
}
stop_all
GROUNDER=ByteDance-Seed/UI-TARS-1.5-7B
start_server "$QWEN_VLLM" $GROUNDER 8001 0.20 || { say "queueM4b FAIL: grounder"; exit 1; }
PLANNER=meituan/EvoCUA-32B-20260105
if ! start_server "$GEMMA_VLLM" $PLANNER 8000 0.58; then
  pkill -f "vllm serve.*--port 8000" || true; sleep 10
  PLANNER=meituan/EvoCUA-8B-20260105
  say "queueM4b: 32B planner failed beside the grounder; planner is now $PLANNER"
  start_server "$GEMMA_VLLM" $PLANNER 8000 0.40 || { say "queueM4b FAIL: planner"; exit 1; }
fi
MODEL="$PLANNER+$GROUNDER"
(cd ~/chronotope && DISPLAY= setsid -f "$A3PY" studies/agents3_service.py --port 8100 --planner "$PLANNER" --planner-url http://127.0.0.1:8000/v1 \
    --grounder "$GROUNDER" --grounder-url http://127.0.0.1:8001/v1 > runs/agents3_service.log 2>&1 < /dev/null)
sleep 5; curl -s -m 3 http://127.0.0.1:8100/models | grep -q "$PLANNER" || { say "queueM4b FAIL: agents3 service"; exit 1; }
STEM=agents3_$(echo "$PLANNER" | sed 's#.*/##; s/-20260105//' | tr 'A-Z-' 'a-z_')
TAG="m4_${STEM}_notification_100"
say "queueM4b: $TAG: gate + P/R/L 5 windows x 100 seeds (notification, prompt 17, $MODEL)"
$PY studies/r1_revealed_target.py --server agents3_service --model "$MODEL" --base-url http://127.0.0.1:8100 --no-thinking --max-tokens 64 \
    --skin notification --prompt-version 17 --seed-start 200 --n-seeds 100 --out "runs/$TAG" --max-calls 1800 > "runs/$TAG.log" 2>&1 || say "queueM4b: $TAG exited $? (gate or budget)"
$PY studies/analyze_r1.py "results/${TAG^^}.md" "runs/$TAG" > /dev/null 2>&1 || say "queueM4b: $TAG analyze exited $?"
TAG2="m4_${STEM}_ratio_100"
say "queueM4b: $TAG2: Live at 0.5,0.75,1,1.25,1.5 x median calibration B->D"
$PY studies/r1_revealed_target.py --server agents3_service --model "$MODEL" --base-url http://127.0.0.1:8100 --no-thinking --max-tokens 64 \
    --skin notification --prompt-version 17 --seed-start 200 --n-seeds 100 --policies live --windows-ratio 0.5,0.75,1,1.25,1.5 \
    --out "runs/$TAG2" --max-calls 600 > "runs/$TAG2.log" 2>&1 || say "queueM4b: $TAG2 exited $?"
$PY studies/analyze_r2.py "runs/$TAG" "runs/$TAG2" "results/${TAG2^^}.md" > /dev/null 2>&1 || say "queueM4b: $TAG2 analyze exited $?"
stop_all
start_server "$QWEN_VLLM" Qwen/Qwen3.8-27B 8000 0.70 --reasoning-parser qwen3 || say "queueM4b: primary server not restored"
say "queueM4b done"
