#!/usr/bin/env bash
# M3d (2026-08-26): Gemma-4-31B-IT-NVFP4 P/R/L, second model family — third attempt.
# M3b: transformers 5.15.1 head_dim error. M3c: the pinned venv (~/gemma-serve, vllm 0.27.1 + transformers 5.14.0)
# lacked `ninja` (engine core init failed) and the restore step used ~/qwen-serve/bin/vllm instead of
# ~/qwen-serve/.venv/bin/vllm, leaving no server up. Both fixed here; liveness is checked only after a settle delay.
# Runs immediately (GPU idle); markers "queueM3d ..."; study guard only, never a queue bash pid.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while pgrep -f "studies/[a-z0-9_]*\.py .*--base-url" > /dev/null; do sleep 5; done  # GPU studies only; scripted CPU pilots may overlap
GEMMA_VLLM="$HOME/gemma-serve/bin/vllm"; QWEN_VLLM="$HOME/qwen-serve/.venv/bin/vllm"
[ -x "$GEMMA_VLLM" ] && [ -x "$QWEN_VLLM" ] || { say "queueM3d FAIL: vllm binaries missing"; exit 1; }
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 vllm binary, $2 model id, rest: extra args
  local vllm="$1" rid="$2"; shift 2; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  local bindir; bindir=$(dirname "$vllm")
  say "queueM3d: starting vLLM ($vllm) for $rid"
  (cd "$HOME" && setsid -f env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$bindir:/usr/local/cuda/bin:$PATH" "$vllm" serve "$snap" \
      --served-model-name "$rid" --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 4 \
      --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 < /dev/null)
  sleep 20
  for i in $(seq 1 180); do
    curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queueM3d: server up: $rid"; return 0; }
    pgrep -f 'vllm serve' > /dev/null || { say "queueM3d FAIL: server process exited ($rid)"; return 1; }
    sleep 10
  done
  say "queueM3d FAIL: $rid did not come up"; return 1
}
stop_server
if start_server "$GEMMA_VLLM" nvidia/Gemma-4-31B-IT-NVFP4; then
  TAG=m3_gemma4_31b_nvfp4_notification_100
  say "queueM3d: $TAG: gate + P/R/L 5 windows x 100 seeds (notification, prompt 4)"
  $PY studies/r1_revealed_target.py --model nvidia/Gemma-4-31B-IT-NVFP4 --base-url $BASE --no-thinking --skin notification --prompt-version 4 \
      --seed-start 200 --n-seeds 100 --out "runs/$TAG" --max-calls 1800 > "runs/$TAG.log" 2>&1 || say "queueM3d: $TAG exited $?"
  $PY studies/analyze_r1.py "results/${TAG^^}.md" "runs/$TAG" > /dev/null 2>&1 || say "queueM3d: $TAG analyze exited $?"
fi
stop_server
start_server "$QWEN_VLLM" Qwen/Qwen3.8-27B --reasoning-parser qwen3 || say "queueM3d: primary server not restored"
say "queueM3d done"
