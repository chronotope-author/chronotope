#!/usr/bin/env bash
# M3c (2026-08-26): Gemma-4-31B-IT-NVFP4 P/R/L, second model family. The M3b attempt failed at server load:
# vLLM 0.27.1 with transformers 5.15.1 raises AmbiguousGlobalPerLayerAttributeError('head_dim') on the gemma4 config;
# transformers 5.14.0 exposes head_dim (checked). This script serves Gemma from a separate venv (~/gemma-serve:
# vllm 0.27.1 + transformers 5.14.0) and restores the primary Qwen server from ~/qwen-serve at the end.
# Gate: the marker "queueM3b done" (5-s polls so this runs before the other queued studies, which gate on
# "queueM3c done"); never a queue bash pid. Markers "queueM3c ...".
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueM3c: waiting for 'queueM3b done'"
while ! grep -q "queueM3b done" "$LOG"; do sleep 5; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py" > /dev/null; do sleep 5; done
[ -x "$HOME/gemma-serve/bin/vllm" ] || { say "queueM3c FAIL: ~/gemma-serve venv missing"; exit 1; }
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 venv dir, $2 model id, rest: extra args
  local venv="$1" rid="$2"; shift 2; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "queueM3c: starting vLLM ($venv) for $rid"
  (cd "$venv" && setsid -f $SERVE_ENV "$venv/bin/vllm" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 < /dev/null)
  for i in $(seq 1 180); do curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queueM3c: server up: $rid"; return 0; }; pgrep -f 'vllm serve' > /dev/null || { say "queueM3c FAIL: server process exited"; return 1; }; sleep 10; done
  say "queueM3c FAIL: $rid did not come up"; return 1
}
stop_server
if start_server "$HOME/gemma-serve" nvidia/Gemma-4-31B-IT-NVFP4; then
  TAG=m3_gemma4_31b_nvfp4_notification_100
  say "queueM3c: $TAG: gate + P/R/L 5 windows x 100 seeds (notification, prompt 4)"
  $PY studies/r1_revealed_target.py --model nvidia/Gemma-4-31B-IT-NVFP4 --base-url $BASE --no-thinking --skin notification --prompt-version 4 \
      --seed-start 200 --n-seeds 100 --out "runs/$TAG" --max-calls 1800 > "runs/$TAG.log" 2>&1 || say "queueM3c: $TAG exited $?"
  $PY studies/analyze_r1.py "results/${TAG^^}.md" "runs/$TAG" > /dev/null 2>&1 || say "queueM3c: $TAG analyze exited $?"
fi
stop_server
start_server "$HOME/qwen-serve" Qwen/Qwen3.8-27B --reasoning-parser qwen3 || say "queueM3c: primary server not restored"
say "queueM3c done"
