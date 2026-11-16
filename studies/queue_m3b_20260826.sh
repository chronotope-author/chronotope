#!/usr/bin/env bash
# M3 queue, corrected copy (2026-08-26; cards/M3-multi-agent-prl.md). The 2026-08-25 script omitted --prompt-version 4
# (the cited R1 run and the qualification card use prompt 4; the skin default is 2), so every gate would have failed.
# Waits for the 'queueR2c done' marker and both downloads, then per model: swap the vLLM server, Paused gate + P/R/L
# (windows 1,2,4,8,16 s x 100 seeds) on the notification rendering with prompt 4 / JSON grammar, analyze.
# Precision is a treatment (deliberation sets the Live threshold): Qwen3.8-27B NVFP4 vs the bf16 primary is a
# same-weights, speed-only contrast; Gemma-4-31B-IT NVFP4 is a second family at matched precision. Markers "queueM3b …".
# Gate on markers and study processes, NOT on queue bash pids.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
say "queueM3b: waiting for 'queueR2c done' and the two downloads"
while ! grep -q "queueR2c done" "$LOG"; do sleep 30; done
while ! grep -q "download done: nvidia/Gemma-4-31B-IT-NVFP4" "$LOG" || ! grep -q "download done: unsloth/Qwen3.8-27B-NVFP4" "$LOG"; do sleep 60; done
while pgrep -f "studies/(c1_|c2_|c3_|c4_|g1_|p1_|r1_|a1_|a2_|b1_|c_prl|o2_|wi010)[a-z_]*\.py" > /dev/null; do sleep 30; done
BASE=http://127.0.0.1:8000/v1; PY="env PYTHONPATH=src .venv/bin/python"
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH"
VLLM="$HOME/qwen-serve/.venv/bin/vllm"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {
  local rid="$1"; shift; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "queueM3b: starting vLLM for $rid"
  (cd "$HOME/qwen-serve" && setsid -f $SERVE_ENV "$VLLM" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 < /dev/null)
  for i in $(seq 1 180); do curl -s -m 3 "$BASE/models" | grep -q "\"$rid\"" && { say "queueM3b: server up: $rid"; return 0; }; sleep 10; done
  say "queueM3b FAIL: $rid did not come up"; return 1
}
run_prl() {  # $1 model id, $2 run tag, $3.. extra server args
  local rid="$1" tag="$2"; shift 2
  stop_server
  if start_server "$rid" "$@"; then
    say "queueM3b: $tag: gate + P/R/L 5 windows x 100 seeds (notification, prompt 4)"
    $PY studies/r1_revealed_target.py --model "$rid" --base-url $BASE --no-thinking --skin notification --prompt-version 4 \
        --seed-start 200 --n-seeds 100 --out "runs/$tag" --max-calls 1800 > "runs/$tag.log" 2>&1 || say "queueM3b: $tag exited $?"
    $PY studies/analyze_r1.py "results/${tag^^}.md" "runs/$tag" > /dev/null 2>&1 || say "queueM3b: $tag analyze exited $?"
  fi
}
run_prl unsloth/Qwen3.8-27B-NVFP4 m3_qwen38_nvfp4_notification_100 --reasoning-parser qwen3
run_prl nvidia/Gemma-4-31B-IT-NVFP4 m3_gemma4_31b_nvfp4_notification_100
stop_server
start_server Qwen/Qwen3.8-27B --reasoning-parser qwen3 || say "queueM3b: primary server not restored"
say "queueM3b done"
