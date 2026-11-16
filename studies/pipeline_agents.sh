#!/usr/bin/env bash
# Unattended GPU pipeline (WI-009 agent cells on the 27B, then WI-007: two more agents through the
# same harness). Models are served ONE AT A TIME (wall-clock B->D is a measured quantity).
# Logs: runs/pipeline.log. Each stage is skipped (not fatal) if its model fails to come up.
set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH"
VLLM="$HOME/qwen-serve/.venv/bin/vllm"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
BASE=http://127.0.0.1:8000/v1
PY="env PYTHONPATH=src .venv/bin/python"

wait_chain() {   # wait for the WI-004/WI-006 chain launched earlier
  while pgrep -f 'studies/wi00[46]_' > /dev/null; do sleep 60; done
  say "earlier study chain finished"
}
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 = hf repo id, $2.. = extra args
  local rid="$1"; shift
  local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "starting vLLM for $rid ($snap)"
  (cd "$HOME/qwen-serve" && nohup $SERVE_ENV "$VLLM" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 &)
  for i in $(seq 1 120); do   # up to 20 min (first load compiles kernels)
    if curl -s -m 3 "$BASE/models" | grep -q "\"$rid\""; then say "server up: $rid"; return 0; fi
    sleep 10
  done
  say "FAIL: $rid did not come up"; return 1
}

wait_chain
say "WI-009 agent cells on Qwen3.8-27B"
$PY studies/c2_balloon_jump.py --agent --model Qwen/Qwen3.8-27B --base-url $BASE --no-thinking --out runs/wi009_qwen38_nothink --max-calls 300 > runs/wi009_qwen38_nothink.log 2>&1 || say "wi009 exited $?"

for RID in ByteDance-Seed/UI-TARS-1.5-7B Qwen/Qwen3-VL-8B-Instruct; do
  TAG=$(echo "$RID" | sed 's#.*/##; s/[^A-Za-z0-9]/_/g' | tr 'A-Z' 'a-z')
  stop_server
  if start_server "$RID"; then
    say "WI-007 ($RID): WI-002 protocol on the notification skin"
    $PY studies/r1_revealed_target.py --model "$RID" --base-url $BASE --no-thinking --skin notification --out "runs/wi007_${TAG}_prl" --max-calls 400 > "runs/wi007_${TAG}_prl.log" 2>&1 || say "prl exited $?"
    say "WI-007 ($RID): WI-004 subset"
    $PY studies/c1_balloon_drift.py --model "$RID" --base-url $BASE --no-thinking --velocities 0,8 --policies frozen,late_bind --no-controls --out "runs/wi007_${TAG}_wi004" --max-calls 200 > "runs/wi007_${TAG}_wi004.log" 2>&1 || say "wi004 exited $?"
    say "WI-007 ($RID): WI-006 subset"
    $PY studies/p1_pending_button.py --model "$RID" --base-url $BASE --no-thinking --effect-delays 2,8 --memories none,harness_pending --no-drop --out "runs/wi007_${TAG}_wi006" --max-calls 600 > "runs/wi007_${TAG}_wi006.log" 2>&1 || say "wi006 exited $?"
  fi
done

stop_server
start_server Qwen/Qwen3.8-27B --reasoning-parser qwen3 || true
say "pipeline done"
