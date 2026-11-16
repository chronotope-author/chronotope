#!/usr/bin/env bash
# WI-007 pipeline v2: Qwen3-VL-8B-Instruct first (Qwen-VL grid prompts 4/5, as the 27B), then UI-TARS-1.5-7B
# with its NATIVE grammar (prompt 7: pixel coordinates, UI-TARS action syntax) — the one declared alternative
# of cards/WI-007-second-agents.md after the shared JSON grammar failed its competence gate (0.60 / 0.00).
set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
SERVE_ENV="env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH=$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH"
VLLM="$HOME/qwen-serve/.venv/bin/vllm"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
BASE=http://127.0.0.1:8000/v1
PY="env PYTHONPATH=src .venv/bin/python"
stop_server() { pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {
  local rid="$1"; shift
  local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  say "starting vLLM for $rid"
  (cd "$HOME/qwen-serve" && nohup $SERVE_ENV "$VLLM" serve "$snap" --served-model-name "$rid" --host 127.0.0.1 --port 8000 \
      --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --limit-mm-per-prompt '{"image":1}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_').log" 2>&1 &)
  for i in $(seq 1 120); do
    if curl -s -m 3 "$BASE/models" | grep -q "\"$rid\""; then say "server up: $rid"; return 0; fi
    sleep 10
  done
  say "FAIL: $rid did not come up"; return 1
}
run_stage() {  # $1 rid, $2 tag, $3 prl prompt, $4 wi004 prompt, $5 wi006 prompt
  local RID="$1" TAG="$2" PP="$3" P4="$4" P6="$5"
  say "WI-007 ($RID): WI-002 protocol, notification skin, prompt $PP"
  $PY studies/r1_revealed_target.py --model "$RID" --base-url $BASE --no-thinking --skin notification --prompt-version $PP --out "runs/wi007_${TAG}_prl" --max-calls 400 > "runs/wi007_${TAG}_prl.log" 2>&1 || say "prl exited $? (competence gate or error)"
  say "WI-007 ($RID): WI-004 subset, prompt $P4"
  $PY studies/c1_balloon_drift.py --model "$RID" --base-url $BASE --no-thinking --prompt-version $P4 --velocities 0,8 --policies frozen,late_bind --no-controls --out "runs/wi007_${TAG}_wi004" --max-calls 200 > "runs/wi007_${TAG}_wi004.log" 2>&1 || say "wi004 exited $?"
  say "WI-007 ($RID): WI-006 subset, prompt $P6"
  $PY studies/p1_pending_button.py --model "$RID" --base-url $BASE --no-thinking --prompt-version $P6 --effect-delays 2,8 --memories none,harness_pending --no-drop --out "runs/wi007_${TAG}_wi006" --max-calls 600 > "runs/wi007_${TAG}_wi006.log" 2>&1 || say "wi006 exited $?"
}
stop_server
if start_server Qwen/Qwen3-VL-8B-Instruct; then run_stage Qwen/Qwen3-VL-8B-Instruct qwen3_vl_8b_instruct 4 5 4; fi
stop_server
if [ -f runs/PROMPT7_READY ]; then
  if start_server ByteDance-Seed/UI-TARS-1.5-7B; then run_stage ByteDance-Seed/UI-TARS-1.5-7B ui_tars_1_5_7b_native 7 7 7; fi
else
  say "UI-TARS native stage skipped: runs/PROMPT7_READY not present"
fi
stop_server
start_server Qwen/Qwen3.8-27B --reasoning-parser qwen3 || true
say "pipeline v2 done"
