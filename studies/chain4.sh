#!/usr/bin/env bash
# UI-TARS-1.5-7B native grammar (prompt 7) with max_tokens 256 (the Thought text needs them), server already up.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
BASE=http://127.0.0.1:8000/v1
PY="env PYTHONPATH=src .venv/bin/python"
RID=ByteDance-Seed/UI-TARS-1.5-7B; TAG=ui_tars_1_5_7b_native
say "chain4: WI-007 ($RID): WI-002 protocol, notification, prompt 7, max_tokens 256"
$PY studies/r1_revealed_target.py --model "$RID" --base-url $BASE --no-thinking --max-tokens 256 --skin notification --prompt-version 7 --out "runs/wi007_${TAG}_prl" --max-calls 400 > "runs/wi007_${TAG}_prl.log" 2>&1 || say "prl exited $? (competence gate or error)"
say "chain4: WI-007 ($RID): WI-004 subset, prompt 7"
$PY studies/c1_balloon_drift.py --model "$RID" --base-url $BASE --no-thinking --max-tokens 256 --prompt-version 7 --velocities 0,8 --policies frozen,late_bind --no-controls --out "runs/wi007_${TAG}_wi004" --max-calls 200 > "runs/wi007_${TAG}_wi004.log" 2>&1 || say "wi004 exited $?"
say "chain4: WI-007 ($RID): WI-006 subset, prompt 7"
$PY studies/p1_pending_button.py --model "$RID" --base-url $BASE --no-thinking --max-tokens 256 --prompt-version 7 --effect-delays 2,8 --memories none,harness_pending --no-drop --out "runs/wi007_${TAG}_wi006" --max-calls 600 > "runs/wi007_${TAG}_wi006.log" 2>&1 || say "wi006 exited $?"
say "chain4: restoring Qwen3.8-27B"
pkill -f 'vllm serve' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5
SNAP=$(ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"*/ | head -1)
(cd "$HOME/qwen-serve" && nohup env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$HOME/qwen-serve/.venv/bin:/usr/local/cuda/bin:$PATH" "$HOME/qwen-serve/.venv/bin/vllm" serve "$SNAP" --served-model-name Qwen/Qwen3.8-27B --host 127.0.0.1 --port 8000 --max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.70 --reasoning-parser qwen3 --limit-mm-per-prompt '{"image":1}' > "$HOME/chronotope/runs/vllm_Qwen_Qwen3.8-27B.log" 2>&1 &)
for i in $(seq 1 120); do curl -s -m 3 "$BASE/models" | grep -q 'Qwen3.8-27B' && { say "server up: Qwen/Qwen3.8-27B"; break; }; sleep 10; done
say "pipeline v2 done"
