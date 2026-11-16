#!/usr/bin/env bash
# PILOT (2026-08-27, user's order 3 -> 1 -> 2, 10 seeds each, ASAP): faster agents, deadlines scaled to each agent's own
# speed, and the moving-target task, for the CUDA-resident software control, EvoCUA-8B (native grammar, no-wait prompt
# 18, thinking capped at 256 tokens) and Agent S3 (planner Qwen3.8-27B no-think + UI-TARS-1.5-7B grounder; falls back
# to UI-TARS as planner if the 27B does not fit beside the grounder). Preempts queue_m4c (relaunch it afterwards).
# Markers "queuePILOT ..."; never a queue bash pid.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
GEMMA_VLLM="$HOME/gemma-serve/bin/vllm"; QWEN_VLLM="$HOME/qwen-serve/.venv/bin/vllm"; A3PY="$HOME/agents3-venv/bin/python"
PY="env PYTHONPATH=src .venv/bin/python"; KPY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_all() { pkill -f 'vllm serve' || true; pkill -f 'agents3_service.py' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 vllm binary, $2 model id, $3 port, $4 gpu fraction, rest: extra args
  local vllm="$1" rid="$2" port="$3" frac="$4"; shift 4; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  local bindir; bindir=$(dirname "$vllm")
  say "queuePILOT: starting vLLM ($vllm) for $rid on :$port (gpu $frac)"
  (cd "$HOME" && setsid -f env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$bindir:/usr/local/cuda/bin:$PATH" "$vllm" serve "$snap" \
      --served-model-name "$rid" --host 127.0.0.1 --port "$port" --max-model-len 8192 --max-num-seqs 4 \
      --gpu-memory-utilization "$frac" --limit-mm-per-prompt '{"image":8}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_')_$port.log" 2>&1 < /dev/null)
  sleep 20
  for i in $(seq 1 180); do
    curl -s -m 3 "http://127.0.0.1:$port/v1/models" | grep -q "\"$rid\"" && { say "queuePILOT: server up: $rid :$port"; return 0; }
    pgrep -f "vllm serve.*--port $port" > /dev/null || { say "queuePILOT FAIL: server process exited ($rid)"; return 1; }
    sleep 10
  done
  say "queuePILOT FAIL: $rid did not come up"; return 1
}
SEEDS="--seed-start 200 --n-seeds 10 --n-calib 10"
RATIO="--policies live --windows-ratio 0.5,0.75,1,1.25,1.5"

say "queuePILOT start (10 seeds; order: faster agents -> ratio windows -> moving target)"
stop_all

# ---- 3c. the resident agent, fast deliberation (no server needed; runs while the EvoCUA server loads) -------------
say "queuePILOT: resident agent, notification, deliberation 1000 ms and 0 ms (k2 r1, 10 seeds)"
(cd ~/chronotope && setsid -f bash -c "$KPY studies/k2_resident_sweep.py --part r1 --seeds 10 --seed0 1000 --delib-ms 1000 --tag _delib1000_pilot > runs/k2_r1_delib1000_pilot.log 2>&1; $KPY studies/k2_resident_sweep.py --part r1 --seeds 10 --seed0 1000 --delib-ms 0 --tag _delib0_pilot > runs/k2_r1_delib0_pilot.log 2>&1; echo '[$(date "+%F %T")] queuePILOT: resident r1 pilots done' >> $LOG" < /dev/null)

# ---- Phase A: EvoCUA-8B alone --------------------------------------------------------------------------------------
E8=meituan/EvoCUA-8B-20260105; BASE=http://127.0.0.1:8000/v1
if start_server "$GEMMA_VLLM" $E8 8000 0.70; then
  T=pilot_evocua8b_t256_notification; say "queuePILOT: 3a $T: EvoCUA-8B prompt 18, max_tokens 256, gate + P/R/L x 10 seeds"
  $PY studies/r1_revealed_target.py --model $E8 --base-url $BASE --thinking --max-tokens 256 --skin notification --prompt-version 18 \
      $SEEDS --out runs/$T --max-calls 400 > runs/$T.log 2>&1 || say "queuePILOT: $T exited $?"
  $PY studies/analyze_r1.py results/${T^^}.md runs/$T > /dev/null 2>&1 || say "queuePILOT: $T analyze exited $?"
  T2=pilot_evocua8b_t256_ratio; say "queuePILOT: 1a $T2: Live at 0.5..1.5 x own median B->D"
  $PY studies/r1_revealed_target.py --model $E8 --base-url $BASE --thinking --max-tokens 256 --skin notification --prompt-version 18 \
      $SEEDS $RATIO --out runs/$T2 --max-calls 200 > runs/$T2.log 2>&1 || say "queuePILOT: $T2 exited $?"
  $PY studies/analyze_r2.py runs/$T runs/$T2 results/${T2^^}.md > /dev/null 2>&1 || say "queuePILOT: $T2 analyze exited $?"
  T3=pilot_evocua8b_t256_balloon; say "queuePILOT: 2a $T3: balloon 0,4,8,16 px/s x {frozen, late_bind}, window 16 s"
  $PY studies/c1_balloon_drift.py --model $E8 --base-url $BASE --thinking --max-tokens 256 --prompt-version 18 --seed-start 200 --n-seeds 10 \
      --skip-calibration --no-controls --velocities 0,4,8,16 --policies frozen,late_bind --window-s 16 --out runs/$T3 --max-calls 200 > runs/$T3.log 2>&1 || say "queuePILOT: $T3 exited $?"
else
  say "queuePILOT FAIL: EvoCUA-8B server; phase A skipped"
fi
stop_all

# ---- Phase B: Agent S3, fast configuration --------------------------------------------------------------------------
GROUNDER=ByteDance-Seed/UI-TARS-1.5-7B; PLANNER=Qwen/Qwen3.8-27B
start_server "$QWEN_VLLM" $GROUNDER 8001 0.20 || { say "queuePILOT FAIL: grounder"; exit 1; }
if ! start_server "$QWEN_VLLM" $PLANNER 8000 0.55 --reasoning-parser qwen3; then
  pkill -f "vllm serve.*--port 8000" || true; sleep 10
  PLANNER=$GROUNDER; say "queuePILOT: 27B planner failed beside the grounder; planner is now $PLANNER"
  start_server "$QWEN_VLLM" $PLANNER 8000 0.40 || { say "queuePILOT FAIL: planner"; exit 1; }
fi
MODEL="$PLANNER+$GROUNDER"
(cd ~/chronotope && DISPLAY= setsid -f "$A3PY" studies/agents3_service.py --port 8100 --planner "$PLANNER" --planner-url http://127.0.0.1:8000/v1 \
    --grounder "$GROUNDER" --grounder-url http://127.0.0.1:8001/v1 > runs/agents3_service_pilot.log 2>&1 < /dev/null)
sleep 5; curl -s -m 3 http://127.0.0.1:8100/models | grep -q "$PLANNER" || { say "queuePILOT FAIL: agents3 service"; exit 1; }
STEM=agents3_$(echo "$PLANNER" | sed 's#.*/##; s/-20260105//' | tr 'A-Z-.' 'a-z__')
T=pilot_${STEM}_notification; say "queuePILOT: 3b $T: Agent S3 ($MODEL), gate + P/R/L x 10 seeds"
$PY studies/r1_revealed_target.py --server agents3_service --model "$MODEL" --base-url http://127.0.0.1:8100 --no-thinking --max-tokens 64 \
    --skin notification --prompt-version 19 $SEEDS --out runs/$T --max-calls 400 > runs/$T.log 2>&1 || say "queuePILOT: $T exited $?"
$PY studies/analyze_r1.py results/${T^^}.md runs/$T > /dev/null 2>&1 || say "queuePILOT: $T analyze exited $?"
T2=pilot_${STEM}_ratio; say "queuePILOT: 1b $T2: Live at 0.5..1.5 x own median B->D"
$PY studies/r1_revealed_target.py --server agents3_service --model "$MODEL" --base-url http://127.0.0.1:8100 --no-thinking --max-tokens 64 \
    --skin notification --prompt-version 19 $SEEDS $RATIO --out runs/$T2 --max-calls 200 > runs/$T2.log 2>&1 || say "queuePILOT: $T2 exited $?"
$PY studies/analyze_r2.py runs/$T runs/$T2 results/${T2^^}.md > /dev/null 2>&1 || say "queuePILOT: $T2 analyze exited $?"
T3=pilot_${STEM}_balloon; say "queuePILOT: 2b $T3: balloon 0,4,8,16 px/s x {frozen, late_bind}, window 32 s"
$PY studies/c1_balloon_drift.py --server agents3_service --model "$MODEL" --base-url http://127.0.0.1:8100 --no-thinking --max-tokens 64 --prompt-version 19 \
    --seed-start 200 --n-seeds 10 --skip-calibration --no-controls --velocities 0,4,8,16 --policies frozen,late_bind --window-s 32 --out runs/$T3 --max-calls 200 > runs/$T3.log 2>&1 || say "queuePILOT: $T3 exited $?"
stop_all
start_server "$QWEN_VLLM" Qwen/Qwen3.8-27B 8000 0.70 --reasoning-parser qwen3 || say "queuePILOT: primary server not restored"
say "queuePILOT done"
