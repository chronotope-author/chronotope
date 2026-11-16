#!/usr/bin/env bash
# D1 gym (cards/D1-distributional-gym.md), every agent, 10-episode frozen manifests (2026-08-27; user rule: 10 runs).
# Waits for 'queuePILOT done'. Phase A EvoCUA-8B (native grammar, no-wait prompt 18, thinking capped 256); phase B Agent S3
# (Qwen3.8-27B no-think planner + UI-TARS-1.5-7B grounder, UI-TARS as planner if the 27B does not fit); phase C Qwen3.8-27B
# no-think (the paper's VLM; the only agent with a typed grammar for the rotating-code families). The resident agent's D1
# rows come from studies/d1_resident_pilot.sh. Reports per family with the fixed-grid SR as a reference column.
# Markers "queueD1 ..."; never a queue bash pid.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
GEMMA_VLLM="$HOME/gemma-serve/bin/vllm"; QWEN_VLLM="$HOME/qwen-serve/.venv/bin/vllm"; A3PY="$HOME/agents3-venv/bin/python"
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
D1="$PY studies/d1_distributional_gym.py"
SNAP() { ls -d "$HOME/.cache/huggingface/hub/models--$1/snapshots/"*/ | head -1; }
stop_all() { pkill -f 'vllm serve' || true; pkill -f 'agents3_service.py' || true; for i in $(seq 1 60); do pgrep -f 'vllm serve' > /dev/null || break; sleep 2; done; sleep 5; }
start_server() {  # $1 vllm binary, $2 model id, $3 port, $4 gpu fraction, rest: extra args
  local vllm="$1" rid="$2" port="$3" frac="$4"; shift 4; local snap; snap=$(SNAP "$(echo "$rid" | sed 's#/#--#')")
  local bindir; bindir=$(dirname "$vllm")
  say "queueD1: starting vLLM ($vllm) for $rid on :$port (gpu $frac)"
  (cd "$HOME" && setsid -f env HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda PATH="$bindir:/usr/local/cuda/bin:$PATH" "$vllm" serve "$snap" \
      --served-model-name "$rid" --host 127.0.0.1 --port "$port" --max-model-len 8192 --max-num-seqs 4 \
      --gpu-memory-utilization "$frac" --limit-mm-per-prompt '{"image":8}' "$@" \
      > "$HOME/chronotope/runs/vllm_$(echo "$rid" | tr '/' '_')_$port.log" 2>&1 < /dev/null)
  sleep 20
  for i in $(seq 1 180); do
    curl -s -m 3 "http://127.0.0.1:$port/v1/models" | grep -q "\"$rid\"" && { say "queueD1: server up: $rid :$port"; return 0; }
    pgrep -f "vllm serve.*--port $port" > /dev/null || { say "queueD1 FAIL: server process exited ($rid)"; return 1; }
    sleep 10
  done
  say "queueD1 FAIL: $rid did not come up"; return 1
}
d1run() {  # $1 family, $2 spec name (specs/d1/$2.json), $3 run name
  local fam="$1" spec="$2" name="$3"
  [ -d "runs/d1/$fam/$name" ] && { say "queueD1: $fam/$name exists, skipped"; return 0; }
  say "queueD1: $fam / $name"
  $D1 run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec.json --name "$name" \
      --conditions live,replay,paused --allow-small --max-calls 300 --out runs/d1/$fam > runs/d1_${fam}_${name}.log 2>&1 \
      || say "queueD1: $fam/$name exited $?"
}

say "queueD1: waiting for the pilot"
until grep -q "] queuePILOT done$" "$LOG"; do sleep 120; done
while pgrep -f "studies/(r1_|c1_|k2_|d1_)[a-z0-9_]*\.py" > /dev/null; do sleep 60; done
stop_all

# ---- Phase A: EvoCUA-8B ---------------------------------------------------------------------------------------------
if start_server "$GEMMA_VLLM" meituan/EvoCUA-8B-20260105 8000 0.70; then
  d1run closing_window evocua8b_t256_direct evocua8b_t256_direct
  d1run static_control evocua8b_t256_late_bind evocua8b_t256_late_bind
  d1run moving_target evocua8b_t256_late_bind evocua8b_t256_late_bind
  d1run moving_target evocua8b_t256_frozen evocua8b_t256_frozen
else say "queueD1 FAIL: EvoCUA-8B server; phase A skipped"; fi
stop_all

# ---- Phase B: Agent S3 ------------------------------------------------------------------------------------------------
GROUNDER=ByteDance-Seed/UI-TARS-1.5-7B; PLANNER=Qwen/Qwen3.8-27B
if start_server "$QWEN_VLLM" $GROUNDER 8001 0.20; then
  if ! start_server "$QWEN_VLLM" $PLANNER 8000 0.55 --reasoning-parser qwen3; then
    pkill -f "vllm serve.*--port 8000" || true; sleep 10
    PLANNER=$GROUNDER; say "queueD1: 27B planner failed beside the grounder; planner is now $PLANNER"
    start_server "$QWEN_VLLM" $PLANNER 8000 0.40 || say "queueD1 FAIL: planner"
  fi
  MODEL="$PLANNER+$GROUNDER"
  (cd ~/chronotope && DISPLAY= setsid -f "$A3PY" studies/agents3_service.py --port 8100 --planner "$PLANNER" --planner-url http://127.0.0.1:8000/v1 \
      --grounder "$GROUNDER" --grounder-url http://127.0.0.1:8001/v1 > runs/agents3_service_d1.log 2>&1 < /dev/null)
  sleep 5
  if curl -s -m 3 http://127.0.0.1:8100/models | grep -q "$PLANNER"; then
    for s in agents3_direct agents3_late_bind agents3_frozen; do sed "s#AGENTS3_MODEL#$MODEL#" specs/d1/$s.json > specs/d1/${s}_live.json; done
    STEM=agents3_$(echo "$PLANNER" | sed 's#.*/##; s/-20260105//' | tr 'A-Z-.' 'a-z__')
    d1run closing_window agents3_direct_live ${STEM}_direct
    d1run static_control agents3_late_bind_live ${STEM}_late_bind
    d1run moving_target agents3_late_bind_live ${STEM}_late_bind
    d1run moving_target agents3_frozen_live ${STEM}_frozen
  else say "queueD1 FAIL: agents3 service; phase B skipped"; fi
else say "queueD1 FAIL: grounder; phase B skipped"; fi
stop_all

# ---- Phase C: Qwen3.8-27B no-think (all five families) ----------------------------------------------------------------
if start_server "$QWEN_VLLM" Qwen/Qwen3.8-27B 8000 0.70 --reasoning-parser qwen3; then
  d1run closing_window qwen38_direct qwen38_direct
  d1run static_control qwen38_late_bind qwen38_late_bind
  d1run moving_target qwen38_late_bind qwen38_late_bind
  d1run moving_target qwen38_frozen qwen38_frozen
  d1run late_binding_natural qwen38_value_late_bind qwen38_value_late_bind
  d1run late_binding_natural qwen38_value_frozen qwen38_value_frozen
  d1run late_binding_gate qwen38_value_late_bind qwen38_value_late_bind
  d1run late_binding_gate qwen38_value_frozen qwen38_value_frozen
else say "queueD1 FAIL: Qwen server; phase C skipped"; fi

# ---- Reports (one per family; every run dir present; the fixed-grid SR of the same agent as a reference column) ------
say "queueD1: reports"
for fam in closing_window moving_target static_control late_binding_natural late_binding_gate; do
  runs=$(ls -d runs/d1/$fam/*/ 2>/dev/null | sed 's#/$##'); [ -n "$runs" ] || continue
  base=$(ls -d runs/d1/$fam/chronotope_late_bind_delib0 2>/dev/null | sed 's#/$##'); base=$(basename "${base:-$(echo $runs | cut -d' ' -f1)}")
  $D1 report --family $fam --runs $runs --baseline "$base" --manifest runs/d1/manifests/$fam.jsonl --out results/D1_${fam}_pilot.md \
      --reference "chronotope_late_bind_delib0=notification fixed grid 10/10 at 1..16 s (K2_r1_delib0_pilot)" \
                  "chronotope_late_bind_delib1000=0/10 at 1 s, 10/10 at 2..16 s (K2_r1_delib1000_pilot)" \
                  "chronotope_late_bind_delib5200=0/100 at 1..4 s, 100/100 at 8,16 s (K2_r1)" \
                  "chronotope_frozen_delib5200=balloon: 0/100 at >=4 px/s (K2_c1 frozen)" \
                  "agents3_qwen3_8_27b_direct=Agent S3 EvoCUA-8B planner: 0/100 at <=8 s, 29/100 at 16 s (M4b)" \
      > runs/d1_report_$fam.log 2>&1 || say "queueD1: report $fam exited $?"
done
say "queueD1 done"
