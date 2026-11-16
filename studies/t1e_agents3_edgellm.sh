#!/usr/bin/env bash
# Agent S3 with NO vLLM: planner Qwen3.8-27B and grounder UI-TARS-1.5-7B served by TensorRT Edge-LLM's own
# OpenAI-compatible experimental server (two engines, :8000 / :8001), studies/agents3_service.py on :8100, then the D1
# rows (closing_window direct; moving_target / static_control late-bound and frozen). Waits for 'T1d done' (UI-TARS engines).
set -u; cd ~/chronotope; LOG=runs/pipeline.log
log() { echo "[$(date '+%F %T')] queueS3E: $*" | tee -a "$LOG"; }
WS=$HOME/tensorrt-edgellm-workspace; EL=$HOME/TensorRT-Edge-LLM; A3PY="$HOME/agents3-venv/bin/python"
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python; D1="env PYTHONPATH=src:chronotope/python .venv/bin/python studies/d1_distributional_gym.py"
until grep -q "queueT1: T1d done" "$LOG"; do sleep 60; done
while pgrep -f "[d]1_distributional_gym.py run" > /dev/null; do sleep 30; done
serve() {  # name port
  local m=$WS/$1
  [ -f "$m/engines/llm/config.json" ] || { log "no engines for $1"; return 1; }
  (cd $EL && setsid -f $PY -m experimental.server --model "$m/engines/llm" --multimodal-engine-dir "$m/engines" --host 127.0.0.1 --port $2 > "$m/server_$2.log" 2>&1 < /dev/null)
  for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:$2/v1/models > /dev/null 2>&1 && { log "Edge-LLM server up: $1 :$2"; return 0; }; sleep 5; done
  log "Edge-LLM server for $1 did not come up"; return 1
}
log "start: Agent S3 over Edge-LLM servers (27B planner :8000, UI-TARS grounder :8001)"
serve Qwen3.8-27B 8000 || exit 1
serve UI-TARS-1.5-7B 8001 || { pkill -f "experimental.server"; exit 1; }
PLANNER=$(curl -s http://127.0.0.1:8000/v1/models | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
GROUNDER=$(curl -s http://127.0.0.1:8001/v1/models | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
log "model ids: planner=$PLANNER grounder=$GROUNDER"
(cd ~/chronotope && DISPLAY= setsid -f "$A3PY" studies/agents3_service.py --port 8100 --planner "$PLANNER" --planner-url http://127.0.0.1:8000/v1 \
    --grounder "$GROUNDER" --grounder-url http://127.0.0.1:8001/v1 > runs/agents3_service_edgellm.log 2>&1 < /dev/null)
sleep 8; curl -s -m 5 http://127.0.0.1:8100/models | grep -q "$PLANNER" || { log "agents3 service did not come up"; pkill -f "experimental.server"; exit 1; }
MODEL="$PLANNER+$GROUNDER"
for s in agents3_direct agents3_late_bind agents3_frozen; do sed "s#AGENTS3_MODEL#$MODEL#" specs/d1/$s.json > specs/d1/${s}_edgellm.json; done
run() { [ -d runs/d1/$1/$3 ] && { log "$1/$3 exists"; return; }; log "$1 / $3"; $D1 run --manifest runs/d1/manifests/$1.jsonl --controller-json specs/d1/$2.json --name $3 --conditions live,replay,paused --allow-small --max-calls 300 --out runs/d1/$1 > runs/d1_$1_$3.log 2>&1 || log "$1/$3 exited $?"; }
run closing_window agents3_direct_edgellm agents3_edgellm_direct
run moving_target agents3_late_bind_edgellm agents3_edgellm_late_bind
run moving_target agents3_frozen_edgellm agents3_edgellm_frozen
run static_control agents3_late_bind_edgellm agents3_edgellm_late_bind
pkill -f "agents3_service.py"; pkill -f "experimental.server"; sleep 3
log "done"
