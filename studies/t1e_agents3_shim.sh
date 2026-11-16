#!/usr/bin/env bash
# Agent S3 with the paper's 27B as planner and grounder, both answered by ONE in-process Edge-LLM engine behind
# studies/edgellm_openai_shim.py (:8002). No vLLM, no Edge-LLM experimental server. Then the D1 rows.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
log() { echo "[$(date '+%F %T')] queueS3S: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; A3PY="$HOME/agents3-venv/bin/python"; E=$HOME/tensorrt-edgellm-workspace/Qwen3.8-27B/engines
log "start: shim for Qwen3.8-27B on :8002"
(setsid -f env PYTHONPATH=src:chronotope/python .venv/bin/python studies/edgellm_openai_shim.py --engines "$E" --model-id Qwen3.8-27B --port 8002 > runs/edgellm_shim_8002.log 2>&1 < /dev/null)
for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8002/v1/models | grep -q Qwen3.8-27B && break; sleep 5; done
curl -s -m 3 http://127.0.0.1:8002/v1/models | grep -q Qwen3.8-27B || { log "shim did not come up"; exit 1; }
log "shim up; click probe:"; $PY - <<'PYEOF' 2>&1 | tail -1 | tee -a "$LOG"
import sys, json, base64, urllib.request; sys.path.insert(0, "src")
from chronotope.png import encode_png; from chronotope.scenario import manifest_from_dict; from chronotope.controller import SYSTEM_PROMPTS
m = manifest_from_dict({"scenario_id":"revealed_target","scenario_seed":200,"canvas_w_px":320,"canvas_h_px":240,"target_size_px":24,"event_time_ns":500000000,"response_window_ns":8000000000,"frame_period_ns":16666667,"episode_horizon_ns":9000000000,"irrelevant_motion":False,"skin":"notification"})
png = base64.b64encode(encode_png(m.render(600000000), 320, 240, 2)).decode()
body = {"model":"Qwen3.8-27B","messages":[{"role":"system","content":SYSTEM_PROMPTS[4].format(w=640,h=480)},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+png}},{"type":"text","text":"Screenshot attached. Click the target's center or wait."}]}],"max_tokens":64,"temperature":0}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8002/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=300))
print("queueS3S: probe answer", r["choices"][0]["message"]["content"][:80], "(target grid ~175,479)", r["usage"])
PYEOF
(cd ~/chronotope && DISPLAY= setsid -f "$A3PY" studies/agents3_service.py --port 8100 --planner Qwen3.8-27B --planner-url http://127.0.0.1:8002/v1 --grounder Qwen3.8-27B --grounder-url http://127.0.0.1:8002/v1 > runs/agents3_service_shim.log 2>&1 < /dev/null)
sleep 8; curl -s -m 5 http://127.0.0.1:8100/models | grep -q Qwen3.8-27B || { log "agents3 service did not come up"; pkill -f edgellm_openai_shim.py; exit 1; }
for s in agents3_direct agents3_late_bind agents3_frozen; do sed "s#AGENTS3_MODEL#Qwen3.8-27B+Qwen3.8-27B#" specs/d1/$s.json > specs/d1/${s}_shim.json; done
run() { [ -d runs/d1/$1/$3 ] && { log "$1/$3 exists"; return; }; log "$1 / $3"; $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/$1.jsonl --controller-json specs/d1/$2.json --name $3 --conditions live,replay,paused --allow-small --max-calls 400 --out runs/d1/$1 > runs/d1_$1_$3.log 2>&1 || log "$1/$3 exited $?"; }
run closing_window agents3_direct_shim agents3_edgellm27b_direct
run moving_target agents3_late_bind_shim agents3_edgellm27b_late_bind
run moving_target agents3_frozen_shim agents3_edgellm27b_frozen
run static_control agents3_late_bind_shim agents3_edgellm27b_late_bind
pkill -f "agents3_service.py"; pkill -f "edgellm_openai_shim.py"; sleep 3
log "done"
