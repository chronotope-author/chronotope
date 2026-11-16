#!/usr/bin/env bash
# queueT1 — TensorRT Edge-LLM as the planner (cards/T1-tensorrt-edge-llm-planner.md). GPU-bound steps only; the
# toolchain (runtime build, exports on CPU) is done. NOT launched by the build session: it needs the GPU to itself
# (engine builds saturate it) and, for the 27B export, ~80 GB of free RAM (no vLLM server up).
#
#   AFTER=queueM4b setsid nohup bash studies/queue_t1_20260826.sh > runs/queue_t1.log 2>&1 & disown
#   (AFTER=queueK1 if K1 was launched: the two must not overlap.)
set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
AFTER=${AFTER:-queueM4b}
WS=$HOME/tensorrt-edgellm-workspace
EL=$HOME/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueT1: $*" | tee -a "$LOG"; }
log "waiting for $AFTER"
while ! grep -q "\] $AFTER done$" "$LOG" 2>/dev/null; do sleep 120; done
while pgrep -f '^\S*python\S* studies/(r1_revealed_target|c1_balloon_drift|g2_live_page|a1_sponsor|b1_live|k1_kgua_cell)\.py' > /dev/null \
      || pgrep -f 'vllm serve' > /dev/null; do log "a study or a vLLM server is still running; waiting"; sleep 60; done

# ---- T1a: the quick-start VLM (Qwen3.5-0.8B): engines, one image request, the kGUA acceptance run
M=$WS/Qwen3.5-0.8B
if [ ! -f "$M/engines/llm/config.json" ]; then
  log "T1a: building engines for Qwen3.5-0.8B"
  $EL/build/examples/llm/llm_build --onnxDir "$M/onnx/llm" --engineDir "$M/engines/llm" --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048 > "$M/llm_build.log" 2>&1 || { log "llm_build FAILED (see $M/llm_build.log)"; }
  $EL/build/examples/multimodal/visual_build --onnxDir "$M/onnx/visual" --engineDir "$M/engines" --minImageTokens 64 --maxImageTokens 1024 --maxImageTokensPerImage 1024 > "$M/visual_build.log" 2>&1 || { log "visual_build FAILED (see $M/visual_build.log)"; }
fi
if [ -f "$M/engines/llm/config.json" ]; then
  log "T1a: one image request through llm_inference (notification frame, prompt 5)"
  PYTHONPATH=src .venv/bin/python - "$M" <<'EOF'
import json, sys, pathlib
sys.path.insert(0, "tests")
from chronotope.png import encode_png
from conftest import base_spec_dict
from chronotope.spec import TrialSpec
m = pathlib.Path(sys.argv[1])
d = base_spec_dict(); d["scenario"]["skin"] = "balloon"; spec = TrialSpec.from_dict(d); man = spec.manifest()
png = encode_png(man.render(man.event_time_ns), man.canvas_w_px, man.canvas_h_px, 2)
(m / "frame_balloon_640x480.png").write_bytes(png)
prompt = ("You control a game by clicking. You are shown one screenshot. A red balloon appears at an unknown position and may drift; it disappears after a short time. "
          "If the balloon is visible, answer with action \"click\" and the coordinates of the center of the red balloon on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; "
          "y: 0 = top edge, 1000 = bottom edge). If no balloon is visible, answer with action \"wait\". Answer with the JSON object only.")
req = {"batch_size": 1, "temperature": 0.0, "max_generate_length": 64, "enable_thinking": False,
       "requests": [{"messages": [{"role": "system", "content": prompt},
                                  {"role": "user", "content": [{"type": "image", "image": str(m / "frame_balloon_640x480.png")},
                                                               {"type": "text", "text": "Screenshot attached. Click the target's center or wait."}]}]}]}
(m / "input_balloon.json").write_text(json.dumps(req, indent=1))
cx, cy = man.target_box().center()
print("target centre (canvas px)", cx, cy, "-> 0-1000 grid", cx * 1000 // man.canvas_w_px, cy * 1000 // man.canvas_h_px)
EOF
  $EL/build/examples/llm/llm_inference --engineDir "$M/engines/llm" --multimodalEngineDir "$M/engines" --inputFile "$M/input_balloon.json" --outputFile "$M/output_balloon.json" > "$M/llm_inference.log" 2>&1
  log "T1a: llm_inference rc=$? ; answer: $(python3 -c "import json;print(json.load(open('$M/output_balloon.json')))" 2>/dev/null | cut -c1-300)"
  log "T1a: chronotope --planner trt acceptance (synthetic target, v = 0)"
  CHRONOTOPE_TRT_ENGINES=$M/engines .venv/bin/python -m pytest chronotope/python/tests/test_kgua_trt.py -q -s -p no:cacheprovider > "$M/chronotope_trt_test.log" 2>&1
  log "T1a: chronotope trt test rc=$? ($(tail -1 "$M/chronotope_trt_test.log"))"
fi

# ---- T1b: the paper's checkpoints. Export needs RAM (1.5 x checkpoint): gate on free memory.
free_gb() { free -g | awk '/^Mem:/{print $7}'; }
export_ckpt() {   # name hf-snapshot-dir
  local name=$1 snap=$2 dst=$WS/ckpt/$name
  [ -d "$dst" ] || { mkdir -p "$WS/ckpt"; cp -rlL "$snap" "$dst" 2>/dev/null || cp -rL "$snap" "$dst"; }
  [ -f "$WS/$name/onnx/llm/config.json" ] && return 0
  log "T1b: exporting $name on CPU (free RAM $(free_gb) GB)"
  $PY -m tensorrt_edgellm.scripts.export "$dst" "$WS/$name/onnx" > "$WS/export_$name.log" 2>&1 || $HOME/edgellm-venv/bin/tensorrt-edgellm-export "$dst" "$WS/$name/onnx" >> "$WS/export_$name.log" 2>&1
  log "T1b: export $name rc=$?"
}
build_engines() {   # name maxInputLen
  local name=$1 m=$WS/$1
  [ -f "$m/engines/llm/config.json" ] && return 0
  log "T1b: building engines for $name"
  $EL/build/examples/llm/llm_build --onnxDir "$m/onnx/llm" --engineDir "$m/engines/llm" --maxBatchSize 1 --maxInputLen ${2:-2048} --maxKVCacheCapacity ${2:-2048} > "$m/llm_build.log" 2>&1
  log "T1b: llm_build $name rc=$?"
  $EL/build/examples/multimodal/visual_build --onnxDir "$m/onnx/visual" --engineDir "$m/engines" --minImageTokens 64 --maxImageTokens 1024 --maxImageTokensPerImage 1024 > "$m/visual_build.log" 2>&1
  log "T1b: visual_build $name rc=$?"
}
G4=$(ls -d ~/.cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots/*/ 2>/dev/null | head -1)
Q27=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)
for spec in "Gemma-4-31B-IT-NVFP4:$G4:50" "Qwen3.8-27B:$Q27:80"; do
  name=${spec%%:*}; rest=${spec#*:}; snap=${rest%%:*}; need=${rest##*:}
  [ -n "$snap" ] || { log "T1b: $name not in the HF cache; skipped"; continue; }
  if [ "$(free_gb)" -lt "$need" ]; then log "T1b: $name needs ~$need GB free RAM for export, have $(free_gb): skipped (re-run when the vLLM server is down)"; continue; fi
  export_ckpt "$name" "$snap" && build_engines "$name" 2048
done
# NVFP4 for the 27B: the docs say an x86 GPU host; tried here, reported either way
if [ -f "$WS/Qwen3.8-27B/onnx/llm/config.json" ] && [ ! -d "$WS/Qwen3.8-27B-nvfp4" ]; then
  log "T1b: trying tensorrt-edgellm-quantize nvfp4 for Qwen3.8-27B on this host"
  $HOME/edgellm-venv/bin/tensorrt-edgellm-quantize llm --model_dir "$WS/ckpt/Qwen3.8-27B" --output_dir "$WS/ckpt/Qwen3.8-27B-nvfp4" --quantization nvfp4 > "$WS/quantize_qwen38_nvfp4.log" 2>&1
  log "T1b: quantize rc=$? (see $WS/quantize_qwen38_nvfp4.log)"
fi

# ---- T1c: the notification P/R/L rows on the Edge-LLM stack through its OpenAI-compatible server (existing study, unchanged)
serve_and_run() {   # name model-id
  local m=$WS/$1
  [ -f "$m/engines/llm/config.json" ] || return 0
  log "T1c: starting the Edge-LLM server for $1"
  (cd $EL && setsid -f $PY -m experimental.server --model "$m/engines/llm" --multimodal-engine-dir "$m/engines" --host 127.0.0.1 --port 8000 > "$m/server.log" 2>&1)
  for i in $(seq 1 60); do curl -s http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && break; sleep 5; done
  log "T1c: r1_revealed_target on $1 (gate 10 seeds Paused, then P/R/L windows 1,2,4,8,16 x 10 seeds)"
  PYTHONPATH=src .venv/bin/python studies/r1_revealed_target.py --model "$2" --base-url http://127.0.0.1:8000/v1 --no-thinking --prompt-version 4 \
      --seed-start 1000 --n-seeds 10 --out "runs/t1_edgellm_$1_notification_10" --max-calls 2400 > "runs/t1_edgellm_$1.log" 2>&1
  log "T1c: r1 rc=$?"
  pkill -f "experimental.server --model $m/engines/llm" ; sleep 5
}
serve_and_run Qwen3.8-27B "Qwen/Qwen3.8-27B-edgellm-fp16"
serve_and_run Gemma-4-31B-IT-NVFP4 "nvidia/Gemma-4-31B-IT-NVFP4-edgellm"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueT1 done" >> "$LOG"
