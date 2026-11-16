set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
WS=$HOME/tensorrt-edgellm-workspace
EL=$HOME/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueT1: $*" | tee -a "$LOG"; }
log "T1a start (engines for Qwen3.5-0.8B; GPU free, no vLLM)"
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

log "T1a done"
