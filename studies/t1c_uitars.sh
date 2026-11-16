#!/usr/bin/env bash
# After T1b: patch Edge-LLM's default LLM path (chronotope/patches/edgellm-0.10.0-default-attn-fp16.patch), export
# UI-TARS-1.5-7B, build engines, then run the chronotope + Edge-LLM D1 rows for UI-TARS and (if built) Qwen3.8-27B.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
log() { echo "[$(date '+%F %T')] queueT1: $*" | tee -a "$LOG"; }
WS=$HOME/tensorrt-edgellm-workspace; EL=$HOME/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
until grep -q "queueT1: T1b done" "$LOG"; do sleep 60; done
F=$EL/tensorrt_edgellm/models/default/modeling_default.py
if ! grep -q "chronotope patch: the plugin's traced output" "$F"; then
  $PY - "$F" <<'PYEOF'
import sys; p=sys.argv[1]; s=open(p).read()
old="        return self.o_proj(attn_output), present_key_value\n"
new=("        if attn_output.dtype != torch.float16:\n"
     "            attn_output = attn_output.to(torch.float16)   # chronotope patch: the plugin's traced output is float32 here\n"
     "        return self.o_proj(attn_output), present_key_value\n")
assert s.count(old) == 1, s.count(old); open(p,"w").write(s.replace(old,new)); print("patched")
PYEOF
  log "T1c: applied edgellm-0.10.0-default-attn-fp16.patch"
fi
name=UI-TARS-1.5-7B; m=$WS/$name
rm -rf "$m/onnx"
log "T1c: exporting $name (free RAM $(free -g | awk '/^Mem:/{print $7}') GB)"
$PY -m tensorrt_edgellm.scripts.export "$WS/ckpt/$name" "$m/onnx" > "$WS/export_${name}_patched.log" 2>&1; log "T1c: export $name rc=$?"
if [ -f "$m/onnx/llm/config.json" ]; then
  $EL/build/examples/llm/llm_build --onnxDir "$m/onnx/llm" --engineDir "$m/engines/llm" --maxBatchSize 1 --maxInputLen 2048 --maxKVCacheCapacity 2048 > "$m/llm_build.log" 2>&1; log "T1c: llm_build rc=$?"
  $EL/build/examples/multimodal/visual_build --onnxDir "$m/onnx/visual" --engineDir "$m/engines" --minImageTokens 64 --maxImageTokens 1024 --maxImageTokensPerImage 1024 > "$m/visual_build.log" 2>&1; log "T1c: visual_build rc=$?"
fi
[ -f "$m/engines/llm/config.json" ] && bash studies/d1_edgellm_pilot.sh uitars7b > runs/d1_edgellm_pilot_uitars.log 2>&1 || log "T1c: no UI-TARS engines; gym rows skipped"
[ -f "$WS/Qwen3.8-27B/engines/llm/config.json" ] && bash studies/d1_edgellm_pilot.sh qwen38_27b > runs/d1_edgellm_pilot_qwen38.log 2>&1 || log "T1c: no 27B engines; gym rows skipped"
log "T1c done"
