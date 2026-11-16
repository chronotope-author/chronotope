#!/usr/bin/env bash
# Fara1.5-9B and -4B (Microsoft, MIT, base Qwen3.5) -> TensorRT Edge-LLM engines (the qwen3_5 export path that built the 0.8B and 27B).
set -u; cd ~/chronotope; LOG=runs/pipeline.log
log() { echo "[$(date '+%F %T')] queueFARA: $*" | tee -a "$LOG"; }
WS=$HOME/tensorrt-edgellm-workspace; EL=$HOME/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
until grep -q "downloaded microsoft/Fara1.5-4B" ~/fara/download.log 2>/dev/null; do sleep 60; done
for name in Fara1.5-9B Fara1.5-4B; do
  snap=$(ls -d $HOME/.cache/huggingface/hub/models--microsoft--$name/snapshots/*/ | head -1); m=$WS/$name
  [ -d "$WS/ckpt/$name" ] || { mkdir -p $WS/ckpt; cp -rlL "$snap" "$WS/ckpt/$name" 2>/dev/null || cp -rL "$snap" "$WS/ckpt/$name"; }
  log "export $name (free RAM $(free -g | awk '/^Mem:/{print $7}') GB)"
  [ -f "$m/onnx/llm/config.json" ] || $PY -m tensorrt_edgellm.scripts.export "$WS/ckpt/$name" "$m/onnx" > "$WS/export_$name.log" 2>&1; log "export $name rc=$?"
  if [ -f "$m/onnx/llm/config.json" ]; then
    [ -f "$m/engines/llm/config.json" ] || $EL/build/examples/llm/llm_build --onnxDir "$m/onnx/llm" --engineDir "$m/engines/llm" --maxBatchSize 1 --maxInputLen 4096 --maxKVCacheCapacity 4096 > "$m/llm_build.log" 2>&1; log "llm_build $name rc=$?"
    ls "$m/engines" | grep -qi visual || $EL/build/examples/multimodal/visual_build --onnxDir "$m/onnx/visual" --engineDir "$m/engines" --minImageTokens 64 --maxImageTokens 2048 --maxImageTokensPerImage 2048 > "$m/visual_build.log" 2>&1; log "visual_build $name rc=$?"
  fi
  log "$name engines: $([ -f "$m/engines/llm/config.json" ] && ls "$m/engines" | grep -qi visual && echo built || echo MISSING)"
done
log "done"
