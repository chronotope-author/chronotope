#!/usr/bin/env bash
# Re-export UI-TARS-1.5-7B's visual encoder with the FP16Linear cast patch, build the visual engine, then run the
# chronotope + Edge-LLM D1 rows: first the 0.8B and 27B balloon-family reruns (prompt 5), then UI-TARS on all three.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
log() { echo "[$(date '+%F %T')] queueT1: $*" | tee -a "$LOG"; }
WS=$HOME/tensorrt-edgellm-workspace; EL=$HOME/TensorRT-Edge-LLM; m=$WS/UI-TARS-1.5-7B
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
log "T1d: UI-TARS visual export with the FP16Linear cast (CPU) while the 0.8B/27B balloon reruns use the GPU"
( $PY -m tensorrt_edgellm.scripts.export --skip-llm "$WS/ckpt/UI-TARS-1.5-7B" "$m/onnx" > "$WS/export_UI-TARS-1.5-7B_visual.log" 2>&1; echo "[$(date '+%F %T')] queueT1: T1d: visual export rc=$?" >> "$LOG" ) &
bash studies/d1_edgellm_pilot.sh qwen35_0p8b > runs/d1_edgellm_pilot_0p8b_balloon.log 2>&1
bash studies/d1_edgellm_pilot.sh qwen38_27b > runs/d1_edgellm_pilot_27b.log 2>&1
wait
if [ -f "$m/onnx/visual/config.json" ]; then
  $EL/build/examples/multimodal/visual_build --onnxDir "$m/onnx/visual" --engineDir "$m/engines" --minImageTokens 64 --maxImageTokens 1024 --maxImageTokensPerImage 1024 > "$m/visual_build.log" 2>&1; log "T1d: visual_build rc=$?"
fi
if [ -f "$m/engines/visual/config.json" ] || ls "$m/engines" | grep -qi visual; then bash studies/d1_edgellm_pilot.sh uitars7b > runs/d1_edgellm_pilot_uitars.log 2>&1; else log "T1d: no UI-TARS visual engine; its rows skipped"; fi
log "T1d done"
