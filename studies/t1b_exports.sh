set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
WS=$HOME/tensorrt-edgellm-workspace
EL=$HOME/TensorRT-Edge-LLM
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
export EDGELLM_PLUGIN_PATH=$EL/build/libNvInfer_edgellm_plugin.so
PY=$HOME/edgellm-venv/bin/python
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueT1: $*" | tee -a "$LOG"; }
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
UT=$(ls -d ~/.cache/huggingface/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/*/ 2>/dev/null | head -1)
Q27=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/ 2>/dev/null | head -1)
log "T1b start: exports + engines for UI-TARS-1.5-7B then Qwen3.8-27B (CPU export, GPU build; no vLLM)"
for spec in "UI-TARS-1.5-7B:$UT:50" "Qwen3.8-27B:$Q27:80"; do
  name=${spec%%:*}; rest=${spec#*:}; snap=${rest%%:*}; need=${rest##*:}
  [ -n "$snap" ] || { log "T1b: $name not in the HF cache; skipped"; continue; }
  if [ "$(free_gb)" -lt "$need" ]; then log "T1b: $name needs ~$need GB free RAM for export, have $(free_gb): skipped"; continue; fi
  export_ckpt "$name" "$snap" && build_engines "$name" 2048
  log "T1b: $name engines: $([ -f "$WS/$name/engines/llm/config.json" ] && echo built || echo MISSING)"
done
log "T1b done"
