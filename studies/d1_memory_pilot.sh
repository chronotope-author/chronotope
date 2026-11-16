#!/usr/bin/env bash
# The memory axis on the D1 pending-effect family (delay ~ log-uniform 0.5–8 s): the C++ agent process with the detector
# planner (then the 0.8B in-process) × memory {none, pending record, pending + staleness gate, pending + ack-aware}.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueD1M: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
run() { [ -d runs/d1/pending_effect/$1 ] && { say "$1 exists"; return; }; say "pending_effect / $1"; $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/pending_effect.jsonl --controller-json specs/d1/$1.json --name $1 --conditions live --allow-small --max-calls 2000 --out runs/d1/pending_effect > runs/d1_pending_effect_$1.log 2>&1 || say "$1 exited $?"; }
say "memory axis: detector planner arms"
for a in mem_none mem_pending mem_pending_stale1s mem_pending_ack; do run cpp_detector_$a; done
say "memory axis: 0.8B in-process arms"
for a in mem_none mem_pending mem_pending_stale1s mem_pending_ack; do run cpp_qwen35_0p8b_detector_$a; done
runs=$(ls -d runs/d1/pending_effect/*/ | sed 's#/$##' | grep -v payloads | tr '\n' ' ')
$PY studies/d1_distributional_gym.py report --family pending_effect --runs $runs --baseline cpp_detector_mem_none --manifest runs/d1/manifests/pending_effect.jsonl --out results/D1_pending_effect_pilot.md > runs/d1_report_pending_effect.log 2>&1 || say "report failed: $(tail -1 runs/d1_report_pending_effect.log | cut -c1-160)"
say "done"
