#!/usr/bin/env bash
# 2026-08-28b: rerun of overnight_100_20260828.sh; every cell had exited 1 because --allow-small was missing (N_MIN=300).
# Overnight (2026-08-28): the 100-episode pass on the D1 gym, no vLLM. Order: the resident detector rows (all families,
# three conditions), the memory arms at 100, the 0.8B in the C++ agent process (closing window: live+replay+paused;
# balloon families: live+replay), then the 27B through the Python binding (live+replay). Reports + master + expanded after.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queue100b: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
while pgrep -f "studies/(d1_|k2_|c1_|r1_)[a-z0-9_]*\.(py|sh)" > /dev/null; do sleep 60; done
run() { # family manifest spec name conditions maxcalls
  local fam=$1 man=$2 spec=$3 name=$4 conds=$5 mc=$6
  [ -d runs/d1_100/$fam/$name ] && [ "$(wc -l < runs/d1_100/$fam/$name/outcomes.jsonl 2>/dev/null || echo 0)" -ge 100 ] && { say "$fam/$name exists"; return; }
  rm -rf runs/d1_100/$fam/$name; say "$fam / $name ($conds)"
  $D1 run --manifest runs/d1/manifests/$man.jsonl --controller-json specs/d1/$spec.json --name $name --conditions $conds --allow-small --max-calls $mc --out runs/d1_100/$fam > runs/d1_100_${fam}_${name}.log 2>&1 || say "$fam/$name exited $?"
}
say "start (100 episodes per cell)"
# 1. the resident detector rows
for fam in closing_window moving_target static_control; do for s in kgua_late_bind_delib0 kgua_frozen_delib0; do run $fam ${fam}_100 $s $s live,replay,paused 0; done; done
run moving_target moving_target_100 kgua_detector_detector kgua_detector_detector live,replay,paused 0
# 2. the memory arms (detector planner, C++ process)
for a in mem_none mem_pending mem_pending_stale1s mem_pending_ack; do run pending_effect pending_effect_100 cpp_detector_$a cpp_detector_$a live 20000; done
# 3. the 0.8B in the C++ agent process
for s in cpp_qwen35_0p8b_detector cpp_qwen35_0p8b_late_bind cpp_qwen35_0p8b_frozen; do run closing_window closing_window_100 $s $s live,replay,paused 40000; done
for fam in moving_target static_control; do for s in cpp_qwen35_0p8b_detector cpp_qwen35_0p8b_late_bind cpp_qwen35_0p8b_frozen; do run $fam ${fam}_100 ${s}_balloon $s live,replay 40000; done; done
for a in mem_none mem_pending mem_pending_ack; do run pending_effect pending_effect_100 cpp_qwen35_0p8b_detector_$a cpp_qwen35_0p8b_detector_$a live 40000; done
# 4. the 27B through the Python binding (live+replay: 100 x 2 x ~6 s ~ 20 min per row)
for s in edgellm_qwen38_27b_detector edgellm_qwen38_27b_late_bind edgellm_qwen38_27b_frozen; do run closing_window closing_window_100 $s $s live,replay 4000; done
for s in edgellm_qwen38_27b_detector edgellm_qwen38_27b_late_bind edgellm_qwen38_27b_frozen; do run moving_target moving_target_100 ${s}_balloon $s live,replay 4000; done
# 5. reports
for fam in closing_window moving_target static_control; do runs=$(ls -d runs/d1_100/$fam/*/ 2>/dev/null | sed 's#/$##' | grep -v payloads | tr '\n' ' '); [ -n "$runs" ] && $D1 report --family $fam --runs $runs --baseline kgua_late_bind_delib0 --manifest runs/d1/manifests/${fam}_100.jsonl --extra-law 1,16 1,64 --out results/D1_${fam}_100.md > runs/d1_100_report_$fam.log 2>&1 || say "report $fam failed"; done
runs=$(ls -d runs/d1_100/pending_effect/*/ 2>/dev/null | sed 's#/$##' | grep -v payloads | tr '\n' ' '); [ -n "$runs" ] && $D1 report --family pending_effect --runs $runs --baseline cpp_detector_mem_none --manifest runs/d1/manifests/pending_effect_100.jsonl --out results/D1_pending_effect_100.md > runs/d1_100_report_pending.log 2>&1 || say "pending report failed"
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1 || say "master failed"
$D1 expanded --runs-root runs/d1_100 --out results/D1_expanded_100.md > /dev/null 2>&1 || say "expanded failed"
$D1 expanded --runs-root runs/d1 --out results/D1_expanded.md > /dev/null 2>&1
say "done"
