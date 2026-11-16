#!/usr/bin/env bash
# Fara1.5 rows after the engines exist: 10 episodes on card/balloon/static (edge_llm in-process, prompt 21), then 100.
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueFARAROWS: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
until grep -q "queueFARA: done" "$LOG"; do sleep 60; done
for tag in fara9b fara4b; do
  name=$([ $tag = fara9b ] && echo Fara1.5-9B || echo Fara1.5-4B); m=$HOME/tensorrt-edgellm-workspace/$name
  { [ -f "$m/engines/llm/config.json" ] && ls "$m/engines" | grep -qi visual; } || { say "$name: no engines; skipped"; continue; }
  say "probe $name on a notification frame"; $PY - "$m/engines" "$name" <<'PYEOF' 2>&1 | grep -v "^\[\|^Using" | tail -3 | tee -a "$LOG"
import sys; from chronotope.controller import SYSTEM_PROMPTS, UITARS_INSTRUCTION_NOWAIT, parse_evocua; from chronotope.png import encode_png; from chronotope.scenario import manifest_from_dict; from chronotope_core.edgellm import EdgeLLMAdapter
ad = EdgeLLMAdapter(engines_root=sys.argv[1], model_id=sys.argv[2], max_tokens=256, temperature=0.0)
for seed in (200, 201, 202):
    m = manifest_from_dict({"scenario_id":"revealed_target","scenario_seed":seed,"canvas_w_px":320,"canvas_h_px":240,"target_size_px":24,"event_time_ns":500000000,"response_window_ns":8000000000,"frame_period_ns":16666667,"episode_horizon_ns":9000000000,"irrelevant_motion":False,"skin":"notification"})
    r = ad.invoke(encode_png(m.render(600000000),320,240,2), SYSTEM_PROMPTS[21], UITARS_INSTRUCTION_NOWAIT["notification"], structured=False, parser=parse_evocua)
    cx, cy = m.target_box().center(); print(f"queueFARAROWS: probe seed {seed}: parsed={r.parsed} target grid=({cx*1000//320},{cy*1000//240}) {r.detail['runtime_ns']/1e6:.0f} ms raw={r.detail['raw_text'][-120:]!r}")
PYEOF
  for fam in closing_window moving_target static_control; do
    case $fam in moving_target|static_control) SK=_balloon;; *) SK=;; esac
    for suf in detector late_bind frozen self_grounder; do spec=edgellm_${tag}_${suf}; [ -d runs/d1/$fam/$spec ] && continue; say "$fam / $spec (10)"
      $D1 run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay,paused --allow-small --max-calls 2000 --out runs/d1/$fam > runs/d1_${fam}_${spec}.log 2>&1 || say "$fam/$spec exited $?"; done
  done
done
for fam in closing_window moving_target static_control; do runs=$(for d in runs/d1/$fam/*/; do d=${d%/}; case $(basename $d) in payloads|*delib1000*|*delib5200*) continue;; esac; [ "$(wc -l < $d/outcomes.jsonl 2>/dev/null)" = "30" ] && echo -n "$d "; done); $D1 report --family $fam --runs $runs --baseline kgua_late_bind_delib0 --manifest runs/d1/manifests/$fam.jsonl --extra-law 1,64 --out results/D1_${fam}_pilot.md > runs/d1_report_$fam.log 2>&1 || say "report $fam failed"; done
$D1 master --out results/D1_master.md > /dev/null 2>&1; $D1 expanded --out results/D1_expanded.md > /dev/null 2>&1
# the 100-episode pass for Fara (card + balloon, live+replay)
for tag in fara9b fara4b; do for fam in closing_window moving_target; do case $fam in moving_target) SK=_balloon;; *) SK=;; esac; for suf in detector late_bind frozen; do spec=edgellm_${tag}_${suf}; [ -f specs/d1/$spec$SK.json ] || continue; [ -d runs/d1_100/$fam/$spec ] && continue; say "$fam / $spec (100)"
  $D1 run --manifest runs/d1/manifests/${fam}_100.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay --max-calls 20000 --out runs/d1_100/$fam > runs/d1_100_${fam}_${spec}.log 2>&1 || say "$fam/$spec (100) exited $?"; done; done; done
say "done"
