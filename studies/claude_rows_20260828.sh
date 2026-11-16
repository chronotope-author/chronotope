#!/usr/bin/env bash
# Claude planners (opus-5, sonnet-5, fable-5) through the Messages API, in the host Python controller and in the CUDA
# core (ctypes path): 10 episodes on card/balloon/static, then 100 (card + balloon, live+replay).
# Needs: ~/.anthropic_key (0600) and ANTHROPIC_WORKSPACE_ID exported (the key is identity-linked).
#   ANTHROPIC_WORKSPACE_ID=wrkspc_... bash studies/claude_rows_20260828.sh
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueCLAUDE: $*" | tee -a "$LOG"; }
[ -n "${ANTHROPIC_WORKSPACE_ID:-}" ] || { echo "set ANTHROPIC_WORKSPACE_ID (the key is identity-linked)"; exit 2; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
say "probe: one frame per model"; $PY - <<'PYEOF' 2>&1 | tail -3 | tee -a "$LOG"
import sys; sys.path.insert(0, "src")
from chronotope.anthropic_adapter import AnthropicAdapter; from chronotope.controller import SYSTEM_PROMPTS; from chronotope.png import encode_png; from chronotope.scenario import manifest_from_dict
m = manifest_from_dict({"scenario_id":"revealed_target","scenario_seed":200,"canvas_w_px":320,"canvas_h_px":240,"target_size_px":24,"event_time_ns":500000000,"response_window_ns":8000000000,"frame_period_ns":16666667,"episode_horizon_ns":9000000000,"irrelevant_motion":False,"skin":"notification"})
png = encode_png(m.render(600000000), 320, 240, 2); cx, cy = m.target_box().center()
for model in ("claude-opus-5", "claude-sonnet-5", "claude-fable-5"):
    try:
        r = AnthropicAdapter(model_id=model, max_tokens=256).invoke(png, SYSTEM_PROMPTS[4].format(w=640, h=480), "Screenshot attached. Click the target's center or wait.", structured=True)
        print(f"queueCLAUDE: {model}: {r.parsed} target=({cx*1000//320},{cy*1000//240}) {r.detail['runtime_ns']/1e6:.0f} ms")
    except Exception as e:
        print(f"queueCLAUDE: {model}: ERROR {type(e).__name__}: {str(e)[:160]}")
PYEOF
run() { local fam=$1 man=$2 spec=$3 name=$4 conds=$5 root=$6; [ -d $root/$fam/$name ] && { say "$fam/$name exists"; return; }; say "$fam / $name ($conds)"
  $D1 run --manifest runs/d1/manifests/$man.jsonl --controller-json specs/d1/$spec.json --name $name --conditions $conds --allow-small --max-calls 6000 --out $root/$fam > runs/${root##*/}_${fam}_${name}.log 2>&1 || say "$fam/$name exited $?"; }
for tag in opus5 sonnet5 fable5; do
  run closing_window closing_window claude_${tag}_direct claude_${tag}_direct live,replay,paused runs/d1
  for suf in detector late_bind frozen; do run closing_window closing_window core_claude_${tag}_${suf} core_claude_${tag}_${suf} live,replay,paused runs/d1; done
  for fam in moving_target static_control; do
    for pol in late_bind frozen; do run $fam $fam claude_${tag}_${pol}_balloon claude_${tag}_${pol} live,replay,paused runs/d1; done
    for suf in detector late_bind frozen; do run $fam $fam core_claude_${tag}_${suf}_balloon core_claude_${tag}_${suf} live,replay,paused runs/d1; done
  done
done
for fam in closing_window moving_target static_control; do runs=$(for d in runs/d1/$fam/*/; do d=${d%/}; case $(basename $d) in payloads|*delib1000*|*delib5200*) continue;; esac; [ "$(wc -l < $d/outcomes.jsonl 2>/dev/null)" = "30" ] && echo -n "$d "; done); $D1 report --family $fam --runs $runs --baseline kgua_late_bind_delib0 --manifest runs/d1/manifests/$fam.jsonl --extra-law 1,64 --out results/D1_${fam}_pilot.md > runs/d1_report_$fam.log 2>&1 || say "report $fam failed"; done
$D1 master --out results/D1_master.md > /dev/null 2>&1; $D1 expanded --out results/D1_expanded.md > /dev/null 2>&1
say "10-episode Claude rows done; starting 100"
for tag in opus5 sonnet5 fable5; do
  run closing_window closing_window_100 claude_${tag}_direct claude_${tag}_direct live,replay runs/d1_100
  for suf in detector late_bind frozen; do run closing_window closing_window_100 core_claude_${tag}_${suf} core_claude_${tag}_${suf} live,replay runs/d1_100; done
  for pol in late_bind frozen; do run moving_target moving_target_100 claude_${tag}_${pol}_balloon claude_${tag}_${pol} live,replay runs/d1_100; done
  for suf in detector late_bind frozen; do run moving_target moving_target_100 core_claude_${tag}_${suf}_balloon core_claude_${tag}_${suf} live,replay runs/d1_100; done
done
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1; $D1 expanded --runs-root runs/d1_100 --out results/D1_expanded_100.md > /dev/null 2>&1
say "done"
