#!/usr/bin/env bash
# D1 gym, the CUDA-resident software control, 10-episode manifests (2026-08-27 pilot). Runs beside the GPU queue
# (the detector planner is light). Families with click actions only (the rotating-code families need typed actions).
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"
say "queueD1R: resident agent on the D1 gym (closing_window, moving_target, static_control; delib 0/1000/5200 ms; late_bind + frozen)"
for fam in closing_window moving_target static_control; do
  for ms in 0 1000 5200; do
    for pol in late_bind frozen; do
      name=chronotope_${pol}_delib${ms}
      [ -d runs/d1/$fam/$name ] && { say "queueD1R: $fam/$name exists, skipped"; continue; }
      $PY studies/d1_distributional_gym.py run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$name.json \
          --name $name --conditions live,replay,paused --allow-small --out runs/d1/$fam > runs/d1_${fam}_${name}.log 2>&1 \
          || say "queueD1R: $fam/$name exited $?"
    done
  done
  say "queueD1R: $fam done"
done
say "queueD1R done"
