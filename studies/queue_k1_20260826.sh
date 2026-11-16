#!/usr/bin/env bash
# queueK1 — K1c (cards/K1-chronotope-control-core.md, U-K3): the binder's latency in the error law, 10 seeds per cell.
# Waits for M4b (the last queued model study) so that the wall-clock timing has the GPU to itself, then runs the
# scripted (detector-planner) cells: no model is served. NOT launched by the build session (2026-08-26): slot it.
#
#   setsid nohup bash studies/queue_k1_20260826.sh > runs/queue_k1.log 2>&1 & disown
set -u
cd "$(dirname "$0")/.."
LOG=runs/pipeline.log
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueK1: $*" | tee -a "$LOG"; }
log "waiting for M4b"
while ! grep -q "\] queueM4b done$" "$LOG" 2>/dev/null; do sleep 120; done
# no other study may be running (anchored guards on the study scripts, never on a queue's bash pid)
while pgrep -f '^\S*python\S* studies/(r1_revealed_target|c1_balloon_drift|g2_live_page|a1_sponsor|b1_live|k1_kgua_cell)\.py' > /dev/null; do
  log "a study is still running; waiting"; sleep 60
done
if [ ! -f chronotope/build/libchronotope.so ]; then
  log "building chronotope"
  (cmake -S chronotope -B chronotope/build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc >/dev/null && make -C chronotope/build -j8 >/dev/null) || { log "chronotope build FAILED"; echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueK1 done" >> "$LOG"; exit 1; }
fi
log "k1_kgua_cell: velocities 8,100,200 x {tracker_py, cuda} late_bind + cuda frozen, 10 seeds, deliberation 1.0 s, window 1.3 s"
PYTHONPATH=src:chronotope/python .venv/bin/python studies/k1_kgua_cell.py --out runs/k1_kgua_cell_10 --n-seeds 10 --seed-start 1000 \
    --velocities 8,100,200 --deliberation-s 1.0 --window-s 1.3 --results results/K1_residency_cell_100.md > runs/k1_kgua_cell_10.log 2>&1
log "k1_kgua_cell rc=$?"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] queueK1 done" >> "$LOG"
