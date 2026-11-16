#!/usr/bin/env bash
# K2 overnight lanes (cards/K2-resident-sweep.md). Three detached lanes share the GPU with G2/M4b/phy3 (user's direction).
#   studies/queue_k2_20260826.sh A|B|C
# A: r1 (1500 trials) + c2 (400)      ~3.5 h
# B: c1 (1400) + c3 (200) + k1c (900) ~3.5 h
# C: p1 (800) + p1_kind (300)         ~5.5 h (17 s episodes)
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:chronotope/python
lane="${1:?lane A|B|C}"
case "$lane" in
  A) parts="r1 c2" ;;
  B) parts="c1 c3 k1c" ;;
  C) parts="p1 p1_kind" ;;
  *) echo "unknown lane $lane"; exit 2 ;;
esac
echo "$(date '+%F %T') k2 lane $lane start pid=$$ parts=$parts" >> pipeline.log
for p in $parts; do
  .venv/bin/python studies/k2_resident_sweep.py --part "$p" --seeds 100 --seed0 1000 > "runs/k2_${p}.log" 2>&1
  echo "$(date '+%F %T') k2 lane $lane part $p exit=$?" >> pipeline.log
done
echo "$(date '+%F %T') k2 lane $lane done" >> pipeline.log
