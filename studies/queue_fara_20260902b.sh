#!/usr/bin/env bash
# Fara1.5 rows, continuation of queueFARAc (2026-09-02, marker queueFARAd). queueFARAc was paused at 09:04 after its
# 9B closing_window pilot cells came back 0/10 in Live, Replay AND Paused: with the parser fixed, every 9B first
# decision on the card frame says "No notification card is currently visible" (0/130 across queueFARAb/c), while the
# 4B clicks the card (118/120) and the 9B clicks the balloon (120/120). studies/fara_card_diag.py asks the engine
# directly (runs/fara9b_card_diag.jsonl). The 10-episode rows finish as before (cells with outcomes.jsonl are kept);
# the 100-episode stage runs the 9B on the balloon only and the 4B on card + balloon: a 100-episode 9B card cell would
# repeat a 0/0/0 row (the 10-episode rows document it; 100 episodes are for cells that will be cited).
set -u; cd ~/chronotope; LOG=runs/pipeline.log
say() { echo "[$(date '+%F %T')] queueFARAd: $*" | tee -a "$LOG"; }
export LD_LIBRARY_PATH=$HOME/trt-prefix/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}; export EDGELLM_PLUGIN_PATH=$HOME/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
PY="env PYTHONPATH=src:chronotope/python .venv/bin/python"; D1="$PY studies/d1_distributional_gym.py"
while pgrep -f "studies/d1_distributional_gym.py" > /dev/null; do sleep 60; done
while pgrep -f "studies/fara_card_diag.py" > /dev/null; do sleep 30; done
say "start"
for tag in fara9b fara4b; do
  for fam in closing_window moving_target static_control; do
    case $fam in moving_target|static_control) SK=_balloon;; *) SK=;; esac
    for suf in detector late_bind frozen self_grounder; do spec=edgellm_${tag}_${suf}
      [ -f runs/d1/$fam/$spec/outcomes.jsonl ] && { say "$fam/$spec exists"; continue; }
      rm -rf runs/d1/$fam/$spec; say "$fam / $spec (10)"
      $D1 run --manifest runs/d1/manifests/$fam.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay,paused --allow-small --max-calls 2000 --out runs/d1/$fam > runs/d1_${fam}_${spec}.log 2>&1 || say "$fam/$spec exited $?"
    done
  done
done
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
say "pilot rows done"
for cell in "fara9b moving_target" "fara4b closing_window" "fara4b moving_target"; do set -- $cell; tag=$1; fam=$2
  case $fam in moving_target) SK=_balloon;; *) SK=;; esac
  for suf in detector late_bind frozen; do spec=edgellm_${tag}_${suf}
    [ -f runs/d1_100/$fam/$spec/outcomes.jsonl ] && [ "$(wc -l < runs/d1_100/$fam/$spec/outcomes.jsonl)" -ge 100 ] && { say "$fam/$spec (100) exists"; continue; }
    rm -rf runs/d1_100/$fam/$spec; say "$fam / $spec (100)"
    $D1 run --manifest runs/d1/manifests/${fam}_100.jsonl --controller-json specs/d1/$spec$SK.json --name $spec --conditions live,replay --allow-small --max-calls 20000 --out runs/d1_100/$fam > runs/d1_100_${fam}_${spec}.log 2>&1 || say "$fam/$spec (100) exited $?"
  done
done
$D1 master --runs-root runs/d1 --out results/D1_master.md > /dev/null 2>&1 || say "master (pilot) failed"
$D1 master --runs-root runs/d1_100 --out results/D1_master_100.md > /dev/null 2>&1 || say "master (100) failed"
say "done"
