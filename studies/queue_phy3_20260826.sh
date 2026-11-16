#!/usr/bin/env bash
# PHY3 queue (2026-08-26; cards/PHY3-system-cell.md). Waits for the 'queueM3b done' marker, then for device access
# (studies/phy_preflight.py rc 0; until then its access instructions are appended here every 10 min under the marker
# "queuePHY3: waiting for device access"), then: instrument (phy_measure.py, gates c1-c5) -> empirical-replay B1 cell
# (software route, measured D->A replayed; CPU) -> system cells on the hardware rig: B1 oracle (100 trials), C1 8 px/s
# with the scripted oracle, then C1 with the model (Qwen/Qwen3.8-27B at http://127.0.0.1:8000/v1, --no-thinking,
# prompt 5 as in C1) x {frozen, late_bind} x 100 seeds. Markers "queuePHY3 ...". Gate on markers and study processes,
# never on queue bash pids. All paths relative to ~/chronotope. The target (Windows laptop) must show the served page
# full-screen for the preflight's null test and for every hardware cell; the preflight prints the URL.
set -u
cd ~/chronotope
LOG=runs/pipeline.log
DATE=20260826
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
PY="env PYTHONPATH=src .venv/bin/python"
GUARD='studies/(c1_|c2_|c3_|c4_|g1_|g2_|p1_|r1_|a1_|a2_|a8_|b1_|c_prl|o2_|o3_|v1_|phy_|l1_|wi010)[a-z_]*\.py'
BASE=http://127.0.0.1:8000/v1
MODEL=Qwen/Qwen3.8-27B
DISPLAY_PROFILE="${PHY_DISPLAY_PROFILE:-target laptop 1920x1080@60, 100% scaling, browser full-screen (F11); pinned for the session}"

wait_idle() { while pgrep -f "$GUARD" > /dev/null; do sleep 30; done; }

say "queuePHY3: waiting for M3d"
while ! grep -q "\] queueM3d done$" "$LOG"; do sleep 60; done
wait_idle

# ---- device access gate: poll the preflight every 10 min --------------------------------------------------------
while true; do
  wait_idle
  out=$($PY studies/phy_preflight.py --null-s 60 --stream-s 10 --fps 30 --out "runs/phy_preflight_$DATE.json" 2>&1); rc=$?
  if [ "$rc" -eq 0 ]; then say "queuePHY3: preflight passed"; echo "$out" >> "$LOG"; break; fi
  if [ "$rc" -eq 2 ]; then
    say "queuePHY3: waiting for device access"
    echo "$out" | tee -a "$LOG"
  else
    say "queuePHY3: preflight failed (rc $rc); retrying in 10 min"
    echo "$out" | tail -20 | tee -a "$LOG"
  fi
  sleep 1800
done

# ---- instrument -----------------------------------------------------------------------------------------------
wait_idle
say "queuePHY3: instrument (phy_measure.py, NV12 30 Hz, null 600 s, 100 events + 30 known-delay)"
$PY studies/phy_measure.py --out "runs/phy_$DATE" --name "$DATE" --display-profile "$DISPLAY_PROFILE" \
    --null-s 600 --events 100 --known-delay-ms 100 --known-delay-events 30 > "runs/phy_$DATE.log" 2>&1 || say "queuePHY3: instrument exited $?"
ROUTE="results/PHY_route_$DATE.json"
if [ ! -f "$ROUTE" ]; then say "queuePHY3 FAIL: $ROUTE not written (gates c1-c5 failed; see runs/phy_$DATE/)"; exit 1; fi
say "queuePHY3: instrument done: $ROUTE"

# ---- system cells -----------------------------------------------------------------------------------------------
# B1: the cell runs the empirical-replay half itself (same seeds on the software route with the measured D->A).
wait_idle
say "queuePHY3: B1 jitter validation, oracle, 100 trials (physical + empirical replay)"
$PY studies/phy_gym_cell.py --cell b1 --route-json "$ROUTE" --display-profile "$DISPLAY_PROFILE" \
    --seed-start 1000 --n-seeds 100 --out "runs/phy3_b1_$DATE" --results-name "$DATE" > "runs/phy3_b1_$DATE.log" 2>&1 || say "queuePHY3: B1 exited $?"
wait_idle
say "queuePHY3: C1 8 px/s, scripted oracle, 100 seeds x {frozen, late_bind}"
$PY studies/phy_gym_cell.py --cell c1 --route-json "$ROUTE" --display-profile "$DISPLAY_PROFILE" \
    --seed-start 300 --n-seeds 100 --out "runs/phy3_c1_oracle_$DATE" --results-name "oracle_$DATE" > "runs/phy3_c1_oracle_$DATE.log" 2>&1 || say "queuePHY3: C1 oracle exited $?"
wait_idle
if curl -s -m 5 "$BASE/models" | grep -q "\"$MODEL\""; then
  for pol in frozen late_bind; do
    wait_idle
    say "queuePHY3: C1 8 px/s, $MODEL no-thinking prompt 5, $pol, 100 seeds"
    $PY studies/phy_gym_cell.py --cell c1 --route-json "$ROUTE" --display-profile "$DISPLAY_PROFILE" \
        --model "$MODEL" --base-url "$BASE" --no-thinking --prompt-version 5 --policy "$pol" \
        --seed-start 300 --n-seeds 100 --max-calls 2000 --out "runs/phy3_c1_${pol}_$DATE" --results-name "${pol}_$DATE" \
        > "runs/phy3_c1_${pol}_$DATE.log" 2>&1 || say "queuePHY3: C1 $pol exited $?"
  done
else
  say "queuePHY3: model server $MODEL not up at $BASE; C1 model cells skipped (rerun the two C1 --model lines by hand)"
fi
say "queuePHY3 done"
