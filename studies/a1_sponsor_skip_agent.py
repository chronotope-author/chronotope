"""A1 sponsor skip (agent) — card cards/A1-sponsor-skip-agent.md. Paused competence gate on the player skin, then
Paused / Replay / Live across response windows with the 27B model; every cell reports (SR, mean G_A).

  PYTHONPATH=src .venv/bin/python studies/a1_sponsor_skip_agent.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/a1_qwen38_nothink_pilot10 --max-calls 600
  (pilot: seeds 200..209; the full run passes --n-seeds 100)

Predeclared: scenario sponsor_skip, skin player 320x240, target 24 px (button 60x24), event 500 ms, seek_delay 1 s,
60 Hz, publication 30 ms, delivery 20 ms; controller vlm_direct, prompt 9 (0-1000 grid), image scale 2;
windows {1, 2, 4, 8, 16} s; calibration seeds 100..109 (Paused, window 16 s).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, scenario, spec, vlm  # noqa: E402

WINDOWS_S = [1, 2, 4, 8, 16]
SEEK_NS = 1 * S


def main() -> int:
    ap = agent_argparser("A1 sponsor skip (agent): gate + P/R/L window sweep", seed_start=200)
    ap.add_argument("--windows", default=",".join(map(str, WINDOWS_S)))
    ap.add_argument("--seek-s", type=float, default=1.0)
    ap.add_argument("--skin", default="player", choices=["player", "player_static"],
                    help="player_static = the paired control with no pre-event motion (Q3, 2026-08-24)")
    a = ap.parse_args()
    st = Study(a, "A1 sponsor skip (agent) — Paused / Replay / Live across windows")
    seek = int(a.seek_s * S)
    ctrl = vlm(a, "vlm_direct", 9)

    def paused_spec(seed):
        return spec(scenario("sponsor_skip", seed, a.skin, 16 * S, seek_delay_ns=seek), ctrl, policy="paused")

    if not st.gate(paused_spec, list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib)), f"{a.skin} skin, prompt 9"):
        return 1
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    st.say(f"## Triplets (seeds {seeds[0]}..{seeds[-1]}; seek delay {a.seek_s:g} s)")
    for w in [int(x) for x in a.windows.split(",")]:
        for seed in seeds:
            st.triplet(spec(scenario("sponsor_skip", seed, a.skin, w * S, seek_delay_ns=seek), ctrl), f"w{w}", seed)
    st.finish()
    out_md = Path("results") / f"A1_{a.out.name.replace('a1_', '')}.md"
    subprocess.run([sys.executable, "studies/analyze_prl.py", str(a.out), str(out_md), "--title", "A1 sponsor skip (agent)",
                    "--cell-order", ",".join(f"w{w}" for w in [int(x) for x in a.windows.split(",")])]
                   + (["--pilot"] if a.n_seeds < 100 else []), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
