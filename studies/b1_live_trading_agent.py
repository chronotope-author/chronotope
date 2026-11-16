"""B1 live trading (agent) — card cards/B1-live-trading-agent.md. Paused competence gate on the chart skin, then the
reversal regimes × Paused / Replay / Live with the 27B model; every cell reports (SR, mean G_B).

  PYTHONPATH=src .venv/bin/python studies/b1_live_trading_agent.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/b1_qwen38_nothink_pilot10 --max-calls 500

Predeclared: scenario live_trading, skin chart 320x240, target 24 px (button 60x24), event 500 ms, expiry 16 s,
utilities (+1, -1, 0), 60 Hz, publication 30 ms, delivery 20 ms; controller vlm_direct, prompt 10, image scale 2;
regimes: rnone (no reversal), r2000 (reversal 2.0 s after the signal: before D, observed), r5200 (5.2 s ≈ the median
receipt: straddle), r10000 (10 s: after A); calibration seeds 100..109 (Paused, no reversal).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, scenario, spec, vlm  # noqa: E402

EXPIRY_NS = 16 * S
# regime -> (reversal ms after the signal, delivery delay ns). Pilot (2026-08-24): the agent's click decision starts at
# ~1.9 s (its first decision is spent on a pre-signal frame: the price line animates), so D ≈ E + 6.7 s and A = D + 20 ms.
# r6680 straddles the median D on the 20-ms route (per-trial split by deliberation jitter); r6930s puts the reversal
# inside (D, A] by using a 500-ms route — invalidated_after_commitment at the model's timescale.
REGIMES = {"rnone": (0, 20_000_000), "r2000": (2000, 20_000_000), "r5200": (5200, 20_000_000),
           "r6680": (6680, 20_000_000), "r6930s": (6930, 500_000_000), "r10000": (10000, 20_000_000)}
PILOT_REGIMES = "rnone,r2000,r5200,r10000"
FULL_REGIMES = "rnone,r2000,r6680,r6930s,r10000"


def main() -> int:
    ap = agent_argparser("B1 live trading (agent): gate + reversal regimes x P/R/L", seed_start=200)
    ap.add_argument("--regimes", default=None, help=f"comma list; default {PILOT_REGIMES} for pilots, {FULL_REGIMES} at >= 100 seeds")
    a = ap.parse_args()
    st = Study(a, "B1 live trading (agent) — reversal regimes × Paused / Replay / Live")
    ctrl = vlm(a, "vlm_direct", 10)
    regimes = REGIMES
    if a.regimes is None:
        a.regimes = FULL_REGIMES if a.n_seeds >= 100 else PILOT_REGIMES

    def sc(seed, reversal_ms):
        return scenario("live_trading", seed, "chart", EXPIRY_NS, reversal_delay_ns=reversal_ms * 1_000_000,
                        u_valid_order=1.0, u_invalid_order=-1.0, u_no_order=0.0)

    if not st.gate(lambda seed: spec(sc(seed, 0), ctrl, policy="paused"),
                   list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib)), "chart skin, prompt 10"):
        return 1
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    st.say(f"## Triplets (seeds {seeds[0]}..{seeds[-1]})")
    order = [r for r in a.regimes.split(",") if r in regimes]
    for reg in order:
        rev_ms, dl = regimes[reg]
        st.say(f"### regime {reg}: reversal {'none' if rev_ms == 0 else f'{rev_ms} ms after the signal'}; delivery {dl/1e6:.0f} ms")
        for seed in seeds:
            st.triplet(spec(sc(seed, rev_ms), ctrl, delivery_ns=dl), reg, seed)
    st.finish()
    out_md = Path("results") / f"B1_{a.out.name.replace('b1_', '')}.md"
    subprocess.run([sys.executable, "studies/analyze_prl.py", str(a.out), str(out_md), "--title", "B1 live trading (agent)",
                    "--cell-order", ",".join(order)] + (["--pilot"] if a.n_seeds < 100 else []), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
