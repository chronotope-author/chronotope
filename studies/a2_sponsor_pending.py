"""A2 sponsor skip (pending) — card cards/A2-sponsor-skip-pending.md. Task A with repeated skips: the memoryless agent
vs the pending record vs Chronotope (binder + record), Live only; every cell reports (SR_exact, mean G_A).

  PYTHONPATH=src .venv/bin/python studies/a2_sponsor_pending.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/a2_qwen38_nothink_pilot10 --max-calls 1500

Predeclared: scenario sponsor_skip_pending, player skin 320x240, target 24 px, event 500 ms, window 16 s, 60 Hz,
publication 30 ms, delivery 20 ms; seek delays {1, 2, 4, 8} s; arms none (vlm_pending/none), record
(vlm_pending/harness_pending), chronotope (vlm_chronotope: late_bind template 12 px NCC 0.8 + harness_pending);
pending_timeout 10 s, radius 10 px; prompt 9; seeds 400..409 (pilot). Gate: Paused on sponsor_skip, seeds 100..109.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, fmt, scenario, spec, vlm  # noqa: E402

WINDOW_NS = 16 * S
SEEKS_S = [1, 2, 4, 8]
ARMS = ["none", "record", "chronotope"]


def controller(a, arm: str) -> dict:
    mem = dict(pending_timeout_ns=10 * S, pending_radius_px=10)
    if arm == "none":
        return vlm(a, "vlm_pending", 9, memory="none", **mem)
    if arm == "record":
        return vlm(a, "vlm_pending", 9, memory="harness_pending", **mem)
    if arm == "chronotope":
        return vlm(a, "vlm_chronotope", 9, memory="harness_pending", template_radius_px=12, min_match_score=0.8, **mem)
    raise ValueError(arm)


def main() -> int:
    ap = agent_argparser("A2 sponsor skip (pending): none vs record vs chronotope", seed_start=400)
    ap.add_argument("--seeks", default=",".join(map(str, SEEKS_S)))
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    st = Study(a, "A2 sponsor skip (pending) — memoryless vs pending record vs Chronotope")
    if not st.gate(lambda seed: spec(scenario("sponsor_skip", seed, "player", WINDOW_NS, seek_delay_ns=1 * S), vlm(a, "vlm_direct", 9), policy="paused"),
                   list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib)), "player skin, prompt 9"):
        return 1
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    st.say(f"## Live trials (seeds {seeds[0]}..{seeds[-1]})")
    for seek in [int(x) for x in a.seeks.split(",")]:
        for arm in a.arms.split(","):
            for seed in seeds:
                name = f"live_k{seek}_{arm}_s{seed}"
                r, t = st.trial(spec(scenario("sponsor_skip_pending", seed, "player", WINDOW_NS, seek_delay_ns=seek * S), controller(a, arm)), name)
                corr = t.all("coordinate_correction")
                bind = (f" bind: {sum(1 for c in corr if c['translation_x'] == 0 and c['translation_y'] == 0)}/{len(corr)} zero-translation,"
                        f" {sum(1 for c in corr if c.get('ambiguous'))} ambiguous") if corr else ""
                st.say(f"- {name}: accepted={r.accepted} dup={r.duplicates} exact={r.exact_success} G_A={fmt(r.return_G)} "
                       f"firstE→A={fmt((r.first_valid_E_A_ns or 0)/1e9)}s invocations={r.n_decisions} suppressed={r.n_suppressed} "
                       f"invalid_clicks={r.invalid_clicks} viol={','.join(r.contract_violations) or '-'}{bind}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.finish()
    out_md = Path("results") / f"A2_{a.out.name.replace('a2_', '')}.md"
    subprocess.run([sys.executable, "studies/analyze_a2.py", str(a.out), str(out_md)] + (["--pilot"] if a.n_seeds < 100 else []), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
