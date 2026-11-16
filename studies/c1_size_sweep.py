"""C1-size sweep (cards/L1-normalized-law.md, design step 2; U60): does frozen success switch at v (t_A - tau) / r = 1
when the hit tolerance r is varied and not only the speed? Balloon half-width r in {6, 12, 24} px
(target_size_px = 2r; the balloon skin has aspect 1, so the box is 2r wide and the scorer accepts any click inside it)
x speed v in {2, 4, 8, 16, 32} px/s x {frozen, late_bind}, Live, 100 seeds per cell. Modelled on c1_balloon_drift.py:
same prompt (5, 0-1000 grid), image scale 2, 60 Hz frames, 30 ms publication, 20 ms delivery, tracker radius 12 px.

  PYTHONPATH=src .venv/bin/python studies/c1_size_sweep.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --out runs/c1_size_qwen38_nothink_100 --seed-start 1000 --n-seeds 100 --max-calls 3200
  UI-TARS subset (r in {6, 24} at v = 8 only; the r = 12 cells exist in runs/c1_uitars_native_100):
  ... --model ByteDance-Seed/UI-TARS-1.5-7B --prompt-version 7 --uitars-subset --out runs/c1_size_uitars_native_100

Window: 16 s as in C1, except that a target must stay on the 320-px canvas for the whole window, so the window is
shortened to floor((canvas_w - 2r - 4) / v) s when that is smaller (v = 32 px/s: 8 s; deliberation is ~5.3 s, so
lateness is still not the failure under study). The window used is written into every trace's manifest.
Traces: live_r<r>_v<v>_<policy>_s<seed>.jsonl. Analysis: studies/assemble_trials.py then studies/analyze_l1.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agentlib import S, Study, agent_argparser, late_binding, scenario, spec  # noqa: E402

HALF_WIDTHS = [6, 12, 24]
VELOCITIES = [2, 4, 8, 16, 32]
POLICIES = ["frozen", "late_bind"]
CANVAS_W, CANVAS_H = 320, 240
WINDOW_S = 16


def window_s(r: int, v: int) -> int:
    return min(WINDOW_S, max(1, (CANVAS_W - 2 * r - 4) // v)) if v > 0 else WINDOW_S


def make_spec(a, seed: int, r: int, v: int, policy: str, clock: str = "live"):
    sc = scenario("moving_target", seed, "balloon", window_s(r, v) * S, target=2 * r, w=CANVAS_W, h=CANVAS_H,
                  velocity_x_px_per_s=v, velocity_y_px_per_s=0)
    return spec(sc, late_binding(a, a.prompt_version, policy), policy=clock)


def main() -> int:
    ap = agent_argparser("C1-size sweep: hit tolerance x speed (L1 card, U60)", seed_start=1000, n_seeds=10)
    ap.add_argument("--prompt-version", type=int, default=5, help="5 = balloon, 0-1000 grid (Qwen-VL); 7 = UI-TARS native grammar")
    ap.add_argument("--half-widths", default=None, help="comma list overriding {6,12,24}")
    ap.add_argument("--velocities", default=None, help="comma list overriding {2,4,8,16,32}")
    ap.add_argument("--policies", default=None, help="comma list overriding {frozen,late_bind}")
    ap.add_argument("--uitars-subset", action="store_true", help="r in {6,24} at v = 8 only")
    a = ap.parse_args()
    rs = [int(x) for x in a.half_widths.split(",")] if a.half_widths else HALF_WIDTHS
    vs = [int(x) for x in a.velocities.split(",")] if a.velocities else VELOCITIES
    pols = a.policies.split(",") if a.policies else POLICIES
    if a.uitars_subset:
        rs, vs = [6, 24], [8]
    st = Study(a, "C1-size sweep: balloon half-width x speed, frozen vs late binding (L1 card)")
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    calib = list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib))
    # Paused competence at the smallest target (the hardest static case), v = 0, frozen
    if not st.gate(lambda s: make_spec(a, s, min(rs), 0, "frozen", "paused"), calib, f"r = {min(rs)} px, v = 0, Paused"):
        return 1
    st.say("## Live trials (one line per trial)")
    summary = []
    for r in rs:
        for v in vs:
            w = window_s(r, v)
            for pol in pols:
                ok = n = 0
                for seed in seeds:
                    name = f"live_r{r}_v{v}_{pol}_s{seed}"
                    res, _ = st.trial(make_spec(a, seed, r, v, pol), name)
                    n += int(res.valid_trial); ok += int(bool(res.valid_trial and res.valid_at_receipt))
                    st.say(f"- {name} (w={w}s): valid={res.valid_at_receipt} err={res.spatial_error_px if res.spatial_error_px is None else round(res.spatial_error_px, 1)} "
                           f"counterfactual_frozen={res.proposed_valid_at_receipt} B→D={(res.b_to_d_ns or 0) / 1e9:.2f}s G={res.return_G} "
                           f"labels={','.join(res.labels) or '-'}" + ("" if res.valid_trial else f" INVALID:{res.invalid_reason}"))
                summary.append((r, v, pol, w, ok, n))
                st.say(f"cell r={r} v={v} {pol}: {ok}/{n}")
    st.say("\n## Summary (Live; SR = valid at A)\n")
    st.say("| r (px) | v (px/s) | policy | window (s) | n | SR |")
    st.say("|---|---|---|---|---|---|")
    for r, v, pol, w, ok, n in summary:
        st.say(f"| {r} | {v} | {pol} | {w} | {n} | {ok}/{n} |")
    st.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
