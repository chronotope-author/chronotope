"""G1 — late binding on a Chromium-rendered page with an asynchronous reflow (cards/G1-realistic-reflow.md).
reflow_page, browser rendering 640x480 (image_scale 1: the model sees the page at its native size), prompt 12.

  PYTHONPATH=src .venv/bin/python studies/g1_reflow.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --n-seeds 100 --out runs/g1_reflow_qwen38_nothink_100 --max-calls 1200

Predeclared: event 0.5 s, window 16 s, 60 Hz, pub 30 ms, delivery 20 ms; the banner is inserted at E + 2 s (during
B->D ≈ 5 s) or at E + 14 s (after A; control); gate Paused on the un-reflowed page, seeds 100..109;
cells: ch2 x {frozen, late_bind, restart_if_moved(6 px)}; ch14 x {frozen, late_bind}; seeds 1000..1099.
Files live_<cell>_<policy>_s<seed>.jsonl; analysis studies/analyze_g1.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, fmt, late_binding, mean, scenario, spec  # noqa: E402

WINDOW = 16 * S
CELLS = [("ch2", 2 * S, ["frozen", "late_bind", "restart_if_moved"]), ("ch14", 14 * S, ["frozen", "late_bind"])]


def sc(seed: int, change_ns: int) -> dict:
    return scenario("reflow_page", seed, "browser", WINDOW, target=37, w=640, h=480, change_delay_ns=change_ns)


def ctrl(a, policy: str) -> dict:
    return {**late_binding(a, 12, policy), "image_scale": 1}


def main() -> int:
    ap = agent_argparser("G1 reflow page: frozen / late_bind / restart_if_moved on a Chromium-rendered button", seed_start=1000)
    a = ap.parse_args()
    st = Study(a, "G1 reflow page (browser rendering, prompt 12)")
    calib = list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib))
    if not st.gate(lambda seed: spec(sc(seed, 30 * S), ctrl(a, "frozen"), policy="paused"), calib, "no reflow, frozen"):
        st.finish(); return 1
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    rows = []
    for cell, change_ns, policies in CELLS:
        for pol in policies:
            st.say(f"## {cell} (banner at E + {change_ns/S:.0f} s), {pol}")
            for seed in seeds:
                s_ = spec(sc(seed, change_ns), ctrl(a, pol))
                r, t = st.trial(s_, f"live_{cell}_{pol}_s{seed}")
                corr = t.first("coordinate_correction"); dy = s_.manifest().reflow_dy_px()
                rows.append((cell, pol, r))
                st.say(f"- live_{cell}_{pol}_s{seed}: valid={r.valid_at_receipt} err={fmt(r.spatial_error_px, 1)} G={fmt(r.return_G)} "
                       f"reflow_dy={dy} "
                       f"translation_y={(corr or {}).get('translation_y')} counterfactual_frozen={r.proposed_valid_at_receipt} "
                       f"restarts={r.n_restarts} B→D={fmt((r.b_to_d_ns or 0)/1e9)}s firstB→D={fmt((r.first_b_to_d_ns or 0)/1e9)}s "
                       f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.say()
    st.say("| cell | policy | n | SR (valid at A) | mean err (px) | mean G (px) | frozen counterfactual SR (same A) | restarts/trial | B→D (s) | first B→D (s) | outdated_at_commitment |")
    st.say("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell, change_ns, policies in CELLS:
        for pol in policies:
            rs = [r for (c, p, r) in rows if (c, p) == (cell, pol) and r.valid_trial]
            n = len(rs)
            if not n:
                st.say(f"| {cell} | {pol} | 0 | - | - | - | - | - | - | - | - |"); continue
            cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
            st.say(f"| {cell} | {pol} | {n} | {sum(bool(r.valid_at_receipt) for r in rs)/n:.2f} | {fmt(mean([r.spatial_error_px for r in rs]), 1)} | "
                   f"{fmt(mean([r.return_G for r in rs]), 1)} | {(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | {mean([r.n_restarts for r in rs]):.1f} | "
                   f"{fmt(mean([(r.b_to_d_ns or 0)/1e9 for r in rs]))} | {fmt(mean([(r.first_b_to_d_ns or 0)/1e9 for r in rs]))} | "
                   f"{sum('outdated_at_commitment' in r.labels for r in rs)} |")
    st.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
