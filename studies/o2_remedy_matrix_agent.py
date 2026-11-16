"""O2 remedy matrix (agent) — card cards/O2-remedy-matrix-agent.md. Three failure columns at the model's timescale ×
four remedy rows, Live only, balloon skin; every cell reports (SR, mean G_C).

  PYTHONPATH=src .venv/bin/python studies/o2_remedy_matrix_agent.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/o2_qwen38_nothink_pilot10 --max-calls 400

Columns: pre_d (jump_target: jump 96 px at 2 s, window 16 s); closing (moving_target v = 0, window 6.5 s);
post_d (moving_target 16 px/s, window 16 s). Rows: none (frozen, delivery 1.5 s); newest (late_bind, 1.5 s);
earlier (frozen, 20 ms); both (late_bind, 20 ms). Predicted SR pattern (card): pre_d 0/1/0/1; closing 0/0/1/1;
post_d 0/0/0/1. Seeds 1000..1009 (pilot).
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, fmt, late_binding, scenario, spec  # noqa: E402

SLOW_NS, FAST_NS = 1_500_000_000, 20_000_000
COLS = {
    "pre_d": lambda seed: scenario("jump_target", seed, "balloon", 16 * S, jump_delay_ns=2 * S, jump_dx_px=96, jump_dy_px=0),
    "closing": lambda seed: scenario("moving_target", seed, "balloon", 6_500_000_000, velocity_x_px_per_s=0, velocity_y_px_per_s=0),
    "post_d": lambda seed: scenario("moving_target", seed, "balloon", 16 * S, velocity_x_px_per_s=16, velocity_y_px_per_s=0),
}
ROWS = {"none": ("frozen", SLOW_NS), "newest": ("late_bind", SLOW_NS), "earlier": ("frozen", FAST_NS), "both": ("late_bind", FAST_NS)}
PREDICTED = {"pre_d": [0, 1, 0, 1], "closing": [0, 0, 1, 1], "post_d": [0, 0, 0, 1]}


def main() -> int:
    ap = agent_argparser("O2 remedy matrix (agent)", seed_start=1000)
    ap.add_argument("--closing-window-s", type=float, default=6.5)
    a = ap.parse_args()
    st = Study(a, "O2 remedy matrix (agent) — three failure columns × four remedies, Live")
    st.say("Gate: balloon skin qualified in C1 (results/model_qualification.md); not re-run here.")
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    cells = {}
    for col, mk in COLS.items():
        for row, (policy, dl) in ROWS.items():
            st.say(f"### {col} × {row} ({policy}, delivery {dl/1e6:.0f} ms)")
            rs = []
            for seed in seeds:
                sc = mk(seed)
                if col == "closing":
                    sc["response_window_ns"] = int(a.closing_window_s * S); sc["episode_horizon_ns"] = sc["event_time_ns"] + sc["response_window_ns"] + 500_000_000
                r, t = st.trial(spec(sc, late_binding(a, 5, policy), delivery_ns=dl), f"o2_{col}_{row}_s{seed}")
                rc = t.first("receipt"); c = t.first("commit")
                rs.append((r, (c["task_time_at_commit_ns"] / 1e9) if c else None, (rc["task_time_at_receipt_ns"] / 1e9) if rc else None))
                st.say(f"- s{seed}: valid={r.valid_at_receipt} G={fmt(r.return_G, 1)} err={fmt(r.spatial_error_px, 1)} D={fmt(rs[-1][1], 3)} A={fmt(rs[-1][2], 3)} "
                       f"S_A={fmt((r.intervals_ns.get('S_A') or 0)/1e9, 3)} labels={','.join(r.labels) or '-'}"
                       + (f" viol={','.join(r.contract_violations)}" if r.contract_violations else "") + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
            cells[(col, row)] = rs
    st.finish()
    L = [f"# O2 remedy matrix (agent) — {a.model} ({a.out.name})" + (" — PILOT, 10 seeds: not a citable cell" if a.n_seeds < 100 else ""), "",
         "Live, balloon skin 320x240, 24-px balloon, prompt 5; slow route 1.5 s, fast route 20 ms; success = valid at receipt; "
         "G_C = −min(click error at receipt, 400 px). Predicted pattern per column (none/newest/earlier/both): "
         + "; ".join(f"{c} {'/'.join(map(str, v))}" for c, v in PREDICTED.items()) + ".", "",
         "## Success rate", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for row in ROWS:
        L.append(f"| {row} | " + " | ".join(f"{sum(bool(r.valid_at_receipt) for r, _, _ in cells[(c, row)])}/{len(cells[(c, row)])}" for c in COLS) + " |")
    L += ["", "## Mean return G_C (px)", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for row in ROWS:
        L.append(f"| {row} | " + " | ".join(f"{statistics.mean(r.return_G for r, _, _ in cells[(c, row)] if r.valid_trial):.1f}" for c in COLS) + " |")
    L += ["", "## Pattern check vs the card's prediction", ""]
    for c, want in PREDICTED.items():
        got = [round(sum(bool(r.valid_at_receipt) for r, _, _ in cells[(c, row)]) / max(1, len(cells[(c, row)]))) for row in ROWS]
        L.append(f"- {c}: predicted {want}, observed (rounded) {got} {'OK' if got == want else 'DEVIATES'}")
    A = [x for (c, row), rs in cells.items() if c == "closing" for _, _, x in rs if x is not None]
    D = [d for (c, row), rs in cells.items() if c == "closing" for _, d, _ in rs if d is not None]
    if A:
        L += ["", f"Closing-column margins: window closes at {0.5 + a.closing_window_s:.2f} s; D ∈ [{min(D):.2f}, {max(D):.2f}] s; "
              f"A ∈ [{min(A):.2f}, {max(A):.2f}] s over all four rows (earlier-receipt rows must have A < close, slow rows A > close)."]
    (Path("results") / f"O2_{a.out.name.replace('o2_', '')}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
