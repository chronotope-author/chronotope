"""C1–C3 Paused / Replay / Live rows — card cards/C-prl-rows.md. Fresh triplets for the step-based agent (the exact
"initial coordinate" controller of the C1–C3 cells: vlm_late_binding / frozen) on C1 drift {4, 8, 16} px/s, C2 jump
at 2 s, C3 named balloon at 8 px/s; plus ONE diagnostic cell, late_bind under Replay at C1 8 px/s (U37), which is
expected to miss by construction (the scorer pins the balloon; the binder follows the replayed frames).

  PYTHONPATH=src .venv/bin/python studies/c_prl_rows.py --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/c_prl_qwen38_nothink_pilot10 --max-calls 600
  (pilot: seeds 1000..1009 = the first ten seeds of the 100-seed Live cells)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, late_binding, scenario, spec  # noqa: E402
from chronotope.scenario import BALLOON_ORDER  # noqa: E402

WINDOW_NS = 16 * S


def cell_spec(a, cell: str, seed: int):
    if cell.startswith("c1v"):
        v = int(cell[3:].rstrip("lb"))
        policy = "late_bind" if cell.endswith("lb") else "frozen"
        return spec(scenario("moving_target", seed, "balloon", WINDOW_NS, velocity_x_px_per_s=v, velocity_y_px_per_s=0),
                    late_binding(a, 5, policy))
    if cell == "c2j2":
        return spec(scenario("jump_target", seed, "balloon", WINDOW_NS, jump_delay_ns=2 * S, jump_dx_px=96, jump_dy_px=0),
                    late_binding(a, 5, "frozen"))
    if cell.startswith("c3v"):
        v = int(cell[3:])
        return spec(scenario("balloon_select", seed, "balloon", WINDOW_NS, named_color=BALLOON_ORDER[seed % 3],
                             velocity_x_px_per_s=v), late_binding(a, 8, "frozen"))
    raise ValueError(cell)


CELLS = ["c1v4", "c1v8", "c1v16", "c2j2", "c3v8"]
DIAG = ["c1v8lb"]


def main() -> int:
    ap = agent_argparser("C1–C3 P/R/L rows for the step-based agent", seed_start=1000)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--no-diagnostic", action="store_true")
    a = ap.parse_args()
    st = Study(a, "C1–C3 Paused / Replay / Live rows (step-based agent)")
    st.say("Gate: the balloon skins were qualified in C1/C3 (results/model_qualification.md); not re-run here.")
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    cells = a.cells.split(",")
    for cell in cells:
        st.say(f"### {cell}")
        for seed in seeds:
            st.triplet(cell_spec(a, cell, seed), cell, seed)
    if not a.no_diagnostic:
        for cell in DIAG:
            st.say(f"### {cell} (diagnostic: late binding under Replay is ill-posed by construction; 10 seeds only)")
            for seed in seeds[:10]:
                st.triplet(cell_spec(a, cell, seed), cell, seed)
    st.finish()
    out_md = Path("results") / f"C_prl_{a.out.name.replace('c_prl_', '')}.md"
    subprocess.run([sys.executable, "studies/analyze_prl.py", str(a.out), str(out_md), "--title", "C1–C3 P/R/L rows (step-based agent)",
                    "--cell-order", ",".join(cells + ([] if a.no_diagnostic else DIAG)), "--diagnostic", ",".join(DIAG)]
                   + (["--pilot"] if a.n_seeds < 100 else []), check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
