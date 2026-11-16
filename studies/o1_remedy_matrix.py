"""O1 study (card: cards/O1-remedy-matrix.md): oracle remedy matrix for tab:results-interventions.
Five failure-class columns x six remedy rows, Live condition, simulated clock, 100 seeds per cell.
Deterministic, CPU-only; reproducible with one command:

  PYTHONPATH=src .venv/bin/python studies/o1_remedy_matrix.py [--n-seeds 100] [--seed-start 1000]
"""
from __future__ import annotations

import argparse
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from conftest import base_spec_dict  # noqa: E402

MS = 1_000_000
PUB_NATIVE = 30 * MS


def revealed(seed, pub, compute, del_, window, binding="state_at_B", exp=None):
    d = base_spec_dict()
    d["scenario"]["scenario_seed"] = seed
    d["scenario"]["response_window_ns"] = window
    d["observation"]["publication_delay_ns"] = pub
    d["controller"].update({"compute_ns": compute, "binding": binding,
                            "expected_observation_delay_ns": exp[0] if exp else None,
                            "expected_delivery_delay_ns": exp[1] if exp else None})
    d["delivery"]["delivery_delay_ns"] = del_
    return TrialSpec.from_dict(d)


def moving(seed, pub, compute, del_, window, vx, binding="state_at_B", exp=None):
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500 * MS, "response_window_ns": window,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 2_000 * MS, "irrelevant_motion": False,
                     "skin": "plain", "velocity_x_px_per_s": vx, "velocity_y_px_per_s": 0}
    d["observation"]["publication_delay_ns"] = pub
    d["controller"].update({"compute_ns": compute, "binding": binding,
                            "expected_observation_delay_ns": exp[0] if exp else None,
                            "expected_delivery_delay_ns": exp[1] if exp else None})
    d["delivery"]["delivery_delay_ns"] = del_
    return TrialSpec.from_dict(d)


def jump(seed, pub, compute, del_, window, binding="state_at_B", exp=None):
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "jump_target", "scenario_seed": seed, "canvas_w_px": 96, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": 500 * MS, "response_window_ns": window,
                     "frame_period_ns": 16_666_667, "episode_horizon_ns": 2_500 * MS, "irrelevant_motion": False,
                     "skin": "plain", "jump_delay_ns": 300 * MS, "jump_dx_px": 40, "jump_dy_px": 0}
    d["observation"]["publication_delay_ns"] = pub
    d["controller"].update({"compute_ns": compute, "binding": binding,
                            "expected_observation_delay_ns": exp[0] if exp else None,
                            "expected_delivery_delay_ns": exp[1] if exp else None})
    d["delivery"]["delivery_delay_ns"] = del_
    return TrialSpec.from_dict(d)


# Column constants from the card (ms): (builder, PUB, compute, DEL, window, extra)
COLS = {
    "transient": dict(build=revealed, pub=150, comp=100, dl=40, win=220,
                      label="missed_evidence:window_closes"),
    "pre_d":     dict(build=jump, pub=30, comp=500, dl=60, win=1500,
                      label="outdated_at_commitment"),
    "closing":   dict(build=revealed, pub=30, comp=300, dl=150, win=400,
                      label="invalidated_after_commitment"),
    "post_d":    dict(build=moving, pub=30, comp=100, dl=100, win=400, vx=120, label=None),
    "combined":  dict(build=moving, pub=30, comp=100, dl=200, win=300, vx=120, label=None),
}

# Rows: remedy name -> per-column knob override, per the card. None = cell keeps column values.
def row_spec(col_key, c, remedy, seed):
    pub, comp, dl, win = c["pub"] * MS, c["comp"] * MS, c["dl"] * MS, c["win"] * MS
    b, exp = "state_at_B", None
    if remedy == "obs":
        pub = (30 if col_key == "transient" else 0) * MS
    elif remedy == "newest":
        b = "newest_at_D"
    elif remedy == "receipt":
        dl = (20 if col_key == "combined" else 0) * MS
    elif remedy == "predict":
        b, exp = "predicted_at_A", (pub, dl)
    elif remedy == "combo":
        if col_key == "transient":
            pub, dl = 30 * MS, 0
        elif col_key == "pre_d":
            b = "newest_at_D"
        elif col_key == "closing":
            dl = 0
        elif col_key == "post_d":
            dl = 0
            b, exp = "predicted_at_A", (pub, dl)
        elif col_key == "combined":
            dl = 20 * MS
            b, exp = "predicted_at_A", (pub, dl)
    kw = dict(pub=pub, compute=comp, del_=dl, window=win, binding=b, exp=exp)
    if "vx" in c:
        kw["vx"] = c["vx"]
    return c["build"](seed, **kw)


ROWS = ["baseline", "obs", "newest", "receipt", "predict", "combo"]
ROW_LABEL = {"baseline": "Baseline", "obs": "Observation before expiry", "newest": "Latest relevant pre-$D$ state",
             "receipt": "Earlier receipt", "predict": "Receipt-state prediction", "combo": "Matched combination"}


def diag_px(spec):
    s = spec.raw["scenario"]
    return math.hypot(s["canvas_w_px"], s["canvas_h_px"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    args = ap.parse_args()
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    out_dir = ROOT / "runs" / "o1_remedy_matrix"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    store = PayloadStore(out_dir / "payloads")

    results = {}   # (row, col) -> list[(valid, G, labels)]
    for ck, c in COLS.items():
        for rk in ROWS:
            cell = []
            for s in seeds:
                spec = row_spec(ck, c, rk, s) if rk != "baseline" else row_spec(ck, c, "keep", s)
                p = run_trial(spec, out_dir / f"{ck}_{rk}_s{s}.jsonl", store)
                r = evaluate_trace(Trace.load(p))
                assert r.valid_trial, f"invalid trial {ck}/{rk}/{s}"
                g = -(r.spatial_error_px if r.spatial_error_px is not None else diag_px(spec))
                cell.append((bool(r.valid_at_receipt), g, tuple(r.labels)))
            results[(rk, ck)] = cell
            k = sum(v for v, _, _ in cell)
            print(f"{ck:9s} {rk:8s} {k}/{len(cell)}  G={statistics.mean(g for _, g, _ in cell):.1f}", flush=True)

    # static negative control: every remedy on a static generous-window variant must be 1.00
    control = {}
    for rk in ROWS:
        cell = []
        for s in seeds:
            c = dict(build=moving, pub=30, comp=100, dl=100, win=1000, vx=0, label=None)
            spec = row_spec("post_d_static", c, rk if rk != "baseline" else "keep", s)
            p = run_trial(spec, out_dir / f"static_{rk}_s{s}.jsonl", store)
            r = evaluate_trace(Trace.load(p))
            cell.append(bool(r.valid_at_receipt))
        control[rk] = (sum(cell), len(cell))
        print(f"static    {rk:8s} {sum(cell)}/{len(cell)}", flush=True)

    # contract check: baseline labels per column
    lab_ok = {}
    for ck, c in COLS.items():
        if c["label"] is None:
            continue
        labs = [set(l) for _, _, l in results[("baseline", ck)]]
        lab_ok[ck] = sum(c["label"] in l for l in labs), len(labs)

    L = ["# O1 — oracle remedy matrix (card: cards/O1-remedy-matrix.md)", "",
         f"Live condition, simulated clock, seeds {args.seed_start}..{args.seed_start + args.n_seeds - 1}. "
         "Success = valid at receipt; G = -e_click at A (display diagonal when the target is absent at receipt).", "",
         "## Success rate", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for rk in ROWS:
        L.append(f"| {ROW_LABEL[rk]} | " + " | ".join(
            f"{sum(v for v, _, _ in results[(rk, ck)])}/{len(results[(rk, ck)])}" for ck in COLS) + " |")
    L += ["", "## Mean return G (px)", "", "| remedy | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 1)]
    for rk in ROWS:
        L.append(f"| {ROW_LABEL[rk]} | " + " | ".join(
            f"{statistics.mean(g for _, g, _ in results[(rk, ck)]):.1f}" for ck in COLS) + " |")
    L += ["", "## Contract checks", ""]
    for ck, (k, n) in lab_ok.items():
        L.append(f"- baseline `{ck}` carries `{COLS[ck]['label']}`: {k}/{n}")
    for rk, (k, n) in control.items():
        L.append(f"- static control `{rk}`: {k}/{n} valid")
    L += ["", "## Predeclared expectation (card)", "",
          "Diagonal: each single remedy fixes exactly its own column; combination fixes all; any deviation is a finding."]
    (ROOT / "results" / "O1_remedy_matrix.md").write_text("\n".join(L) + "\n")
    print("\nwritten: results/O1_remedy_matrix.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
