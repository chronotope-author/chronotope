"""L1 step (1): one trial-level table from the immutable traces of every 100-seed cell the paper cites
(cards/L1-normalized-law.md). Pure analysis: reads runs/<run>/*.jsonl, never writes there.

  PYTHONPATH=src .venv/bin/python studies/assemble_trials.py [--runs-dir runs] [--out results/L1_trials.csv]
      [--coverage results/L1_assemble.md] [--only r1_qwen38_notification_100,...]

One row per trace (gate/calibration files skipped). Times are seconds. Harness-clock columns (t_O, B, D, A) are
offsets from the harness origin; task-clock columns (t_E, tau, tau_eff, t_close, A_task) are task time. Under Live
the two clocks coincide; under Paused/Replay the task clock is held, so S_A and the state-change columns are
computed for Live rows only (the evaluator's convention).

Columns
  run, agent, task, family, cell, policy (paused|replay|live), controller, seed
  t_E      task time of the response-triggering event (E)
  t_O      harness offset of the first observation carrying E (O); missing when E never reached O
  tau      source time of the observation the COMMITTING decision was invoked on (Appendix A.3 join)
  tau_eff  source time of the frame that supplied the committed action: tau for a frozen/restarted decision,
           tau_D (the binder's frame at D) when the action was updated before D and the binder resolved
  B, D, A  harness offsets of decision start, commitment, receipt (A missing for no-action / dropped trials)
  A_task   task time at receipt
  t_close  task time at which the response stops being valid (window close; the reversal for trading)
  v        speed of the target along its motion axis (px/s; 0 for static, jump, reflow, code)
  r        the scorer's hit tolerance along the motion axis: half of the target box (px), from the manifest,
           NOT the binder's template radius (12 px); None for typed-value tasks
  x_norm   normalized state change between tau_eff and A: |displacement(A) - displacement(tau_eff)| / r for
           coordinate tasks; 1/0 (code changed / not) for the typed code; 0 for windowed tasks
  update_before_D   the committed action was filled from a frame newer than tau (binder resolved)
  D_A      A - D (s); S_A = t_close - A_task (Live only); R = A - t_E (total latency)
  delivery_ms   the route's delay from the spec
  success  valid at receipt (exact success for the multi-action pending tasks); G the episodic return
  labels   evaluator labels (';'-joined); valid_trial; invalid_reason
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.evaluate import evaluate_pending_trace, evaluate_trace  # noqa: E402
from chronotope.scenario import EVENT_TARGET_APPEARS, manifest_from_dict  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

NS = 1e9

# run name -> (family, filename regex). Groups: cell, seed, optional policy, optional ctrl. Files that do not match
# (gates, calibration) are skipped. The controller comes from the header (at_commit_policy / kind) unless the
# filename names an arm (O2, P1).
RUNS: dict[str, tuple[str, re.Pattern]] = {
    "r1_qwen38_notification_100": ("A notification", re.compile(r"eval_w(?P<w>\d+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "r2_window_sweep_100c": ("A notification (fine sweep)", re.compile(r"eval_w(?P<w>\d+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "a1_qwen38_nothink_100": ("A player", re.compile(r"prl_(?P<cell>[a-z0-9]+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "a1_static_qwen38_nothink_100": ("A player (static)", re.compile(r"prl_(?P<cell>[a-z0-9]+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "b1_qwen38_nothink_100": ("B trading", re.compile(r"prl_(?P<cell>[a-z0-9]+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "c_prl_qwen38_nothink_100": ("C P/R/L rows", re.compile(r"prl_(?P<cell>[a-z0-9]+)_s(?P<seed>\d+)_(?P<policy>paused|replay|live)\.jsonl$")),
    "c1_qwen38_nothink_100": ("C1 drift", re.compile(r"live_(?P<cell>v\d+(?:_irr)?)_(?:frozen|restart_if_moved|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "c1_uitars_native_100": ("C1 drift (UI-TARS)", re.compile(r"live_(?P<cell>v\d+(?:_irr)?)_(?:frozen|restart_if_moved|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "c2_qwen38_nothink_100": ("C2 jump", re.compile(r"live_(?P<cell>jump\d+s)_(?:frozen|restart_if_moved|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "c3_qwen38_nothink_100": ("C3 select", re.compile(r"live_(?P<cell>v\d+)_(?:frozen|restart_if_moved|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "c4_qwen38_nothink_100": ("D typed code", re.compile(r"live_(?P<cell>ch2|ch14|static)_(?:frozen|restart_if_changed|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "g1_reflow_qwen38_nothink_100": ("C Chromium page", re.compile(r"live_(?P<cell>ch2|ch14)_(?:frozen|restart_if_moved|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "o2_qwen38_nothink_100": ("O2 remedy matrix", re.compile(r"o2_(?P<cell>pre_d|closing|post_d)_(?P<ctrl>none|newest|earlier|both)_s(?P<seed>\d+)\.jsonl$")),
    "c1_size_qwen38_nothink_100": ("C1 size sweep", re.compile(r"live_(?P<cell>r\d+_v\d+)_(?:frozen|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "c1_size_uitars_native_100": ("C1 size sweep (UI-TARS)", re.compile(r"live_(?P<cell>r\d+_v\d+)_(?:frozen|late_bind)_s(?P<seed>\d+)\.jsonl$")),
    "p1_qwen38_nothink_100": ("P1 pending effect", re.compile(r"live_(?P<cell>d\d+(?:_drop)?)_(?P<ctrl>none|harness_pending|model_informed)(?:_drop)?_s(?P<seed>\d+)\.jsonl$")),
}
M3_GLOB = "m3_*_100"   # the M3 agents, as they land (same file convention as r1)
M3_RE = RUNS["r1_qwen38_notification_100"][1]

AGENT = {"Qwen/Qwen3.8-27B": "Qwen3.8-27B", "unsloth/Qwen3.8-27B-NVFP4": "Qwen3.8-27B-NVFP4",
         "ByteDance-Seed/UI-TARS-1.5-7B": "UI-TARS-1.5-7B", "nvidia/Gemma-4-31B-IT-NVFP4": "Gemma-4-31B-NVFP4"}

FIELDS = ["run", "agent", "task", "family", "cell", "policy", "controller", "seed", "t_E", "t_O", "tau", "tau_eff",
          "B", "D", "A", "A_task", "t_close", "v", "r", "x_norm", "update_before_D", "D_A", "S_A", "R", "delivery_ms",
          "success", "G", "labels", "valid_trial", "invalid_reason", "n_decisions", "binding"]


def agent_name(model_id: str) -> str:
    return AGENT.get(model_id, model_id.split("/")[-1])


def hit_tolerance(m, dx: float, dy: float) -> float | None:
    """Half of the target box along the motion axis (the scorer accepts any click inside the box)."""
    if dx and not dy:
        return m.target_w_px() / 2
    if dy and not dx:
        return m.target_size_px / 2
    if dx and dy:
        return math.hypot(m.target_w_px() / 2, m.target_size_px / 2)
    # no motion in this trial: report the along-x tolerance (the direction every drifting cell uses)
    return m.target_w_px() / 2


def state_change(m, tau_eff_ns: int | None, a_task_ns: int | None) -> tuple[float | None, float | None]:
    """(x_norm, r) — normalized state change of the acted-on field between tau_eff and receipt."""
    if tau_eff_ns is None or a_task_ns is None:
        return None, None
    sid = m.scenario_id
    if sid == "rotating_code":
        a, b = m.code_at(tau_eff_ns), m.code_at(a_task_ns)
        return (0.0 if (a is None or b is None or a == b) else 1.0), None
    if sid in ("live_trading",):
        return 0.0, None
    if not hasattr(m, "displacement_at"):
        return 0.0, None
    d0 = m.displacement_at(tau_eff_ns)
    d1 = m.displacement_at(a_task_ns)
    dx, dy = d1[0] - d0[0], d1[1] - d0[1]
    r = hit_tolerance(m, dx, dy)
    return math.hypot(dx, dy) / r, r


def close_time_ns(m) -> int:
    return m.valid_until_ns() if m.scenario_id == "live_trading" else m.window_close_ns()


def row_from_trace(p: Path, run: str, family: str, mt: re.Match) -> dict:
    t = Trace.load(p)
    h = t.header()
    spec = h["spec"]
    m = manifest_from_dict(spec["scenario"])
    h0 = h["harness_origin_ns"]
    policy = h["task_clock_policy"]
    pending = bool(getattr(m, "multi_action", False))
    res = evaluate_pending_trace(t) if pending else evaluate_trace(t)
    ctrl = spec["controller"]
    controller = mt.groupdict().get("ctrl") or ctrl.get("at_commit_policy") or ctrl["kind"]
    cell = mt.groupdict().get("cell") or (f"w{int(mt.group('w')) / NS:g}s" if "w" in mt.groupdict() else "")
    live = policy == "live"

    obs = t.all("observation")
    evid = [o for o in obs if o["source_event_id"] == EVENT_TARGET_APPEARS]
    o1 = evid[0] if evid else None
    commits = t.all("commit")
    receipts = t.all("receipt")
    row: dict = dict(run=run, agent=agent_name(ctrl["model_id"]), task=m.scenario_id, family=family, cell=cell,
                     policy=policy, controller=controller, seed=int(mt.group("seed")),
                     t_E=m.event_time_ns / NS, t_O=(o1["publication_time_ns"] - h0) / NS if o1 else None,
                     tau=None, tau_eff=None, B=None, D=None, A=None, A_task=None, t_close=close_time_ns(m) / NS,
                     v=float(getattr(m, "velocity_x_px_per_s", 0) or 0), r=hit_tolerance(m, 1, 0), x_norm=None,
                     update_before_D=False, D_A=None, S_A=None, R=None,
                     delivery_ms=spec["delivery"]["delivery_delay_ns"] / 1e6,
                     success=None, G=res.return_G, labels=";".join(getattr(res, "labels", [])),
                     valid_trial=res.valid_trial, invalid_reason=res.invalid_reason, n_decisions=res.n_decisions,
                     binding=getattr(res, "binding", None))
    if pending:
        row["success"] = bool(res.exact_success) if res.valid_trial else None
        row["labels"] = ";".join(k for k in ("accepted", "duplicates", "dropped") if getattr(res, k))
    else:
        row["success"] = bool(res.valid_at_receipt) if res.valid_trial else None
    if not commits:
        return row
    c = commits[0]
    d0 = next((d for d in t.all("decision_begin") if d["decision_id"] == c["decision_id"]), None)
    ob = next((o for o in obs if d0 and o["observation_id"] == d0["observation_id"]), None)
    corr = next((r for r in t.all("coordinate_correction") if r["decision_id"] == c["decision_id"]), None)
    tau_ns = ob["source_time_ns"] if ob else None
    tau_eff_ns = tau_ns
    if corr and corr.get("binding") in ("newest_at_D", "tracked_at_D", "decoded_at_D") and corr.get("matched", True) \
            and corr["corrected_action"] == c["action"]:
        ob2 = next((o for o in obs if o["observation_id"] == corr["observation_id"]), None)
        if ob2 is not None:
            tau_eff_ns = ob2["source_time_ns"]
            row["update_before_D"] = ob2["source_time_ns"] > (tau_ns or -1)
    r = next((x for x in receipts if x["action_id"] == c["action_id"]), None)
    row.update(tau=None if tau_ns is None else tau_ns / NS, tau_eff=None if tau_eff_ns is None else tau_eff_ns / NS,
               B=(d0["begin_time_ns"] - h0) / NS if d0 else None, D=(c["commitment_time_ns"] - h0) / NS)
    if r is None:
        return row
    A = r["receipt_time_ns"]
    a_task = r["task_time_at_receipt_ns"]
    row.update(A=(A - h0) / NS, A_task=a_task / NS, D_A=(A - c["commitment_time_ns"]) / NS, R=(A - h0) / NS - row["t_E"])
    if live:
        row["S_A"] = (close_time_ns(m) - a_task) / NS
        x, rr = state_change(m, tau_eff_ns, a_task)
        row["x_norm"] = x
        if rr is not None:
            row["r"] = rr
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results/L1_trials.csv"))
    ap.add_argument("--coverage", type=Path, default=Path("results/L1_assemble.md"))
    ap.add_argument("--only", default=None, help="comma list of run names")
    a = ap.parse_args()
    runs = dict(RUNS)
    for d in sorted(a.runs_dir.glob(M3_GLOB)):
        if d.is_dir() and d.name not in runs:
            runs[d.name] = ("A notification (M3 agent)", M3_RE)
    if a.only:
        keep = set(a.only.split(","))
        runs = {k: v for k, v in runs.items() if k in keep}
    rows: list[dict] = []
    cover: list[tuple[str, int, int, int, dict]] = []
    for run, (family, rx) in runs.items():
        d = a.runs_dir / run
        if not d.is_dir():
            cover.append((run, 0, 0, 0, {"missing run dir": 1}))
            print(f"[skip] {run}: no directory", flush=True)
            continue
        n_files = n_rows = n_err = 0
        missing: dict[str, int] = defaultdict(int)
        for p in sorted(d.glob("*.jsonl")):
            mt = rx.search(p.name)
            if not mt:
                continue
            n_files += 1
            try:
                row = row_from_trace(p, run, family, mt)
            except Exception as e:  # a malformed trace is reported, never silently dropped
                n_err += 1
                print(f"[error] {p}: {e!r}", flush=True)
                continue
            for k in ("t_O", "tau", "B", "D", "A", "x_norm", "S_A"):
                if row[k] is None:
                    missing[k] += 1
            rows.append(row)
            n_rows += 1
        cover.append((run, n_files, n_rows, n_err, dict(missing)))
        print(f"{run}: {n_rows} rows ({n_err} errors)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in FIELDS})
    L = ["# L1 — trial table assembly coverage", "",
         f"Source: `{a.runs_dir}`; output `{a.out}` ({len(rows)} rows). Gate/calibration files are not rows. "
         "Columns are defined in the docstring of studies/assemble_trials.py. Missing counts: t_O = E never reached O; "
         "tau/B/D = no committed action (wait / no decision); A = no receipt (dropped or no action); "
         "x_norm/S_A = defined for Live rows with a receipt only.", "",
         "| run | traces matched | rows | errors | Live rows | success (Live) | missing t_O | missing tau | missing B | missing D | missing A | missing x_norm | missing S_A |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for run, nf, nr, ne, miss in cover:
        live = [r for r in rows if r["run"] == run and r["policy"] == "live"]
        sr = f"{sum(bool(r['success']) for r in live)}/{len(live)}" if live else "-"
        L.append(f"| {run} | {nf} | {nr} | {ne} | {len(live)} | {sr} | " + " | ".join(str(miss.get(k, 0)) for k in ("t_O", "tau", "B", "D", "A", "x_norm", "S_A")) + " |")
    L += ["", "## Rows per (run, cell, controller, policy)", "", "| run | cell | controller | policy | n | SR | mean G |", "|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["run"], r["cell"], r["controller"], r["policy"])].append(r)
    for k in sorted(groups):
        g = [r for r in groups[k] if r["valid_trial"]]
        sr = sum(bool(r["success"]) for r in g)
        gs = [r["G"] for r in g if r["G"] is not None]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(groups[k])} | {sr}/{len(g)} | {sum(gs)/len(gs):.2f} |" if gs else
                 f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {len(groups[k])} | {sr}/{len(g)} | - |")
    a.coverage.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} and {a.coverage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
