"""K2: the GPU-resident agent (chronotope, detector planner, software track) on every task family of the paper, overnight.
Predeclared in cards/K2-resident-sweep.md. Same software path as every other cell in the paper (simulated 30 ms
publication, 20 ms delivery, no HID). The planner is the resident colour detector with an EMULATED deliberation
(compute_ns, a wall-clock sleep after B, 5.2 s = the 27B's median), so the cells measure the control loop, not a model.

  PYTHONPATH=src:chronotope/python .venv/bin/python studies/k2_resident_sweep.py --part r1 [--seeds 100 --seed0 1000] [--smoke]

parts (each writes runs/k2_<part>/ and results/K2_<part>.md):
  r1       revealed_target (notification skin) x windows {1,2,4,8,16} s x Paused/Replay/Live (run_triplet), late_bind
  c1       moving_target (balloon) v {0,4,8,16} px/s x {frozen, late_bind, restart_if_moved}; irrelevant-motion v=0 controls
  c2       jump_target (balloon) jump (96,0) at {2, 14} s x {frozen, late_bind}
  c3       balloon_select v=8 x {frozen, late_bind}  (rgb template, named colour = BALLOON_ORDER[seed % 3])
  k1c      moving_target v {8,100,200} x binder {cuda, tracker_py} late_bind + cuda frozen; deliberation 1.0 s, window 1.3 s
  p1       pending_effect (notification) effect delay {1,2,8} s x memory {none, harness_pending}; drop cell (2 s, drop_first)
  p1_kind  pending_effect delay {1,2,8} s, harness_pending + validate_kind (the stale-proposal guard at D)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "chronotope" / "python"))

from chronotope.__main__ import run_triplet  # noqa: E402
from chronotope.evaluate import evaluate_pending_trace, evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from chronotope_core.controller import build  # noqa: E402
from chronotope_core.glue import controller_dict, register  # noqa: E402

register()

PERIOD = 16_666_667
PUB = 30_000_000
DELIVERY = 20_000_000
DELIB = 5_200_000_000          # the 27B's median deliberation (CLAIMS.md), emulated
WINDOW = 16_000_000_000
BALLOON_ORDER = ("red", "blue", "yellow")   # spec schema of balloon_select
PIPELINE_LOG = ROOT / "pipeline.log"


def say(s: str = "") -> None:
    print(s, flush=True)


def mark(s: str) -> None:
    with PIPELINE_LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} k2 {s}\n")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def ms(x):
    return "-" if x is None else f"{x / 1e6:.0f}"


# ---------------------------------------------------------------- specs
def scenario(scenario_id: str, seed: int, window_ns: int, skin: str, **extra) -> dict:
    return {"scenario_id": scenario_id, "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
            "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": window_ns,
            "frame_period_ns": PERIOD, "episode_horizon_ns": 500_000_000 + window_ns + 500_000_000,
            "irrelevant_motion": extra.pop("irrelevant_motion", False), "skin": skin, **extra}


def spec(scn: dict, ctrl: dict, route: str = "simulated_fixed_delay") -> TrialSpec:
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": scn, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": ctrl, "observation": {"boundary_version": 1, "publication_delay_ns": PUB},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": route, "delivery_delay_ns": DELIVERY, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


def ctrl(policy: str, binder: str = "cuda", delib: int | None = None, **kw) -> dict:
    delib = DELIB if delib is None else delib   # DELIB may be overridden by --delib-ms (pilot 2026-08-27)
    return controller_dict(planner="detector", binder=binder, at_commit_policy=policy, compute_ns=delib,
                           restart_threshold_px=6 if policy == "restart_if_moved" else None, **kw)


def factory(s: TrialSpec):
    return build(s)


# ---------------------------------------------------------------- one trial
def one(s: TrialSpec, out: Path, store: PayloadStore, name: str) -> dict:
    """A single-response trial; the per-trial record used by every single-response table."""
    c = build(s)
    try:
        p = run_trial(s, out / f"{name}.jsonl", store, controller=c)
    finally:
        c.close()
    t = Trace.load(p)
    r = evaluate_trace(t)
    corr = t.first("coordinate_correction") or {}
    iv = r.intervals_ns or {}
    return {"name": name, "valid_trial": r.valid_trial, "sr": bool(r.valid_at_receipt), "sr_proposed": bool(r.proposed_valid_at_receipt),
            "G": r.return_G, "err": getattr(r, "spatial_error_px", None), "err_proposed": r.proposed_spatial_error_px,
            "E_A": iv.get("E_A"), "B_D": r.first_b_to_d_ns, "restarts": r.n_restarts, "labels": list(r.labels),
            "binder": corr.get("binder"), "binding": corr.get("binding"), "matched": corr.get("matched"),
            "moved_px": corr.get("moved_px"), "track_ns": corr.get("track_ns"), "residency_ok": corr.get("residency_ok"),
            "evidence_age_ns": corr.get("evidence_age_ns"), "violations": list(r.contract_violations)}


def cell_row(tag: str, rows: list[dict]) -> str:
    n = len(rows)
    v = [r for r in rows if r["valid_trial"]]
    k = sum(r["sr"] for r in v)
    kp = sum(r["sr_proposed"] for r in v)
    lo, hi = wilson(k, len(v))
    labs: dict[str, int] = {}
    for r in v:
        for l in r["labels"]:
            labs[l] = labs.get(l, 0) + 1
    labs_s = ",".join(f"{a}:{b}" for a, b in sorted(labs.items())) or "-"
    res = sum(1 for r in v if r["residency_ok"] is True)
    return (f"| {tag} | {k}/{len(v)} [{lo:.2f},{hi:.2f}] | {kp}/{len(v)} | {f(mean([r['G'] for r in v]), 1)} | "
            f"{f(mean([r['err'] for r in v]), 1)} | {f(mean([r['err_proposed'] for r in v]), 1)} | {ms(mean([r['E_A'] for r in v]))} | "
            f"{ms(mean([r['B_D'] for r in v]))} | {f(mean([r['restarts'] for r in v]), 2)} | "
            f"{f((median([r['track_ns'] for r in v]) or 0) / 1e3, 0)} | {res}/{len(v)} | {n - len(v)} | {labs_s} |")


HEAD = ("| cell | SR [Wilson 95%] | SR proposed (counterfactual) | mean G | mean err px | proposed err px | mean E→A ms | "
        "mean B→D ms | restarts | median bind µs | residency ok | invalid | labels |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def run_cells(cells: list[tuple[str, callable]], seeds: list[int], out: Path, store: PayloadStore, log) -> dict[str, list[dict]]:
    """cells: (tag, seed -> TrialSpec). Every trial is guarded: an exception is logged and the trial recorded as invalid."""
    results: dict[str, list[dict]] = {tag: [] for tag, _ in cells}
    total, done, t0 = len(cells) * len(seeds), 0, time.time()
    for seed in seeds:                       # seed-major so that every cell fills at the same rate (a partial night is still balanced)
        for tag, mk in cells:
            name = f"{tag}_s{seed}"
            try:
                rec = one(mk(seed), out, store, name)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"name": name, "valid_trial": False, "sr": False, "sr_proposed": False, "G": None, "err": None,
                       "err_proposed": None, "E_A": None, "B_D": None, "restarts": 0, "labels": [], "binder": None, "binding": None,
                       "matched": None, "moved_px": None, "track_ns": None, "residency_ok": None, "evidence_age_ns": None,
                       "violations": [f"exception: {e!r}"]}
            results[tag].append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            done += 1
            if done % 10 == 0 or done == total:
                el = time.time() - t0
                say(f"  {done}/{total} trials, {el / 60:.1f} min, eta {el / done * (total - done) / 60:.1f} min")
    return results


# ---------------------------------------------------------------- parts
def part_r1(seeds, out, store, log, smoke):
    windows = [1, 2, 4, 8, 16]
    rows = {w: [] for w in windows}
    for seed in seeds:
        for w in windows:
            s = spec(scenario("revealed_target", seed, w * 1_000_000_000, "notification"), ctrl("late_bind"))
            tag = f"w{w}_s{seed}_"
            try:
                res = run_triplet(s, out, tag, factory)
                r = {p: res["results"][p] for p in ("paused", "replay", "live")}
                pair = res["pair"]
                pair_ok = pair.get("valid", pair.get("ok")) if isinstance(pair, dict) else getattr(pair, "valid", getattr(pair, "ok", None))
                rec = {"seed": seed, "w": w, "pair_ok": pair_ok, "pair": json.loads(json.dumps(pair, default=str)),
                       **{p: {"valid_trial": r[p].valid_trial, "sr": bool(r[p].valid_at_receipt), "G": r[p].return_G,
                              "E_A": (r[p].intervals_ns or {}).get("E_A"), "labels": list(r[p].labels)} for p in r}}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"seed": seed, "w": w, "pair_ok": None, "pair": repr(e),
                       **{p: {"valid_trial": False, "sr": False, "G": None, "E_A": None, "labels": []} for p in ("paused", "replay", "live")}}
            rows[w].append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
        say(f"  seed {seed} done ({seeds.index(seed) + 1}/{len(seeds)})")
    L = ["| window s | Paused SR | Replay SR | Live SR [Wilson] | Live mean G | Live mean E→A ms | pairs valid | Live labels |",
         "|---|---|---|---|---|---|---|---|"]
    for w in windows:
        rs = rows[w]

        def sr(p):
            v = [x[p] for x in rs if x[p]["valid_trial"]]
            return f"{sum(x['sr'] for x in v)}/{len(v)}"
        lv = [x["live"] for x in rs if x["live"]["valid_trial"]]
        k = sum(x["sr"] for x in lv)
        lo, hi = wilson(k, len(lv))
        labs: dict[str, int] = {}
        for x in lv:
            for l in x["labels"]:
                labs[l] = labs.get(l, 0) + 1
        L.append(f"| {w} | {sr('paused')} | {sr('replay')} | {k}/{len(lv)} [{lo:.2f},{hi:.2f}] | {f(mean([x['G'] for x in lv]), 1)} | "
                 f"{ms(mean([x['E_A'] for x in lv]))} | {sum(1 for x in rs if x['pair_ok'])}/{len(rs)} | "
                 f"{','.join(f'{a}:{b}' for a, b in sorted(labs.items())) or '-'} |")
    return ["## revealed_target (notification), chronotope detector planner, late_bind, deliberation 5.2 s, seeds "
            f"{seeds[0]}..{seeds[-1]}", "", "Paused/Replay/Live per window (Replay derived from the Live trace).", ""] + L


def part_c1(seeds, out, store, log, smoke):
    cells = []
    for v in (0, 4, 8, 16):
        for pol in ("frozen", "late_bind", "restart_if_moved"):
            cells.append((f"v{v}_{pol}", lambda seed, v=v, pol=pol: spec(
                scenario("moving_target", seed, WINDOW, "balloon", velocity_x_px_per_s=v, velocity_y_px_per_s=0), ctrl(pol))))
    for pol in ("frozen", "late_bind"):
        cells.append((f"v0_irrelevant_{pol}", lambda seed, pol=pol: spec(
            scenario("moving_target", seed, WINDOW, "balloon", irrelevant_motion=True, velocity_x_px_per_s=0, velocity_y_px_per_s=0), ctrl(pol))))
    res = run_cells(cells, seeds, out, store, log)
    return [f"## moving_target (balloon), chronotope detector planner, deliberation 5.2 s, window 16 s, seeds {seeds[0]}..{seeds[-1]}", "",
            HEAD] + [cell_row(tag, res[tag]) for tag, _ in cells]


def part_c2(seeds, out, store, log, smoke):
    cells = []
    for jd in (2, 14):
        for pol in ("frozen", "late_bind"):
            cells.append((f"jump{jd}s_{pol}", lambda seed, jd=jd, pol=pol: spec(
                scenario("jump_target", seed, WINDOW, "balloon", jump_delay_ns=jd * 1_000_000_000, jump_dx_px=96, jump_dy_px=0), ctrl(pol))))
    res = run_cells(cells, seeds, out, store, log)
    return [f"## jump_target (balloon), jump (96, 0) px at 2 s (inside B→D) and 14 s (after A), deliberation 5.2 s, seeds {seeds[0]}..{seeds[-1]}",
            "", HEAD] + [cell_row(tag, res[tag]) for tag, _ in cells]


def part_c3(seeds, out, store, log, smoke):
    cells = []
    for pol in ("frozen", "late_bind"):
        cells.append((f"select_v8_{pol}", lambda seed, pol=pol: spec(
            scenario("balloon_select", seed, WINDOW, "balloon", velocity_x_px_per_s=8, named_color=BALLOON_ORDER[seed % 3]), ctrl(pol))))
    res = run_cells(cells, seeds, out, store, log)
    return [f"## balloon_select (three balloons, named colour = BALLOON_ORDER[seed % 3]), v = 8 px/s, rgb template, deliberation 5.2 s, seeds {seeds[0]}..{seeds[-1]}",
            "", HEAD] + [cell_row(tag, res[tag]) for tag, _ in cells]


def part_k1c(seeds, out, store, log, smoke):
    W, DL = 1_300_000_000, 1_000_000_000
    cells = []
    for v in (8, 100, 200):
        for binder in ("cuda", "tracker_py"):
            cells.append((f"v{v}_late_bind_{binder}", lambda seed, v=v, b=binder: spec(
                scenario("moving_target", seed, W, "balloon", velocity_x_px_per_s=v, velocity_y_px_per_s=0), ctrl("late_bind", binder=b, delib=DL))))
        cells.append((f"v{v}_frozen_cuda", lambda seed, v=v: spec(
            scenario("moving_target", seed, W, "balloon", velocity_x_px_per_s=v, velocity_y_px_per_s=0), ctrl("frozen", delib=DL))))
    res = run_cells(cells, seeds, out, store, log)
    return [f"## K1c: moving_target (balloon), deliberation 1.0 s, window 1.3 s, resident NCC (cuda) vs host tracker.py binder, seeds {seeds[0]}..{seeds[-1]}",
            "", HEAD] + [cell_row(tag, res[tag]) for tag, _ in cells]


def pending_spec(seed: int, delay_s: int, memory: str, route: str, validate_kind: bool = False) -> TrialSpec:
    return spec(scenario("pending_effect", seed, WINDOW, "notification", effect_delay_ns=delay_s * 1_000_000_000),
                ctrl("late_bind", memory=memory, pending_timeout_ns=10_000_000_000, pending_radius_px=10, validate_kind=validate_kind), route)


def run_pending(cells, seeds, out, store, log, title):
    res: dict[str, list[dict]] = {tag: [] for tag, _ in cells}
    total, done, t0 = len(cells) * len(seeds), 0, time.time()
    for seed in seeds:
        for tag, mk in cells:
            name = f"{tag}_s{seed}"
            try:
                s = mk(seed)
                c = build(s)
                try:
                    p = run_trial(s, out / f"{name}.jsonl", store, controller=c)
                finally:
                    c.close()
                t = Trace.load(p)
                r = evaluate_pending_trace(t)
                rec = {"name": name, "valid_trial": r.valid_trial, "accepted": r.accepted, "duplicates": r.duplicates,
                       "exact": r.exact_success, "invalid_clicks": r.invalid_clicks, "dropped": r.dropped, "clicks": r.n_clicks_committed,
                       "suppressed": r.n_suppressed, "decisions": r.n_decisions, "waits": r.n_waits, "events": dict(r.pending_events),
                       "violations": list(r.contract_violations)}
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                rec = {"name": name, "valid_trial": False, "accepted": False, "duplicates": 0, "exact": False, "invalid_clicks": 0,
                       "dropped": 0, "clicks": 0, "suppressed": 0, "decisions": 0, "waits": 0, "events": {}, "violations": [f"exception: {e!r}"]}
            res[tag].append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            done += 1
            if done % 5 == 0 or done == total:
                el = time.time() - t0
                say(f"  {done}/{total} trials, {el / 60:.1f} min, eta {el / done * (total - done) / 60:.1f} min")
    L = [title, "", "| cell | accepted | exact (accepted, 0 duplicates) [Wilson] | mean duplicates | mean invalid clicks | dropped | "
         "mean clicks | mean suppressed | mean decisions | invalid trials |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tag, _ in cells:
        rs = [r for r in res[tag] if r["valid_trial"]]
        k = sum(r["exact"] for r in rs)
        lo, hi = wilson(k, len(rs))
        L.append(f"| {tag} | {sum(r['accepted'] for r in rs)}/{len(rs)} | {k}/{len(rs)} [{lo:.2f},{hi:.2f}] | "
                 f"{f(mean([r['duplicates'] for r in rs]))} | {f(mean([r['invalid_clicks'] for r in rs]))} | {sum(r['dropped'] for r in rs)} | "
                 f"{f(mean([r['clicks'] for r in rs]))} | {f(mean([r['suppressed'] for r in rs]))} | {f(mean([r['decisions'] for r in rs]))} | "
                 f"{len(res[tag]) - len(rs)} |")
    return L


def part_p1(seeds, out, store, log, smoke):
    cells = []
    for d in (1, 2, 8):
        for mem in ("none", "harness_pending"):
            cells.append((f"delay{d}s_{mem}", lambda seed, d=d, mem=mem: pending_spec(seed, d, mem, "simulated_fixed_delay")))
    for mem in ("none", "harness_pending"):
        cells.append((f"drop_first_delay2s_{mem}", lambda seed, mem=mem: pending_spec(seed, 2, mem, "simulated_drop_first")))
    return run_pending(cells, seeds, out, store, log,
                       f"## pending_effect (notification), chronotope detector planner, multi-action (a decision on every changed frame), "
                       f"deliberation 5.2 s, pending record radius 10 px / timeout 10 s, seeds {seeds[0]}..{seeds[-1]}")


def part_p1_kind(seeds, out, store, log, smoke):
    cells = [(f"delay{d}s_harness_pending_validate_kind", lambda seed, d=d: pending_spec(seed, d, "harness_pending", "simulated_fixed_delay", True))
             for d in (1, 2, 8)]
    return run_pending(cells, seeds, out, store, log,
                       "## pending_effect with the kind check at D (a proposal whose target is gone in the newest frame is discarded), "
                       f"harness_pending, deliberation 5.2 s, seeds {seeds[0]}..{seeds[-1]}")


PARTS = {"r1": part_r1, "c1": part_c1, "c2": part_c2, "c3": part_c3, "k1c": part_k1c, "p1": part_p1, "p1_kind": part_p1_kind}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=sorted(PARTS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="1 seed per cell into runs/k2_smoke_<part>/")
    ap.add_argument("--delib-ms", type=int, default=None, help="override the emulated deliberation (default 5200 ms)")
    ap.add_argument("--tag", default="", help="suffix for the run/results name (required with --delib-ms)")
    a = ap.parse_args()
    if a.delib_ms is not None:
        global DELIB
        DELIB = a.delib_ms * 1_000_000
        if not a.tag:
            ap.error("--delib-ms needs --tag so runs/k2_<part> is not overwritten")
    seeds = [a.seed0] if a.smoke else list(range(a.seed0, a.seed0 + a.seeds))
    name = f"k2_{'smoke_' if a.smoke else ''}{a.part}{a.tag}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(out / "payloads")
    mark(f"{name} start pid={os.getpid()} seeds={seeds[0]}..{seeds[-1]}")
    t0 = time.time()
    say(f"{name}: seeds {seeds[0]}..{seeds[-1]} -> {out}")
    with (out / "trials.jsonl").open("w") as log:
        lines = PARTS[a.part](seeds, out, store, log, a.smoke)
    el = time.time() - t0
    head = [f"# K2 {a.part}{' (SMOKE, 1 seed)' if a.smoke else ''}: the GPU-resident agent on the software track", "",
            f"Generated by studies/k2_resident_sweep.py --part {a.part} on {time.strftime('%Y-%m-%d %H:%M')} ({el / 60:.1f} min).",
            "Software track: simulated 30 ms publication, 20 ms delivery, 60 Hz frames, no HID. Planner: the resident colour detector",
            "with an emulated deliberation (a wall-clock sleep after B). Binder: the resident NCC (chronotope-1 = tracker.py v2 on the device)",
            "unless the cell says tracker_py. The GPU was shared with the running G2 (vLLM Qwen3.8-27B) study and with the other K2 lanes;",
            "wall-clock intervals (E→A, bind µs) carry that contention. Seeds are the same across cells; Wilson 95% intervals.", ""]
    md = ROOT / "results" / f"K2_{'smoke_' if a.smoke else ''}{a.part}{a.tag}.md"
    md.write_text("\n".join(head + lines) + "\n")
    say(f"wrote {md}")
    mark(f"{name} done ({el / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
