"""K1 (U-K2): replay every recorded binding through the resident binder and through tracker.py v2.

For each trace with a coordinate_correction record, the frame at B (the proposal's observation), the
newest frame at D (the correction's observation), the proposed click, the template radius and the
channels are read back from the trace and its payload store; both binders run on exactly those inputs.
Agreement = same (x, y), same matched and ambiguous verdicts; scores compared to 1e-4. The recorded
commit is compared as well (older runs used tracker v1, which lacked the second-peak guard).

  PYTHONPATH=src:chronotope/python .venv/bin/python studies/k1_binder_replay.py --runs c1_qwen38_nothink_100 ... \\
      --out results/K1_binder_replay.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "chronotope" / "python"))

import chronotope_core  # noqa: E402
from chronotope.png import to_rgb  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from chronotope.tracker import AMBIGUITY_MARGIN, PLATEAU_MAX, cut_template, track  # noqa: E402

DEFAULT_RUNS = ["c1_qwen38_nothink_100", "c2_qwen38_nothink_100", "c3_qwen38_nothink_100", "c1_uitars_native_100",
                "g1_reflow_qwen38_nothink_100"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "K1_binder_replay.md")
    ap.add_argument("--limit", type=int, default=None, help="traces per run (default all)")
    a = ap.parse_args()
    cores: dict[tuple, chronotope_core.Core] = {}
    rows = []
    per_run = defaultdict(lambda: {"n": 0, "xy": 0, "verdict": 0, "score_max": 0.0, "second_max": 0.0, "plateau": 0,
                                   "recorded_xy": 0, "recorded_n": 0, "py_ms": [], "cu_ms": [], "disagree": []})
    for run in a.runs:
        rd = ROOT / "runs" / run
        if not rd.exists():
            print(f"skip {run}: no such run"); continue
        store = rd / "payloads"
        paths = sorted(rd.glob("*.jsonl"))
        if a.limit:
            paths = paths[: a.limit]
        for p in paths:
            t = Trace.load(p)
            corrs = [c for c in t.all("coordinate_correction") if c.get("binding") == "tracked_at_D"]
            if not corrs:
                continue
            hdr = t.header()
            w, h = hdr["manifest"]["canvas_w_px"], hdr["manifest"]["canvas_h_px"]
            obs = {o["observation_id"]: o for o in t.all("observation")}
            for c in corrs:
                ob_b, ob_d = obs.get(c["proposal_observation_id"]), obs.get(c["observation_id"])
                if ob_b is None or ob_d is None:
                    continue
                fb = to_rgb((store / f"{ob_b['payload_hash']}.bin").read_bytes(), w, h)
                fd = to_rgb((store / f"{ob_d['payload_hash']}.bin").read_bytes(), w, h)
                px, py_ = c["proposed_action"]["x"], c["proposed_action"]["y"]
                radius, rgb = c["template_radius_px"], c.get("binder_channels", "grey") == "rgb"
                min_score = c["min_match_score"]
                key = (w, h, radius)
                if key not in cores:
                    cores[key] = chronotope_core.Core(width=w, height=h, template_radius_px=radius, min_match_score=min_score)
                core = cores[key]
                t0 = time.perf_counter_ns()
                m = track(cut_template(fb, w, h, px, py_, radius, rgb=rgb), fd, w, h)
                py_ms = (time.perf_counter_ns() - t0) / 1e6
                t0 = time.perf_counter_ns()
                cu = core.track(fb, fd, px, py_, radius, rgb=rgb)
                cu_ms = (time.perf_counter_ns() - t0) / 1e6
                py_amb = m.second >= m.score - AMBIGUITY_MARGIN or m.plateau > PLATEAU_MAX
                py_matched = m.score >= min_score and not py_amb
                cu_amb, cu_matched = bool(cu["ambiguous"]), bool(cu["matched"])
                s = per_run[run]
                s["n"] += 1
                xy_ok = (cu["x"], cu["y"]) == (m.x, m.y)
                v_ok = (cu_matched, cu_amb) == (py_matched, py_amb)
                s["xy"] += xy_ok; s["verdict"] += v_ok
                s["score_max"] = max(s["score_max"], abs(cu["score"] - m.score))
                s["second_max"] = max(s["second_max"], abs(cu["second"] - m.second))
                s["plateau"] += cu["plateau"] == m.plateau
                s["py_ms"].append(py_ms); s["cu_ms"].append(cu_ms)
                rec_x, rec_y = c["corrected_action"]["x"], c["corrected_action"]["y"]
                cu_final = (cu["x"], cu["y"]) if cu_matched else (px, py_)
                cu_final = (min(max(0, cu_final[0]), w - 1), min(max(0, cu_final[1]), h - 1))
                s["recorded_n"] += 1; s["recorded_xy"] += (cu_final == (rec_x, rec_y))
                if not (xy_ok and v_ok):
                    s["disagree"].append(f"{p.name}: cuda ({cu['x']},{cu['y']}) score {cu['score']:.5f} second {cu['second']:.5f} plateau {cu['plateau']} "
                                         f"matched {cu_matched} amb {cu_amb} | py ({m.x},{m.y}) score {m.score:.5f} second {m.second:.5f} plateau {m.plateau} "
                                         f"matched {py_matched} amb {py_amb}")
                elif cu_final != (rec_x, rec_y):
                    s["disagree"].append(f"{p.name}: both binders ({cu_final[0]},{cu_final[1]}) but the recorded commit was ({rec_x},{rec_y}) "
                                         f"(binder_version {c.get('binder_version', 1)})")
            rows.append(p.name)
    L = ["# K1 — the resident binder replayed on the recorded bindings (U-K2)", "",
         f"chronotope core {chronotope_core.version()}; tracker.py v2 (AMBIGUITY_MARGIN {AMBIGUITY_MARGIN}, PLATEAU_MAX {PLATEAU_MAX}). "
         "Inputs per binding: the frame at B, the proposal, the radius and channels, the newest frame at D, all from the trace and its payload store.", "",
         "| run | bindings | same (x, y) | same verdict (matched, ambiguous) | same plateau | max |Δscore| | max |Δsecond| | same as the recorded commit | tracker.py median ms | resident probe median ms |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    tot = {"n": 0, "xy": 0, "verdict": 0, "rec": 0, "recn": 0}
    for run in a.runs:
        s = per_run.get(run)
        if not s or not s["n"]:
            continue
        L.append(f"| {run} | {s['n']} | {s['xy']}/{s['n']} | {s['verdict']}/{s['n']} | {s['plateau']}/{s['n']} | {s['score_max']:.2e} | {s['second_max']:.2e} | "
                 f"{s['recorded_xy']}/{s['recorded_n']} | {statistics.median(s['py_ms']):.1f} | {statistics.median(s['cu_ms']):.2f} |")
        tot["n"] += s["n"]; tot["xy"] += s["xy"]; tot["verdict"] += s["verdict"]; tot["rec"] += s["recorded_xy"]; tot["recn"] += s["recorded_n"]
    L.append(f"| **all** | {tot['n']} | {tot['xy']}/{tot['n']} | {tot['verdict']}/{tot['n']} | | | | {tot['rec']}/{tot['recn']} | | |")
    L += ["", "The probe path copies both frames to the device per call (not the resident path, whose frames are already there); "
          "its time is an upper bound on the resident bind.", "", "## Disagreements", ""]
    any_d = False
    for run in a.runs:
        s = per_run.get(run)
        if s and s["disagree"]:
            any_d = True
            L.append(f"### {run}"); L += [f"- {d}" for d in s["disagree"]]; L.append("")
    if not any_d:
        L.append("None.")
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    for c in cores.values():
        c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
