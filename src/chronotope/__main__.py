"""CLI.

  python -m chronotope triplet SPEC.json OUT_DIR        run live, replay(live), paused; evaluate; validate pair
  python -m chronotope evaluate TRACE.jsonl              evaluate one trace
  python -m chronotope validate-pair REPLAY.jsonl LIVE.jsonl
  python -m chronotope sweep SPEC.json OUT_DIR --compute-ns a,b,.. --delivery-ns c,d,..
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .evaluate import evaluate_trace, summarize
from .pair import validate_pair
from .runner import PayloadStore, run_trial
from .spec import TrialSpec
from .trace import Trace


def run_triplet(spec: TrialSpec, out: Path, tag: str = "", controller_factory=None) -> dict:
    """Live first; Replay is derived from the Live trace; Paused is independent.
    `controller_factory(spec) -> Controller` builds a fresh controller per condition (VLM runs)."""
    if spec.policy != "live":
        raise SystemExit("triplet base spec must have task_clock_policy == live")
    out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(out / "payloads")
    mk = (lambda s: None) if controller_factory is None else controller_factory
    live_p = run_trial(spec, out / f"{tag}live.jsonl", store, controller=mk(spec))
    live = Trace.load(live_p)
    replay_spec = spec.derive("replay", live.file_hash())
    replay_p = run_trial(replay_spec, out / f"{tag}replay.jsonl", store, live_trace=live, controller=mk(replay_spec))
    paused_spec = spec.derive("paused", None)
    paused_p = run_trial(paused_spec, out / f"{tag}paused.jsonl", store, controller=mk(paused_spec))
    res = {p: evaluate_trace(Trace.load(out / f"{tag}{p}.jsonl")) for p in ("paused", "replay", "live")}
    verdict = validate_pair(Trace.load(replay_p), live)
    return {"results": res, "pair": verdict}


def _fmt_row(tag: str, res: dict) -> str:
    cells = []
    for p in ("paused", "replay", "live"):
        r = res[p]
        v = "INVALID" if not r.valid_trial else ("1" if r.valid_at_receipt else "0")
        cells.append(f"{p}={v}")
    live = res["live"]
    iv = live.intervals_ns
    labs = ",".join(live.labels) or "-"
    return (f"{tag:<28} {' '.join(cells)}  live E→O={iv.get('E_O')} O→D={iv.get('O_D')} D→A={iv.get('D_A')} "
            f"E→A={iv.get('E_A')} S_A={iv.get('S_A')}  labels[live]={labs}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="chronotope")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("triplet"); s.add_argument("spec", type=Path); s.add_argument("out", type=Path)
    s = sub.add_parser("evaluate"); s.add_argument("trace", type=Path)
    s = sub.add_parser("validate-pair"); s.add_argument("replay", type=Path); s.add_argument("live", type=Path)
    s = sub.add_parser("sweep"); s.add_argument("spec", type=Path); s.add_argument("out", type=Path)
    s.add_argument("--compute-ns", required=True); s.add_argument("--delivery-ns", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "evaluate":
        print(json.dumps(evaluate_trace(Trace.load(a.trace)).to_dict(), indent=2))
        return 0
    if a.cmd == "validate-pair":
        v = validate_pair(Trace.load(a.replay), Trace.load(a.live))
        print(json.dumps(v.__dict__, indent=2))
        return 0 if v.valid else 1
    spec = TrialSpec.from_path(a.spec)
    if a.cmd == "triplet":
        out = run_triplet(spec, a.out)
        for p, r in out["results"].items():
            print(json.dumps(r.to_dict(), indent=2))
        print("pair:", json.dumps(out["pair"].__dict__))
        return 0 if out["pair"].valid else 1
    if a.cmd == "sweep":
        computes = [int(x) for x in a.compute_ns.split(",")]
        delays = [int(x) for x in a.delivery_ns.split(",")]
        m = spec.manifest()
        print(f"event_time_ns={m.event_time_ns} response_window_ns={m.response_window_ns} "
              f"frame_period_ns={m.frame_period_ns} publication_delay_ns={spec.raw['observation']['publication_delay_ns']}")
        all_invalid_pairs = 0
        for c in computes:
            for d in delays:
                raw = json.loads(json.dumps(spec.raw))
                raw["controller"]["compute_ns"] = c
                raw["delivery"]["delivery_delay_ns"] = d
                sp = TrialSpec.from_dict(raw)
                tag = f"c{c}_d{d}_"
                out = run_triplet(sp, a.out, tag)
                all_invalid_pairs += int(not out["pair"].valid)
                print(_fmt_row(f"compute={c} delivery={d}", out["results"]))
        print(f"invalid pairs: {all_invalid_pairs}")
        return 0 if all_invalid_pairs == 0 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
