"""Replay–Live cadence check (round-8 critique §1): for every retained Replay/Live pair in the paper's 100-seed
P/R/L runs, verify that Replay published the same observation identifiers (source event, source time, payload hash)
at the same harness offsets relative to episode origin as the Live trace it was built from.

Reads traces only; writes results/replay_cadence_check.md.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.pair import validate_pair  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

RUNS = ["a1_qwen38_nothink_100", "a1_static_qwen38_nothink_100", "r1_qwen38_notification_100",
        "c_prl_qwen38_nothink_100", "b1_qwen38_nothink_100"]


def offsets(t: Trace) -> list[tuple[int, int, str, int]]:
    h0 = t.header()["harness_origin_ns"]
    return [(o["source_event_id"], o["source_time_ns"], o["payload_hash"], o["publication_time_ns"] - h0)
            for o in t.all("observation")]


def main() -> None:
    lines = ["# Replay–Live cadence check", "",
             "For each Replay trace, its source Live trace (by recorded hash) is loaded; observation identity "
             "(source event id, source time, payload hash) and publication offset from harness origin are compared "
             "index by index over the observations Replay published (Replay may stop earlier because the "
             "response stays valid).", "",
             "| Run | Pairs | validate_pair ok | Identity mismatches | Max \\|Δ offset\\| (ns) | Pairs with Δ ≠ 0 |",
             "|---|---:|---:|---:|---:|---:|"]
    grand = defaultdict(int)
    for run in RUNS:
        d = ROOT / "runs" / run
        if not d.exists():
            lines.append(f"| {run} | (missing) | | | | |")
            continue
        live_by_hash = {}
        for p in d.glob("*_live.jsonl"):
            t = Trace.load(p)
            live_by_hash[t.file_hash()] = t
        n = ok = mism = nonzero = 0
        maxd = 0
        for p in sorted(d.glob("*_replay.jsonl")):
            r = Trace.load(p)
            live = live_by_hash.get(r.header()["replay_source_live_trace_sha256"])
            if live is None:
                mism += 1
                continue
            n += 1
            v = validate_pair(r, live)
            ok += v.valid
            orr, ol = offsets(r), offsets(live)
            for a, b in zip(orr, ol):
                if a[:3] != b[:3]:
                    mism += 1
                    break
            ds = [abs(a[3] - b[3]) for a, b in zip(orr, ol)]
            m = max(ds) if ds else 0
            maxd = max(maxd, m)
            nonzero += m != 0
        lines.append(f"| {run} | {n} | {ok} | {mism} | {maxd} | {nonzero} |")
        for k, v in (("pairs", n), ("ok", ok), ("mism", mism), ("nonzero", nonzero)):
            grand[k] += v
        grand["maxd"] = max(grand["maxd"], maxd)
    lines += ["", f"Total: {grand['pairs']} pairs; {grand['ok']} valid; {grand['mism']} identity mismatches; "
              f"max |Δ offset| {grand['maxd']} ns; {grand['nonzero']} pairs with any nonzero offset difference."]
    out = ROOT / "results" / "replay_cadence_check.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
