"""Regression check for the binder's uniqueness guard (2026-08-24): re-run track() with the current tracker on the
frame pairs recorded in existing late-binding traces and compare with the recorded corrections. Reports any
binding whose (x, y, matched) would change, i.e. any recorded match that the guard now calls ambiguous.

  PYTHONPATH=src .venv/bin/python studies/check_binder_uniqueness.py runs/c1_qwen38_nothink_100 runs/c2_qwen38_nothink_100 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.runner import PayloadStore  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from chronotope.tracker import AMBIGUITY_MARGIN, PLATEAU_MAX, cut_template, track  # noqa: E402


def main(runs: list[Path]) -> int:
    total = changed = ambiguous = 0
    seconds = []
    for run in runs:
        store = PayloadStore(run / "payloads")
        for p in sorted(run.glob("live_*.jsonl")):
            t = Trace.load(p)
            corrs = t.all("coordinate_correction")
            if not corrs:
                continue
            h = t.header()
            sc = h["spec"]["scenario"]; c = h["spec"]["controller"]
            w, hh = sc["canvas_w_px"], sc["canvas_h_px"]
            rgb = sc["scenario_id"] == "balloon_select"
            obs = {o["observation_id"]: o for o in t.all("observation")}
            decs = {d["decision_id"]: d for d in t.all("decision_begin")}
            for cr in corrs:
                if cr["binding"] != "tracked_at_D":
                    continue
                total += 1
                frame_b = store.get(obs[decs[cr["decision_id"]]["observation_id"]]["payload_hash"])
                frame_d = store.get(obs[cr["observation_id"]]["payload_hash"])
                px, py = cr["proposed_action"]["x"], cr["proposed_action"]["y"]
                tpl = cut_template(frame_b, w, hh, px, py, c["template_radius_px"], rgb=rgb)
                m = track(tpl, frame_d, w, hh)
                seconds.append((m.score - m.second, p.name, cr["decision_id"]))
                amb = m.second >= m.score - AMBIGUITY_MARGIN or m.plateau > PLATEAU_MAX
                matched = m.score >= c["min_match_score"] and not amb
                x, y = (m.x, m.y) if matched else (px, py)
                x, y = min(max(0, x), w - 1), min(max(0, y), hh - 1)
                ambiguous += int(amb)
                if (x, y) != (cr["corrected_action"]["x"], cr["corrected_action"]["y"]) or matched != cr["matched"]:
                    changed += 1
                    print(f"CHANGED {p.name} dec {cr['decision_id']}: recorded {cr['corrected_action']} matched={cr['matched']} "
                          f"score={cr['match_score']} -> now ({x},{y}) matched={matched} score={m.score:.3f} second={m.second:.3f}")
    seconds.sort()
    if seconds:
        print("smallest (best - second) gaps:", ", ".join(f"{g:.3f} ({n} d{d})" for g, n, d in seconds[:8]))
        print(f"bindings re-run: {total}; changed: {changed}; ambiguous under the guard (margin {AMBIGUITY_MARGIN}): "
              f"{ambiguous}; gap median {seconds[len(seconds)//2][0]:.3f}")
    return 0 if changed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main([Path(x) for x in sys.argv[1:]]))
