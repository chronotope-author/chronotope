"""Fara1.5 card-blindness diagnosis (2026-09-02). In every closing_window episode of queueFARAb/c the 9B answers "No
notification card is currently visible" on the first frame after the card appears (0/130 first decisions), while the 4B
clicks the card (118/120) and the 9B clicks the balloon on moving_target / static_control (120/120). Same engine
build parameters (min 64 / max 2048 image tokens), same prompt 21, same image_scale 2. The card canvas is 320x240
(640x480 to the model, 300 image tokens); the balloon canvas is 640x480 (1280x960, 1200 tokens). This script asks one
engine directly, on the rendered first target frame of the manifest's episodes, at image scales 2 / 3 / 4 and with the
balloon frame as a control. Same adapter, prompts and PNG path as the D1 rows; no harness, no scoring.

  PYTHONPATH=src:chronotope/python .venv/bin/python studies/fara_card_diag.py --tag fara9b --n 3 --out runs/fara9b_card_diag.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, "src"); sys.path.insert(0, "chronotope/python")
from chronotope.controller import FARA_USER_TEMPLATE, SYSTEM_PROMPTS, UITARS_INSTRUCTION_NOWAIT, parse_evocua
from chronotope.distributional.manifest import EpisodeRow, build_manifest, first_target_frame_ns
from chronotope.png import encode_png
from chronotope_core.edgellm import EdgeLLMAdapter

ENGINES = {"fara9b": ("Fara1.5-9B", "~/tensorrt-edgellm-workspace/Fara1.5-9B/engines"),
           "fara4b": ("Fara1.5-4B", "~/tensorrt-edgellm-workspace/Fara1.5-4B/engines")}


def rows(fam: str, n: int) -> list[EpisodeRow]:
    out = []
    for line in open(f"runs/d1/manifests/{fam}.jsonl"):
        r = json.loads(line)
        if r.get("record") == "manifest_header":
            continue
        r.pop("record", None); out.append(EpisodeRow.from_dict(r))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=sorted(ENGINES), required=True); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--scales", default="2,3,4"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id, root = ENGINES[a.tag]
    ad = EdgeLLMAdapter(engines_root=root, model_id=model_id, max_tokens=256, temperature=0.0)
    system = SYSTEM_PROMPTS[21]
    variants = []
    for row in rows("closing_window", a.n):
        m = build_manifest(row); t = first_target_frame_ns(m.event_time_ns, m.frame_period_ns); rgb = m.render(t)
        user = FARA_USER_TEMPLATE.format(instruction=UITARS_INSTRUCTION_NOWAIT["notification"])
        for s in (int(x) for x in a.scales.split(",")):
            variants.append((row.episode_id, f"card_x{s}", encode_png(rgb, m.canvas_w_px, m.canvas_h_px, s), user, m.target_box_at(t).center(), s))
    for row in rows("static_control", 1):
        m = build_manifest(row); t = first_target_frame_ns(m.event_time_ns, m.frame_period_ns); rgb = m.render(t)
        user = FARA_USER_TEMPLATE.format(instruction=UITARS_INSTRUCTION_NOWAIT["balloon"])
        variants.append((row.episode_id, "balloon_x2", encode_png(rgb, m.canvas_w_px, m.canvas_h_px, 2), user, m.target_box_at(t).center(), 2))
    with open(a.out, "w") as f:
        for ep, name, png, user, (cx, cy), s in variants:
            r = ad.invoke(png, system, user, structured=False, parser=parse_evocua)
            p = r.parsed if isinstance(r.parsed, dict) else {}
            rec = {"tag": a.tag, "episode": ep, "variant": name, "answer": p.get("action"), "x_1000": p.get("x"), "y_1000": p.get("y"),
                   "target_center_canvas": [cx, cy], "output_tokens": r.detail.get("output_tokens"),
                   "runtime_s": round((r.detail.get("runtime_ns") or 0) / 1e9, 2), "raw_text": r.detail.get("raw_text")}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps({k: v for k, v in rec.items() if k != "raw_text"}), "::", (rec["raw_text"] or "")[:150].replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
