"""M4c y-diagnostic (2026-08-27): why did EvoCUA-32B in its native grammar (prompt 16) put every click on the image's
vertical midline (raw y = 500..545 of 999) while x was within 7 px of the button? Same frame, same request path as the
M4 cells, four variants; prints the raw 0-999 answer next to the button's true grid position. No harness, no scoring.
  as_is     640x480 (image_scale 2), prompt 18 (16 + the no-wait sentence)
  scale3    960x720
  wide      the 320x240 canvas padded on the right to 16:9 (427x240) before the x2 upscale (854x480)
  history   as_is with one previous action listed ("wait") instead of "None"
"""
from __future__ import annotations
import argparse, json, sys
sys.path.insert(0, "src")
from chronotope.controller import (EVOCUA_USER_TEMPLATE, UITARS_INSTRUCTION_NOWAIT, LocalOpenAIAdapter, SYSTEM_PROMPTS,
                                   parse_evocua)
from chronotope.png import encode_png, to_rgb
from chronotope.scenario import manifest_from_dict

def manifest(seed: int):
    return manifest_from_dict({"scenario_id": "revealed_target", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                               "target_size_px": 24, "event_time_ns": 500_000_000, "response_window_ns": 8_000_000_000,
                               "frame_period_ns": 16_666_667, "episode_horizon_ns": 9_000_000_000, "irrelevant_motion": False,
                               "skin": "notification"})

def pad_right(rgb: bytes, w: int, h: int, new_w: int, fill=(233, 236, 239)) -> bytes:
    row = bytes(fill) * (new_w - w)
    return b"".join(rgb[y * w * 3:(y + 1) * w * 3] + row for y in range(h))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--seed-start", type=int, default=200); ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ad = LocalOpenAIAdapter(base_url=a.base_url, model_id=a.model, max_tokens=4096, temperature=0.0, enable_thinking=True, reasoning_effort=None)
    system = SYSTEM_PROMPTS[18]
    user = EVOCUA_USER_TEMPLATE.format(instruction=UITARS_INSTRUCTION_NOWAIT["notification"])
    user_hist = user.replace("Previous actions:\nNone", "Previous actions:\nStep 1: wait")
    rows = []
    for seed in range(a.seed_start, a.seed_start + a.n_seeds):
        m = manifest(seed); rgb = to_rgb(m.render(600_000_000), 320, 240); cx, cy = m.target_box().center()
        variants = {
            "as_is": (encode_png(rgb, 320, 240, 2), 320, 240, user),
            "scale3": (encode_png(rgb, 320, 240, 3), 320, 240, user),
            "wide": (encode_png(pad_right(rgb, 320, 240, 427), 427, 240, 2), 427, 240, user),
            "history": (encode_png(rgb, 320, 240, 2), 320, 240, user_hist),
        }
        for name, (png, w, h, utext) in variants.items():
            r = ad.invoke(png, system, utext, structured=False, parser=parse_evocua)
            p = r.parsed if isinstance(r.parsed, dict) else {}
            gx, gy = (round(cx * 999 / w), round(cy * 999 / h))
            row = {"seed": seed, "variant": name, "answer": p.get("action"), "raw_x": p.get("x"), "raw_y": p.get("y"),
                   "true_grid_x": gx, "true_grid_y": gy,
                   "err_x_px": None if p.get("x") is None else round(p["x"] * w / 999 - cx),
                   "err_y_px": None if p.get("y") is None else round(p["y"] * h / 999 - cy),
                   "raw_text": r.detail.get("raw_text")}
            rows.append(row); print(json.dumps({k: v for k, v in row.items() if k != "raw_text"}), flush=True)
    with open(a.out, "w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    print("# summary (variant: clicks, |err_x|<=12 px, |err_y|<=12 px)")
    for name in ("as_is", "scale3", "wide", "history"):
        rs = [r for r in rows if r["variant"] == name and r["answer"] == "click"]
        print(name, len(rs), sum(abs(r["err_x_px"]) <= 12 for r in rs), sum(abs(r["err_y_px"]) <= 12 for r in rs))

if __name__ == "__main__":
    main()
