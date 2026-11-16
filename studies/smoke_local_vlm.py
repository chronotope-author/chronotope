"""One-off smoke test of the local VLM adapter: render one frame with the target, ask the model,
print latency, raw answer, and whether the click lands. Usage:
  PYTHONPATH=src .venv/bin/python studies/smoke_local_vlm.py [--thinking] [--effort low] [--n 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chronotope.controller import SYSTEM_PROMPTS, LocalOpenAIAdapter  # noqa: E402
from chronotope.png import encode_png  # noqa: E402
from chronotope.scenario import Manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--thinking", action="store_true")
ap.add_argument("--effort", default=None)
ap.add_argument("--max-tokens", type=int, default=None)
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--skin", default="plain", choices=["plain", "notification", "balloon"])
ap.add_argument("--prompt", type=int, default=None)
a = ap.parse_args()
SK = {"plain": (64, 48, 6, 10, 1), "notification": (320, 240, 24, 2, 2), "balloon": (320, 240, 24, 2, 3)}
W, H, TS, SCALE, PV = SK[a.skin]
if a.prompt: PV = a.prompt
from chronotope.controller import PROMPT_COORDS  # noqa: E402
mt = a.max_tokens or (4096 if a.thinking else 64)
ad = LocalOpenAIAdapter(base_url="http://127.0.0.1:8000/v1", model_id=a.model, max_tokens=mt, temperature=0.0,
                        enable_thinking=a.thinking, reasoning_effort=a.effort)
scale = SCALE
for seed in range(100, 100 + a.n):
    m = Manifest(scenario_seed=seed, canvas_w_px=W, canvas_h_px=H, target_size_px=TS, event_time_ns=0,
                 response_window_ns=10**9, frame_period_ns=16_666_667, episode_horizon_ns=2 * 10**9, irrelevant_motion=False, skin=a.skin)
    png = encode_png(m.render(0), W, H, scale)
    sysmsg = SYSTEM_PROMPTS[PV].format(w=W * scale, h=H * scale)
    t0 = time.monotonic()
    r = ad.invoke(png, sysmsg, "Screenshot attached. Click the target's center or wait.")
    dt = time.monotonic() - t0
    box = m.target_box()
    hit = None
    if isinstance(r.parsed, dict) and r.parsed.get("action") == "click":
        if PROMPT_COORDS[PV] == "norm1000":
            hit = box.contains(int(r.parsed["x"] * W // 1000), int(r.parsed["y"] * H // 1000))
        else:
            hit = box.contains(r.parsed["x"] // scale, r.parsed["y"] // scale)
    tc = [c*scale for c in box.center()] if PROMPT_COORDS[PV] != "norm1000" else [box.center()[0]*1000//W, box.center()[1]*1000//H]
    print(f"seed={seed} {dt*1000:.0f} ms parsed={r.parsed} target_center={tc} "
          f"hit={hit} tokens_in={r.detail['input_tokens']} out={r.detail['output_tokens']} "
          f"reasoning={r.detail['reasoning_tokens']} finish={r.detail['finish_reason']}")
