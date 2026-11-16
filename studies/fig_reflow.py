"""Appendix figure: the Chromium-rendered page before and after the banner reflow (assets/skin_browser, seed 1000)."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
A = ROOT / "assets/skin_browser"
meta = json.loads((A / f"640x480_s{seed}.json").read_text())
ims = []
for name, key in (("before", "button_before"), ("after", "button_after")):
    im = Image.frombytes("RGB", (meta["w"], meta["h"]), (A / f"640x480_s{seed}_{name}.bin").read_bytes())
    d = ImageDraw.Draw(im); b = meta[key]
    d.rectangle([b["x"] - 3, b["y"] - 3, b["x"] + b["width"] + 3, b["y"] + b["height"] + 3], outline=(200, 30, 30), width=3)
    ims.append(im)
out = Image.new("RGB", (2 * meta["w"] + 20, meta["h"]), "white")
out.paste(ims[0], (0, 0)); out.paste(ims[1], (meta["w"] + 20, 0))
p = ROOT / "paper_chronotope_iclr2027/figures/reflow_before_after.png"; out.save(p); print(p, "dy", meta["dy"])
