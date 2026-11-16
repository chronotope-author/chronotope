"""G1 (cards/G1-realistic-reflow.md): bake the browser rendering. For each seed, Chromium renders a seeded article page
with a "Continue" button; a notice banner inserted above it (asynchronously in the real page; here as the second
state) pushes the button down by a seeded displacement. Two frames per seed (before / after) plus the DOM geometry
of the button, read from the page, become the assets that scenario `reflow_page` renders as a pure function of time
(the same mechanism as the calendar rendering). The button colour is unique in the page so the scripted oracle can
find it by colour, as on every other rendering.

  .venv/bin/python site/make_reflow_assets.py --seeds 100-109,1000-1099
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "skin_browser"
W, H = 640, 480
BUTTON_RGB = (79, 70, 229)          # indigo; TARGET_COLOR["browser"]
TITLES = ["Quarterly report", "Release notes", "Terms update", "Meeting summary", "Order confirmation", "Draft proposal",
          "Weekly digest", "Invoice 2049", "Onboarding guide", "Security notice"]
NOTICES = ["A new version of this page is available.", "Cookies help us improve this site.",
           "Your session will expire in 5 minutes.", "3 new comments on this document.", "Connection restored."]
WORDS = ("the project ships a controller that keeps the decision open until the newest frame is admitted and fills the "
         "field before commitment so that a click aimed at the frame it saw is still right when it lands").split()


def page_html(seed: int) -> tuple[str, int]:
    rng = random.Random(seed)
    title = rng.choice(TITLES); notice = rng.choice(NOTICES)
    paras = []
    for _ in range(rng.randint(1, 3)):
        n = rng.randint(18, 40)
        paras.append(" ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + ".")
    top_pad = rng.randint(8, 40)
    left = rng.randint(0, 300)                      # seeded horizontal position of the button
    # keep the pushed-down button inside the viewport: text height ~ 20 px/line * ~ lines + margins
    est_text = sum(20 * (len(p) // 70 + 1) + 12 for p in paras)
    room = H - (48 + top_pad + est_text + 37 + 12 + 24)
    banner_h = rng.randint(60, max(60, min(200, room)))
    body = "".join(f"<p>{p}</p>" for p in paras)
    html = f"""<!doctype html><html><head><meta charset=utf-8><style>
    * {{ box-sizing: border-box; }} body {{ margin:0; font-family: Arial, Helvetica, sans-serif; background:#f3f4f6; color:#111827; }}
    header {{ height:48px; background:#1f2937; color:#fff; display:flex; align-items:center; padding:0 16px; font-size:16px; }}
    main {{ width:560px; margin:0 auto; padding:{top_pad}px 0 0 0; }}
    p {{ font-size:14px; line-height:20px; margin:0 0 12px 0; }}
    #banner {{ display:none; height:{banner_h}px; background:#fef3c7; border:1px solid #f59e0b; border-radius:6px; padding:12px; font-size:14px; margin:0 0 12px 0; }}
    #cta {{ display:inline-block; margin-left:{left}px; background:rgb({BUTTON_RGB[0]},{BUTTON_RGB[1]},{BUTTON_RGB[2]}); color:#fff; border:none; border-radius:4px; padding:10px 22px; font-size:15px; }}
    </style></head><body><header>{title}</header><main>{body}<div id=banner>{notice}</div><button id=cta>Continue</button></main></body></html>"""
    return html, banner_h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109,1000-1099")
    ap.add_argument("--exe", default="/usr/bin/chromium-browser")
    a = ap.parse_args()
    seeds = []
    for part in a.seeds.split(","):
        lo, _, hi = part.partition("-"); seeds += list(range(int(lo), int(hi or lo) + 1))
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.exe, headless=True, args=["--no-sandbox", "--disable-gpu", "--force-device-scale-factor=1"])
        pg = b.new_page(viewport={"width": W, "height": H})
        for seed in seeds:
            html, banner_h = page_html(seed)
            pg.set_content(html)
            pg.evaluate("document.fonts.ready")
            before = pg.locator("#cta").bounding_box()
            png_before = pg.screenshot()
            pg.evaluate("document.getElementById('banner').style.display='block'")
            after = pg.locator("#cta").bounding_box()
            png_after = pg.screenshot()
            from PIL import Image
            import io
            for name, png in (("before", png_before), ("after", png_after)):
                im = Image.open(io.BytesIO(png)).convert("RGB")
                assert im.size == (W, H), im.size
                (OUT / f"{W}x{H}_s{seed}_{name}.bin").write_bytes(im.tobytes())
            meta = {"seed": seed, "w": W, "h": H, "button_before": before, "button_after": after,
                    "dy": round(after["y"] - before["y"]), "banner_h": banner_h, "button_rgb": BUTTON_RGB}
            (OUT / f"{W}x{H}_s{seed}.json").write_text(json.dumps(meta))
            # exact-colour check: the button colour must occur only inside the button box (oracle contract)
            im = Image.open(io.BytesIO(png_after)).convert("RGB"); px = im.load()
            outside = sum(1 for y in range(0, H, 2) for x in range(0, W, 2) if px[x, y] == BUTTON_RGB
                          and not (after["x"] <= x < after["x"] + after["width"] and after["y"] <= y < after["y"] + after["height"]))
            print(f"seed {seed}: button {before['x']:.0f},{before['y']:.0f} {before['width']:.0f}x{before['height']:.0f} -> dy {meta['dy']} "
                  f"(banner {banner_h}); off-button colour pixels {outside}", flush=True)
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
