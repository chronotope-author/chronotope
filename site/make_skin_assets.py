"""Bake the 'calendar' skin assets for WI-010 (cards/P1-reschedule.md).

For each scenario seed, three full-canvas RGB states at the manifest's own seeded geometry:
  plain  — busy calendar week, no toast (pre-appearance / after the window closes)
  toast  — + chat notification from Priya with the Reschedule button on target_box()
           (the countdown bar is NOT baked; the renderer draws it per frame)
  done   — toast replaced by the 'Rescheduled' card; the Priya block moved to Fri 9:00

    /usr/bin/python3 site/make_skin_assets.py 100-109 400-409
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.scenario import manifest_from_dict  # noqa: E402

OUT = ROOT / "assets" / "skin_calendar"
OUT.mkdir(parents=True, exist_ok=True)
BTN_BLUE = (37, 99, 235)  # must equal TARGET_COLOR-for-clicking: unique on screen
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTB if bold else FONT, size)


def manifest(seed: int):
    return manifest_from_dict({
        "scenario_id": "pending_effect", "scenario_seed": seed, "canvas_w_px": 640, "canvas_h_px": 480,
        "target_size_px": 48, "event_time_ns": 500_000_000, "response_window_ns": 16_000_000_000,
        "frame_period_ns": 16_666_667, "episode_horizon_ns": 17_000_000_000, "irrelevant_motion": False,
        "skin": "calendar", "effect_delay_ns": 8_000_000_000})


def draw_state(m, state: str) -> Image.Image:
    w, h = m.canvas_w_px, m.canvas_h_px
    im = Image.new("RGB", (w, h), (248, 249, 250))
    d = ImageDraw.Draw(im)
    bar_h = h // 12
    d.rectangle([0, 0, w, bar_h], fill=(52, 58, 64))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 24, bar_h // 2 - 7, 28 + i * 24, bar_h // 2 + 7], fill=c)
    d.text((96, bar_h // 2 - 9), "Calendar — week of Aug 17", fill=(222, 226, 230), font=f(15, True))
    gut = 52
    days = ["Mon 17", "Tue 18", "Wed 19", "Thu 20", "Fri 21"]
    col_w = (w - gut - 6) / 5
    for i, day in enumerate(days):
        d.text((gut + i * col_w + col_w / 2 - d.textlength(day, font=f(11, True)) / 2, bar_h + 4), day,
               fill=(95, 99, 104), font=f(11, True))
    grid_y0, grid_y1 = bar_h + 22, h - 6
    hours = ["9", "10", "11", "12", "1", "2", "3", "4"]
    row_h = (grid_y1 - grid_y0) / len(hours)
    for i, hr in enumerate(hours):
        y = grid_y0 + i * row_h
        d.line([gut, y, w - 6, y], fill=(226, 228, 231), width=1)
        d.text((gut - d.textlength(hr, font=f(10)) - 8, y - 4), hr, fill=(150, 155, 160), font=f(10))
    for i in range(6):
        d.line([gut + i * col_w, grid_y0, gut + i * col_w, grid_y1], fill=(226, 228, 231), width=1)

    def event(day, h0, h1, title, fill_c, edge_c, ink=(50, 60, 70), bold=False):
        x0, x1 = gut + day * col_w + 3, gut + (day + 1) * col_w - 3
        y0, y1 = grid_y0 + h0 * row_h + 2, grid_y0 + h1 * row_h - 2
        d.rectangle([x0, y0, x1, y1], fill=fill_c)
        d.rectangle([x0, y0, x0 + 3, y1], fill=edge_c)
        d.text((x0 + 7, y0 + 3), title, fill=ink, font=f(10, bold))

    blue, bluee = (210, 227, 252), (66, 133, 244)
    green, greene = (206, 234, 214), (52, 168, 83)
    purple, purplee = (233, 222, 248), (161, 66, 244)
    amberf, ambere = (254, 239, 195), (244, 180, 0)
    event(0, 0, 1, "Standup", green, greene)
    event(0, 2, 4, "Paper writing", blue, bluee)
    event(0, 5, 6, "CI triage", amberf, ambere)
    event(1, 0, 1, "Standup", green, greene)
    event(1, 1, 3, "Seed sweep review", purple, purplee)
    event(1, 4, 6, "Demo prep", blue, bluee)
    event(2, 0, 1, "Standup", green, greene)
    event(2, 2, 3, "1:1 with Kai", amberf, ambere)
    if state == "done":
        event(2, 4, 5, "(moved)", (255, 255, 255), (200, 205, 210), ink=(170, 175, 180))
        event(4, 0, 1, "Meeting with Priya", blue, bluee, bold=True)
    else:
        event(2, 4, 5, "Meeting with Priya", blue, bluee, bold=True)
    event(3, 0, 2, "Grant report", purple, purplee)
    event(3, 3, 4, "Standup (infra)", green, greene)
    event(3, 5, 7, "Figures pass", blue, bluee)
    event(4, 1, 2, "Standup", green, greene)
    event(4, 2, 4, "Reading group", amberf, ambere)
    event(4, 5, 6, "Retro", purple, purplee)

    if state == "plain":
        return im
    b, th = m.target_box(), m.target_size_px
    cx0, cy0, cx1, cy1 = b.x0 - th // 2, b.y0 - 2 * th, b.x1 + th // 2, b.y1 + th // 2
    d.rectangle([cx0 - 2, cy0 - 2, cx1 + 2, cy1 + 2], fill=(160, 168, 176))
    d.rectangle([cx0, cy0, cx1, cy1], fill=(255, 255, 255))
    if state == "done":
        d.rectangle([cx0, cy0, cx0 + 5, cy1], fill=(22, 163, 74))
        d.text((cx0 + 12, cy0 + 8), "Rescheduled", fill=(52, 58, 64), font=f(14, True))
        d.text((cx0 + 12, cy0 + 30), "Meeting with Priya", fill=(120, 126, 133), font=f(11))
        d.text((cx0 + 12, cy0 + 46), "moved to Fri 9:00 AM", fill=(120, 126, 133), font=f(11))
        d.rectangle([b.x0, b.y0 + th // 3, b.x1, b.y1 - th // 3], fill=(22, 163, 74))
        d.text((b.x0 + 10, b.y0 + th // 3 + 1), "✓ done", fill=(255, 255, 255), font=f(11, True))
        return im
    d.rectangle([cx0, cy0, cx0 + 5, cy1], fill=(97, 31, 105))
    d.ellipse([cx0 + 12, cy0 + 10, cx0 + 38, cy0 + 36], fill=(97, 31, 105))
    d.text((cx0 + 18, cy0 + 16), "PC", fill=(255, 255, 255), font=f(11, True))
    d.text((cx0 + 46, cy0 + 10), "Priya Chakrabarti", fill=(29, 28, 29), font=f(12, True))
    d.text((cx0 + 46, cy0 + 26), "now", fill=(140, 145, 150), font=f(10))
    for i, ln in enumerate(["Sorry — last meeting is", "running late. Can we", "reschedule?"]):
        d.text((cx0 + 12, cy0 + 46 + i * 15, ), ln, fill=(70, 74, 78), font=f(11))
    d.rectangle([b.x0, b.y0, b.x1 - 1, b.y1 - 1], fill=BTN_BLUE)
    tw = d.textlength("Reschedule", font=f(13, True))
    d.text(((b.x0 + b.x1) / 2 - tw / 2, (b.y0 + b.y1) / 2 - 8), "Reschedule", fill=(255, 255, 255), font=f(13, True))
    return im


def main() -> None:
    seeds = []
    for arg in sys.argv[1:]:
        if "-" in arg:
            a, b = arg.split("-"); seeds += list(range(int(a), int(b) + 1))
        else:
            seeds.append(int(arg))
    for seed in seeds:
        m = manifest(seed)
        for state in ("plain", "toast", "done"):
            im = draw_state(m, state)
            (OUT / f"{m.canvas_w_px}x{m.canvas_h_px}_s{seed}_{state}.bin").write_bytes(im.tobytes())
        print(f"seed {seed}: baked at box {m.target_box()}")


if __name__ == "__main__":
    main()
