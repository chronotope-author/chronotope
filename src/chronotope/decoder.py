"""Value decoder: the separately identified component that binds a provisional `type(value)` to the newest frame
at D (cards/C4-late-bound-text.md), the non-spatial analogue of tracker.py. It sees only pixels of admitted frames
and knows the glyph font and the panel colour; it never reads the manifest.

Decoding: find the bounding box of the panel colour, cut the digit row at the fixed padding, sample each glyph
cell at its 5x7 centres, and take the digit whose bitmap is nearest in Hamming distance. A cell whose nearest
glyph is farther than MAX_HAMMING (or whose two nearest glyphs tie) fails to decode; a masked panel ('----')
decodes to None as well. Pure Python over bytes; a 320x240 frame takes ~1-3 ms."""
from __future__ import annotations

from dataclasses import dataclass

from .glyphs import DIGITS, GAP, GLYPH_H, GLYPH_W, INK_COLOR, PAD, PANEL_COLOR, SCALE

DECODER_VERSION = 1
MAX_HAMMING = 4      # of 35 cells; the rendered glyphs are exact, so anything above this is not a digit


@dataclass(frozen=True)
class Decoded:
    value: str | None      # the digit string, or None when the panel is absent or not decodable
    panel: tuple[int, int, int, int] | None
    hamming: int           # summed distance to the chosen glyphs (0 for an exact render)
    reason: str


def _panel_box(payload: bytes, w: int, h: int) -> tuple[int, int, int, int] | None:
    pc = bytes(PANEL_COLOR)
    x0 = y0 = 10 ** 9; x1 = y1 = -1
    for yy in range(h):
        row = payload[yy * w * 3:(yy + 1) * w * 3]
        i = row.find(pc)
        if i < 0 or i % 3:
            # an aligned match only; a colour straddling pixel boundaries is not this panel
            j = i
            while j >= 0 and j % 3:
                j = row.find(pc, j + 1)
            if j < 0:
                continue
            i = j
        x0 = min(x0, i // 3); y0 = min(y0, yy); y1 = max(y1, yy)
        k = row.rfind(pc)
        while k % 3:
            k = row.rfind(pc, 0, k)
        x1 = max(x1, k // 3)
    return None if x1 < 0 else (x0, y0, x1 + 1, y1 + 1)


def decode(payload: bytes, w: int, h: int, n_digits: int = 4) -> Decoded:
    box = _panel_box(payload, w, h)
    if box is None:
        return Decoded(None, None, 0, "no_panel")
    px0, py0, px1, py1 = box
    exp_w = n_digits * GLYPH_W * SCALE + (n_digits - 1) * GAP + 2 * PAD
    exp_h = GLYPH_H * SCALE + 2 * PAD
    if (px1 - px0, py1 - py0) != (exp_w, exp_h):
        return Decoded(None, box, 0, "panel_size")
    ink = bytes(INK_COLOR)
    out: list[str] = []
    total = 0
    for i in range(n_digits):
        gx = px0 + PAD + i * (GLYPH_W * SCALE + GAP)
        gy = py0 + PAD
        bits = []
        for r in range(GLYPH_H):
            for c in range(GLYPH_W):
                xx, yy = gx + c * SCALE + SCALE // 2, gy + r * SCALE + SCALE // 2
                o = (yy * w + xx) * 3
                bits.append("1" if payload[o:o + 3] == ink else "0")
        got = "".join(bits)
        dists = sorted((sum(a != b for a, b in zip(got, "".join(rows))), d) for d, rows in DIGITS.items())
        if dists[0][0] > MAX_HAMMING or (len(dists) > 1 and dists[1][0] == dists[0][0]):
            return Decoded(None, box, dists[0][0], "glyph_unreadable" if dists[0][0] > MAX_HAMMING else "glyph_ambiguous")
        out.append(dists[0][1]); total += dists[0][0]
    return Decoded("".join(out), box, total, "ok")
