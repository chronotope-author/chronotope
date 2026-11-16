"""5x7 digit glyphs shared by the rotating_code renderer and the value decoder (cards/C4-late-bound-text.md).
The decoder is allowed to know the font and the panel colour — the analogue of the coordinate binder knowing
what a pixel template is — and nothing about the manifest."""
from __future__ import annotations

DIGITS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
GLYPH_W, GLYPH_H = 5, 7
SCALE = 3                      # each glyph cell is 3x3 canvas pixels: digits are 15x21 px, legible at 2x upscale
GAP = 2 * SCALE                # between digits
PAD = 2 * SCALE                # panel padding around the digit row
PANEL_COLOR = (254, 249, 195)  # unique light yellow; no other pixel in the notification skin uses it
INK_COLOR = (17, 24, 39)
MASK_COLOR = (156, 163, 175)


def digit_row_size(n: int) -> tuple[int, int]:
    return n * GLYPH_W * SCALE + (n - 1) * GAP, GLYPH_H * SCALE


def panel_size(n: int) -> tuple[int, int]:
    w, h = digit_row_size(n)
    return w + 2 * PAD, h + 2 * PAD


def draw_digits(buf: bytearray, w: int, h: int, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    """Draw `text` (digits or '-') with the glyph's top-left at (x, y)."""
    for i, ch in enumerate(text):
        gx = x + i * (GLYPH_W * SCALE + GAP)
        rows = DIGITS.get(ch) or ["00000", "00000", "00000", "11111", "00000", "00000", "00000"]
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    for yy in range(y + r * SCALE, y + (r + 1) * SCALE):
                        if 0 <= yy < h:
                            for xx in range(gx + c * SCALE, gx + (c + 1) * SCALE):
                                if 0 <= xx < w:
                                    o = (yy * w + xx) * 3
                                    buf[o:o + 3] = bytes(color)
