"""Template tracker: the separately identified component that binds a provisional click to the
newest frame at D (WI-004; §9.3 "an initial coordinate proposal is corrected by a tracker before
commitment"). It sees only pixels of admitted frames — never the manifest.

Frames are row-major RGB bytes (w*h*3). Matching is normalized cross-correlation on grey, over
the whole frame (no search radius: one fewer parameter, and a wrong far match is visible in the
score). Pure numpy; a 320x240 frame with a 25x25 template takes ~20-50 ms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .png import to_rgb

# v1 (C1–C3 runs of 2026-08-23): best NCC peak, matched iff score >= min_match_score.
# v2 (2026-08-24): additionally reports the second-best peak outside the winner's footprint; the controller treats
# a binding as ambiguous — and commits the proposal unchanged — when that peak is within AMBIGUITY_MARGIN of the
# best. Re-running v2 on all 1000 recorded v1 bindings changes exactly the one wrong v1 binding (C1 irrelevant-motion
# control, seed 1079: a template that re-occurred perfectly 17 px away) — see studies/check_binder_uniqueness.py.
BINDER_VERSION = 2   # 2.1 (2026-08-24 evening): plateau test added; recorded as 2 with `plateau` in the record
AMBIGUITY_MARGIN = 0.05
PLATEAU_TOL = 0.02   # offsets scoring within this of the best count as the same match
PLATEAU_MAX = 4      # a peak spans at most this many offsets; more is a plateau (the template can slide)


def _grey(payload: bytes, w: int, h: int) -> np.ndarray:
    a = np.frombuffer(to_rgb(payload, w, h), dtype=np.uint8)
    if a.size == w * h * 3:
        a = a.reshape(h, w, 3).astype(np.float32)
        return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114
    if a.size == w * h:
        return a.reshape(h, w).astype(np.float32)
    raise ValueError(f"payload size {a.size} is neither {w*h} nor {w*h*3}")


def _rgb(payload: bytes, w: int, h: int) -> np.ndarray:
    a = np.frombuffer(to_rgb(payload, w, h), dtype=np.uint8)
    if a.size == w * h * 3:
        return a.reshape(h, w, 3).astype(np.float32)
    if a.size == w * h:
        return np.repeat(a.reshape(h, w, 1).astype(np.float32), 3, axis=2)
    raise ValueError(f"payload size {a.size} is neither {w*h} nor {w*h*3}")


@dataclass(frozen=True)
class Template:
    pixels: np.ndarray        # (2r+1, 2r+1) grey or (2r+1, 2r+1, 3) RGB, as cut from the frame at B
    dx: int                   # click offset from the template's top-left (so the click need not be its centre)
    dy: int

    @property
    def channels(self) -> str:
        return "rgb" if self.pixels.ndim == 3 else "grey"


def cut_template(payload: bytes, w: int, h: int, x: int, y: int, radius_px: int, rgb: bool = False) -> Template:
    """Patch of side 2r+1 around (x, y), clamped into the frame; remembers where the click sits in it.
    rgb=True keeps colour (needed when targets differ only by colour, e.g. balloon_select)."""
    g = _rgb(payload, w, h) if rgb else _grey(payload, w, h)
    side = 2 * radius_px + 1
    x0 = min(max(0, x - radius_px), max(0, w - side))
    y0 = min(max(0, y - radius_px), max(0, h - side))
    return Template(g[y0:y0 + side, x0:x0 + side].copy(), x - x0, y - y0)


@dataclass(frozen=True)
class Match:
    x: int            # where the click lands in the new frame (template top-left + click offset)
    y: int
    score: float      # NCC in [-1, 1]; 1 = identical patch
    second: float = -1.0   # best NCC outside the winning window's footprint: a template with too little structure
                           # (an edge, flat background) scores ~1 in many places and cannot bind a coordinate
    plateau: int = 1       # offsets inside the footprint scoring within PLATEAU_TOL of the best: 1-4 is a peak, more
                           # means the template slides inside a uniform region (a wide button) and cannot bind either


def _valid_xcorr(g: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cross-correlation of image g (H,W) with template t (th,tw) at every valid offset, via FFT."""
    H, W = g.shape; th, tw = t.shape
    fh, fw = H + th - 1, W + tw - 1
    G = np.fft.rfft2(g, s=(fh, fw)); T = np.fft.rfft2(t, s=(fh, fw))
    full = np.fft.irfft2(G * np.conj(T), s=(fh, fw))
    return full[:H - th + 1, :W - tw + 1]


def _window_sums(g: np.ndarray, th: int, tw: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum and sum-of-squares over every (th, tw) window, via integral images (float64)."""
    g64 = g.astype(np.float64)
    S = np.zeros((g.shape[0] + 1, g.shape[1] + 1)); S[1:, 1:] = g64.cumsum(0).cumsum(1)
    Q = np.zeros_like(S); Q[1:, 1:] = (g64 * g64).cumsum(0).cumsum(1)
    s = S[th:, tw:] - S[:-th, tw:] - S[th:, :-tw] + S[:-th, :-tw]
    q = Q[th:, tw:] - Q[:-th, tw:] - Q[th:, :-tw] + Q[:-th, :-tw]
    return s, q


def track(template: Template, payload: bytes, w: int, h: int) -> Match:
    """Normalized cross-correlation of the template over the whole frame (grey, or jointly over RGB when the
    template was cut in colour), computed with FFT correlation + integral-image window statistics. The best
    location plus the stored click offset is the translated click."""
    t = template.pixels
    chans = [t[..., c] for c in range(3)] if t.ndim == 3 else [t]
    g = _rgb(payload, w, h) if t.ndim == 3 else _grey(payload, w, h)
    gchans = [g[..., c] for c in range(3)] if t.ndim == 3 else [g]
    th, tw = chans[0].shape
    tmean = float(np.mean(t))                       # joint mean over all channels
    tz = [c - tmean for c in chans]
    tn = float(np.sqrt(sum((z * z).sum() for z in tz)))
    if tn == 0.0:  # flat template (e.g. cut from empty background): nothing to track
        return Match(template.dx, template.dy, 0.0, 1.0, 10**6)
    n = th * tw * len(chans)
    num = None; s = None; ss = None
    for gc, zc in zip(gchans, tz):
        nc = _valid_xcorr(gc.astype(np.float64), zc.astype(np.float64))
        sc, qc = _window_sums(gc, th, tw)
        num = nc if num is None else num + nc
        s = sc if s is None else s + sc
        ss = qc if ss is None else ss + qc
    var = ss - s * s / n
    # windows with (numerically) no variance — flat sky — cannot match anything: exclude them explicitly
    # (FFT/integral-image round-off would otherwise turn 0/0 into a spurious maximum)
    valid = var > 1e-6 * (tn * tn)
    den = np.sqrt(np.clip(var, 0.0, None)) * tn
    ncc = np.where(valid, num / np.where(valid, den, 1.0), -1.0)
    iy, ix = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    best = float(min(1.0, max(-1.0, ncc[iy, ix])))
    # second-best peak outside the winner's own footprint (one template size around it)
    win = ncc[max(0, iy - th + 1):iy + th, max(0, ix - tw + 1):ix + tw]
    plateau = int((win >= best - PLATEAU_TOL).sum())
    rest = ncc.copy()
    rest[max(0, iy - th + 1):iy + th, max(0, ix - tw + 1):ix + tw] = -1.0
    second = float(min(1.0, max(-1.0, rest.max()))) if rest.size else -1.0
    return Match(int(ix) + template.dx, int(iy) + template.dy, best, second, plateau)
