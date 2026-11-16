"""A simulated target machine for --simulate runs and tests (no hardware).

It stands in for the Windows laptop, its browser showing the frame server's page, the capture card
looking at its screen, and the KMBox moving its cursor:

  display   shows the newest served frame with a fixed display-chain delay eo_ms (what E->O measures) and
            paints the echo panel from the page's state; render(ts) returns the Y plane the capture card
            would dequeue at kernel time ts (feed it to capture.FakeSource(pixfmt="NV12"));
  page      on a click, reports the scenario coordinate under the cursor da_ms after the serial write
            (POST /input over HTTP when `post_url` is given, else straight into the ReceiptSink), and
            flips the echo panel at the same moment; `lose_every` drops every n-th page report (the
            echo fallback then carries the receipt) and `mute_page` drops them all;
  hid       the KMBox interface (move/click/key/close/version) with a cursor on screen_w x screen_h,
            clamped at the edges as a real relative-motion cursor is.
"""
from __future__ import annotations

import io
import json
import threading
import time
import urllib.request
from collections import deque

from PIL import Image, ImageDraw

from .gymserve import BLOCK, CELLS, PANEL_Y, X0, H, W, FrameServer

MS = 1_000_000


class SimTarget:
    def __init__(self, server: FrameServer, *, eo_ms: float = 40.0, da_ms: float = 30.0, fit: float = 1.0,
                 page_ox: float = 0.0, page_oy: float = 0.0, screen_w: int = 1920, screen_h: int = 1080,
                 lose_every: int | None = None, mute_page: bool = False, post_url: str | None = None,
                 da_jitter_ms: float = 0.0, seed: int = 0) -> None:
        self.server = server
        self.eo_ns, self.da_ns = int(eo_ms * MS), int(da_ms * MS)
        self.fit, self.page_ox, self.page_oy = fit, page_ox, page_oy
        self.screen_w, self.screen_h = screen_w, screen_h
        self.lose_every, self.mute_page, self.post_url = lose_every, mute_page, post_url
        import random
        self.rng, self.da_jitter_ns = random.Random(seed), int(da_jitter_ms * MS)
        self.cx = self.cy = 0
        self.n_clicks = 0
        self._flips: deque[tuple[int, bool]] = deque(maxlen=1024)   # (t_ns, panel state after)
        self._echo = False
        self._shown: deque[tuple[int, int, Image.Image]] = deque(maxlen=64)   # (t_ns, k, gray image)
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._display_loop, daemon=True)
        self.writes: list[str] = []

    # -- display ---------------------------------------------------------------------------------
    def start(self) -> "SimTarget":
        self._thread.start()
        return self

    def _display_loop(self) -> None:
        seq = 0
        while not self._stop:
            got = self.server.wait_newer(seq, 0.2)
            if got is None:
                continue
            seq, jpeg = got
            img = Image.open(io.BytesIO(jpeg)).convert("L")
            with self._lock:
                self._shown.append((time.monotonic_ns(), seq, img))

    def render(self, ts_kernel_ns: int) -> bytes:
        """The Y plane the capture card dequeues at kernel time ts: the frame the display was showing
        eo_ms earlier, with the panel state of that moment."""
        t = ts_kernel_ns - self.eo_ns
        with self._lock:
            shown = [s for s in self._shown if s[0] <= t]
            base = shown[-1][2] if shown else None
            echo = False
            for tf, st in self._flips:
                if tf <= t:
                    echo = st
        img = Image.new("L", (W, H), 128) if base is None else base.copy()
        if self.fit != 1.0 or self.page_ox or self.page_oy:
            scr = Image.new("L", (self.screen_w, self.screen_h), 128)
            scr.paste(img.resize((int(W * self.fit), int(H * self.fit)), Image.NEAREST), (int(self.page_ox), int(self.page_oy)))
            img = scr
        d = ImageDraw.Draw(img)
        x0, y0 = self.page_ox + X0 * self.fit, self.page_oy + PANEL_Y * self.fit
        d.rectangle([x0, y0, x0 + CELLS * BLOCK * self.fit - 1, y0 + 6 * BLOCK * self.fit - 1], fill=255 if echo else 0)
        return img.tobytes()

    # -- hid (the KMBox interface) ---------------------------------------------------------------
    def move(self, dx: int, dy: int, step: int = 120) -> int:
        self.writes.append(f"km.move({dx},{dy})")
        self.cx = max(0, min(self.screen_w - 1, self.cx + int(dx)))
        self.cy = max(0, min(self.screen_h - 1, self.cy + int(dy)))
        return time.monotonic_ns()

    def click(self, button: str = "left", hold_ms: int = 20) -> int:
        t_write = time.monotonic_ns()
        self.writes.append(f"kmc(km.{button},{hold_ms})")
        self.n_clicks += 1
        n = self.n_clicks
        # scenario coordinate under the cursor, as the page computes it
        fx, fy = (self.cx - self.page_ox) / self.fit, (self.cy - self.page_oy) / self.fit
        from .gymserve import IMG_SCALE, IMG_X, IMG_Y
        sx, sy = (fx - IMG_X) / IMG_SCALE, (fy - IMG_Y) / IMG_SCALE
        da = self.da_ns + (self.rng.randint(-self.da_jitter_ns, self.da_jitter_ns) if self.da_jitter_ns else 0)
        t_recv = t_write + da
        lost = self.mute_page or (self.lose_every is not None and n % self.lose_every == 0)
        threading.Thread(target=self._page_receives, args=(t_recv, sx, sy, n, lost), daemon=True).start()
        return t_write

    def _page_receives(self, t_recv: int, x: float, y: float, n: int, lost: bool) -> None:
        dt = t_recv - time.monotonic_ns()
        if dt > 0:
            time.sleep(dt / 1e9)
        with self._lock:
            self._echo = not self._echo
            self._flips.append((t_recv, self._echo))
        if lost:
            return
        body = {"t_ms": t_recv / MS, "x": x, "y": y, "kind": "click", "count": n & 0xFFFF}
        if self.post_url:
            req = urllib.request.Request(self.post_url.rstrip("/") + "/input", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=2.0).read()
            except OSError:
                pass
        else:
            self.server.receipts.push({"t_ns": t_recv, "x": int(round(x)), "y": int(round(y)), "kind": "click",
                                       "count": n & 0xFFFF, "arrived_ns": time.monotonic_ns()})

    def key(self, name: str, hold_ms: int = 60) -> int:
        self.writes.append(f"kmk({name},{hold_ms})")
        return time.monotonic_ns()

    def version(self) -> str:
        return "sim"

    def close(self) -> None:
        self._stop = True
