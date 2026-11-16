"""The physical implementations of the runner's two seams (cards/PHY2 part 1, PHY3 part 1).

CaptureFrameSource (runner.FrameSource)
    Frames come from the capture card. For each captured frame the frame-counter strip is decoded from
    the Y plane; the counter resolves (through the frame server's registry) to the served frame's
    source_event_id, source_time_ns and payload; O = the kernel CLOCK_MONOTONIC buffer timestamp of the
    captured frame (the same clock as the harness). A counter seen before is not re-published: the
    trial's observations are the distinct served frames, at the time each was first seen.

PhysicalRoute (runner.Route)
    deliver() moves the cursor to the action's scenario coordinate (through a four-corner calibration
    fitted at session start), clicks through the KMBox, and waits for A. A is the target page's own
    input time (POST /input on the frame server, agent clock via the page's Cristian sync); when no
    page report arrives, the echo strip is the fallback: the first captured frame whose echo panel
    flipped, with the display chain (E->O median from the instrument) subtracted. deliver() returns
    None when neither arrives within receipt_timeout_ns: the runner records a lost delivery, exactly
    as SimulatedDropFirstRoute does. The serial-write time is never reported as A.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass

from ..runner import Frame
from .gymserve import IMG_SCALE, IMG_X, IMG_Y, ReceiptSink
from .probe import Code, DecodeError, decode_capture

MS = 1_000_000


# ------------------------------------------------------------------------------- frame source
class CaptureFrameSource:
    """runner.FrameSource over a capture stream (capture.ElgatoSource or capture.FakeSource)."""

    name = "capture"

    def __init__(self, source, resolve, *, horizon_ns: int, decode=decode_capture, reduce: int = 4,
                 origin_ns: int | None = None, on_origin=None) -> None:
        self.source, self.resolve, self.decode, self.reduce = source, resolve, decode, reduce
        self.horizon_ns = horizon_ns
        self.h0 = origin_ns
        self.on_origin = on_origin      # called with h0 at trial start (the cell starts its real-time publisher here)
        self._q: deque[Frame] = deque()
        self._cv = threading.Condition()
        self._seen: set[tuple[int, int]] = set()
        self._closed = False
        self.n_frames = self.n_decoded = self.n_unresolved = self.n_repeat = 0
        self.codes: deque[tuple[int, Code]] = deque(maxlen=4096)   # (ts_kernel_ns, code): the echo watcher's input
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "CaptureFrameSource":
        self._thread.start()
        return self

    def bind_origin(self, h0: int) -> None:
        self.h0 = h0
        if self.on_origin is not None:
            self.on_origin(h0)

    def reset(self, horizon_ns: int | None = None) -> None:
        """Start a new trial on the same capture stream: forget queued and seen frames."""
        with self._cv:
            self._q.clear(); self._seen.clear(); self.codes.clear()
            self.h0 = None
            if horizon_ns is not None:
                self.horizon_ns = horizon_ns

    def _run(self) -> None:
        try:
            for fr in self.source.frames():
                if self._closed:
                    break
                self.n_frames += 1
                try:
                    code = self.decode(fr, self.reduce)
                except (DecodeError, ValueError, OSError):
                    continue
                self.n_decoded += 1
                with self._cv:
                    self.codes.append((fr.ts_kernel_ns, code))
                    self._cv.notify_all()
                key = (code.count, code.draw_ms)
                if key in self._seen:
                    self.n_repeat += 1
                    continue
                ent = self.resolve(code)
                if ent is None:
                    self.n_unresolved += 1
                    continue
                self._seen.add(key)
                ev, t_src, payload = ent
                with self._cv:
                    self._q.append(Frame(fr.ts_kernel_ns, ev, t_src, payload))
                    self._cv.notify_all()
        finally:
            with self._cv:
                self._closed = True
                self._cv.notify_all()

    def _deadline_passed(self) -> bool:
        return self.h0 is not None and time.monotonic_ns() > self.h0 + self.horizon_ns + 500 * MS

    def peek(self) -> Frame | None:
        """The oldest unpublished captured frame. Blocks until one arrives; None once the stream has
        closed or the episode horizon (plus 0.5 s) has passed with nothing pending — so a quiet capture
        never ends a trial early, and a dead capture cannot hang it."""
        with self._cv:
            while not self._q:
                if self._closed or self._deadline_passed():
                    return None
                self._cv.wait(0.05)
            return self._q[0]

    def peek_due(self, upto_h: int, slack_ns: int = 5 * MS) -> Frame | None:
        """The oldest unpublished frame if one is queued; otherwise None as soon as no frame with a kernel stamp
        <= upto_h can still be in the capture pipeline (now - slack > upto_h), without waiting for the next
        frame. Used by the runner's publish_due(): the catch-up at D costs the pipeline latency, not a frame."""
        with self._cv:
            while not self._q:
                if self._closed or time.monotonic_ns() - slack_ns > upto_h:
                    return None
                self._cv.wait(0.005)
            return self._q[0]

    def pop(self) -> Frame:
        with self._cv:
            return self._q.popleft()

    def wait_echo_after(self, ts_ns: int, timeout_s: float) -> int | None:
        """Echo fallback: the kernel stamp of the first frame after ts_ns whose panel state differs from
        the last state seen at or before ts_ns. None if no flip within timeout_s."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                before = [c.echo for t, c in self.codes if t <= ts_ns and c.echo is not None]
                base = before[-1] if before else None
                for t, c in self.codes:
                    if t > ts_ns and c.echo is not None and base is not None and c.echo != base:
                        return t
                rem = deadline - time.monotonic()
                if rem <= 0 or self._closed:
                    return None
                self._cv.wait(min(rem, 0.05))

    def close(self) -> None:
        self._closed = True
        self.source.close()


# ------------------------------------------------------------------------------- calibration
@dataclass
class Calibration:
    """Affine map from scenario px to target screen px (axis-aligned: scale + offset per axis)."""
    sx: float
    sy: float
    ox: float
    oy: float
    residual_px: float = 0.0

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(round(self.sx * x + self.ox)), int(round(self.sy * y + self.oy))

    @classmethod
    def fit(cls, pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> "Calibration":
        """pairs: ((scenario_x, scenario_y), (screen_x, screen_y)); least squares per axis, >= 2 points."""
        if len(pairs) < 2:
            raise ValueError("calibration needs at least two points")
        def fit_axis(us, vs):
            n = len(us); mu, mv = sum(us) / n, sum(vs) / n
            var = sum((u - mu) ** 2 for u in us)
            if var == 0:
                raise ValueError("degenerate calibration points")
            s = sum((u - mu) * (v - mv) for u, v in zip(us, vs)) / var
            return s, mv - s * mu
        sx, ox = fit_axis([p[0][0] for p in pairs], [p[1][0] for p in pairs])
        sy, oy = fit_axis([p[0][1] for p in pairs], [p[1][1] for p in pairs])
        c = cls(sx, sy, ox, oy)
        c.residual_px = max(max(abs(c.sx * p[0][0] + c.ox - p[1][0]), abs(c.sy * p[0][1] + c.oy - p[1][1])) for p in pairs)
        return c

    @classmethod
    def nominal(cls, fit_scale: float = 1.0, page_ox: float = 0.0, page_oy: float = 0.0) -> "Calibration":
        """The page's own geometry (served frame shown at fit_scale, top-left at page_ox/page_oy)."""
        return cls(IMG_SCALE * fit_scale, IMG_SCALE * fit_scale, page_ox + IMG_X * fit_scale, page_oy + IMG_Y * fit_scale)


class Cursor:
    """Dead-reckoned cursor on the target screen. home() rams the cursor into the top-left corner
    (relative HID moves clamp at the screen edge), after which relative moves are tracked exactly if the
    target has no pointer acceleration (F38: the calibration's verification click checks this)."""

    def __init__(self, hid, screen_w: int = 1920, screen_h: int = 1080) -> None:
        self.hid, self.w, self.h = hid, screen_w, screen_h
        self.x = self.y = 0
        self.homed = False

    def home(self) -> None:
        self.hid.move(-(self.w + 200), -(self.h + 200))
        self.x = self.y = 0
        self.homed = True

    def move_to(self, x: int, y: int) -> None:
        if not self.homed:
            self.home()
        x = max(0, min(self.w - 1, int(x))); y = max(0, min(self.h - 1, int(y)))
        self.hid.move(x - self.x, y - self.y)
        self.x, self.y = x, y


def four_corner_calibration(cursor: Cursor, hid, receipts: ReceiptSink, screen_points: list[tuple[int, int]] | None = None,
                            timeout_s: float = 2.0, settle_s: float = 0.15, tolerance_px: float = 2.0) -> Calibration:
    """Click at four screen points; the page reports each click's scenario coordinate; fit scenario->screen.
    A fifth click at the fitted centre verifies the map (residual must be <= tolerance_px)."""
    pts = screen_points or [(cursor.w // 4, cursor.h // 4), (3 * cursor.w // 4, cursor.h // 4),
                            (cursor.w // 4, 3 * cursor.h // 4), (3 * cursor.w // 4, 3 * cursor.h // 4)]
    pairs = []
    for sx, sy in pts:
        cursor.move_to(sx, sy)
        time.sleep(settle_s)
        t = hid.click()
        r = receipts.wait_after(t, timeout_s, kind="click")
        if r is None:
            raise TimeoutError(f"calibration: no page report for the click at screen ({sx},{sy})")
        pairs.append(((r["x"], r["y"]), (sx, sy)))
    cal = Calibration.fit(pairs)
    # verification click at the scenario point that maps to the screen centre
    cx, cy = (cursor.w // 2 - cal.ox) / cal.sx, (cursor.h // 2 - cal.oy) / cal.sy
    cursor.move_to(*cal.to_screen(cx, cy))
    time.sleep(settle_s)
    t = hid.click()
    r = receipts.wait_after(t, timeout_s, kind="click")
    if r is None:
        raise TimeoutError("calibration: no page report for the verification click")
    err = max(abs(r["x"] - cx), abs(r["y"] - cy)) * max(cal.sx, cal.sy)
    cal.residual_px = max(cal.residual_px, err)
    if cal.residual_px > tolerance_px:
        raise RuntimeError(f"calibration residual {cal.residual_px:.1f} px > {tolerance_px} px (pointer acceleration? F38)")
    return cal


# ------------------------------------------------------------------------------- route
class PhysicalRoute:
    route = "physical"

    def __init__(self, hid, cursor: Cursor, calibration: Calibration, receipts: ReceiptSink, *,
                 receipt_timeout_ns: int, expected_delivery_delay_ns: int, echo_source: CaptureFrameSource | None = None,
                 display_chain_ns: int | None = None, settle_s: float = 0.0, echo_check_s: float = 0.0) -> None:
        self.hid, self.cursor, self.cal, self.receipts = hid, cursor, calibration, receipts
        self.echo_check_s = echo_check_s
        self.timeout_ns = receipt_timeout_ns
        self.delay = expected_delivery_delay_ns
        self.echo, self.display_chain_ns, self.settle_s = echo_source, display_chain_ns, settle_s
        self.log: list[dict] = []     # per attempt: write stamp, receipt source, A (consumed by the cell's report)

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        action = json.loads(action_bytes)
        kind = action.get("kind")
        if kind != "click":
            raise ValueError(f"PhysicalRoute: unsupported action kind {kind!r} (clicks only)")
        self.cursor.move_to(*self.cal.to_screen(action["x"], action["y"]))
        if self.settle_s:
            time.sleep(self.settle_s)
        t_write = self.hid.click()
        entry = {"attempt": attempt, "submit_ns": submit_ns, "t_write_ns": t_write, "receipt": None, "a_ns": None}
        self.log.append(entry)
        r = self.receipts.wait_after(t_write, self.timeout_ns / 1e9, kind="click")
        if r is not None:
            entry["receipt"], entry["a_ns"] = "page", r["t_ns"]
            if self.echo is not None and self.echo_check_s > 0:
                # the independent second measurement of A (echo strip through the capture), for the concordance report
                t_flip = self.echo.wait_echo_after(t_write, self.echo_check_s)
                entry["a_echo_ns"] = None if t_flip is None else t_flip - (self.display_chain_ns or 0)
            received = {"kind": "click", "x": r["x"], "y": r["y"]}       # what the target actually got
            return r["t_ns"], json.dumps(received, sort_keys=True, separators=(",", ":")).encode()
        if self.echo is not None and self.display_chain_ns is not None:
            rem = max(0.0, (t_write + self.timeout_ns - time.monotonic_ns()) / 1e9)
            t_flip = self.echo.wait_echo_after(t_write, rem)
            if t_flip is not None:
                entry["receipt"], entry["a_ns"] = "echo", t_flip - self.display_chain_ns
                return t_flip - self.display_chain_ns, action_bytes
        entry["receipt"] = "lost"
        return None
