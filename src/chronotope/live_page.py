"""A live Chromium page as the environment (cards/G2-live-page.md).

The page (site/live_page.html) runs in a headless Chromium driven by Playwright. Nothing about it is a pure function
of task time: the harness observes it through the O boundary (screenshots taken on a schedule, PNG bytes), acts on
it through the browser's input path (page.mouse.click), and learns its events (reveal, banner insert, step change,
completion) and geometry (DOM rects) from the page's own log. Three pieces:

  LivePageSource   FrameSource over the page: frames are screenshots captured every `period_ns` on the harness
                   clock; payload = PNG bytes; source_time_ns = the page's performance.now() read right after the
                   capture returned (the depicted state is never later than the stamp), mapped to the harness clock
                   and expressed in task time; publication (O) = capture completion.
  PlaywrightRoute  delivery adapter: deliver(action_bytes, attempt, submit_ns) clicks at the coordinates; the
                   receipt A is the page's own pointerdown event.timeStamp mapped to the harness clock; None (a lost
                   delivery) when the page reports no pointerdown within the timeout.
  LivePageManifest the scenario "live_page": its events, target boxes and validity come from the page log (E = the
                   banner insert reported by the page; boxes = DOM rects logged with each event); during a live trial
                   it reads the session's log as it grows, and the runner appends what it learned as a
                   `scenario_trailer` record so the evaluator rebuilds the same manifest from the trace alone.

Clock mapping (H-G2-3): the page's performance.now() and the harness's monotonic clock are related by one offset,
measured at page open as the best-of-n round trip (offset uncertainty <= round trip / 2, recorded). Input events use
the same origin as performance.now(), so A = offset + event.timeStamp. performance.timeOrigin is recorded for the
trace. Paused is not available: the page cannot hold its response-triggering state without instrumenting it, so a
live page has Replay (the recorded screenshot stream through ReplayFrameSource) and Live only.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode

from .hashing import hash_obj
from .runner import Frame
from .scenario import (EVENT_TARGET_APPEARS, EVENT_TARGET_JUMPS, EVENT_WINDOW_CLOSES, SCENARIOS, Event, Manifest,
                       State, TargetBox)

EVENT_DIALOG_SHOWN = 7      # step 2: the confirm dialog appeared (a consequence of the first click)
EVENT_TASK_COMPLETE = 8     # the page reports the task complete
PAGE_PATH = Path(__file__).resolve().parents[2] / "site" / "live_page.html"
CHROMIUM = "/usr/bin/chromium-browser"
CHROMIUM_ARGS = ["--no-sandbox", "--disable-gpu", "--force-device-scale-factor=1"]


# ---- clock mapping ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClockMap:
    """harness_ns = offset_ns + page_ms * 1e6, where page_ms is a performance.now()-based page timestamp."""
    offset_ns: int
    roundtrip_ns: int          # round trip of the sample that set the offset (its uncertainty is at most half of it)
    time_origin_ms: float      # performance.timeOrigin of the page (wall clock, ms), recorded for the trace

    def harness_ns(self, page_ms: float) -> int:
        return self.offset_ns + int(round(page_ms * 1e6))

    def page_ms(self, harness_ns: int) -> float:
        return (harness_ns - self.offset_ns) / 1e6

    def to_dict(self) -> dict[str, Any]:
        return {"offset_ns": self.offset_ns, "roundtrip_ns": self.roundtrip_ns, "time_origin_ms": self.time_origin_ms}


def sync_clock(samples: list[tuple[int, float, int]], time_origin_ms: float) -> ClockMap:
    """samples: (monotonic before, performance.now() as returned, monotonic after). The page time is placed at the
    midpoint of the shortest round trip."""
    if not samples:
        raise ValueError("clock sync needs at least one sample")
    t0, ms, t1 = min(samples, key=lambda s: s[2] - s[0])
    return ClockMap(offset_ns=(t0 + t1) // 2 - int(round(ms * 1e6)), roundtrip_ns=t1 - t0, time_origin_ms=time_origin_ms)


# ---- the page log -> what the manifest needs ------------------------------------------------------------------------

def _box(r: dict | None) -> list[int] | None:
    return None if r is None else [int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])]


def measured_from_log(log: list[dict]) -> dict[str, Any]:
    """The page's event log (entries carry `t_ns`, task time) reduced to the quantities the manifest uses. Pure."""
    m: dict[str, Any] = {"load_ns": None, "reveal_ns": None, "banner_ns": None, "button_before": None, "button_after": None,
                         "banner_h_px": None, "step2_ns": None, "dialog_t0_ns": None, "confirm_start": None,
                         "confirm_end": None, "slide_ns": None, "dialog_stop_ns": None, "confirm_stopped": None,
                         "complete_ns": None, "clicks": [], "time_origin_ms": None}
    for e in log:
        k, t = e["kind"], int(e["t_ns"])
        if k == "load":
            m["load_ns"], m["time_origin_ms"] = t, e.get("time_origin")
        elif k == "reveal":
            m["reveal_ns"], m["button_before"] = t, _box(e["cta"])
        elif k == "banner":
            m["banner_ns"], m["button_after"], m["banner_h_px"] = t, _box(e["cta"]), e.get("banner_h")
        elif k == "pointerdown":
            m["clicks"].append({"t_ns": t, "x": int(e["x"]), "y": int(e["y"]), "target": e.get("target", ""),
                                "step": int(e.get("step", 1)), "hit": bool(e.get("hit"))})
        elif k == "step" and int(e.get("step", 0)) == 2:
            m["step2_ns"], m["confirm_start"], m["confirm_end"] = t, _box(e["confirm_start"]), _box(e["confirm_end"])
            m["dialog_t0_ns"] = int(e["t0_ns"]) if "t0_ns" in e else t
            m["slide_ns"] = int(e["slide_ms"] * 1e6)
        elif k == "dialog_stop":
            m["dialog_stop_ns"], m["confirm_stopped"] = t, _box(e["confirm"])
        elif k == "complete":
            m["complete_ns"] = t
    return m


# ---- the manifest ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LivePageManifest(Manifest):
    """live_page. Seeded page parameters are the spec; everything time- or geometry-dependent is `measured` (from
    the page log). `event_time_ns` is the planned reveal (task time); the measured reveal replaces it once known.
    Attach a LivePageSession with attach() to read the log live; with_trailer() rebuilds the evaluated manifest."""
    scenario_id = "live_page"
    reveal_delay_ns: int = 500_000_000      # page: content appears this long after load
    banner_delay_ns: int = 2_000_000_000    # page: banner inserted this long after load
    banner_h_px: int = 100
    dialog_dx_px: int = 40
    dialog_dy_px: int = -120
    slide_dx_px: int = 0
    slide_dy_px: int = 80
    slide_ns: int = 2_000_000_000
    two_step: bool = False
    measured: dict | None = None

    def __post_init__(self) -> None:
        if self.skin != "browser":
            raise ValueError("live_page is drawn by the browser (skin browser)")
        if self.target_size_px <= 0 or self.frame_period_ns <= 0 or self.response_window_ns <= 0:
            raise ValueError("target_size_px, frame_period_ns, response_window_ns must be positive")
        if self.episode_horizon_ns < self.event_time_ns + self.response_window_ns:
            raise ValueError("episode_horizon_ns must cover the response window")
        if self.banner_delay_ns <= 0 or self.slide_ns <= 0 or self.banner_h_px <= 0:
            raise ValueError("banner_delay_ns, slide_ns, banner_h_px must be positive")

    # identity: the seeded page, not what was measured on it (Replay must match Live)
    def manifest_hash(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("measured", "event_time_ns")}
        return hash_obj({"scenario_id": self.scenario_id, **d})

    @property
    def multi_action(self) -> bool:      # type: ignore[override]
        return self.two_step

    def page_query(self) -> dict[str, int]:
        return {"seed": self.scenario_seed, "reveal": self.reveal_delay_ns // 1_000_000, "t_banner": self.banner_delay_ns // 1_000_000,
                "banner_h": self.banner_h_px, "two_step": int(self.two_step), "dialog_dx": self.dialog_dx_px,
                "dialog_dy": self.dialog_dy_px, "slide_dx": self.slide_dx_px, "slide_dy": self.slide_dy_px,
                "slide": self.slide_ns // 1_000_000}

    # --- measured state: the attached session's log while live, the stored dict afterwards ------------------------
    def attach(self, session: "LivePageSession") -> None:
        object.__setattr__(self, "_session", session)

    def _m(self) -> dict[str, Any]:
        if self.measured is not None:
            return self.measured
        s = getattr(self, "_session", None)
        return measured_from_log(s.log()) if s is not None and s.started else measured_from_log([])

    def trailer(self) -> dict[str, Any]:
        s = getattr(self, "_session", None)
        extra = s.trailer_fields() if s is not None and self.measured is None else {}
        return {"measured": self._m(), **extra}

    def with_trailer(self, trailer: dict[str, Any]) -> "LivePageManifest":
        me = trailer["measured"]
        ev = me["reveal_ns"] if me.get("reveal_ns") is not None else self.event_time_ns
        return replace(self, measured=me, event_time_ns=ev)

    def t_reveal(self) -> int:
        r = self._m().get("reveal_ns")
        return self.event_time_ns if r is None else r

    def window_close_ns(self) -> int:
        return self.t_reveal() + self.response_window_ns

    # --- events and state ---------------------------------------------------------------------------------------------
    def events(self) -> tuple[Event, ...]:
        me = self._m()
        ev = [Event(EVENT_TARGET_APPEARS, "target_appears", self.t_reveal())]
        if me["banner_ns"] is not None:
            ev.append(Event(EVENT_TARGET_JUMPS, "target_jumps", me["banner_ns"]))
        if me["step2_ns"] is not None:
            ev.append(Event(EVENT_DIALOG_SHOWN, "dialog_shown", me["step2_ns"]))
        if me["complete_ns"] is not None:
            ev.append(Event(EVENT_TASK_COMPLETE, "task_complete", me["complete_ns"]))
        ev.append(Event(EVENT_WINDOW_CLOSES, "window_closes", self.window_close_ns()))
        return tuple(ev)

    def state_at(self, task_time_ns: int) -> State:
        t, me = task_time_ns, self._m()
        if t < self.t_reveal():
            return State(t, False, False, 0)
        # consequences of a click (step change, completion) hold strictly after the click's own time
        if me["complete_ns"] is not None and t > me["complete_ns"]:
            return State(t, False, False, EVENT_TASK_COMPLETE)
        if t >= self.window_close_ns():
            return State(t, False, False, EVENT_WINDOW_CLOSES)
        if me["step2_ns"] is not None and t > me["step2_ns"]:
            return State(t, True, True, EVENT_DIALOG_SHOWN)
        if me["banner_ns"] is not None and t >= me["banner_ns"]:
            return State(t, True, True, EVENT_TARGET_JUMPS)
        return State(t, True, True, EVENT_TARGET_APPEARS)

    def target_box(self) -> TargetBox:
        b = self._m()["button_before"]
        return TargetBox(*b) if b else TargetBox(0, 0, 1, 1)

    def target_box_at(self, task_time_ns: int) -> TargetBox:
        t, me = task_time_ns, self._m()
        if me["step2_ns"] is not None and t > me["step2_ns"] and me["confirm_start"] is not None:
            f = min(1.0, max(0.0, (t - me["dialog_t0_ns"]) / me["slide_ns"]))
            s, e = me["confirm_start"], me["confirm_end"]
            return TargetBox(*[int(round(a + (b - a) * f)) for a, b in zip(s, e)])
        if me["banner_ns"] is not None and t >= me["banner_ns"] and me["button_after"] is not None:
            return TargetBox(*me["button_after"])
        return self.target_box()

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        b0, b1 = self.target_box(), self.target_box_at(task_time_ns)
        return (b1.x0 - b0.x0, b1.y0 - b0.y0)

    def reflow_dy_px(self) -> int | None:
        me = self._m()
        return None if me["button_after"] is None else me["button_after"][1] - me["button_before"][1]

    def click_record(self, task_time_ns: int, action: dict) -> dict | None:
        """The page's own record of a click received at exactly this task time and these coordinates."""
        for c in self._m()["clicks"]:
            if c["t_ns"] == task_time_ns and c["x"] == action.get("x") and c["y"] == action.get("y"):
                return c
        return None

    def geometry_valid_at(self, task_time_ns: int, action: dict) -> bool:
        st = self.state_at(task_time_ns)
        return st.target_visible and self.target_box_at(task_time_ns).contains(action["x"], action["y"])

    def action_valid_at(self, task_time_ns: int, action: dict) -> bool:
        """Application-time validity. For a click the page recorded, the page's own hit-test decides; for any other
        (time, action) the DOM geometry at that time (the frames' state)."""
        if self.validate_action(action) is not None:
            return False
        c = self.click_record(task_time_ns, action)
        return bool(c["hit"]) if c is not None else self.geometry_valid_at(task_time_ns, action)

    def render(self, task_time_ns: int) -> bytes:
        raise NotImplementedError("a live page is not rendered by the harness; frames come from LivePageSource")

    # --- multi-action interface (two_step): the page owns the environment state ---------------------------------------
    def state_env(self, task_time_ns: int, accepted_at_ns: int | None) -> State:
        return self.state_at(task_time_ns)

    def effect_visible_ns(self, accepted_at_ns: int | None) -> int | None:
        """The first page state change at or after the accepted click: the dialog (after Continue) or completion."""
        if accepted_at_ns is None:
            return None
        me = self._m()
        cands = [x for x in (me["step2_ns"], me["complete_ns"]) if x is not None and x >= accepted_at_ns]
        return min(cands) if cands else self.episode_horizon_ns

    @property
    def POST_EFFECT_NS(self) -> int:    # noqa: N802 — the runner reads this name; the page decides when the episode ends
        return 0 if self._m()["complete_ns"] is not None else self.episode_horizon_ns

    def receipt_outcome(self, task_time_ns: int, action: dict, accepted_at_ns: int | None) -> str:
        if self.validate_action(action) is not None:
            return "invalid"
        return "accepted" if self.action_valid_at(task_time_ns, action) else "invalid"

    def return_pending(self, accepted_at_ns: int | None) -> float:
        """Two-step return: minus the seconds from the reveal to the page's completion; the full window if never."""
        c = self._m()["complete_ns"]
        return -self.response_window_ns / 1e9 if c is None else -(c - self.t_reveal()) / 1e9


SCENARIOS["live_page"] = LivePageManifest


# ---- the browser ----------------------------------------------------------------------------------------------------

class _Loop:
    """An asyncio loop on its own thread; Playwright's async API lives there, the harness calls in synchronously."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="playwright-loop", daemon=True)
        self.thread.start()

    def run(self, coro, timeout: float | None = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)


class LiveBrowser:
    """One headless Chromium for a study; open() gives a fresh page per trial."""

    def __init__(self, executable_path: str = CHROMIUM, headless: bool = True) -> None:
        self._loop = _Loop()
        self._pw = self._browser = None
        self._loop.run(self._start(executable_path, headless))

    async def _start(self, exe: str, headless: bool) -> None:
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(executable_path=exe, headless=headless, args=CHROMIUM_ARGS)

    def open(self, url: str, w: int, h: int) -> "LivePageSession":
        return LivePageSession(self, url, w, h)

    def close(self) -> None:
        async def _close():
            if self._browser is not None:
                await self._browser.close()
            if self._pw is not None:
                await self._pw.stop()
        try:
            self._loop.run(_close(), timeout=20)
        finally:
            self._loop.stop()


def page_url(manifest: LivePageManifest, path: Path = PAGE_PATH) -> str:
    return path.resolve().as_uri() + "?" + urlencode(manifest.page_query())


class LivePageSession:
    """One page: opened blank on construction (the harness binding is installed), navigated by begin(h0), which is
    called by the runner's source factory so that page load happens after the harness origin."""

    def __init__(self, browser: LiveBrowser, url: str, w: int, h: int) -> None:
        self._lb, self._loop, self.url, self.w, self.h = browser, browser._loop, url, w, h
        self._cv = threading.Condition()
        self._entries: list[dict] = []          # page log entries as they arrive (page timestamps)
        self._frames: deque = deque()           # (publication_harness_ns, source_harness_ns, png)
        self.clock: ClockMap | None = None
        self.h0: int | None = None
        self.started = False
        self._capture_done = False
        self._capture_future = None
        self.capture: dict[str, Any] = {"n_frames": 0, "skipped_ticks": 0, "capture_errors": 0, "capture_ms": [], "period_ns": None,
                                        "start_ns": None, "end_ns": None}
        self._loop.run(self._open())

    async def _open(self) -> None:
        self.page = await self._lb._browser.new_page(viewport={"width": self.w, "height": self.h})
        await self.page.expose_binding("__harness_event", self._on_event)
        self._cdp = await self.page.context.new_cdp_session(self.page)
        await self._cdp.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})   # warm the capture path

    def _on_event(self, source, entry) -> None:
        with self._cv:
            self._entries.append(dict(entry))
            self._cv.notify_all()

    # --- lifecycle ---------------------------------------------------------------------------------------------------
    def begin(self, h0: int, period_ns: int, horizon_ns: int) -> None:
        """Navigate to the page (load happens after h0), sync the clock, start the capture schedule."""
        self.h0 = h0
        self._loop.run(self._navigate_and_sync())
        self.started = True
        self.capture.update(period_ns=period_ns, start_ns=time.monotonic_ns())
        self._capture_future = self._loop.submit(self._capture(h0, period_ns, h0 + horizon_ns))

    async def _navigate_and_sync(self) -> None:
        await self.page.goto(self.url, wait_until="load")
        samples = []
        for _ in range(7):
            t0 = time.monotonic_ns()
            ms = await self.page.evaluate("performance.now()")
            samples.append((t0, float(ms), time.monotonic_ns()))
        origin = float(await self.page.evaluate("performance.timeOrigin"))
        self.clock = sync_clock(samples, origin)

    async def _capture(self, h0: int, period_ns: int, until_ns: int) -> None:
        k = 0
        try:
            while True:
                due = h0 + k * period_ns
                now = time.monotonic_ns()
                if now >= until_ns:
                    break
                if now > due + period_ns:                    # missed ticks (a slow capture): resume on the schedule
                    skipped = (now - due) // period_ns
                    self.capture["skipped_ticks"] += skipped
                    k += skipped
                    continue
                if now < due:
                    await asyncio.sleep((due - now) / 1e9)
                t_a = time.monotonic_ns()
                try:
                    r = await self._cdp.send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
                except Exception as e:   # a transient "Unable to capture screenshot" (compositor busy): skip this tick
                    self.capture["capture_errors"] += 1
                    if self.capture["capture_errors"] > 20:
                        raise
                    k += 1
                    continue
                # source time: the page's clock right after the capture returned. The depicted state is never later than
                # this stamp; a change inside the capture window shows as a frame stamped after the change that depicts the
                # state before it, which the controllers here never act on (the change reaches them in the next frame).
                ms = await self.page.evaluate("performance.now()")
                t_b = time.monotonic_ns()
                png = base64.b64decode(r["data"])
                with self._cv:
                    self._frames.append((t_b, self.clock.harness_ns(float(ms)), png))
                    self._cv.notify_all()
                self.capture["n_frames"] += 1
                self.capture["capture_ms"].append((t_b - t_a) / 1e6)
                k += 1
        except Exception as e:      # recorded in the trailer; the source then reports no more frames
            self.capture["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            self.capture["end_ns"] = time.monotonic_ns()
            with self._cv:
                self._capture_done = True
                self._cv.notify_all()

    def stop_capture(self) -> None:
        with self._cv:
            self._capture_done = True
        if self._capture_future is not None:
            self._capture_future.cancel()

    def close(self) -> None:
        self.stop_capture()
        try:
            self._loop.run(self.page.close(), timeout=20)
        except Exception:
            pass

    # --- the page log ---------------------------------------------------------------------------------------------------
    def log(self) -> list[dict]:
        """Entries with `harness_ns` (mapped) and `t_ns` (task time) added; the page's own fields untouched."""
        if self.clock is None or self.h0 is None:
            return []
        with self._cv:
            raw = list(self._entries)
        out = []
        for e in raw:
            d = dict(e)
            d["harness_ns"] = self.clock.harness_ns(float(e["t"]))
            d["t_ns"] = d["harness_ns"] - self.h0
            if "t0" in d:
                d["t0_ns"] = self.clock.harness_ns(float(d["t0"])) - self.h0
            out.append(d)
        return out

    def wait_for(self, pred: Callable[[dict], bool], timeout_s: float, start_index: int = 0) -> dict | None:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                for e in self._entries[start_index:]:
                    if pred(e):
                        return dict(e)
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)

    def n_entries(self) -> int:
        with self._cv:
            return len(self._entries)

    # --- page access ---------------------------------------------------------------------------------------------------
    def evaluate(self, js: str):
        return self._loop.run(self.page.evaluate(js), timeout=30)

    def target_rect(self) -> dict | None:
        """The current clickable target's DOM rect (None before the reveal / after completion) — the scripted oracle."""
        return self.evaluate("window.__target_rect()")

    def click(self, x: int, y: int) -> None:
        self._loop.run(self.page.mouse.click(x, y), timeout=30)

    def achieved_hz(self) -> float | None:
        c = self.capture
        end = c["end_ns"] if c["end_ns"] is not None else time.monotonic_ns()
        if c["start_ns"] is None or end <= c["start_ns"] or not c["n_frames"]:
            return None
        return c["n_frames"] / ((end - c["start_ns"]) / 1e9)

    def trailer_fields(self) -> dict[str, Any]:
        c = self.capture
        ms = sorted(c["capture_ms"])
        return {"clock": self.clock.to_dict() if self.clock else None, "page_url": self.url,
                "capture": {"n_frames": c["n_frames"], "skipped_ticks": c["skipped_ticks"], "capture_errors": c["capture_errors"],
                            "error": c.get("error"), "period_ns": c["period_ns"],
                            "achieved_hz": self.achieved_hz(),
                            "capture_ms_median": ms[len(ms) // 2] if ms else None, "capture_ms_max": ms[-1] if ms else None},
                "page_log": self.log()}


# ---- FrameSource and delivery route ---------------------------------------------------------------------------------------

class LivePageSource:
    """Frames = the session's captured screenshots, oldest first. peek() blocks until the next capture exists (the
    runner reads None as "no more frames"), and returns None only once the capture schedule has ended."""

    name = "live_page"

    def __init__(self, session: LivePageSession, manifest: LivePageManifest, h0: int) -> None:
        self.s, self.m, self.h0 = session, manifest, h0

    def peek(self) -> Frame | None:
        s = self.s
        with s._cv:
            while not s._frames and not s._capture_done:
                s._cv.wait(1.0)
            if not s._frames:
                return None
            t_pub, src_h, png = s._frames[0]
        st = src_h - self.h0
        return Frame(t_pub, self.m.state_at(st).latest_event_id, st, png)

    def peek_due(self, upto_harness_ns: int) -> Frame | None:
        """The oldest captured frame, waiting only while a frame due by `upto_harness_ns` can still arrive (a capture
        completes at the harness's now, so nothing new is due once now >= upto); None when nothing is queued."""
        s = self.s
        with s._cv:
            while not s._frames and not s._capture_done:
                left = upto_harness_ns - time.monotonic_ns()
                if left <= 0:
                    return None
                s._cv.wait(min(0.05, left / 1e9))
            if not s._frames:
                return None
            t_pub, src_h, png = s._frames[0]
        st = src_h - self.h0
        return Frame(t_pub, self.m.state_at(st).latest_event_id, st, png)

    def pop(self) -> Frame:
        f = self.peek()
        assert f is not None
        with self.s._cv:
            self.s._frames.popleft()
        return f


def live_source_factory(session: LivePageSession, manifest: LivePageManifest, period_ns: int):
    """For run_trial(source=...): called with (h0, task_clock) once the harness origin exists."""
    def make(h0: int, task_clock) -> LivePageSource:
        session.begin(h0, period_ns, manifest.episode_horizon_ns)
        return LivePageSource(session, manifest, h0)
    return make


class PlaywrightRoute:
    """deliver() clicks through the browser's input path; the receipt is the page's pointerdown timestamp."""

    route = "playwright_page"

    def __init__(self, session: LivePageSession, receipt_timeout_ns: int) -> None:
        self.s = session
        self.delay = receipt_timeout_ns          # the runner's bound on a failed attempt (publish_due(submit + delay))
        self.lost = 0
        self.latencies_ns: list[int] = []        # A - submit, per delivered attempt

    def deliver(self, action_bytes: bytes, attempt: int, submit_ns: int) -> tuple[int, bytes] | None:
        a = json.loads(action_bytes)
        x, y = int(a["x"]), int(a["y"])
        start = self.s.n_entries()
        self.s.click(x, y)
        e = self.s.wait_for(lambda e: e.get("kind") == "pointerdown" and e.get("x") == x and e.get("y") == y,
                            self.delay / 1e9, start_index=start)
        if e is None:
            self.lost += 1
            return None
        if e.get("hit"):   # the page logs the consequence (step / complete) in the same handler; let it arrive before the trailer
            self.s.wait_for(lambda c: c.get("kind") in ("step", "complete") and c.get("t") == e["t"], 1.0, start_index=start)
        a_ns = self.s.clock.harness_ns(float(e["t"]))
        self.latencies_ns.append(a_ns - submit_ns)
        return a_ns, action_bytes


# ---- the scripted oracle: the DOM instead of the model -------------------------------------------------------------------

class ScriptedPageOracle:
    """Stands in for LocalOpenAIAdapter in --scripted runs: a perceiver of the screenshot it is given (the target
    buttons' unique colours, as the scripted oracle does on every other rendering), answering the button's center on a
    0-1000 grid (prompts 12/15) after a fixed compute delay. The controllers around it (late binding, restart, pending
    record) are the real ones. perceive="dom" answers from the page's DOM at invocation instead (window.__target_rect);
    that is NOT a stand-in for the model — the DOM at B can already show a state the frame at B does not (the dialog
    the just-delivered click opened), and the binder then cuts its template from the wrong pixels — it exists only to
    check the page log against the DOM. Under Replay there is no page; the frame is the only source."""

    # the page's target colours, first found wins: green (Confirm's outline and text, step 2) before purple Continue (step 1)
    COLORS = ((22, 163, 74), (79, 70, 229))

    def __init__(self, session: LivePageSession | None, compute_s: float, model_id: str = "scripted-page-oracle",
                 colors: tuple[tuple[int, int, int], ...] = COLORS, perceive: str = "frame") -> None:
        if perceive not in ("frame", "dom"):
            raise ValueError(perceive)
        self.s, self.compute_s, self.model_id, self.colors, self.perceive = session, compute_s, model_id, colors, perceive
        self.calls = 0

    def spec_fields(self) -> dict[str, Any]:
        return {"server": "scripted_page_oracle", "model_id": self.model_id, "compute_s": self.compute_s, "perceive": self.perceive}

    @staticmethod
    def frame_box(png: bytes, colors) -> tuple[tuple[int, int, int, int] | None, int, int]:
        import io
        import numpy as np
        from PIL import Image
        im = Image.open(io.BytesIO(png)).convert("RGB")
        a = np.asarray(im)
        for color in colors:
            ys, xs = np.nonzero(np.all(a == np.array(color, dtype=np.uint8), axis=2))
            if xs.size:
                return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), im.width, im.height
        return None, im.width, im.height

    def invoke(self, png: bytes, system: str, user_text: str, structured: bool = True, schema: dict | None = None):
        from .controller import RawInvocation
        self.calls += 1
        if self.perceive == "dom" and self.s is not None:
            r = self.s.target_rect()
            box = None if r is None else (r["x0"], r["y0"], r["x1"], r["y1"])
            w, h = self.s.w, self.s.h
        else:
            box, w, h = self.frame_box(png, self.colors)
        if self.compute_s > 0:
            time.sleep(self.compute_s)
        if box is None:
            parsed = {"action": "wait"}
        else:
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            parsed = {"action": "click", "x": cx * 1000 // w, "y": cy * 1000 // h}
        return RawInvocation(parsed=parsed, invocation_id=f"dom_{self.calls}",
                             detail={"model_reported": self.model_id, "finish_reason": "stop", "raw_text": json.dumps(parsed)})


# ---- evaluation of the two-step episode -------------------------------------------------------------------------------------

@dataclass
class LiveTwoStepResult:
    valid_trial: bool
    invalid_reason: str | None
    completed: bool
    seconds_to_completion: float | None      # reveal -> the page's completion
    return_G: float | None                   # minus seconds to completion; minus the window if never completed
    n_decisions: int = 0
    n_clicks: int = 0
    n_hits: int = 0
    n_lost: int = 0
    n_restarts: int = 0
    clicks: list[dict] = field(default_factory=list)   # per committed click: step, hit, labels, b_to_d_ns, moved_px
    step2_labels: list[str] = field(default_factory=list)   # labels of the first click aimed at the dialog
    step2_b_to_d_ns: int | None = None
    dialog_moved_px: float | None = None     # confirm-button displacement between that click's frame at B and its receipt
    contract_violations: list[str] = field(default_factory=list)

    @property
    def step1_labels(self) -> list[str]:
        return [lab for c in self.clicks if c["step"] == 1 for lab in c["labels"]]


def click_labels(m: Manifest, t, commit: dict, receipt: dict | None) -> list[str]:
    """The per-click timing labels of evaluate_trace (outdated_at_commitment, invalidated_after_commitment, late), for
    one committed click of a multi-click trace; semantic validity uses the DOM geometry at the frame's source time."""
    obs = t.all("observation")
    d0 = next((d for d in t.all("decision_begin") if d["decision_id"] == commit["decision_id"]), None)
    corr = next((c for c in t.all("coordinate_correction") if c["decision_id"] == commit["decision_id"]), None)
    action = commit["action"]
    labels: list[str] = []
    if d0 is None:
        return ["missing_B"]
    ob = next((o for o in obs if o["observation_id"] == d0["observation_id"]), None)
    if ob is None:
        return ["decision_references_unknown_observation"]
    tau = ob["source_time_ns"]
    if corr is not None:
        ob2 = next((o for o in obs if o["observation_id"] == corr["observation_id"]), None)
        semantic = m.geometry_valid_at(ob2["source_time_ns"], action) if ob2 else None
    else:
        semantic = m.geometry_valid_at(tau, action)
    if semantic is False:
        labels.append("semantic_error")
    if receipt is None:
        return labels + ["dropped"]
    D, t_a = commit["commitment_time_ns"], receipt["task_time_at_receipt_ns"]
    valid_a = bool(receipt["valid_at_receipt"])
    if t_a >= m.window_close_ns():
        labels.append("late")
    if m.geometry_valid_at(commit["task_time_at_commit_ns"], action) and not valid_a:
        labels.append("invalidated_after_commitment")
    if semantic and corr is None:
        if any(o["publication_time_ns"] < D and o["source_time_ns"] > tau and not m.geometry_valid_at(o["source_time_ns"], action)
               for o in obs):
            labels.append("outdated_at_commitment")
    return labels


def evaluate_live_two_step(t) -> LiveTwoStepResult:
    from .evaluate import evaluate_pending_trace
    h = t.header()
    tr = t.first("scenario_trailer")
    if tr is None:
        return LiveTwoStepResult(False, "no scenario_trailer", False, None, None)
    from .scenario import manifest_from_dict
    m = manifest_from_dict(h["manifest"]).with_trailer(tr)
    pend = evaluate_pending_trace(t)
    me = m.measured
    commits, receipts = t.all("commit"), {r["action_id"]: r for r in t.all("receipt")}
    decisions = t.all("decision_begin")
    completed = me["complete_ns"] is not None
    secs = (me["complete_ns"] - m.t_reveal()) / 1e9 if completed else None
    res = LiveTwoStepResult(pend.valid_trial, pend.invalid_reason, completed, secs, m.return_pending(me["complete_ns"]),
                            n_decisions=len(decisions), n_clicks=len(commits),
                            n_hits=sum(bool(receipts[c["action_id"]]["valid_at_receipt"]) for c in commits if c["action_id"] in receipts),
                            n_lost=pend.dropped, n_restarts=len(t.all("decision_restarted")),
                            contract_violations=list(pend.contract_violations))
    s2 = me["step2_ns"]
    obs = {o["observation_id"]: o for o in t.all("observation")}
    for c in commits:
        r = receipts.get(c["action_id"])
        labs = click_labels(m, t, c, r)
        t_a = r["task_time_at_receipt_ns"] if r else c["task_time_at_commit_ns"]
        d0 = next(d for d in decisions if d["decision_id"] == c["decision_id"])
        ob = obs.get(d0["observation_id"])
        tau = ob["source_time_ns"] if ob else None
        # a click is aimed at the dialog when the frame it was decided on already showed it
        aimed_step = 2 if (s2 is not None and tau is not None and tau > s2) else 1
        moved = None
        if ob is not None and r is not None:
            b0, b1 = m.target_box_at(tau), m.target_box_at(t_a)
            moved = math.hypot(b1.x0 - b0.x0, b1.y0 - b0.y0)
        res.clicks.append({"step": aimed_step, "hit": bool(r and r["valid_at_receipt"]), "labels": labs,
                           "b_to_d_ns": c["commitment_time_ns"] - d0["begin_time_ns"], "moved_px": moved,
                           "received_after_step2": bool(s2 is not None and t_a > s2)})
        if aimed_step == 2 and res.step2_b_to_d_ns is None:
            res.step2_labels, res.step2_b_to_d_ns, res.dialog_moved_px = labs, c["commitment_time_ns"] - d0["begin_time_ns"], moved
    return res
