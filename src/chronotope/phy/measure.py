"""PHY measurement session: serve the probe page, watch frames, send inputs, record everything.

Per frame (records/frames.jsonl): seq, ts_kernel_ns, t_dq_ns, decoded code or the decode error.
Per input (records/inputs.jsonl): t_write_ns (D), the first frame whose count advanced, A (receipt_ms
on the agent clock), D→A, E→O of that frame, sync bound (rtt/2), known delay applied.

Tests (cards/PHY-physical-path.md, contract_before_measurement):
  1. clock identity — capture.ElgatoSource raises unless both stamps are CLOCK_MONOTONIC.
  2. distributions from ≥100 events with the quantization floor (one capture frame) stated.
  3. null test — no input sent for `null_s` seconds: the count must never advance.
  4. known-delay test — +delay_ms inserted in software before the serial write; the measured D→A
     must rise by delay_ms within one capture frame.
  5. summary regenerates from the records (summarize() reads the jsonl, nothing else).
"""
from __future__ import annotations

import json
import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterator

from .capture import Frame
from .probe import Code, DecodeError, decode_capture, page_html

MS = 1_000_000


# ------------------------------------------------------------------------------------- server
class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *a, page: str = "", **kw):
        self._page = page
        super().__init__(*a, **kw)

    def log_message(self, *a) -> None:  # quiet
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/t"):
            body = str(time.monotonic_ns()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
        else:
            body = self._page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int, known_delay_ms: int = 0) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("0.0.0.0", port), partial(_Handler, page=page_html(known_delay_ms)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


# ------------------------------------------------------------------------------------ records
@dataclass
class FrameRecord:
    seq: int
    ts_kernel_ns: int
    t_dq_ns: int
    ok: bool
    draw_ms: int | None = None
    receipt_ms: int | None = None
    count: int | None = None
    rtt_ms: int | None = None
    flags: int | None = None
    error: str | None = None

    @property
    def eo_ms(self) -> float | None:
        """E→O for this frame: kernel capture stamp minus the agent-clock time it was drawn."""
        if not self.ok or self.draw_ms in (None, 0):
            return None
        return _unwrap_ms(self.ts_kernel_ns // MS, self.draw_ms)


def _unwrap_ms(ref_ms: int, enc_ms: int) -> float:
    """enc_ms is the agent clock mod 2^32; recover the signed difference ref - enc."""
    d = (ref_ms - enc_ms) % (1 << 32)
    if d > (1 << 31):
        d -= 1 << 32
    return float(d)


@dataclass
class InputRecord:
    kind: str
    t_write_ns: int          # D: the decision-side stamp (the serial write, or the stamp before an inserted known delay)
    known_delay_ms: int
    found: bool
    seq: int | None = None
    ts_kernel_ns: int | None = None
    receipt_ms: int | None = None
    count_before: int | None = None
    count_after: int | None = None
    rtt_ms: int | None = None
    da_ms: float | None = None
    eo_ms: float | None = None
    rt_ms: float | None = None   # serial write → frame showing the echo, at the kernel stamp


class FrameWatcher:
    """Consumes a frame source on a thread, decodes every frame, keeps the latest state."""

    def __init__(self, source, out: Path, decode: Callable[[Frame], Code] = decode_capture):
        self.source, self.out, self.decode = source, out, decode
        self.latest: FrameRecord | None = None
        self.n_ok = self.n_bad = 0
        self._lock = threading.Condition()
        self._stop = False
        self._f = open(out / "frames.jsonl", "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "FrameWatcher":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            for fr in self.source.frames():
                rec = self._decode(fr)
                self._f.write(json.dumps(asdict(rec)) + "\n")
                with self._lock:
                    self.latest = rec
                    self._lock.notify_all()
                if self._stop:
                    break
        finally:
            self._f.flush()

    def _decode(self, fr: Frame) -> FrameRecord:
        try:
            c = self.decode(fr)
            self.n_ok += 1
            return FrameRecord(fr.seq, fr.ts_kernel_ns, fr.t_dq_ns, True, c.draw_ms, c.receipt_ms, c.count, c.rtt_ms, c.flags)
        except (DecodeError, OSError, ValueError) as e:
            self.n_bad += 1
            return FrameRecord(fr.seq, fr.ts_kernel_ns, fr.t_dq_ns, False, error=str(e)[:120])

    def wait_for(self, pred: Callable[[FrameRecord], bool], timeout_s: float) -> FrameRecord | None:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            seen = self.latest
            while True:
                if self.latest is not None and self.latest is not seen:
                    seen = self.latest
                if seen is not None and pred(seen):
                    return seen
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return None
                self._lock.wait(rem)

    def stop(self) -> None:
        if self._f.closed:
            return
        self._stop = True
        self.source.close()
        if self._thread.ident is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise TimeoutError("frame watcher did not stop after closing the source")
        self._f.flush()
        self._f.close()


# ------------------------------------------------------------------------------------ session
class Session:
    def __init__(self, watcher: FrameWatcher, hid, out: Path, frame_period_ms: float = 1000 / 60):
        self.w, self.hid, self.out = watcher, hid, out
        self.frame_period_ms = frame_period_ms
        self._inputs = open(out / "inputs.jsonl", "a", encoding="utf-8")

    def wait_synced(self, timeout_s: float = 30.0) -> FrameRecord:
        rec = self.w.wait_for(lambda r: r.ok and r.draw_ms not in (None, 0), timeout_s)
        if rec is None:
            raise TimeoutError("no decodable, synced probe frame — is the target showing the page?")
        return rec

    def one_input(self, kind: str = "click", known_delay_ms: int = 0, timeout_s: float = 2.0) -> InputRecord:
        before = self.wait_synced()
        # Contract c3: D is stamped, THEN the known delay is inserted in software, THEN the serial write
        # happens; D->A measured from D must rise by the inserted delay (within one capture frame).
        t_d = time.monotonic_ns()
        if known_delay_ms:
            time.sleep(known_delay_ms / 1000)
        t_write = self.hid.click() if kind == "click" else self.hid.key("space")
        if not known_delay_ms:
            t_d = t_write
        target = (before.count + 1) & 0xFFFF
        after = self.w.wait_for(lambda r: r.ok and r.count == target and r.ts_kernel_ns > t_write, timeout_s)
        rec = InputRecord(kind, t_d, known_delay_ms, after is not None, count_before=before.count)
        if after is not None:
            rec.seq, rec.ts_kernel_ns, rec.receipt_ms = after.seq, after.ts_kernel_ns, after.receipt_ms
            rec.count_after, rec.rtt_ms = after.count, after.rtt_ms
            rec.da_ms = _unwrap_ms(after.receipt_ms, (t_d // MS) & 0xFFFFFFFF)
            rec.eo_ms = after.eo_ms
            rec.rt_ms = (after.ts_kernel_ns - t_d) / MS
        self._inputs.write(json.dumps(asdict(rec)) + "\n")
        self._inputs.flush()
        return rec

    def null_test(self, seconds: float) -> dict:
        """Contract 3: no input; the count must never advance and decodes must keep succeeding."""
        start = self.wait_synced()
        t_end = time.monotonic() + seconds
        advanced = 0
        n = 0
        last = start
        while time.monotonic() < t_end:
            rec = self.w.wait_for(lambda r: r is not last, min(1.0, max(0.01, t_end - time.monotonic())))
            if rec is None:
                continue
            n += 1
            if rec.ok and rec.count != start.count:
                advanced += 1
            last = rec
        res = {"seconds": seconds, "frames": n, "count_advanced_frames": advanced, "decode_failures": self.w.n_bad}
        (self.out / "null_test.json").write_text(json.dumps(res, indent=1))
        return res

    def close(self) -> None:
        self._inputs.close()


# ------------------------------------------------------------------------------------ summary
def _dist(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = lambda p: s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"n": len(s), "min": s[0], "median": statistics.median(s), "p90": q(0.9), "p99": q(0.99), "max": s[-1]}


def summarize(out: Path, frame_period_ms: float = 1000 / 60) -> dict:
    """Everything below is regenerated from the jsonl records (contract 5)."""
    frames = [json.loads(l) for l in (out / "frames.jsonl").read_text().splitlines() if l.strip()]
    inputs = [json.loads(l) for l in (out / "inputs.jsonl").read_text().splitlines() if l.strip()] if (out / "inputs.jsonl").exists() else []
    eo = []
    seen = set()
    for f in frames:
        if f["ok"] and f["draw_ms"] not in (None, 0) and f["draw_ms"] not in seen:
            seen.add(f["draw_ms"])
            eo.append(_unwrap_ms(f["ts_kernel_ns"] // MS, f["draw_ms"]))
    by_delay: dict[int, list[float]] = {}
    for i in inputs:
        if i["found"]:
            by_delay.setdefault(i["known_delay_ms"], []).append(i["da_ms"])
    rtts = [f["rtt_ms"] for f in frames if f["ok"] and f["rtt_ms"] is not None]
    res = {
        "frames": len(frames), "decode_ok": sum(f["ok"] for f in frames), "frame_period_ms": frame_period_ms,
        "sync_bound_ms_half_rtt": (min(rtts) / 2 if rtts else None),
        "E_to_O_ms": _dist(eo),
        "inputs": len(inputs), "inputs_found": sum(i["found"] for i in inputs),
        "D_to_A_ms": {str(k): _dist(v) for k, v in sorted(by_delay.items())},
        "round_trip_ms": _dist([i["rt_ms"] for i in inputs if i["found"]]),
    }
    if 0 in by_delay and any(k > 0 for k in by_delay):
        base = statistics.median(by_delay[0])
        res["known_delay_test"] = {str(k): {"expected_shift_ms": k, "measured_shift_ms": statistics.median(v) - base,
                                            "pass": abs(statistics.median(v) - base - k) <= frame_period_ms}
                                   for k, v in by_delay.items() if k > 0}
    if (out / "null_test.json").exists():
        res["null_test"] = json.loads((out / "null_test.json").read_text())
    return res


def summary_markdown(res: dict, title: str) -> str:
    L = [f"# {title}", ""]
    L.append(f"Frames {res['frames']} (decoded {res['decode_ok']}); capture period {res['frame_period_ms']:.2f} ms = quantization floor; "
             f"clock-sync bound ±{res['sync_bound_ms_half_rtt']:.2f} ms (half the best RTT)." if res.get("sync_bound_ms_half_rtt") is not None
             else f"Frames {res['frames']} (decoded {res['decode_ok']}); no synced frames.")
    L += ["", "| interval | n | min | median | p90 | p99 | max |", "|---|---|---|---|---|---|---|"]
    def row(name, d):
        if d.get("n"):
            L.append(f"| {name} | {d['n']} | {d['min']:.1f} | {d['median']:.1f} | {d['p90']:.1f} | {d['p99']:.1f} | {d['max']:.1f} |")
        else:
            L.append(f"| {name} | 0 | - | - | - | - | - |")
    row("E→O (ms)", res["E_to_O_ms"])
    for k, d in res["D_to_A_ms"].items():
        row(f"D→A (ms), software delay +{k} ms", d)
    row("serial write → echo frame (ms)", res["round_trip_ms"])
    if "known_delay_test" in res:
        L += ["", "## Known-delay test", ""]
        for k, d in res["known_delay_test"].items():
            L.append(f"- +{k} ms inserted before the write: measured shift {d['measured_shift_ms']:+.1f} ms → {'PASS' if d['pass'] else 'FAIL'} (tolerance one frame)")
    if "null_test" in res:
        n = res["null_test"]
        L += ["", "## Null test", "", f"- {n['seconds']:.0f} s, {n['frames']} frames, count advanced in {n['count_advanced_frames']} frames "
              f"({'PASS' if n['count_advanced_frames'] == 0 else 'FAIL'}); decode failures {n['decode_failures']}"]
    return "\n".join(L) + "\n"
