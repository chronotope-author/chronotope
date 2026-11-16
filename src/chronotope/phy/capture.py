"""V4L2 capture (Elgato 4K X) in NV12 or MJPG with kernel timestamps on CLOCK_MONOTONIC.

NV12 is the default (cards/PHY3-system-cell.md): the frame's first width*height bytes are the Y
plane, and every probe (frame-counter strip, echo panel) is luminance-coded, so decoding never
touches the chroma plane and never needs a colour conversion. MJPG stays available (--pixfmt MJPG).

Adapted from personalgui-accel-old/gua_eval/capture.py (audited 2026-08-23) with the card's fixes:
  * BOTH the kernel buffer timestamp and the dequeue time are propagated per frame (ns);
  * clock identity is VERIFIED at runtime (contract item 1), not assumed: the driver must flag the
    buffer timestamp as CLOCK_MONOTONIC and Python's monotonic clock must be CLOCK_MONOTONIC;
  * truncated JPEGs are counted, never silently dropped.

The device is opened lazily; construction never touches hardware, so everything imports and tests
without the card. FakeSource provides a deterministic frame stream for tests.
"""
from __future__ import annotations

import fcntl
import mmap
import os
import select
import struct
import time
from dataclasses import dataclass
from typing import Iterator, Protocol


def _ioc(d: int, t: str, nr: int, size: int) -> int:
    return (d << 30) | (ord(t) << 8) | nr | (size << 16)


_VIDIOC_ENUM_FMT = _ioc(3, "V", 2, 64)
_VIDIOC_S_FMT = _ioc(3, "V", 5, 208)
_VIDIOC_REQBUFS = _ioc(3, "V", 8, 20)
_VIDIOC_QUERYBUF = _ioc(3, "V", 9, 88)
_VIDIOC_QBUF = _ioc(3, "V", 15, 88)
_VIDIOC_DQBUF = _ioc(3, "V", 17, 88)
_VIDIOC_STREAMON = _ioc(1, "V", 18, 4)
_VIDIOC_STREAMOFF = _ioc(1, "V", 19, 4)
_VIDIOC_S_PARM = _ioc(3, "V", 22, 204)
_BUF_TYPE_CAPTURE = 1
_MEMORY_MMAP = 1
_TS_MASK, _TS_MONOTONIC = 0x0000E000, 0x00002000


def python_clock_is_monotonic() -> bool:
    """True iff time.monotonic_ns() is CLOCK_MONOTONIC (Linux CPython: clock_gettime(CLOCK_MONOTONIC))."""
    info = time.get_clock_info("monotonic")
    return "CLOCK_MONOTONIC" in info.implementation and "RAW" not in info.implementation


@dataclass(frozen=True)
class Frame:
    seq: int
    ts_kernel_ns: int        # driver buffer timestamp (CLOCK_MONOTONIC when ts_monotonic)
    t_dq_ns: int             # time.monotonic_ns() right after dequeue
    ts_monotonic: bool       # driver flagged the stamp as CLOCK_MONOTONIC
    data: bytes              # MJPG: the JPEG; NV12: Y plane (w*h bytes) followed by interleaved UV
    pixfmt: str = "MJPG"
    width: int = 0
    height: int = 0

    @property
    def jpeg(self) -> bytes:
        """The JPEG bytes of an MJPG frame (kept for the measurement code and older callers)."""
        if self.pixfmt != "MJPG":
            raise ValueError(f"frame is {self.pixfmt}, not MJPG")
        return self.data

    def luma(self) -> tuple[bytes, int, int]:
        """(Y plane bytes, width, height). NV12: a slice, no decode. MJPG: decoded to 8-bit gray."""
        if self.pixfmt == "NV12":
            n = self.width * self.height
            return self.data[:n], self.width, self.height
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(self.data)).convert("L")
        return img.tobytes(), img.width, img.height


class RawFrameSource(Protocol):
    """A device-level frame stream (not the trial-level runner.FrameSource)."""
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


FrameSource = RawFrameSource   # older name


def jpeg_complete(data: bytes) -> bool:
    return data[:2] == b"\xff\xd8" and data.rstrip(b"\x00")[-2:] == b"\xff\xd9"


def nv12_size(width: int, height: int) -> int:
    return width * height * 3 // 2


def frame_complete(pixfmt: str, data: bytes, width: int, height: int) -> bool:
    return jpeg_complete(data) if pixfmt == "MJPG" else len(data) >= nv12_size(width, height)


def nv12_from_luma(y: bytes, width: int, height: int) -> bytes:
    """Pack a Y plane into an NV12 buffer with neutral chroma (tests and the simulated target)."""
    assert len(y) == width * height, (len(y), width, height)
    return bytes(y) + bytes([128]) * (width * height // 2)


class _V4L2:
    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self.path = path
        self.bufs: list[mmap.mmap] = []
        self.streaming = False

    def enum_formats(self) -> list[str]:
        """Pixel formats the device offers (VIDIOC_ENUM_FMT), e.g. ['NV12', 'MJPG', 'YUYV']."""
        out = []
        for i in range(32):
            d = bytearray(64)
            struct.pack_into("<II", d, 0, i, _BUF_TYPE_CAPTURE)
            try:
                fcntl.ioctl(self.fd, _VIDIOC_ENUM_FMT, d)
            except OSError:
                break
            out.append(bytes(d[44:48]).decode("ascii", "replace"))
        return out

    def set_format(self, pixfmt: str, width: int, height: int) -> tuple[str, int, int]:
        f = bytearray(208)
        struct.pack_into("<I", f, 0, _BUF_TYPE_CAPTURE)
        struct.pack_into("<II4sI", f, 8, width, height, pixfmt.encode("ascii"), 0)
        fcntl.ioctl(self.fd, _VIDIOC_S_FMT, f)
        w, h = struct.unpack_from("<II", f, 8)
        self.pixfmt, self.width, self.height = bytes(f[16:20]).decode("ascii", "replace"), w, h
        return self.pixfmt, w, h

    def set_fps(self, fps: float) -> float:
        p = bytearray(204)
        struct.pack_into("<I", p, 0, _BUF_TYPE_CAPTURE)
        num, den = (1, int(round(fps))) if abs(fps - round(fps)) < 1e-3 else (1001, int(round(fps * 1001)))
        struct.pack_into("<II", p, 12, num, den)
        fcntl.ioctl(self.fd, _VIDIOC_S_PARM, p)
        n, d = struct.unpack_from("<II", p, 12)
        return (d / n) if n else 0.0

    def start(self, n_buffers: int = 4) -> None:
        req = bytearray(20)
        struct.pack_into("<III", req, 0, n_buffers, _BUF_TYPE_CAPTURE, _MEMORY_MMAP)
        fcntl.ioctl(self.fd, _VIDIOC_REQBUFS, req)
        (count,) = struct.unpack_from("<I", req, 0)
        for i in range(count):
            b = bytearray(88)
            struct.pack_into("<II", b, 0, i, _BUF_TYPE_CAPTURE)
            struct.pack_into("<I", b, 60, _MEMORY_MMAP)
            fcntl.ioctl(self.fd, _VIDIOC_QUERYBUF, b)
            (offset,) = struct.unpack_from("<I", b, 64)
            (length,) = struct.unpack_from("<I", b, 72)
            mm = mmap.mmap(self.fd, length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=offset)
            self.bufs.append(mm)
            fcntl.ioctl(self.fd, _VIDIOC_QBUF, b)
        fcntl.ioctl(self.fd, _VIDIOC_STREAMON, struct.pack("<i", _BUF_TYPE_CAPTURE))
        self.streaming = True

    def _buf(self) -> bytearray:
        b = bytearray(88)
        struct.pack_into("<I", b, 4, _BUF_TYPE_CAPTURE)
        struct.pack_into("<I", b, 60, _MEMORY_MMAP)
        return b

    def dq(self, timeout_s: float) -> Frame:
        while True:
            r, _, _ = select.select([self.fd], [], [], timeout_s)
            if not r:
                raise TimeoutError(f"{self.path}: no frame within {timeout_s}s — HDMI source active?")
            b = self._buf()
            try:
                fcntl.ioctl(self.fd, _VIDIOC_DQBUF, b)
            except BlockingIOError:
                continue
            t_dq = time.monotonic_ns()
            index, _, bytesused, flags = struct.unpack_from("<IIII", b, 0)
            sec, usec = struct.unpack_from("<qq", b, 24)
            (seq,) = struct.unpack_from("<I", b, 56)
            data = bytes(self.bufs[index][:bytesused])
            fcntl.ioctl(self.fd, _VIDIOC_QBUF, b)
            return Frame(seq=seq, ts_kernel_ns=sec * 1_000_000_000 + usec * 1000, t_dq_ns=t_dq,
                         ts_monotonic=(flags & _TS_MASK) == _TS_MONOTONIC, data=data,
                         pixfmt=getattr(self, "pixfmt", "MJPG"), width=getattr(self, "width", 0),
                         height=getattr(self, "height", 0))

    def close(self) -> None:
        if self.streaming:
            try:
                fcntl.ioctl(self.fd, _VIDIOC_STREAMOFF, struct.pack("<i", _BUF_TYPE_CAPTURE))
            except OSError:
                pass
            self.streaming = False
        for mm in self.bufs:
            try:
                mm.close()
            except BufferError:
                pass
        self.bufs = []
        os.close(self.fd)


def list_formats(device: str = "/dev/video0") -> list[str]:
    cap = _V4L2(device)
    try:
        return cap.enum_formats()
    finally:
        os.close(cap.fd)


class ElgatoSource:
    """Live capture (NV12 by default, or MJPG). frames() yields complete frames with kernel + dequeue stamps.

    Contract item 1: the FIRST dequeued frame must be flagged CLOCK_MONOTONIC and the Python clock
    must be CLOCK_MONOTONIC, else ClockIdentityError — a measurement on two clocks is not a measurement.
    """

    def __init__(self, device: str = "/dev/video0", width: int = 1920, height: int = 1080, fps: float = 60.0,
                 n_buffers: int = 4, timeout_s: float = 3.0, pixfmt: str = "NV12"):
        if pixfmt not in ("NV12", "MJPG"):
            raise ValueError(f"pixfmt {pixfmt!r}: NV12 or MJPG")
        self.device, self.width, self.height, self.fps, self.pixfmt = device, width, height, fps, pixfmt
        self.n_buffers, self.timeout_s = n_buffers, timeout_s
        self.n_truncated = 0
        self.n_frames = 0
        self._cap: _V4L2 | None = None
        self.clock_verified = False

    def _ensure_open(self) -> None:
        if self._cap is not None:
            return
        if not python_clock_is_monotonic():
            raise ClockIdentityError(f"time.monotonic is {time.get_clock_info('monotonic').implementation}, not CLOCK_MONOTONIC")
        cap = _V4L2(self.device)
        try:
            got, w, h = cap.set_format(self.pixfmt, self.width, self.height)
            if got != self.pixfmt:
                raise RuntimeError(f"{self.device}: driver substituted pixfmt {got} for {self.pixfmt}")
            self.width, self.height = w, h
            self.fps = cap.set_fps(self.fps) or self.fps
            cap.start(self.n_buffers)
        except BaseException:
            cap.close()
            raise
        self._cap = cap

    def frames(self) -> Iterator[Frame]:
        self._ensure_open()
        assert self._cap is not None
        while True:
            f = self._cap.dq(self.timeout_s)
            if not self.clock_verified:
                if not f.ts_monotonic:
                    raise ClockIdentityError(f"{self.device}: buffer timestamps are not CLOCK_MONOTONIC (flags)")
                if abs(f.t_dq_ns - f.ts_kernel_ns) > 2_000_000_000:
                    raise ClockIdentityError(f"{self.device}: kernel stamp and monotonic_ns differ by "
                                             f"{(f.t_dq_ns - f.ts_kernel_ns) / 1e9:.1f} s — not the same clock")
                self.clock_verified = True
            if not frame_complete(f.pixfmt, f.data, f.width, f.height):
                self.n_truncated += 1
                continue
            self.n_frames += 1
            yield f

    def close(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.close()


class ClockIdentityError(RuntimeError):
    pass


class FakeSource:
    """Deterministic frame stream for tests: a callable renders the frame bytes for a given time
    (a JPEG for pixfmt MJPG; an NV12 buffer, or just its Y plane, for pixfmt NV12)."""

    def __init__(self, render, period_ns: int = 16_666_667, start_ns: int | None = None, n: int | None = None,
                 realtime: bool = False, pixfmt: str = "MJPG", width: int = 1920, height: int = 1080):
        self.render, self.period_ns, self.n, self.realtime = render, period_ns, n, realtime
        self.t0 = time.monotonic_ns() if start_ns is None else start_ns
        self.n_truncated = 0
        self.n_frames = 0
        self.pixfmt, self.width, self.height, self.fps = pixfmt, width, height, 1e9 / period_ns
        self.clock_verified = True
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        i = 0
        while self.n is None or i < self.n:
            ts = self.t0 + i * self.period_ns
            if self.realtime:
                now = time.monotonic_ns()
                if ts > now:
                    time.sleep((ts - now) / 1e9)
            if self._closed:
                return
            data = self.render(ts)
            if self.pixfmt == "NV12" and len(data) == self.width * self.height:
                data = nv12_from_luma(data, self.width, self.height)
            self.n_frames += 1
            yield Frame(seq=i, ts_kernel_ns=ts, t_dq_ns=ts + 1_000_000, ts_monotonic=True, data=data,
                        pixfmt=self.pixfmt, width=self.width, height=self.height)
            i += 1

    def close(self) -> None:
        self._closed = True
