"""KMBox B serial HID actuator — trimmed from personalgui-accel-setup-old/hid_kmbox.py with the
card's fixes: echo verification is MANDATORY (a rejected command raises), the serial-write timestamp
is the decision-side stamp D and is NEVER reported as A, and the REPL is interrupted on open so a
continuation-mode device cannot masquerade as a working one.

The measurement needs only two primitives: a click where the cursor already is (the probe page
listens everywhere, so no coordinate calibration and no F38 exposure) and a key press (F57: the
keyboard channel is subject to the target's input method; clicks are the primary channel).

    with KMBox() as hid:
        t_write_ns = hid.click()      # D on this machine's CLOCK_MONOTONIC
"""
from __future__ import annotations

import glob
import time
from collections import deque

# Port autodetect (PHY3): the box now enumerates as a Silicon Labs CP2102N; the older CH340 (1a86) id is
# the fallback. An explicit --port always wins.
PORT_PATTERNS = ("/dev/serial/by-id/*CP2102N*", "/dev/serial/by-id/usb-1a86_*")


def find_port(explicit: str | None = None, patterns: tuple[str, ...] = PORT_PATTERNS, _glob=glob.glob) -> str:
    """The serial device path: `explicit` if given, else the first pattern that matches (in order)."""
    if explicit:
        return explicit
    for pat in patterns:
        hits = sorted(_glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no KMBox serial port: none of {patterns} matched (pass --port)")

_CMDS = {
    "init": "import km",
    "dev_click": "kmc(km.{button},{hold_ms})",
    "dev_key": "kmk({code},{ms})",
    "move": "km.move({dx},{dy})",
    "key_down": "km.down({code})",
    "key_up": "km.up({code})",
}
_DEVICE_DEFS = (
    "import time",
    'exec("def kmc(b,ms):\\n b(1)\\n time.sleep_ms(ms)\\n b(0)")',
    'exec("def kmk(c,ms):\\n km.down(c)\\n time.sleep_ms(ms)\\n km.up(c)")',
)
_ERR_MARKERS = (b"Traceback", b"Error", b"NameError", b"SyntaxError")
_EOL = "\r\n"
KEYCODES = {**{chr(ord("a") + i): 4 + i for i in range(26)}, **{str(d): 29 + d for d in range(1, 10)}, "0": 39,
            "enter": 40, "esc": 41, "backspace": 42, "tab": 43, "space": 44, "minus": 45, "-": 45, "=": 46,
            "[": 47, "]": 48, ";": 51, "'": 52, ",": 54, ".": 55, "/": 56, "f11": 68,
            "right": 79, "left": 80, "down": 81, "up": 82, "ctrl": 224, "shift": 225, "alt": 226}
_SHIFTED = {":": ";", "?": "/", "_": "-", "+": "=", "\"": "'", "<": ",", ">": ".", "{": "[", "}": "]"}


class DeviceRejected(RuntimeError):
    pass


class KMBox:
    def __init__(self, port: str | None = None, baud: int = 115200, settle_s: float = 0.01, serial_factory=None):
        self.port_path, self.baud, self.settle_s = port, baud, settle_s
        self._port = None
        self._serial_factory = serial_factory
        self.replies: deque[dict] = deque(maxlen=512)
        self.last_write_ns: int | None = None

    # -- lifecycle --------------------------------------------------------------------------------
    def open(self) -> "KMBox":
        if self._port is None:
            if self.port_path is None:
                self.port_path = find_port()
            if self._serial_factory is not None:
                self._port = self._serial_factory(self.port_path, self.baud)
            else:
                from serial import Serial  # lazy: pyserial is only needed with hardware
                self._port = Serial(self.port_path, self.baud, timeout=0)
            self.interrupt()
            self._send("init", settle_s=0.05)
            for line in _DEVICE_DEFS:
                self._send_raw(line, settle_s=0.05)
        return self

    def interrupt(self) -> str:
        """Raw 0x03 to the REPL (NOT a HID Ctrl+C); raises if it stays in continuation mode."""
        port = self._port
        assert port is not None
        port.write(b"\x03")
        time.sleep(0.3)
        port.write(b"\r\n")
        time.sleep(0.3)
        reply = port.read(512).decode("utf-8", "replace")
        if reply.rstrip().endswith("..."):
            raise DeviceRejected(f"REPL still in continuation mode after interrupt: {reply[-80:]!r}")
        return reply

    def close(self) -> None:
        if self._port is not None:
            try:
                self._port.write(b"km.zero(0)\r\n")
                self._port.flush()
            finally:
                self._port.close()
                self._port = None

    def __enter__(self) -> "KMBox":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- protocol ---------------------------------------------------------------------------------
    def _send(self, cmd: str, settle_s: float | None = None, **kw) -> int:
        return self._send_raw(_CMDS[cmd].format(**kw), settle_s)

    def _send_raw(self, line: str, settle_s: float | None = None) -> int:
        """Write one REPL line; return the monotonic_ns stamp taken immediately BEFORE the write.
        The echo is read after the settle window and verified: an error marker raises DeviceRejected."""
        if self._port is None:
            self.open()
        assert self._port is not None
        data = (line + _EOL).encode("utf-8")
        t0 = time.monotonic_ns()
        self._port.write(data)
        self.last_write_ns = t0
        time.sleep(self.settle_s if settle_s is None else settle_s)
        n = self._port.in_waiting
        reply = self._port.read(n) if n else b""
        self.replies.append({"t_ns": t0, "sent": line, "reply": reply.decode("utf-8", "replace")})
        if any(m in reply for m in _ERR_MARKERS):
            raise DeviceRejected(f"device rejected {line!r}: {reply!r}")
        return t0

    # -- primitives -------------------------------------------------------------------------------
    def click(self, button: str = "left", hold_ms: int = 20) -> int:
        """Press+release where the cursor is. Returns the serial-write stamp (D), never A."""
        return self._send("dev_click", button=button, hold_ms=int(hold_ms))

    def key(self, name: str, hold_ms: int = 60) -> int:
        return self._send("dev_key", code=KEYCODES[name], ms=int(hold_ms))

    def move(self, dx: int, dy: int, step: int = 120) -> int:
        """Relative move, split into HID-sized steps (|d| <= step per report). Returns the first write stamp."""
        dx, dy, t0 = int(dx), int(dy), None
        while dx or dy:
            sx = max(-step, min(step, dx)); sy = max(-step, min(step, dy))
            t = self._send("move", dx=sx, dy=sy)
            t0 = t if t0 is None else t0
            dx -= sx; dy -= sy
        return t0 if t0 is not None else time.monotonic_ns()

    def version(self) -> str:
        """The device's own version string (preflight: proves the REPL answers, not just that the port opens)."""
        self._send_raw("print(km.version())", settle_s=0.2)
        return self.replies[-1]["reply"].strip()

    def chord(self, names: list[str], hold_ms: int = 60) -> None:
        codes = [KEYCODES[n] for n in names]
        for c in codes:
            self._send("key_down", code=c)
        time.sleep(hold_ms / 1000)
        for c in reversed(codes):
            self._send("key_up", code=c)

    def type_text(self, text: str, per_char_s: float = 0.03) -> None:
        """Type ASCII via HID usage ids (F57 caveat: the target's input method can rewrite this)."""
        for ch in text:
            if ch in _SHIFTED or ch.isupper():
                base = _SHIFTED.get(ch, ch.lower())
                self.chord(["shift", base])
            else:
                self.key(ch if ch != " " else "space", hold_ms=20)
            time.sleep(per_char_s)


class FakeSerial:
    """In-memory serial port for tests: echoes commands and optionally injects errors."""

    def __init__(self, port: str = "", baud: int = 0, reject: set[str] | None = None, on_write=None):
        self.buf = bytearray()
        self.writes: list[bytes] = []
        self.reject = reject or set()
        self.on_write = on_write

    def write(self, data: bytes) -> None:
        self.writes.append(data)
        line = data.decode("utf-8", "replace").strip()
        if data == b"\x03":
            self.buf += b"\r\n>>> "
            return
        # reject only actual command invocations ("kmk(...)"), never the def upload ("def kmk...")
        if not line.startswith(("exec(", "import", "def ")) and any(r in line for r in self.reject):
            self.buf += (line + "\r\nTraceback (most recent call last):\r\nNameError\r\n>>> ").encode()
        else:
            self.buf += (line + "\r\n>>> ").encode()
        if self.on_write:
            self.on_write(line)

    def flush(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        return len(self.buf)

    def read(self, n: int) -> bytes:
        out, self.buf = bytes(self.buf[:n]), self.buf[n:]
        return out

    def close(self) -> None:
        pass
