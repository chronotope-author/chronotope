"""Serve a gym scenario's frames to the target's browser and take the page's input reports (PHY3 part 3).

The server composes, for every environment frame k, one 1920x1080 gray image:

  rows 0-3 at (X0, Y0)   the probe code (probe.py layout): draw_ms = source time of frame k in ms,
                         count = k (16 bits), receipt/rtt rows 0, flags 0 — the frame-counter strip
  panel at (X0, PANEL_Y) left gray by the server; the PAGE paints it black/white and flips it on
                         every pointerdown (the echo strip: PhysicalRoute's fallback receipt)
  image at (IMG_X, IMG_Y) the scenario render, enlarged IMG_SCALE times (nearest neighbour)

and streams it as MJPEG (`/stream`, multipart/x-mixed-replace) or one-shot (`/frame`). The page shows
the stream on a canvas at fit-to-viewport scale, keeps a Cristian clock sync against `/t`, and on
pointerdown POSTs `/input` {"t_ms": agent-clock ms, "x", "y": scenario px, "count"} — the page-reported
input time that PhysicalRoute uses as A. Frame k's registry entry (source_event_id, source_time_ns,
payload = the scenario's RGB canvas) is what CaptureFrameSource resolves a decoded counter to, so the
observation's payload hash is the served frame's, and Replay stays available (U-P5).
"""
from __future__ import annotations

import io
import json
import socket
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from .probe import BLOCK, CELLS, PANEL_Y, X0, Y0, Code, draw_code

W, H = 1920, 1080
IMG_X, IMG_Y = 40, 600         # the scenario image's top-left in the served frame
IMG_SCALE = 2                  # 320x240 -> 640x480 (fits beneath the panel)
JPEG_QUALITY = 90


class FrameRegistry:
    """k -> (source_event_id, source_time_ns, payload). Bounded; the counter row is 16 bits."""

    def __init__(self, keep: int = 4096) -> None:
        self._d: dict[int, tuple[int, int, bytes]] = {}
        self._keep = keep
        self._lock = threading.Lock()

    def put(self, k: int, source_event_id: int, source_time_ns: int, payload: bytes) -> None:
        with self._lock:
            self._d[k & 0xFFFF] = (source_event_id, source_time_ns, payload)
            if len(self._d) > self._keep:
                for old in sorted(self._d)[: len(self._d) - self._keep]:
                    del self._d[old]

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def resolve(self, code: Code) -> tuple[int, int, bytes] | None:
        """The registry entry for a decoded strip, or None if the counter/draw_ms pair is unknown."""
        with self._lock:
            e = self._d.get(code.count & 0xFFFF)
        if e is None or ((e[1] // 1_000_000) & 0xFFFFFFFF) != code.draw_ms:
            return None
        return e


class ReceiptSink:
    """Page-reported inputs, in arrival order: {"t_ns", "x", "y", "kind", "count", "arrived_ns"}."""

    def __init__(self) -> None:
        self._items: list[dict] = []
        self._cv = threading.Condition()

    def push(self, item: dict) -> None:
        with self._cv:
            self._items.append(item)
            self._cv.notify_all()

    def wait_after(self, t_ns: int, timeout_s: float, kind: str | None = None) -> dict | None:
        """The first report whose page time is >= t_ns (and of `kind`), waiting up to timeout_s."""
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                for it in self._items:
                    if it["t_ns"] >= t_ns and (kind is None or it.get("kind") == kind):
                        return it
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return None
                self._cv.wait(rem)

    def all(self) -> list[dict]:
        with self._cv:
            return list(self._items)


def compose(code: Code, payload: bytes | None, canvas_w: int = 0, canvas_h: int = 0) -> Image.Image:
    """payload: the manifest's row-major RGB canvas (canvas_w x canvas_h), as the trace stores it."""
    img = Image.new("L", (W, H), 128)
    draw_code(img, code, echo_on=None)          # the page paints the panel
    if payload is not None:
        sc = Image.frombytes("RGB", (canvas_w, canvas_h), payload).convert("L")
        sc = sc.resize((sc.width * IMG_SCALE, sc.height * IMG_SCALE), Image.NEAREST)
        img.paste(sc, (IMG_X, IMG_Y))
    return img


def strip_code(k: int, source_time_ns: int) -> Code:
    return Code(draw_ms=(source_time_ns // 1_000_000) & 0xFFFFFFFF, receipt_ms=0, count=k & 0xFFFF, rtt_ms=0, flags=0)


class FrameServer:
    """Publish frames; serve the page, the stream, the clock and the input POSTs."""

    def __init__(self, canvas_w: int, canvas_h: int, port: int = 8766, host: str = "0.0.0.0") -> None:
        self.canvas_w, self.canvas_h, self.port, self.host = canvas_w, canvas_h, port, host
        self.registry = FrameRegistry()
        self.receipts = ReceiptSink()
        self._jpeg: bytes = b""
        self._k = -1
        self._seq = 0                  # publication sequence: increases across trials (k restarts at 0 per trial)
        self._cv = threading.Condition()
        self._httpd: ThreadingHTTPServer | None = None
        self.n_published = 0

    # -- publishing ------------------------------------------------------------------------------
    def publish(self, k: int, source_event_id: int, source_time_ns: int, payload: bytes) -> bytes:
        """Compose frame k and make it the current stream frame. Returns the JPEG."""
        img = compose(strip_code(k, source_time_ns), payload, self.canvas_w, self.canvas_h)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        jpeg = out.getvalue()
        self.registry.put(k, source_event_id, source_time_ns, payload)
        with self._cv:
            self._jpeg, self._k = jpeg, k
            self._seq += 1
            self.n_published += 1
            self._cv.notify_all()
        return jpeg

    def latest(self) -> tuple[int, bytes]:
        with self._cv:
            return self._k, self._jpeg

    def wait_newer(self, seq: int, timeout_s: float) -> tuple[int, bytes] | None:
        """(publication seq, jpeg) of the first publication after `seq`, or None on timeout."""
        with self._cv:
            if self._seq <= seq and not self._cv.wait_for(lambda: self._seq > seq, timeout_s):
                return None
            return self._seq, self._jpeg

    # -- http ------------------------------------------------------------------------------------
    def start(self) -> "FrameServer":
        self._httpd = ThreadingHTTPServer((self.host, self.port), partial(_Handler, server_obj=self))
        self.port = self._httpd.server_address[1]
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def url(self) -> str:
        return f"http://{lan_ip()}:{self.port}/"


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, server_obj: FrameServer, **kw):
        self.srv = server_obj
        super().__init__(*a, **kw)

    def log_message(self, *a) -> None:
        pass

    def _send(self, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.startswith("/t"):
            self._send(str(time.monotonic_ns()).encode(), "text/plain")
        elif self.path.startswith("/frame"):
            k, jpeg = self.srv.latest()
            self._send(jpeg, "image/jpeg", {"X-Frame-K": str(k)})
        elif self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seq = 0
            try:
                while True:
                    got = self.srv.wait_newer(seq, 5.0)
                    if got is None:
                        continue
                    seq, jpeg = got
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg))
                    self.wfile.write(jpeg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(page_html().encode(), "text/html; charset=utf-8")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        arrived = time.monotonic_ns()
        try:
            d = json.loads(raw)
            item = {"t_ns": int(round(float(d["t_ms"]) * 1_000_000)), "x": int(round(float(d["x"]))),
                    "y": int(round(float(d["y"]))), "kind": str(d.get("kind", "click")),
                    "count": int(d.get("count", 0)), "arrived_ns": arrived}
        except (ValueError, KeyError, TypeError) as e:
            self.send_response(400); self.end_headers(); self.wfile.write(str(e).encode())
            return
        self.srv.receipts.push(item)
        self._send(b"ok", "text/plain")


def page_html() -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>chronotope gym</title>
<style>html,body{{margin:0;height:100%;background:#808080;overflow:hidden;cursor:none}}canvas{{display:block}}</style></head>
<body><canvas id="c"></canvas>
<script>
(() => {{
const W={W}, H={H}, X0={X0}, PANEL_Y={PANEL_Y}, BLOCK={BLOCK}, CELLS={CELLS}, IMG_X={IMG_X}, IMG_Y={IMG_Y}, IMG_SCALE={IMG_SCALE};
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let dpr=1, fit=1, ox=0, oy=0;
function resize(){{ dpr=window.devicePixelRatio||1; cv.width=Math.round(innerWidth*dpr); cv.height=Math.round(innerHeight*dpr);
  cv.style.width=innerWidth+'px'; cv.style.height=innerHeight+'px';
  fit=Math.min(innerWidth/W, innerHeight/H); ox=(innerWidth-W*fit)/2; oy=(innerHeight-H*fit)/2; }}
addEventListener('resize',resize); resize();
let offset=null, rttMin=1e9;
async function sync(){{ try{{ const p0=performance.now(); const r=await fetch('/t',{{cache:'no-store'}}); const txt=await r.text();
  const p1=performance.now(); const rtt=p1-p0; const agentMs=Number(BigInt(txt.trim())/1000000n);
  if(rtt<rttMin){{ rttMin=rtt; offset=agentMs-(p0+p1)/2; }} }}catch(e){{}} }}
sync(); setInterval(sync, 500);
const img=new Image(); img.src='/stream';
let echoOn=false, count=0;
addEventListener('pointerdown', ev=>{{ if(offset===null) return; const t=ev.timeStamp+offset;
  const fx=(ev.clientX-ox)/fit, fy=(ev.clientY-oy)/fit;           // served-frame px
  const x=(fx-IMG_X)/IMG_SCALE, y=(fy-IMG_Y)/IMG_SCALE;             // scenario px (may be outside the image)
  echoOn=!echoOn; count=(count+1)&0xFFFF;
  fetch('/input',{{method:'POST',headers:{{'Content-Type':'application/json'}},
    body:JSON.stringify({{t_ms:t, x:x, y:y, kind:'click', count:count}})}}).catch(()=>{{}}); }}, {{passive:true}});
function draw(){{ ctx.setTransform(dpr,0,0,dpr,0,0); ctx.imageSmoothingEnabled=false;
  ctx.fillStyle='#808080'; ctx.fillRect(0,0,innerWidth,innerHeight);
  if(img.complete && img.naturalWidth>0) ctx.drawImage(img, ox, oy, W*fit, H*fit);
  ctx.fillStyle=echoOn?'#fff':'#000'; ctx.fillRect(ox+X0*fit, oy+PANEL_Y*fit, CELLS*BLOCK*fit, 6*BLOCK*fit);
  requestAnimationFrame(draw); }}
requestAnimationFrame(draw);
}})();
</script></body></html>"""


class EnvPublisher:
    """Real-time environment: publishes frame k at harness time h0 + k*period until the horizon.
    The environment is a pure function of task time, so what the target displays at k is m.render(k*period)."""

    def __init__(self, server: FrameServer, manifest, period_ns: int, horizon_ns: int) -> None:
        self.server, self.m, self.period_ns, self.horizon_ns = server, manifest, period_ns, horizon_ns
        self._stop = False
        self._thread: threading.Thread | None = None
        self.h0: int | None = None
        self.n_late = 0

    def start(self, h0: int) -> "EnvPublisher":
        self.h0 = h0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        k = 0
        assert self.h0 is not None
        while not self._stop:
            t = k * self.period_ns
            if t > self.horizon_ns:
                return
            due = self.h0 + t
            dt = due - time.monotonic_ns()
            if dt > 0:
                time.sleep(dt / 1e9)
            elif -dt > self.period_ns:
                self.n_late += 1
            st = self.m.state_at(t)
            self.server.publish(k, st.latest_event_id, t, self.m.render(t))
            k += 1

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
