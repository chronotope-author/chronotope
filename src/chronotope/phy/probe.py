"""The probe page and its pixel code.

The target's browser shows probe.html full-screen. Every animation frame it draws, in device
pixels, four rows of BLOCK-sized black/white cells (LSB first, left to right):

  row 0  sync pattern 1010...10 (32 cells) — locates the code, measures the cell size
  row 1  draw_ms      : agent-clock time (ms mod 2^32) at which THIS frame was drawn
  row 2  receipt_ms   : agent-clock time (ms mod 2^32) at which the LAST input event was received
  row 3  count (16 bits) | rtt_ms of the best sync sample (8 bits) | flags (8 bits)
         flags: bit0 = page-side known-delay mode; bit1 = parity(row1); bit2 = parity(row2)

Agent-clock time on the page = performance.now() + offset, where offset comes from Cristian's
algorithm against GET /t (this machine's monotonic_ns) using the minimum-RTT sample; the page keeps
re-sampling so the bound only tightens. Beneath the code, a large panel flips black/white on every
input event so a human (or a GIF) can see the echo.

The same layout is implemented twice: in JavaScript (the page) and in Python (`encode_frame`, for
tests and for the FakeSource that simulates the target), and decoded once (`decode_frame`).
Decoding is scale-free: the sync row yields x0, y0 and the cell size, so a target at another
resolution, or a capture at another size, still decodes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageDraw

BLOCK = 40          # device px per cell in the page; the decoder measures the real size
CELLS = 32
X0, Y0 = 40, 40     # page origin of the code
ROW_PITCH = 1.5     # rows are 1.5 cells apart (a half-cell gray gap between rows)
ROWS = 4
PANEL_Y = 320       # echo panel top (page px)


def parity(v: int) -> int:
    return bin(v & 0xFFFFFFFF).count("1") & 1


@dataclass(frozen=True)
class Code:
    draw_ms: int
    receipt_ms: int
    count: int
    rtt_ms: int
    flags: int
    echo: bool | None = None    # decoded only: the echo panel's state (None = not sampled / not in frame)

    @property
    def known_delay_mode(self) -> bool:
        return bool(self.flags & 1)

    def row3(self) -> int:
        return (self.count & 0xFFFF) | ((min(self.rtt_ms, 255) & 0xFF) << 16) | ((self.flags & 0xFF) << 24)


def rows_for(code: Code) -> list[int]:
    flags = (code.flags & 1) | (parity(code.draw_ms) << 1) | (parity(code.receipt_ms) << 2)
    c = Code(code.draw_ms, code.receipt_ms, code.count, code.rtt_ms, flags)
    sync = sum(1 << i for i in range(0, CELLS, 2))   # cell 0 white, 1 black, ...
    return [sync, c.draw_ms & 0xFFFFFFFF, c.receipt_ms & 0xFFFFFFFF, c.row3()]


# ------------------------------------------------------------------------------- Python encoder
def draw_code(img: Image.Image, code: Code, echo_on: bool | None = False, scale: float = 1.0,
              x0: float = X0, y0: float = Y0) -> None:
    """Draw the four code rows (and the echo panel unless echo_on is None) onto an 'L' image in place."""
    d = ImageDraw.Draw(img)
    b = BLOCK * scale
    for r, bits in enumerate(rows_for(code)):
        y = y0 * scale + r * ROW_PITCH * b
        for i in range(CELLS):
            x = x0 * scale + i * b
            d.rectangle([x, y, x + b - 1, y + b - 1], fill=255 if (bits >> i) & 1 else 0)
    if echo_on is not None:
        d.rectangle([x0 * scale, PANEL_Y * scale, x0 * scale + CELLS * b - 1, PANEL_Y * scale + 6 * b],
                    fill=255 if echo_on else 0)


def render_luma(code: Code, w: int = 1920, h: int = 1080, echo_on: bool | None = False, scale: float = 1.0) -> Image.Image:
    """What the page would show, as an 8-bit gray image. `scale` mimics a target at another
    resolution being scaled to the capture size."""
    img = Image.new("L", (w, h), 128)
    draw_code(img, code, echo_on, scale)
    return img


def encode_frame(code: Code, w: int = 1920, h: int = 1080, echo_on: bool = False, quality: int = 85,
                 scale: float = 1.0) -> bytes:
    """The page as a JPEG (the MJPG capture format)."""
    out = io.BytesIO()
    render_luma(code, w, h, echo_on, scale).save(out, format="JPEG", quality=quality)
    return out.getvalue()


def encode_nv12(code: Code, w: int = 1920, h: int = 1080, echo_on: bool = False, scale: float = 1.0) -> bytes:
    """The page as an NV12 buffer (the default capture format): Y plane then neutral chroma."""
    y = render_luma(code, w, h, echo_on, scale).tobytes()
    return y + bytes([128]) * (w * h // 2)


# ------------------------------------------------------------------------------------- decoder
class DecodeError(ValueError):
    pass


def _find_sync(img: Image.Image) -> tuple[float, float, float]:
    """Locate the sync row: returns (x0, y_center, cell) in image px. Scans horizontal scanlines
    for a run of >=CELLS-1 evenly spaced transitions forming the 1010... pattern; the sync row's
    cells give the origin and cell size, so decoding is resolution- and scale-independent."""
    w, h = img.size
    px = img.load()
    for y in range(2, h // 2):
        vals = [px[x, y] for x in range(w)]
        if max(vals) - min(vals) < 100:
            continue
        thr = (max(vals) + min(vals)) / 2
        bits = [v >= thr for v in vals]
        trans = [x for x in range(1, w) if bits[x] != bits[x - 1]]
        # scan runs of consecutive transitions with near-constant spacing
        i = 0
        while i < len(trans) - 1:
            j = i + 1
            gaps = []
            while j < len(trans):
                g = trans[j] - trans[j - 1]
                if gaps and abs(g - gaps[0]) > max(2.0, 0.25 * gaps[0]):
                    break
                gaps.append(g)
                j += 1
            n_trans = j - i          # transitions in this run
            if n_trans >= CELLS - 1 and gaps:
                cell = sum(gaps) / len(gaps)
                if cell < 3:
                    i = j
                    continue
                # transitions run from cell 0/1 boundary onward; x0 is one cell left of the first
                x0 = trans[i] - cell
                # cell 0 (x0 .. x0+cell) must be white (its centre above threshold)
                cx = int(round(x0 + cell / 2))
                if 0 <= cx < w and bits[cx]:
                    # refine y to the band centre
                    yc_probe = int(round(x0 + 0.5 * cell))
                    white = (thr + max(vals)) / 2     # the band is white cells; the gap above/below is gray
                    y_top = y_bot = y
                    while y_bot + 1 < h and px[yc_probe, y_bot + 1] >= white:
                        y_bot += 1
                    while y_top - 1 >= 0 and px[yc_probe, y_top - 1] >= white:
                        y_top -= 1
                    return x0, (y_top + y_bot) / 2, cell
            i = j if j > i + 1 else i + 1
    raise DecodeError("sync row not found")


def decode_image(img: Image.Image) -> Code:
    img = img.convert("L")
    x0, yc, cell = _find_sync(img)
    px = img.load()
    rows = []
    for r in range(ROWS):
        y = int(round(yc + r * ROW_PITCH * cell))
        v = 0
        for i in range(CELLS):
            x = int(round(x0 + (i + 0.5) * cell))
            # sample a 3x3 neighbourhood to survive JPEG ringing
            s = sum(px[min(max(x + dx, 0), img.width - 1), min(max(y + dy, 0), img.height - 1)]
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)) / 9
            if 80 < s < 176:
                raise DecodeError(f"ambiguous cell r{r} c{i}: luma {s:.0f}")
            v |= (1 if s >= 128 else 0) << i
        rows.append(v)
    sync = sum(1 << i for i in range(0, CELLS, 2))
    if rows[0] != sync:
        raise DecodeError("sync row mismatch")
    draw_ms, receipt_ms, r3 = rows[1], rows[2], rows[3]
    flags = (r3 >> 24) & 0xFF
    if ((flags >> 1) & 1) != parity(draw_ms) or ((flags >> 2) & 1) != parity(receipt_ms):
        raise DecodeError("parity mismatch")
    # echo panel: its centre, in cell units from the sync origin (scale-free like the rows)
    ex = int(round(x0 + CELLS * cell / 2))
    ey = int(round(yc + ((PANEL_Y - Y0) / BLOCK + 3.0) * cell))
    echo = None
    if 0 <= ex < img.width and 0 <= ey < img.height:
        s = sum(px[min(max(ex + dx, 0), img.width - 1), min(max(ey + dy, 0), img.height - 1)]
                for dx in (-2, 0, 2) for dy in (-2, 0, 2)) / 9
        echo = None if 80 < s < 176 else s >= 128
    return Code(draw_ms=draw_ms, receipt_ms=receipt_ms, count=r3 & 0xFFFF, rtt_ms=(r3 >> 16) & 0xFF, flags=flags, echo=echo)


def decode_frame(jpeg: bytes, reduce: int = 4) -> Code:
    """Decode a captured MJPEG frame. JPEG draft mode decodes at 1/`reduce` scale (fast); cells
    of 40 px become 10 px, which is plenty."""
    img = Image.open(io.BytesIO(jpeg))
    img.draft("L", (img.width // reduce, img.height // reduce))
    return decode_image(img)


def decode_luma(y, width: int, height: int, reduce: int = 4) -> Code:
    """Decode from a luminance plane alone: `y` is width*height bytes (an NV12 frame's Y plane, or
    any buffer whose first width*height bytes are it) or a 2-D uint8 array. No chroma, no JPEG."""
    if hasattr(y, "tobytes") and hasattr(y, "shape"):
        y = y.tobytes()
    buf = memoryview(y)[: width * height]
    if len(buf) != width * height:
        raise DecodeError(f"luma buffer {len(buf)} bytes < {width}x{height}")
    img = Image.frombuffer("L", (width, height), buf, "raw", "L", 0, 1)
    if reduce > 1:
        img = img.reduce(reduce)
    return decode_image(img)


def decode_capture(frame, reduce: int = 4) -> Code:
    """Decode a capture.Frame of either pixel format."""
    if frame.pixfmt == "NV12":
        return decode_luma(frame.data, frame.width, frame.height, reduce)
    return decode_frame(frame.data, reduce)


# --------------------------------------------------------------------------------------- page
def page_html(known_delay_ms: int = 0) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>chronotope probe</title>
<style>html,body{{margin:0;height:100%;background:#808080;overflow:hidden;cursor:none}}canvas{{display:block}}</style></head>
<body><canvas id="c"></canvas>
<script>
(() => {{
const BLOCK={BLOCK}, CELLS={CELLS}, X0={X0}, Y0={Y0}, PITCH={ROW_PITCH}, PANEL_Y={PANEL_Y}, KNOWN_DELAY_MS={known_delay_ms};
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
let dpr=1;
function resize(){{ dpr=window.devicePixelRatio||1; cv.width=Math.round(innerWidth*dpr); cv.height=Math.round(innerHeight*dpr);
  cv.style.width=innerWidth+'px'; cv.style.height=innerHeight+'px'; }}
addEventListener('resize',resize); resize();
// ---- clock sync (Cristian, min-RTT sample) -----------------------------------------------
let offset=null, rttMin=1e9;
async function sync(){{ try{{ const p0=performance.now(); const r=await fetch('/t',{{cache:'no-store'}}); const txt=await r.text();
  const p1=performance.now(); const rtt=p1-p0; const agentMs=Number(BigInt(txt.trim())/1000000n);
  if(rtt<rttMin){{ rttMin=rtt; offset=agentMs-(p0+p1)/2; }} }}catch(e){{}} }}
sync(); setInterval(sync, 500);
// ---- input echo ---------------------------------------------------------------------------
let count=0, receiptMs=0, echoOn=false;
function onInput(ev){{ const t=ev.timeStamp; // performance.now()-based on modern browsers
  const stamp=()=>{{ if(offset===null) return; receiptMs=Math.round(t+offset); count=(count+1)&0xFFFF; echoOn=!echoOn; }};
  if(KNOWN_DELAY_MS>0) setTimeout(stamp, KNOWN_DELAY_MS); else stamp(); }}
addEventListener('pointerdown', onInput, {{passive:true}});
addEventListener('keydown', ev=>{{ if(!ev.repeat) onInput(ev); }}, {{passive:true}});
function parity(v){{ let p=0; v=v>>>0; while(v){{ p^=v&1; v>>>=1; }} return p; }}
function row(y,bits){{ for(let i=0;i<CELLS;i++){{ ctx.fillStyle=((bits>>>i)&1)?'#fff':'#000'; ctx.fillRect(X0+i*BLOCK,y,BLOCK,BLOCK); }} }}
function draw(){{ ctx.setTransform(dpr,0,0,dpr,0,0); ctx.fillStyle='#808080'; ctx.fillRect(0,0,innerWidth,innerHeight);
  const drawMs = offset===null ? 0 : (Math.round(performance.now()+offset)>>>0);
  const rec=receiptMs>>>0; const rtt=Math.min(255, Math.round(rttMin)); const flags=(KNOWN_DELAY_MS>0?1:0)|(parity(drawMs)<<1)|(parity(rec)<<2);
  const r3=((count&0xFFFF)|((rtt&0xFF)<<16)|((flags&0xFF)<<24))>>>0;
  let sync=0; for(let i=0;i<CELLS;i+=2) sync|=(1<<i); sync>>>=0;
  row(Y0,sync); row(Y0+PITCH*BLOCK,drawMs); row(Y0+2*PITCH*BLOCK,rec); row(Y0+3*PITCH*BLOCK,r3);
  ctx.fillStyle=echoOn?'#fff':'#000'; ctx.fillRect(X0,PANEL_Y,CELLS*BLOCK,6*BLOCK);
  ctx.fillStyle='#000'; ctx.font='16px monospace';
  ctx.fillText(offset===null?'syncing to agent clock…':('agent '+drawMs+' ms  rtt '+rttMin.toFixed(1)+' ms  inputs '+count+(KNOWN_DELAY_MS?'  known delay '+KNOWN_DELAY_MS+' ms':'')), X0, PANEL_Y+7*BLOCK);
  requestAnimationFrame(draw); }}
requestAnimationFrame(draw);
}})();
</script></body></html>"""
