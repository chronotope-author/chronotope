"""Minimal 8-bit grayscale PNG encoder with integer nearest-neighbour upscaling (stdlib only)."""
from __future__ import annotations

import struct
import zlib


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def is_png(payload: bytes) -> bool:
    return payload[:8] == PNG_SIGNATURE


def decode_png_rgb(payload: bytes, w: int, h: int) -> bytes:
    """A PNG payload (live-page screenshots, cards/G2-live-page.md) as row-major RGB bytes of the declared size."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(payload)).convert("RGB")
    if im.size != (w, h):
        raise ValueError(f"PNG payload is {im.size}, expected {(w, h)}")
    return im.tobytes()


def to_rgb(payload: bytes, w: int, h: int) -> bytes:
    """Row-major RGB or grey bytes for any payload the harness stores (raw frames or PNG screenshots)."""
    return decode_png_rgb(payload, w, h) if is_png(payload) else payload


def encode_png(payload: bytes, w: int, h: int, scale: int) -> bytes:
    """8-bit grayscale (w*h bytes) or RGB (w*h*3 bytes), nearest-neighbour upscaled by `scale`."""
    if is_png(payload):
        if scale == 1:
            return payload          # a screenshot is already the PNG the model sees
        payload = decode_png_rgb(payload, w, h)
    if scale < 1 or len(payload) not in (w * h, w * h * 3):
        raise ValueError("payload size or scale invalid")
    ch = len(payload) // (w * h)
    raw = bytearray()
    for y in range(h):
        row = payload[y * w * ch:(y + 1) * w * ch]
        up = b"".join(row[i * ch:(i + 1) * ch] * scale for i in range(w))
        line = b"\x00" + up  # filter type 0
        for _ in range(scale):
            raw += line
    ihdr = struct.pack(">IIBBBBB", w * scale, h * scale, 8, 0 if ch == 1 else 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b""))


encode_gray_png = encode_png  # backward-compatible name
