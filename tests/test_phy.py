"""PHY contract tests without hardware (cards/PHY-physical-path.md)."""
from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from PIL import Image

from chronotope.phy import measure as M
from chronotope.phy.capture import FakeSource
from chronotope.phy.kmbox import DeviceRejected, FakeSerial, KMBox
from chronotope.phy.probe import Code, DecodeError, decode_frame, decode_image, encode_frame, page_html


def test_encode_decode_roundtrip_through_jpeg():
    c = Code(draw_ms=0xDEADBEEF, receipt_ms=123456789, count=4321, rtt_ms=3, flags=1)
    d = decode_frame(encode_frame(c, quality=70))
    assert (d.draw_ms, d.receipt_ms, d.count, d.rtt_ms, d.known_delay_mode) == (c.draw_ms, c.receipt_ms, c.count, c.rtt_ms, True)


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.0])
def test_decoder_is_scale_free(scale):
    """A target at another resolution scaled to the capture size still decodes."""
    c = Code(1_000_000, 999_999, 7, 1, 0)
    d = decode_frame(encode_frame(c, scale=scale), reduce=2)
    assert (d.draw_ms, d.receipt_ms, d.count) == (c.draw_ms, c.receipt_ms, c.count)


def test_decoder_rejects_corrupted_parity():
    c = Code(5, 6, 1, 1, 0)
    img = Image.open(io.BytesIO(encode_frame(c))).convert("L")
    # flip one data cell of row 1 (draw_ms bit 3)
    from chronotope.phy.probe import BLOCK, X0, Y0, ROW_PITCH
    x, y = X0 + 3 * BLOCK, int(Y0 + ROW_PITCH * BLOCK)
    for i in range(BLOCK):
        for j in range(BLOCK):
            img.putpixel((x + i, y + j), 255 - img.getpixel((x + i, y + j)))
    with pytest.raises(DecodeError):
        decode_image(img)


def test_decoder_fails_loud_without_page():
    img = Image.new("L", (640, 360), 128)
    with pytest.raises(DecodeError):
        decode_image(img)


def test_page_html_embeds_layout_and_known_delay():
    h = page_html(100)
    assert "KNOWN_DELAY_MS=100" in h and "BLOCK=40" in h and "fetch('/t'" in h


def test_kmbox_echo_verification_and_write_stamp():
    fake = FakeSerial()
    box = KMBox(port="fake", serial_factory=lambda p, b: fake).open()
    t0 = time.monotonic_ns()
    t = box.click()
    assert t0 <= t <= time.monotonic_ns()
    assert any(b"kmc(km.left,20)" in w for w in fake.writes)
    assert fake.writes[0] == b"\x03"      # REPL interrupted on open
    box.close()


def test_kmbox_raises_on_rejected_command():
    fake = FakeSerial(reject={"kmk"})
    box = KMBox(port="fake", serial_factory=lambda p, b: fake).open()
    with pytest.raises(DeviceRejected):
        box.key("space")


class _Target:
    def __init__(self, eo_ms=40.0, da_ms=30.0):
        self.eo_ns, self.da_ns = int(eo_ms * 1e6), int(da_ms * 1e6)
        self.count, self.receipt_ms = 0, 0

    def click(self):
        t = time.monotonic_ns()
        self.count += 1
        self.receipt_ms = ((t + self.da_ns) // 1_000_000) & 0xFFFFFFFF
        return t

    def key(self, name):
        return self.click()

    def render(self, ts):
        return encode_frame(Code(((ts - self.eo_ns) // 1_000_000) & 0xFFFFFFFF, self.receipt_ms, self.count, 2, 0), quality=75)


def test_session_recovers_true_delays_and_known_delay(tmp_path: Path):
    tgt = _Target(eo_ms=40.0, da_ms=30.0)
    src = FakeSource(tgt.render, period_ns=16_666_667, realtime=True)
    w = M.FrameWatcher(src, tmp_path).start()
    s = M.Session(w, tgt, tmp_path)
    s.wait_synced(5)
    null = s.null_test(0.5)
    assert null["count_advanced_frames"] == 0
    recs = [s.one_input("click") for _ in range(5)] + [s.one_input("click", known_delay_ms=100) for _ in range(5)]
    s.close(); w.stop()
    assert all(r.found for r in recs)
    res = M.summarize(tmp_path)
    assert abs(res["E_to_O_ms"]["median"] - 40.0) <= 2.0
    assert abs(res["D_to_A_ms"]["0"]["median"] - 30.0) <= 2.0
    kd = res["known_delay_test"]["100"]
    assert kd["pass"], kd
    assert res["null_test"]["count_advanced_frames"] == 0
    md = M.summary_markdown(res, "t")
    assert "PASS" in md and "D→A" in md


def test_unwrap_handles_wraparound():
    assert M._unwrap_ms(5, (1 << 32) - 5) == 10.0
    assert M._unwrap_ms((1 << 32) - 5, 5) == -10.0


def test_frame_watcher_stop_waits_before_closing_records(tmp_path: Path):
    c = Code(draw_ms=1, receipt_ms=0, count=0, rtt_ms=1, flags=0)
    source = FakeSource(lambda ts: encode_frame(c), period_ns=3_000_000, realtime=True, n=10)
    watcher = M.FrameWatcher(source, tmp_path).start()
    watcher.stop()
    assert not watcher._thread.is_alive()
    assert watcher._f.closed
    watcher.stop()  # stopping an already stopped source is harmless
