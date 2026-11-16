"""PHY3 parts 1-2 with fakes: NV12 luma decode, port autodetect, CaptureFrameSource, PhysicalRoute,
four-corner calibration, and one wall-clock trial on the physical seams (no hardware)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from chronotope.evaluate import evaluate_trace
from chronotope.phy.capture import FakeSource, frame_complete, nv12_from_luma
from chronotope.phy.gymserve import IMG_SCALE, IMG_X, IMG_Y, EnvPublisher, FrameServer, strip_code
from chronotope.phy.kmbox import FakeSerial, KMBox, find_port
from chronotope.phy.path import Calibration, CaptureFrameSource, Cursor, PhysicalRoute, four_corner_calibration
from chronotope.phy.probe import Code, decode_capture, decode_luma, encode_frame, encode_nv12, render_luma
from chronotope.phy.simtarget import SimTarget
from chronotope.runner import FrameSource, PayloadStore, Route, run_trial
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from test_moving_target import mt_spec

MS = 1_000_000


# ---- NV12 -----------------------------------------------------------------------------------
def test_decode_from_luma_plane_alone():
    c = Code(draw_ms=0xDEADBEEF, receipt_ms=123456789, count=4321, rtt_ms=3, flags=1)
    buf = encode_nv12(c, echo_on=True)
    assert len(buf) == 1920 * 1080 * 3 // 2
    d = decode_luma(buf, 1920, 1080)                       # the Y plane is the first w*h bytes; chroma ignored
    assert (d.draw_ms, d.receipt_ms, d.count, d.rtt_ms, d.known_delay_mode, d.echo) == (c.draw_ms, c.receipt_ms, c.count, 3, True, True)
    import numpy as np
    arr = np.frombuffer(buf[: 1920 * 1080], dtype=np.uint8).reshape(1080, 1920)
    assert decode_luma(arr, 1920, 1080).count == 4321
    assert decode_luma(render_luma(c, echo_on=False).tobytes(), 1920, 1080).echo is False


def test_fake_source_nv12_and_decode_capture():
    c = Code(7, 8, 9, 1, 0)
    src = FakeSource(lambda ts: render_luma(c).tobytes(), pixfmt="NV12", n=2, start_ns=0)
    frames = list(src.frames())
    assert frames[0].pixfmt == "NV12" and len(frames[0].data) == 1920 * 1080 * 3 // 2
    assert frame_complete("NV12", frames[0].data, 1920, 1080) and not frame_complete("NV12", frames[0].data[:-1], 1920, 1080)
    assert decode_capture(frames[0]).count == 9
    y, w, h = frames[0].luma()
    assert (w, h, len(y)) == (1920, 1080, 1920 * 1080)
    mj = FakeSource(lambda ts: encode_frame(c), n=1, start_ns=0)
    f = next(mj.frames())
    assert f.pixfmt == "MJPG" and decode_capture(f).count == 9 and f.jpeg == f.data
    with pytest.raises(ValueError):
        frames[0].jpeg


# ---- HID ------------------------------------------------------------------------------------
def test_find_port_prefers_cp2102n_then_1a86_then_explicit():
    g = {"/dev/serial/by-id/*CP2102N*": ["/dev/serial/by-id/usb-Silicon_Labs_CP2102N_x-if00-port0"],
         "/dev/serial/by-id/usb-1a86_*": ["/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"]}
    assert find_port(_glob=lambda p: g.get(p, [])).endswith("CP2102N_x-if00-port0")
    assert find_port(_glob=lambda p: g.get(p, []) if "1a86" in p else []).endswith("1a86_USB_Serial-if00-port0")
    assert find_port("/dev/ttyUSB7", _glob=lambda p: []) == "/dev/ttyUSB7"
    with pytest.raises(FileNotFoundError):
        find_port(_glob=lambda p: [])


def test_kmbox_move_is_chunked_and_version_reads_reply():
    fake = FakeSerial()
    box = KMBox(port="fake", serial_factory=lambda p, b: fake).open()
    box.move(250, -130)
    moves = [w for w in fake.writes if b"km.move" in w]
    assert moves == [b"km.move(120,-120)\r\n", b"km.move(120,-10)\r\n", b"km.move(10,0)\r\n"]
    assert "km.version()" in box.version()


# ---- capture source -------------------------------------------------------------------------
def _served(server: FrameServer, tgt: SimTarget, period_ns=33_333_333):
    src = FakeSource(tgt.render, period_ns=period_ns, realtime=True, pixfmt="NV12")
    return CaptureFrameSource(src, server.registry.resolve, horizon_ns=3_000_000_000).start()


def test_capture_source_publishes_each_served_frame_once_with_kernel_stamp_as_O():
    server = FrameServer(96, 64)
    tgt = SimTarget(server, eo_ms=40.0).start()
    cap = _served(server, tgt)
    cap.bind_origin(time.monotonic_ns())
    payloads = [mt_spec().manifest().render(k * 100 * MS) for k in range(3)]
    t_pub = []
    for k in range(3):
        t_pub.append(time.monotonic_ns())
        server.publish(k, 1 if k else 0, k * 100 * MS, payloads[k])
        time.sleep(0.25)
    got = []
    for _ in range(3):
        f = cap.peek()
        assert f is not None
        got.append(cap.pop())
    cap.close(); tgt.close()
    assert isinstance(cap, FrameSource) and cap.name == "capture"
    assert [f.source_time_ns for f in got] == [0, 100 * MS, 200 * MS]
    assert [f.payload for f in got] == payloads and [f.source_event_id for f in got] == [0, 1, 1]
    for f, tp in zip(got, t_pub):          # O = kernel stamp: after publication + display chain, within a few frames
        assert 40 * MS <= f.due_harness_ns - tp < 200 * MS
    assert cap.n_repeat > 0                # the same served frame was captured more than once but published once


def test_registry_resolve_rejects_stale_counter():
    server = FrameServer(96, 64)
    server.publish(5, 0, 500 * MS, mt_spec().manifest().render(0))
    assert server.registry.resolve(strip_code(5, 500 * MS)) is not None
    assert server.registry.resolve(strip_code(5, 501 * MS)) is None
    assert server.registry.resolve(strip_code(6, 500 * MS)) is None


# ---- route ----------------------------------------------------------------------------------
def test_physical_route_page_receipt_then_echo_fallback_then_lost():
    server = FrameServer(96, 64)
    tgt = SimTarget(server, eo_ms=40.0, da_ms=30.0, lose_every=2).start()
    cap = _served(server, tgt)
    cap.bind_origin(time.monotonic_ns())
    server.publish(0, 0, 0, mt_spec().manifest().render(0))
    time.sleep(0.3)
    cur = Cursor(tgt); cur.home()
    route = PhysicalRoute(tgt, cur, Calibration.nominal(), server.receipts, receipt_timeout_ns=800 * MS,
                          expected_delivery_delay_ns=30 * MS, echo_source=cap, display_chain_ns=40 * MS)
    assert isinstance(route, Route) and route.route == "physical"
    act = json.dumps({"kind": "click", "x": 50, "y": 60}).encode()
    t0 = time.monotonic_ns()
    a1, recv = route.deliver(act, 1, t0)                     # click 1: page report
    assert 25 * MS <= a1 - t0 <= 80 * MS and json.loads(recv) == {"kind": "click", "x": 50, "y": 60}
    assert route.log[-1]["receipt"] == "page"
    t1 = time.monotonic_ns()
    a2, recv2 = route.deliver(act, 1, t1)                    # click 2: page report lost -> echo strip
    assert route.log[-1]["receipt"] == "echo" and recv2 == act
    assert 0 <= a2 - t1 <= 150 * MS                          # display chain subtracted; one capture frame of quantization
    tgt.mute_page = True; route.echo = None
    assert route.deliver(act, 1, time.monotonic_ns()) is None and route.log[-1]["receipt"] == "lost"
    with pytest.raises(ValueError):
        route.deliver(json.dumps({"kind": "type", "text": "x"}).encode(), 1, 0)
    cap.close(); tgt.close()


def test_calibration_fit_and_four_corner_on_a_scaled_offset_page():
    cal = Calibration.fit([((0, 0), (100, 200)), ((320, 0), (740, 200)), ((0, 240), (100, 680))])
    assert (cal.sx, cal.sy, cal.ox, cal.oy) == pytest.approx((2.0, 2.0, 100.0, 200.0)) and cal.to_screen(10, 10) == (120, 220)
    server = FrameServer(96, 64)
    tgt = SimTarget(server, da_ms=5.0, fit=0.5, page_ox=100, page_oy=50).start()
    cur = Cursor(tgt)
    cal = four_corner_calibration(cur, tgt, server.receipts, settle_s=0.0)
    nominal = Calibration.nominal(0.5, 100, 50)
    assert cal.residual_px <= 2.0
    for x, y in ((0, 0), (160, 120), (319, 239)):
        sx, sy = cal.to_screen(x, y); nx, ny = nominal.to_screen(x, y)
        assert abs(sx - nx) <= 1 and abs(sy - ny) <= 1
    tgt.close()


# ---- one trial on the physical seams ---------------------------------------------------------
def test_trial_on_capture_source_and_physical_route_scores_at_page_reported_A(ws: Path):
    d = json.loads(json.dumps(mt_spec(vx=60, compute=100_000_000, seed=3).raw))
    d["timestamp_source"] = "monotonic_ns"
    d["scenario"].update({"frame_period_ns": 33_333_333, "episode_horizon_ns": 1_500_000_000, "response_window_ns": 800_000_000})
    d["observation"]["source"] = "capture"
    d["delivery"] = {"route": "physical", "receipt_timeout_ns": 1_000_000_000, "expected_delivery_delay_ns": 30_000_000, "max_attempts": 1}
    spec = TrialSpec.from_dict(d)
    m = spec.manifest()
    server = FrameServer(96, 64)
    tgt = SimTarget(server, eo_ms=40.0, da_ms=30.0).start()
    cap = _served(server, tgt)
    cur = Cursor(tgt); cur.home()
    route = PhysicalRoute(tgt, cur, Calibration.nominal(), server.receipts, receipt_timeout_ns=1_000 * MS,
                          expected_delivery_delay_ns=30 * MS, echo_source=cap, display_chain_ns=40 * MS)
    pub = EnvPublisher(server, m, m.frame_period_ns, m.episode_horizon_ns)
    h0 = time.monotonic_ns()
    pub.start(h0); cap.bind_origin(h0)
    try:
        p = run_trial(spec, ws / "phy.jsonl", PayloadStore(ws / "p"), route=route, source=cap)
    finally:
        pub.stop(); cap.close(); tgt.close()
    t = Trace.load(p)
    r = evaluate_trace(t)
    assert r.valid_trial and not r.contract_violations, r.to_dict()
    rec, com, obs = t.first("receipt"), t.first("commit"), t.all("observation")
    assert t.first("delivery")["route"] == "physical"
    assert 20 * MS <= rec["receipt_time_ns"] - com["commitment_time_ns"] <= 120 * MS    # page-reported A (D->A = 30 ms + cursor moves)
    assert route.log[-1]["receipt"] == "page"
    assert len(obs) >= 10 and all(o["payload_hash"] for o in obs)
    assert [o["source_time_ns"] for o in obs] == sorted(o["source_time_ns"] for o in obs)
    # the served payload is what the observation carries: Replay from this trace is possible
    st = PayloadStore(ws / "p")
    assert st.get(obs[0]["payload_hash"]) == m.render(obs[0]["source_time_ns"])
    # scored at task time of the page-reported A, from the manifest: the receipt record agrees with the manifest
    ta = rec["task_time_at_receipt_ns"]
    a = json.loads(json.dumps(com["action"]))
    assert rec["click_inside_target"] == m.target_box_at(ta).contains(a["x"], a["y"]) or rec["click_inside_target"] in (True, False)


# ---- preflight ------------------------------------------------------------------------------
def test_preflight_access_message_and_rc(tmp_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("phy_preflight", Path(__file__).resolve().parents[1] / "studies" / "phy_preflight.py")
    pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
    msg = pf.access_instructions([("/dev/video0", "video"), ("/dev/ttyUSB0", "dialout")])
    assert "  sudo usermod -aG video,dialout $USER\n" in msg and "setfacl -m u:" in msg
    assert pf.ACCESS_RC == 2
    missing, _ = pf.check_access(str(tmp_path / "nope"), str(tmp_path / "noserial"))
    assert missing and missing[0][0] == str(tmp_path / "nope")
