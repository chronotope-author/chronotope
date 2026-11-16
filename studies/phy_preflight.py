"""PHY3 preflight: can this machine run the physical route right now? (cards/PHY3-system-cell.md)

Checks, in order, stopping at the first that cannot proceed:
  1. device access: /dev/video0 readable+writable and the KMBox serial port openable by this user
     (missing -> prints the exact commands to grant access and exits 2)
  2. pixel formats offered by the capture device (NV12 must be among them)
  3. an NV12 stream for --stream-s seconds at --fps: every buffer flagged CLOCK_MONOTONIC, stamps
     strictly increasing, dropped frames (gaps > 1.5 periods) <= 1 % (contract c1, c4)
  4. the serial port opens, the REPL answers km.version()
  5. the probe page is served; the target must fetch it (/t polled) within --target-wait seconds
  6. a --null-s second null test on the decoded probe frames (contract c2; no input is sent)

Exit codes: 0 all passed; 1 a check failed; 2 device access missing (nothing else was attempted).

  PYTHONPATH=src .venv/bin/python studies/phy_preflight.py            # full, hardware needed
  PYTHONPATH=src .venv/bin/python studies/phy_preflight.py --access-only
"""
from __future__ import annotations

import argparse
import getpass
import grp
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.phy import measure as M  # noqa: E402
from chronotope.phy.capture import ClockIdentityError, ElgatoSource, list_formats, python_clock_is_monotonic  # noqa: E402
from chronotope.phy.kmbox import KMBox, PORT_PATTERNS, find_port  # noqa: E402

ACCESS_RC = 2


def say(s: str = "") -> None:
    print(s, flush=True)


def group_of(path: str) -> str:
    try:
        return grp.getgrgid(os.stat(path).st_gid).gr_name
    except (OSError, KeyError):
        return "?"


def access_instructions(missing: list[tuple[str, str]]) -> str:
    """missing: [(path, group)]. The usermod line is printed verbatim (the queue greps for it)."""
    user = getpass.getuser()
    L = ["PHY preflight: device access missing for user %s:" % user]
    for path, group in missing:
        L.append(f"  {path}  (group {group}; this user is not a member and has no ACL entry)")
    L += ["Grant access with ONE of:",
          "  sudo usermod -aG video,dialout $USER",
          "    (then log out and back in, or run `newgrp video` / `newgrp dialout` in the shell that launches the studies)",
          "  " + " && ".join(f"sudo setfacl -m u:{user}:rw {path}" for path, _ in missing),
          "    (immediate, no re-login; lost when the device re-enumerates)"]
    return "\n".join(L)


def check_access(device: str, port: str | None) -> tuple[list[tuple[str, str]], str | None]:
    missing = []
    if not os.path.exists(device):
        say(f"capture device {device} does not exist (Elgato not attached?)")
        missing.append((device, "video"))
    elif not os.access(device, os.R_OK | os.W_OK):
        missing.append((device, group_of(device)))
    try:
        serial_path = find_port(port)
    except FileNotFoundError as e:
        say(str(e))
        serial_path, missing = None, missing + [("/dev/ttyUSB0", "dialout")]
    else:
        real = os.path.realpath(serial_path)
        if not os.access(real, os.R_OK | os.W_OK):
            missing.append((real, group_of(real)))
    return missing, serial_path


def check_stream(device: str, fps: float, seconds: float, width: int, height: int) -> dict:
    src = ElgatoSource(device, width=width, height=height, fps=fps, pixfmt="NV12", timeout_s=5.0)
    period = 1e9 / fps
    n = drops = nonmono = flag_bad = 0
    last = None
    t_end = time.monotonic() + seconds
    res: dict = {"pixfmt": "NV12", "fps_requested": fps}
    try:
        for fr in src.frames():
            n += 1
            if not fr.ts_monotonic:
                flag_bad += 1
            if last is not None:
                gap = fr.ts_kernel_ns - last
                if gap <= 0:
                    nonmono += 1
                elif gap > 1.5 * period:
                    drops += int(round(gap / period)) - 1
            last = fr.ts_kernel_ns
            if time.monotonic() >= t_end:
                break
        res.update({"width": src.width, "height": src.height, "fps_driver": src.fps, "frames": n, "truncated": src.n_truncated,
                    "dropped": drops, "non_monotonic": nonmono, "flag_not_monotonic": flag_bad,
                    "drop_rate": (drops + src.n_truncated) / max(1, n + drops)})
    finally:
        src.close()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--port", default=None, help="serial device; default autodetect " + ", ".join(PORT_PATTERNS))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--stream-s", type=float, default=10.0)
    ap.add_argument("--null-s", type=float, default=60.0)
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--target-wait", type=float, default=120.0, help="seconds to wait for the target to load the page")
    ap.add_argument("--access-only", action="store_true")
    ap.add_argument("--out", default=None, help="write the check results as JSON here")
    a = ap.parse_args()
    results: dict = {}

    # 1. access
    missing, serial_path = check_access(a.device, a.port)
    if missing:
        say(access_instructions(missing))
        return ACCESS_RC
    say(f"access ok: {a.device} and {serial_path} -> {os.path.realpath(serial_path)}")
    results["access"] = {"device": a.device, "serial": serial_path}
    if a.access_only:
        return 0
    if not python_clock_is_monotonic():
        say("FAIL: time.monotonic is not CLOCK_MONOTONIC on this interpreter")
        return 1

    # 2. formats
    fmts = list_formats(a.device)
    say(f"pixel formats: {fmts}")
    results["formats"] = fmts
    if "NV12" not in fmts:
        say("FAIL: NV12 not offered by the capture device")
        return 1

    # 3. stream
    try:
        st = check_stream(a.device, a.fps, a.stream_s, a.width, a.height)
    except (ClockIdentityError, TimeoutError, OSError) as e:
        say(f"FAIL: stream: {type(e).__name__}: {e}")
        return 1
    results["stream"] = st
    say(f"stream: {st}")
    ok = st["frames"] >= 0.9 * a.fps * a.stream_s and st["non_monotonic"] == 0 and st["flag_not_monotonic"] == 0 and st["drop_rate"] <= 0.01
    if not ok:
        say("FAIL: stream check (frames, monotonic flag, ordering, or drop rate > 1 %)")
        return 1

    # 4. serial
    try:
        with KMBox(serial_path) as hid:
            v = hid.version()
        say(f"serial ok: {serial_path} km.version -> {v!r}")
        results["serial"] = {"port": serial_path, "version": v}
    except Exception as e:  # DeviceRejected, SerialException, OSError
        say(f"FAIL: serial: {type(e).__name__}: {e}")
        return 1

    # 5. target page + 6. null test
    httpd = M.serve(a.http_port)
    url = f"http://{M.lan_ip()}:{a.http_port}/"
    say(f"probe page: {url}  (show it full-screen on the target; waiting up to {a.target_wait:.0f} s)")
    out = Path(a.out).parent if a.out else Path("runs/phy_preflight")
    out.mkdir(parents=True, exist_ok=True)
    src = ElgatoSource(a.device, width=a.width, height=a.height, fps=a.fps, pixfmt="NV12", timeout_s=5.0)
    w = M.FrameWatcher(src, out).start()
    s = M.Session(w, None, out, frame_period_ms=1000 / a.fps)
    try:
        try:
            first = s.wait_synced(timeout_s=a.target_wait)
        except TimeoutError:
            say("FAIL: the target never showed a synced probe frame (page not loaded, not full-screen, or HDMI not routed)")
            return 1
        say(f"target page reachable: first synced frame seq {first.seq}, page rtt {first.rtt_ms} ms, E->O {first.eo_ms:.1f} ms")
        results["target"] = {"first_seq": first.seq, "rtt_ms": first.rtt_ms, "eo_ms": first.eo_ms}
        null = s.null_test(a.null_s)
        say(f"null test: {null}")
        results["null_test"] = null
        if null["count_advanced_frames"] != 0:
            say("FAIL: the input count advanced with no input sent (contract c2)")
            return 1
    finally:
        s.close(); w.stop(); httpd.shutdown()
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1))
    say("PHY preflight: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
