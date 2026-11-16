"""PHY instrument — measure E→O and D→A on the physical two-machine path (cards/PHY2 part 2, PHY3 part 2).

Prerequisites on this machine: access to /dev/video0 and the KMBox serial port (studies/phy_preflight.py
prints the commands if it is missing); the Elgato in NV12 at --fps; the KMBox autodetected on
/dev/serial/by-id/*CP2102N* (then the older 1a86 id; --serial overrides). On the target (Windows laptop):
a browser showing http://<this LAN ip>:<port>/ full-screen (F11); --display-profile records the target's
pinned display state (contract c5; F38). `--navigate` types that URL through the KMBox (F57 caveat).

  PYTHONPATH=src .venv/bin/python studies/phy_measure.py --out runs/phy_<date> --name <date> \\
      --display-profile "1920x1080@60 100% scaling, Edge F11" --null-s 600 --events 100 \\
      --known-delay-ms 100 --known-delay-events 30

  PYTHONPATH=src .venv/bin/python studies/phy_measure.py --simulate --out runs/phy_sim      # no hardware

Gates (all must pass before results/PHY_route_<name>.json is written; a failed gate writes
<out>/PHY_route_<name>.REJECTED.json instead and exits 1):
  c1 clock identity verified at runtime (driver monotonic flag + Python CLOCK_MONOTONIC)
  c2 null test: no input sent -> the input count never advances for --null-s seconds
  c3 known-delay test: +--known-delay-ms inserted between D and the write is recovered within one capture frame
  c4 truncated or dropped frames <= 1 % over the session
  c5 the target's display profile is recorded (--display-profile; "simulated" under --simulate)
Output: "eo_ms" and "da_ms" sample lists (known-delay events excluded from da_ms) with per-sample
quantization notes (one capture frame; the page's clock-sync bound), consumed by the runner's
"simulated_empirical" route (results/PHY_route_<name>.json, field "da_ms").
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.phy import measure as M  # noqa: E402
from chronotope.phy.capture import ElgatoSource, FakeSource, python_clock_is_monotonic  # noqa: E402
from chronotope.phy.kmbox import KMBox, find_port  # noqa: E402
from chronotope.phy.probe import Code, render_luma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class SimTarget:
    """A simulated probe target for --simulate: draws frames with true E→O = eo_ms and receives inputs
    da_ms (± da_jitter_ms, seeded) after the serial write."""

    def __init__(self, eo_ms: float = 45.0, da_ms: float = 32.0, rtt_ms: int = 2, da_jitter_ms: float = 8.0, seed: int = 1):
        import random
        self.eo_ns, self.da_ns, self.rtt = int(eo_ms * 1e6), int(da_ms * 1e6), rtt_ms
        self.jit_ns, self.rng = int(da_jitter_ms * 1e6), random.Random(seed)
        self.count, self.receipt_ms = 0, 0
        self._pending: list[int] = []     # receipt times not yet shown (the page learns of a click only at A)

    def on_click(self, t_write_ns: int) -> int:
        da = self.da_ns + (self.rng.randint(-self.jit_ns, self.jit_ns) if self.jit_ns else 0)
        self._pending.append(t_write_ns + da)
        return t_write_ns

    def render(self, ts_kernel_ns: int) -> bytes:
        t_draw = ts_kernel_ns - self.eo_ns
        for t_recv in sorted(self._pending):
            if t_recv <= t_draw:
                self.count = (self.count + 1) & 0xFFFF
                self.receipt_ms = (t_recv // 1_000_000) & 0xFFFFFFFF
                self._pending.remove(t_recv)
        draw_ms = (t_draw // 1_000_000) & 0xFFFFFFFF
        return render_luma(Code(draw_ms, self.receipt_ms, self.count, self.rtt, 0)).tobytes()   # the Y plane


class SimHID:
    def __init__(self, target: SimTarget):
        self.t = target

    def click(self) -> int:
        return self.t.on_click(time.monotonic_ns())

    def key(self, name: str) -> int:
        return self.click()

    def version(self) -> str:
        return "sim"

    def close(self) -> None:
        pass


def dropped_frames(out: Path, frame_period_ms: float) -> tuple[int, int]:
    """(frames seen, frames missing) from the kernel stamps in frames.jsonl: a gap of n periods = n-1 missing."""
    ts = [json.loads(l)["ts_kernel_ns"] for l in (out / "frames.jsonl").read_text().splitlines() if l.strip()]
    missing = 0
    for a, b in zip(ts, ts[1:]):
        gap = (b - a) / 1e6
        if gap > 1.5 * frame_period_ms:
            missing += int(round(gap / frame_period_ms)) - 1
    return len(ts), missing


def route_json(out: Path, res: dict, a, gates: dict, name: str, pixfmt: str, n_truncated: int) -> dict:
    inputs = [json.loads(l) for l in (out / "inputs.jsonl").read_text().splitlines() if l.strip()]
    frame_ms = 1000 / a.fps
    samples = []
    for i in inputs:
        if not i["found"]:
            continue
        samples.append({"da_ms": i["da_ms"], "eo_ms": i["eo_ms"], "rt_ms": i["rt_ms"], "known_delay_ms": i["known_delay_ms"],
                        "quantization_ms": frame_ms, "sync_bound_ms": (i["rtt_ms"] or 0) / 2,
                        "note": f"D->A from the page's receipt stamp (sync bound +-{(i['rtt_ms'] or 0) / 2:.1f} ms); "
                                f"E->O quantized to one capture frame ({frame_ms:.1f} ms)"})
    return {
        "name": name, "date": dt.date.today().isoformat(), "simulated": bool(a.simulate), "pixfmt": pixfmt,
        "fps": a.fps, "frame_period_ms": frame_ms, "display_profile": a.display_profile,
        "gates": gates, "gates_passed": all(g["pass"] for g in gates.values()),
        "eo_ms": [s["eo_ms"] for s in samples if s["known_delay_ms"] == 0 and s["eo_ms"] is not None],
        "da_ms": [s["da_ms"] for s in samples if s["known_delay_ms"] == 0],
        "da_ms_known_delay": {str(k): [s["da_ms"] for s in samples if s["known_delay_ms"] == k]
                              for k in sorted({s["known_delay_ms"] for s in samples} - {0})},
        "samples": samples, "summary": res, "truncated_frames": n_truncated,
        "quantization": {"capture_frame_ms": frame_ms, "note": "E->O and the echo round trip are quantized to one capture "
                         "frame; D->A comes from the page's own input stamp (browser event time + clock-sync offset) and is "
                         "bounded by the sync half-RTT per sample, not by the frame"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="results/PHY_route_<name>.json (default: today's date)")
    ap.add_argument("--port", type=int, default=8765, help="HTTP port of the probe page")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--serial", default=None, help="KMBox serial port (default: autodetect CP2102N, then 1a86)")
    ap.add_argument("--pixfmt", default="NV12", choices=["NV12", "MJPG"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--null-s", type=float, default=600.0)
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--known-delay-ms", type=int, default=100)
    ap.add_argument("--known-delay-events", type=int, default=30)
    ap.add_argument("--gap-s", type=float, default=0.5, help="pause between inputs")
    ap.add_argument("--display-profile", default=None, help="c5: the target's pinned display state (required unless --simulate)")
    ap.add_argument("--navigate", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    if not a.simulate and not a.display_profile:
        ap.error("--display-profile is required (contract c5: pin and record the target's display state)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.name or ("sim" if a.simulate else dt.date.today().isoformat())
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    httpd = None
    if a.simulate:
        a.display_profile = a.display_profile or "simulated"
        tgt = SimTarget()
        src = FakeSource(tgt.render, period_ns=int(1e9 / a.fps), realtime=True, pixfmt="NV12")
        hid = SimHID(tgt)
        a.null_s = min(a.null_s, 3.0)
        a.gap_s = min(a.gap_s, 0.05)
    else:
        httpd = M.serve(a.port)
        url = f"http://{M.lan_ip()}:{a.port}/"
        print(f"probe page: {url}", flush=True)
        src = ElgatoSource(a.device, fps=a.fps, pixfmt=a.pixfmt)
        hid = KMBox(find_port(a.serial)).open()
        print(f"serial: {hid.port_path}  km.version -> {hid.version()!r}", flush=True)
        if a.navigate:
            hid.chord(["ctrl", "l"]); time.sleep(0.4); hid.type_text(url); time.sleep(0.3); hid.key("enter")
            time.sleep(5.0); hid.key("f11"); time.sleep(2.0)
    w = M.FrameWatcher(src, out).start()
    s = M.Session(w, hid, out, frame_period_ms=1000 / a.fps)
    try:
        first = s.wait_synced(timeout_s=60)
        print(f"synced: first decodable frame seq {first.seq}, page rtt {first.rtt_ms} ms, E→O {first.eo_ms:.1f} ms", flush=True)
        if a.null_s > 0:
            print(f"null test for {a.null_s:.0f} s …", flush=True)
            print("null:", s.null_test(a.null_s), flush=True)
        for i in range(a.events):
            r = s.one_input("click")
            print(f"event {i}: found={r.found} D→A {r.da_ms} ms  E→O {r.eo_ms} ms  RT {r.rt_ms} ms", flush=True)
            time.sleep(a.gap_s)
        for i in range(a.known_delay_events):
            r = s.one_input("click", known_delay_ms=a.known_delay_ms)
            print(f"known-delay event {i}: found={r.found} D→A {r.da_ms} ms", flush=True)
            time.sleep(a.gap_s)
    finally:
        s.close(); w.stop(); hid.close()
        if httpd:
            httpd.shutdown()
    res = M.summarize(out, frame_period_ms=1000 / a.fps)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    md = M.summary_markdown(res, f"PHY — {'SIMULATED target' if a.simulate else 'physical path'} ({out.name})")
    (out / "SUMMARY.md").write_text(md)
    print(md)

    # ---- gates c1-c5 ------------------------------------------------------------------------------
    n_frames, n_missing = dropped_frames(out, 1000 / a.fps)
    n_trunc = getattr(src, "n_truncated", 0)
    kd = res.get("known_delay_test", {}).get(str(a.known_delay_ms), {})
    gates = {
        "c1_clock_identity": {"pass": bool(getattr(src, "clock_verified", False)) and python_clock_is_monotonic(),
                              "detail": "driver buffer stamps flagged CLOCK_MONOTONIC and time.monotonic is CLOCK_MONOTONIC"},
        "c2_null_test": {"pass": res.get("null_test", {}).get("count_advanced_frames", 1) == 0 and a.null_s > 0,
                         "detail": res.get("null_test")},
        "c3_known_delay": {"pass": bool(kd.get("pass")), "detail": kd},
        "c4_drop_rate": {"pass": (n_missing + n_trunc) / max(1, n_frames + n_missing) <= 0.01,
                         "detail": {"frames": n_frames, "missing": n_missing, "truncated": n_trunc}},
        "c5_display_pinned": {"pass": bool(a.display_profile), "detail": a.display_profile},
        "n_samples": {"pass": res["D_to_A_ms"].get("0", {}).get("n", 0) >= (a.events if not a.simulate else 1),
                      "detail": {"da_n": res["D_to_A_ms"].get("0", {}).get("n", 0), "eo_n": res["E_to_O_ms"].get("n", 0)}},
    }
    rj = route_json(out, res, a, gates, name, getattr(src, "pixfmt", "NV12"), n_trunc)
    for k, g in gates.items():
        print(f"gate {k}: {'PASS' if g['pass'] else 'FAIL'}  {g['detail']}", flush=True)
    if rj["gates_passed"]:
        dest = (out / f"PHY_route_{name}.json") if a.simulate else (ROOT / "results" / f"PHY_route_{name}.json")
        dest.write_text(json.dumps(rj, indent=1))
        print(f"wrote {dest}  (da_ms n={len(rj['da_ms'])}, eo_ms n={len(rj['eo_ms'])})", flush=True)
        return 0
    dest = out / f"PHY_route_{name}.REJECTED.json"
    dest.write_text(json.dumps(rj, indent=1))
    print(f"gates failed; wrote {dest}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
