"""PHY3 system cells: a gym scenario on the physical route end to end (cards/PHY3-system-cell.md part 3).

This machine serves the scenario's frames (frame-counter strip added server-side) to the target's browser
over LAN, captures the target's screen back (CaptureFrameSource: O = the capture card's kernel stamp,
source time from the decoded counter, payload = the served frame), runs the controller, delivers clicks
by HID at target screen coordinates (four-corner calibration at session start), and scores at the
page-reported A (PhysicalRoute; echo strip as the fallback receipt; lost after --receipt-timeout-ms).

Cells
  b1   jitter validation (H-P2): live trading, scripted oracle, 100 trials on the physical route. The
       reversal is placed at the expected receipt (E->O median + compute + D->A median from the instrument's
       JSON), so trials split by their own measured receipt time; the same seeds run on the software route
       with the measured D->A replayed (delivery.route = simulated_empirical) for the trial-level comparison.
  c1   balloon drift 8 px/s (H-P4), {frozen, late_bind}: --model runs the VLM late-binding controller
       (Qwen3.8-27B, --no-thinking, prompt 5 as in C1); without --model the scripted oracle stands in
       (frozen = state_at_B, late_bind = newest_at_D). Reports (SR, mean G), the error law
       |error| vs v (t_A - tau), and the page/echo concordance of A.

  PYTHONPATH=src .venv/bin/python studies/phy_gym_cell.py --cell b1 --simulate --n-seeds 10 --out runs/phy_sim/cell_b1
  PYTHONPATH=src .venv/bin/python studies/phy_gym_cell.py --cell c1 --simulate --n-seeds 10 --out runs/phy_sim/cell_c1
  PYTHONPATH=src .venv/bin/python studies/phy_gym_cell.py --cell b1 --route-json results/PHY_route_<date>.json \\
      --display-profile "..." --out runs/phy3_b1_<date> --n-seeds 100
  ... --cell c1 --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 --no-thinking --policy late_bind

--simulate replaces the target with chronotope.phy.simtarget.SimTarget (display chain eo_ms, page receipt
da_ms +- jitter, page reports over HTTP to the real frame server) and the capture card with an NV12 FakeSource.
Live only: the physical route has no paused/replay policy (Replay from these traces stays available).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.phy.capture import ElgatoSource, FakeSource  # noqa: E402
from chronotope.phy.gymserve import EnvPublisher, FrameServer  # noqa: E402
from chronotope.phy.kmbox import KMBox, find_port  # noqa: E402
from chronotope.phy.path import Calibration, CaptureFrameSource, Cursor, PhysicalRoute, four_corner_calibration  # noqa: E402
from chronotope.phy.simtarget import SimTarget  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402
from conftest import base_spec_dict  # noqa: E402

MS = 1_000_000
lines: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    lines.append(s)


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------------- specs
B1 = dict(E=500 * MS, WINDOW=2_000 * MS, PERIOD=100 * MS, COMPUTE=200 * MS, U=(1.0, -1.0, 0.0))
C1 = dict(E=500 * MS, WINDOW=16_000 * MS, PERIOD=33_333_333, VX=8)


def b1_spec(seed: int, reversal_ns: int, route: dict, source: str, pub_ns: int) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "live_trading", "scenario_seed": seed, "canvas_w_px": 128, "canvas_h_px": 64,
                     "target_size_px": 6, "event_time_ns": B1["E"], "response_window_ns": B1["WINDOW"],
                     "frame_period_ns": B1["PERIOD"], "episode_horizon_ns": B1["E"] + B1["WINDOW"] + 500 * MS,
                     "irrelevant_motion": False, "skin": "chart", "reversal_delay_ns": reversal_ns,
                     "u_valid_order": B1["U"][0], "u_invalid_order": B1["U"][1], "u_no_order": B1["U"][2]}
    d["controller"]["compute_ns"] = B1["COMPUTE"]
    d["observation"] = {"boundary_version": 1, "publication_delay_ns": pub_ns, "source": source}
    d["timestamp_source"] = "monotonic_ns" if source == "capture" else "simulated"
    d["delivery"] = route
    return TrialSpec.from_dict(d)


def c1_spec(a, seed: int, policy: str, route: dict, pub_ns: int) -> TrialSpec:
    d = base_spec_dict()
    d["scenario"] = {"scenario_id": "moving_target", "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240,
                     "target_size_px": 24, "event_time_ns": C1["E"], "response_window_ns": C1["WINDOW"],
                     "frame_period_ns": C1["PERIOD"], "episode_horizon_ns": C1["E"] + C1["WINDOW"] + 500 * MS,
                     "irrelevant_motion": False, "skin": "balloon", "velocity_x_px_per_s": C1["VX"], "velocity_y_px_per_s": 0}
    if a.model:
        d["controller"] = {"kind": "vlm_late_binding", "server": "openai_compatible_local", "base_url": a.base_url,
                           "model_id": a.model, "prompt_version": a.prompt_version, "image_scale": 2, "max_tokens": a.max_tokens,
                           "temperature": 0.0, "enable_thinking": not a.no_thinking, "reasoning_effort": None,
                           "invoke_on": "frame_change", "at_commit_policy": policy, "template_radius_px": 12,
                           "min_match_score": 0.8, "restart_threshold_px": None}
    else:
        d["controller"].update({"compute_ns": a.oracle_compute_ms * MS, "binding": {"frozen": "state_at_B", "late_bind": "newest_at_D"}[policy]})
    d["observation"] = {"boundary_version": 1, "publication_delay_ns": pub_ns, "source": "capture"}
    d["timestamp_source"] = "monotonic_ns"
    d["delivery"] = route
    return TrialSpec.from_dict(d)


# ---------------------------------------------------------------------------------- rig
class Rig:
    """Frame server + capture source + HID + calibration, shared by every trial of the session."""

    def __init__(self, a, canvas_w: int, canvas_h: int, eo_ns: int, da_ns: int) -> None:
        self.a = a
        self.server = FrameServer(canvas_w, canvas_h, port=a.http_port).start()
        self.eo_ns, self.da_ns = eo_ns, da_ns
        if a.simulate:
            self.tgt = SimTarget(self.server, eo_ms=eo_ns / MS, da_ms=da_ns / MS, da_jitter_ms=a.sim_da_jitter_ms,
                                 post_url=f"http://127.0.0.1:{self.server.port}", seed=a.seed_start).start()
            self.raw = FakeSource(self.tgt.render, period_ns=int(1e9 / a.fps), realtime=True, pixfmt="NV12")
            self.hid = self.tgt
            say(f"simulated target: E->O {eo_ns / MS:.0f} ms, D->A {da_ns / MS:.0f} +- {a.sim_da_jitter_ms:.0f} ms; page reports over HTTP")
        else:
            self.raw = ElgatoSource(a.device, fps=a.fps, pixfmt="NV12")
            self.hid = KMBox(find_port(a.serial)).open()
            say(f"serial {self.hid.port_path} km.version -> {self.hid.version()!r}")
            say(f"target page: {self.server.url()}   show it full-screen on the target (F11); display profile: {a.display_profile}")
        self.publisher: EnvPublisher | None = None
        self.cap = CaptureFrameSource(self.raw, self.server.registry.resolve, horizon_ns=1_000 * MS,
                                      on_origin=lambda h0: self.publisher.start(h0)).start()
        self.cursor = Cursor(self.hid, a.screen_w, a.screen_h)
        self.cursor.home()
        # a blank frame so the page has something to show while calibrating
        self.server.publish(0, 0, 0, bytes(canvas_w * canvas_h * 3))
        self.cal = self.calibrate()
        self.route = PhysicalRoute(self.hid, self.cursor, self.cal, self.server.receipts,
                                   receipt_timeout_ns=a.receipt_timeout_ms * MS, expected_delivery_delay_ns=da_ns,
                                   echo_source=self.cap, display_chain_ns=eo_ns, settle_s=a.settle_ms / 1000,
                                   echo_check_s=0.3)

    def calibrate(self) -> Calibration:
        for i in range(5):
            try:
                cal = four_corner_calibration(self.cursor, self.hid, self.server.receipts, settle_s=self.a.settle_ms / 1000)
                say(f"calibration: scenario->screen sx {cal.sx:.3f} sy {cal.sy:.3f} ox {cal.ox:.1f} oy {cal.oy:.1f} residual {cal.residual_px:.2f} px")
                return cal
            except TimeoutError as e:
                say(f"calibration attempt {i + 1}: {e}; waiting for the target page")
                time.sleep(10.0 if not self.a.simulate else 0.5)
        raise SystemExit("calibration failed: the target page is not reporting clicks")

    def run(self, spec: TrialSpec, out: Path, name: str, controller=None) -> Path:
        m = spec.manifest()
        self.server.registry.clear()
        self.cap.reset(horizon_ns=m.episode_horizon_ns)
        self.publisher = EnvPublisher(self.server, m, m.frame_period_ns, m.episode_horizon_ns)
        self.route.log.clear()
        try:
            return run_trial(spec, out / f"{name}.jsonl", PayloadStore(out / "payloads"), route=self.route,
                             source=self.cap, controller=controller)
        finally:
            self.publisher.stop()

    def close(self) -> None:
        self.cap.close(); self.server.stop()
        try:
            self.hid.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------------- per-trial facts
def trial_facts(p: Path, m, route_log: list[dict]) -> dict:
    t = Trace.load(p)
    r = evaluate_trace(t)
    com, rec = t.first("commit"), t.first("receipt")
    corr = t.first("coordinate_correction")
    dec = t.first("decision_begin")
    obs = {o["observation_id"]: o for o in t.all("observation")}
    f = {"valid_trial": r.valid_trial, "success": bool(r.valid_at_receipt) if r.valid_trial else None, "G": r.return_G,
         "error_px": r.spatial_error_px, "end": t.first("trial_end")["reason"], "n_obs": len(obs),
         "labels": list(r.labels), "violations": list(r.contract_violations)}
    if com and rec:
        f["d_ms"] = com["task_time_at_commit_ns"] / MS
        f["a_ms"] = rec["task_time_at_receipt_ns"] / MS
        f["da_ms"] = (rec["receipt_time_ns"] - com["commitment_time_ns"]) / MS
        src_obs = obs.get(corr["observation_id"]) if corr and corr.get("observation_id") else (obs.get(dec["observation_id"]) if dec else None)
        if src_obs is not None:
            f["tau_ms"] = src_obs["source_time_ns"] / MS
            f["eo_ms"] = (src_obs["publication_time_ns"] - t.header()["harness_origin_ns"] - src_obs["source_time_ns"]) / MS
    last = route_log[-1] if route_log else {}
    f["receipt_via"] = last.get("receipt")
    if last.get("a_ns") is not None and last.get("a_echo_ns") is not None:
        f["page_echo_diff_ms"] = (last["a_echo_ns"] - last["a_ns"]) / MS
    f["lost"] = f["end"] == "delivery_exhausted"
    return f


def sr_g(fs: list[dict]) -> tuple[str, str]:
    v = [f for f in fs if f["valid_trial"]]
    sr = f"{sum(1 for f in v if f['success'])}/{len(v)}"
    g = statistics.mean(f["G"] for f in v) if v else None
    return sr, fmt(g, 2)


# ---------------------------------------------------------------------------------- cells
def cell_b1(a, out: Path, eo_ns: int, da_ns: int, route_json: Path) -> None:
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    half_frame = int(1e9 / a.fps / 2)                       # capture quantization: a served frame is seen half a capture frame late on average
    settle = a.settle_ms * MS
    d_task = B1["E"] + eo_ns + half_frame + B1["COMPUTE"]   # expected commitment on the physical route
    reversal = d_task + settle + da_ns - B1["E"]            # reversal at the expected receipt: trials split by their own A
    say(f"# PHY3 B1 — live trading on the physical route ({'SIMULATED target' if a.simulate else 'hardware'}), {len(seeds)} trials")
    say(f"reversal at E + {reversal / MS:.0f} ms = expected A (E->O median {eo_ns / MS:.0f} + half a capture frame {half_frame / MS:.0f} + compute "
        f"{B1['COMPUTE'] / MS:.0f} + settle {a.settle_ms} + D->A median {da_ns / MS:.0f} ms); "
        f"utilities (valid, invalid, none) = {B1['U']}; seeds {seeds[0]}..{seeds[-1]}; 10 Hz frames; capture {a.fps:.0f} Hz.")
    phys_route = {"route": "physical", "receipt_timeout_ns": a.receipt_timeout_ms * MS, "expected_delivery_delay_ns": da_ns, "max_attempts": 1}
    rig = Rig(a, 128, 64, eo_ns, da_ns)
    facts: dict[int, dict] = {}
    try:
        for s in seeds:
            spec = b1_spec(s, reversal, phys_route, "capture", eo_ns)
            p = rig.run(spec, out, f"b1_phys_s{s}")
            facts[s] = trial_facts(p, spec.manifest(), rig.route.log)
            f = facts[s]
            say(f"seed {s}: end={f['end']} success={f['success']} G={f['G']} D={fmt(f.get('d_ms'))} A={fmt(f.get('a_ms'))} "
                f"D->A={fmt(f.get('da_ms'))} ms via {f['receipt_via']} page-echo {fmt(f.get('page_echo_diff_ms'))} ms obs={f['n_obs']}")
    finally:
        rig.close()
    # software route with the measured distribution replayed (same seeds; CPU, simulated clock)
    # Its observation delay carries the physical path's pre-write constants (E->O median + half a capture frame +
    # settle), so that only the drawn D->A differs between the two rows.
    emp = {"route": "simulated_empirical", "route_name": str(route_json), "delivery_seed": 0, "max_attempts": 1}
    rep: dict[int, dict] = {}
    for s in seeds:
        emp["delivery_seed"] = s
        spec = b1_spec(s, reversal, emp, "environment", eo_ns + half_frame + settle)
        p = run_trial(spec, out / f"b1_emp_s{s}.jsonl", PayloadStore(out / "payloads_emp"))
        rep[s] = trial_facts(p, spec.manifest(), [])
    phys = list(facts.values()); emps = list(rep.values())
    say(""); say("D->A on the physical route is receipt minus commitment: it includes the harness's frame catch-up at D "
                 "(up to one capture frame plus decode before the click is written), the cursor move and settle, and the "
                 "instrument's D->A; the replay row draws the instrument's D->A alone, with E->O + half a capture frame + settle "
                 "as its observation delay.")
    say(""); say("| route | SR | mean G | n | lost | D->A median (ms) | A before reversal | A after reversal |")
    say("|---|---|---|---|---|---|---|---|")
    for name, fs in (("physical (page-reported A)", phys), ("software, measured D->A replayed", emps)):
        sr, g = sr_g(fs)
        before = sum(1 for f in fs if f.get("a_ms") is not None and f["a_ms"] < (B1["E"] + reversal) / MS)
        after = sum(1 for f in fs if f.get("a_ms") is not None and f["a_ms"] >= (B1["E"] + reversal) / MS)
        say(f"| {name} | {sr} | {g} | {len(fs)} | {sum(f['lost'] for f in fs)} | {fmt(median([f.get('da_ms') for f in fs]))} | {before} | {after} |")
    conc = [f for f in phys if f.get("page_echo_diff_ms") is not None]
    if conc:
        diffs = [f["page_echo_diff_ms"] for f in conc]
        say(""); say(f"Page-reported vs echo-strip A: n={len(conc)}, median difference {statistics.median(diffs):+.1f} ms, "
                     f"max |difference| {max(abs(x) for x in diffs):.1f} ms (quantization floor one capture frame = {1000 / a.fps:.1f} ms).")
    # per-trial: outcome sign vs the sign predicted from the trial's own receipt time (both routes)
    say(""); say("| seed | physical D->A (ms) | physical A (task ms) | physical G | predicted from own A | replay D->A (ms) | replay G |")
    say("|---|---|---|---|---|---|---|")
    disc = 0
    for s in seeds:
        f, e = facts[s], rep[s]
        pred = None if f.get("a_ms") is None else (B1["U"][0] if f["a_ms"] < (B1["E"] + reversal) / MS else B1["U"][1])
        if pred is not None and f["G"] is not None and pred != f["G"]:
            disc += 1
        say(f"| {s} | {fmt(f.get('da_ms'))} | {fmt(f.get('a_ms'))} | {fmt(f['G'], 2)} | {fmt(pred, 2)} | {fmt(e.get('da_ms'))} | {fmt(e['G'], 2)} |")
    say(""); say(f"Discordant trials (physical outcome vs the sign of its own page-reported A): {disc}/{len(seeds)}.")
    (out / "facts.json").write_text(json.dumps({"physical": facts, "empirical_replay": rep, "reversal_ns": reversal}, indent=1))


def cell_c1(a, out: Path, eo_ns: int, da_ns: int) -> None:
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    policies = [a.policy] if a.policy else ["frozen", "late_bind"]
    ctrl = f"{a.model} late-binding controller (prompt {a.prompt_version}, {'no thinking' if a.no_thinking else 'thinking'})" if a.model else \
        f"scripted oracle (compute {a.oracle_compute_ms} ms; frozen = state_at_B, late_bind = newest_at_D)"
    say(f"# PHY3 C1 — balloon drift {C1['VX']} px/s on the physical route ({'SIMULATED target' if a.simulate else 'hardware'}), {len(seeds)} seeds x {policies}")
    say(f"controller: {ctrl}; balloon 320x240, target 24 px, event 500 ms, window 16 s, served at 30 Hz, capture {a.fps:.0f} Hz; "
        f"instrument medians E->O {eo_ns / MS:.0f} ms, D->A {da_ns / MS:.0f} ms; clicks by HID at calibrated screen coordinates; A = page-reported.")
    route = {"route": "physical", "receipt_timeout_ns": a.receipt_timeout_ms * MS, "expected_delivery_delay_ns": da_ns, "max_attempts": 1}
    rig = Rig(a, 320, 240, eo_ns, da_ns)
    budget = CallBudget(a.max_calls)
    facts: dict[str, dict[int, dict]] = {p: {} for p in policies}
    try:
        for pol in policies:
            for s in seeds:
                spec = c1_spec(a, s, pol, route, eo_ns)
                ctl = build_controller(spec, budget=budget) if a.model else None
                p = rig.run(spec, out, f"c1_{pol}_s{s}", controller=ctl)
                f = trial_facts(p, spec.manifest(), rig.route.log)
                facts[pol][s] = f
                say(f"{pol} seed {s}: end={f['end']} success={f['success']} G={fmt(f['G'], 2)} err={fmt(f.get('error_px'))} px "
                    f"tau={fmt(f.get('tau_ms'))} A={fmt(f.get('a_ms'))} D->A={fmt(f.get('da_ms'))} ms via {f['receipt_via']} obs={f['n_obs']} labels={f['labels']}")
    finally:
        rig.close()
    say(""); say("| policy | SR | mean G | n | lost | mean |error| (px) | mean v(t_A - tau) (px) | D->A median (ms) | E->O median (ms) |")
    say("|---|---|---|---|---|---|---|---|---|")
    for pol in policies:
        fs = list(facts[pol].values())
        sr, g = sr_g(fs)
        errs = [f["error_px"] for f in fs if f.get("error_px") is not None]
        law = [C1["VX"] * (f["a_ms"] - f["tau_ms"]) / 1000 for f in fs if f.get("a_ms") is not None and f.get("tau_ms") is not None]
        say(f"| {pol} | {sr} | {g} | {len(fs)} | {sum(f['lost'] for f in fs)} | {fmt(statistics.mean(errs) if errs else None)} | "
            f"{fmt(statistics.mean(law) if law else None)} | {fmt(median([f.get('da_ms') for f in fs]))} | {fmt(median([f.get('eo_ms') for f in fs]))} |")
    conc = [f for p in policies for f in facts[p].values() if f.get("page_echo_diff_ms") is not None]
    if conc:
        diffs = [f["page_echo_diff_ms"] for f in conc]
        say(""); say(f"Page-reported vs echo-strip A: n={len(conc)}, median difference {statistics.median(diffs):+.1f} ms, "
                     f"max |difference| {max(abs(x) for x in diffs):.1f} ms (floor one capture frame = {1000 / a.fps:.1f} ms).")
    if a.model:
        say(""); say(f"Model calls used: {budget.calls} of {a.max_calls}.")
    (out / "facts.json").write_text(json.dumps(facts, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["b1", "c1"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--route-json", default=None, help="instrument output (results/PHY_route_<date>.json); default runs/phy_sim/PHY_route_sim.json under --simulate")
    ap.add_argument("--seed-start", type=int, default=300)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--policy", default=None, choices=["frozen", "late_bind"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--prompt-version", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-calls", type=int, default=2000)
    ap.add_argument("--oracle-compute-ms", type=int, default=250)
    ap.add_argument("--http-port", type=int, default=8766)
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--screen-w", type=int, default=1920)
    ap.add_argument("--screen-h", type=int, default=1080)
    ap.add_argument("--receipt-timeout-ms", type=int, default=1000)
    ap.add_argument("--settle-ms", type=int, default=30, help="pause after the cursor move before the click")
    ap.add_argument("--sim-da-jitter-ms", type=float, default=15.0)
    ap.add_argument("--display-profile", default=None)
    ap.add_argument("--results-name", default=None, help="results/PHY3_<cell>_<name>.md (default: not written under --simulate)")
    a = ap.parse_args()
    if not a.simulate and not a.display_profile:
        ap.error("--display-profile is required on hardware (contract c5)")
    if a.simulate and a.model:
        ap.error("--simulate runs the scripted oracle; the model cell needs the hardware rig")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    rj = Path(a.route_json) if a.route_json else (ROOT / "runs" / "phy_sim" / "PHY_route_sim.json" if a.simulate else None)
    if rj is None or not rj.exists():
        ap.error(f"--route-json {rj} not found: run studies/phy_measure.py first (the instrument's medians place the cell)")
    r = json.loads(rj.read_text())
    if not r.get("gates_passed"):
        ap.error(f"{rj}: the instrument's gates did not pass")
    eo_ns, da_ns = int(statistics.median(r["eo_ms"]) * MS), int(statistics.median(r["da_ms"]) * MS)
    say(f"instrument: {rj} ({r.get('date')}, {'simulated' if r.get('simulated') else 'hardware'}; E->O n={len(r['eo_ms'])}, D->A n={len(r['da_ms'])})")
    t0 = time.time()
    if a.cell == "b1":
        cell_b1(a, out, eo_ns, da_ns, rj)
    else:
        cell_c1(a, out, eo_ns, da_ns)
    say(""); say(f"Wall time {time.time() - t0:.0f} s. Traces under {out}.")
    text = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(text)
    if a.results_name:
        (ROOT / "results" / f"PHY3_{a.cell}_{a.results_name}.md").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
