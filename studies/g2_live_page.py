"""G2 — a live Chromium page through the O boundary (cards/G2-live-page.md).

The page (site/live_page.html) runs in headless Chromium; the harness sees it only as screenshots taken on a schedule
(default 15 Hz, PNG), clicks it through the browser's input path (Playwright mouse.click), and reads events and
geometry from the page's own log (E = the banner insert the page reports; boxes = DOM rects). A (receipt) is the
page's pointerdown timestamp mapped to the harness clock by a performance.now() sync at page open.

Cells (seed -> banner time 2 s +- 0.3 s after load, banner height 60-200 px, dialog offset and slide vector):
  step 1   frozen / late_bind / restart_if_moved (vlm_late_binding, prompt 12, image_scale 1): a Continue click
           while the banner reflows the page during deliberation; Live and Replay (the recorded screenshot stream
           through ReplayFrameSource, the response held valid at the triggering frame).
  two-step frozen (vlm_pending, memory none) / late_bind (vlm_chronotope, memory none) / chronotope (vlm_chronotope,
           harness_pending), prompt 15: the Continue click opens a confirm dialog placed from the click's own
           coordinates (a later observation depends on an earlier action) which slides for --slide-s (8 s) and stops;
           the Confirm click completes the task. Live only (the evidence depends on the agent's actions). Why 8 s and
           not the card's 2 s: the decision after the Continue click is spent on the reflowed frame that the click
           preceded (the frame changed during the first deliberation; the pending problem — memory none re-clicks the
           now-hidden Continue, harness_pending suppresses the duplicate, both pay the call), so the first decision
           that sees the dialog begins one deliberation after it appeared; a 2 s slide would always have stopped by
           then and step 2 would test nothing. At 8 s the dialog still moves for any B->D between ~1 and 8 s.

Paused is not available on a live page: holding the response-triggering state until the action arrives would mean
instrumenting the page (stopping its timers and the browser's input path), which is exactly what this task does not
do; the model's competence on this page family was established by G1's Paused check (prompt 12, same rendering).
Results lead with (SR, mean G): step 1 G = minus the click error in px at receipt (minus the display diagonal when
no valid click); two-step G = minus the seconds from the reveal to the page's completion (minus the window if never).

  # scripted pipeline test (no model; a colour perceiver of the screenshot with a 5 s compute delay inside the real
  # controllers; the DOM is the evaluator's truth through the page log)
  PYTHONPATH=src .venv/bin/python studies/g2_live_page.py --scripted --n-seeds 10 --out runs/g2_scripted_pilot
  # the model
  PYTHONPATH=src .venv/bin/python studies/g2_live_page.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \\
      --no-thinking --n-seeds 100 --out runs/g2_live_qwen38_nothink_100 --max-calls 1500
Files live_step1_<policy>_s<seed>.jsonl, replay_step1_<policy>_s<seed>.jsonl, live_two_step_<policy>_s<seed>.jsonl;
analysis studies/analyze_g2.py."""
from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, fmt, late_binding, mean, vlm  # noqa: E402

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.live_page import (CHROMIUM, LiveBrowser, PlaywrightRoute, ScriptedPageOracle, evaluate_live_two_step,  # noqa: E402
                                  live_source_factory, page_url)
from chronotope.pair import validate_pair  # noqa: E402
from chronotope.runner import PayloadStore, SimulatedFixedDelayRoute, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

W, H, TARGET = 640, 480, 37
STEP1_POLICIES = ["frozen", "late_bind", "restart_if_moved"]
TWO_STEP_POLICIES = ["frozen", "late_bind", "chronotope"]
PENDING = dict(pending_timeout_ns=10 * S, pending_radius_px=10)


def seed_params(seed: int) -> dict:
    rng = random.Random(seed)
    return {"banner_delay_ns": (2000 + rng.randint(-300, 300)) * 1_000_000, "banner_h_px": rng.randint(60, 200),
            "dialog_dx_px": rng.randint(-150, 150), "dialog_dy_px": rng.randint(-160, 40),
            "slide_dx_px": rng.choice([-1, 1]) * rng.randint(40, 120), "slide_dy_px": rng.choice([-1, 1]) * rng.randint(40, 120)}


def scenario(seed: int, two_step: bool, window_ns: int, slide_ns: int = 2 * S) -> dict:
    return {"scenario_id": "live_page", "scenario_seed": seed, "canvas_w_px": W, "canvas_h_px": H, "target_size_px": TARGET,
            "event_time_ns": 500_000_000, "response_window_ns": window_ns, "frame_period_ns": 66_666_667,
            "episode_horizon_ns": 500_000_000 + window_ns + 1 * S, "irrelevant_motion": False, "skin": "browser",
            "reveal_delay_ns": 500_000_000, "slide_ns": slide_ns, "two_step": two_step, **seed_params(seed)}


def controller(a, cell: str, pol: str) -> dict:
    if cell == "step1":
        return {**late_binding(a, 12, pol), "image_scale": 1}
    if pol == "frozen":
        return vlm(a, "vlm_pending", 15, memory="none", image_scale=1, **PENDING)
    return vlm(a, "vlm_chronotope", 15, memory=("harness_pending" if pol == "chronotope" else "none"), image_scale=1,
               template_radius_px=12, min_match_score=0.8, **PENDING)


def spec(sc: dict, ctrl: dict, receipt_timeout_ns: int) -> TrialSpec:
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": sc, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": ctrl, "observation": {"boundary_version": 1, "publication_delay_ns": 0},   # O = capture completion
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": "playwright_page", "receipt_timeout_ns": receipt_timeout_ns, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


def argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="G2 live Chromium page: step-1 reflow and the two-step closed loop")
    ap.add_argument("--scripted", action="store_true", help="DOM oracle instead of the model (pipeline test)")
    ap.add_argument("--compute-s", type=float, default=5.0, help="scripted oracle compute delay (s)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, default=2000)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=10, help="10 = pilot; a cited cell needs >= 100")
    ap.add_argument("--hz", type=float, default=15.0, help="screenshot schedule")
    ap.add_argument("--cells", default="step1,two_step")
    ap.add_argument("--no-replay", action="store_true", help="skip the Replay of every step-1 Live trial")
    ap.add_argument("--window-s", type=float, default=16.0, help="step-1 response window")
    ap.add_argument("--two-step-window-s", type=float, default=30.0)
    ap.add_argument("--slide-s", type=float, default=8.0, help="two-step: the dialog slides this long, then stops")
    ap.add_argument("--receipt-timeout-s", type=float, default=2.0)
    ap.add_argument("--chromium", default=CHROMIUM)
    return ap


class G2Study(Study):
    def __init__(self, a, browser: LiveBrowser) -> None:
        super().__init__(a, "G2 live Chromium page (screenshots on a schedule, clicks through the browser, events from the page log)")
        self.browser, self.store = browser, PayloadStore(a.out / "payloads")
        self.period_ns = int(round(1e9 / a.hz))

    def make_controller(self, s: TrialSpec, session=None):
        if self.a.scripted:
            return build_controller(s, vlm_adapter=ScriptedPageOracle(session, self.a.compute_s), budget=self.budget)
        return self.mk(s)

    def live(self, s: TrialSpec, name: str):
        m = s.manifest()
        session = self.browser.open(page_url(m), W, H)
        m.attach(session)
        route = PlaywrightRoute(session, s.raw["delivery"]["receipt_timeout_ns"])
        try:
            p = run_trial(s, self.out / f"{name}.jsonl", self.store, controller=self.make_controller(s, session), route=route,
                          source=live_source_factory(session, m, self.period_ns), manifest=m)
        finally:
            session.close()
        return Trace.load(p), route, session.achieved_hz()

    def replay(self, s: TrialSpec, live: Trace, name: str):
        rs = s.derive("replay", live.file_hash())
        m = rs.manifest().with_trailer(live.first("scenario_trailer"))
        d, r = live.first("delivery"), live.first("receipt")
        d_a = (r["receipt_time_ns"] - d["submit_time_ns"]) if (d and r) else 0
        p = run_trial(rs, self.out / f"{name}.jsonl", self.store, live_trace=live, controller=self.make_controller(rs),
                      route=SimulatedFixedDelayRoute(max(0, d_a)), manifest=m)
        return Trace.load(p)


def main() -> int:
    a = argparser().parse_args()
    if a.scripted:
        a.model, a.base_url, a.thinking = "scripted-page-oracle", "http://127.0.0.1:0/v1", False
    elif not a.model or not a.base_url or not (a.thinking or a.no_thinking):
        raise SystemExit("--model, --base-url and --thinking/--no-thinking are required without --scripted")
    cells = a.cells.split(",")
    browser = LiveBrowser(a.chromium)
    st = G2Study(a, browser)
    st.say(f"Screenshot schedule {a.hz:g} Hz; receipt timeout {a.receipt_timeout_s:g} s; seeds {a.seed_start}..{a.seed_start + a.n_seeds - 1}"
           + (f"; scripted page oracle (screenshot colour perceiver), compute {a.compute_s:g} s" if a.scripted else "") + ". Paused is not available on a live page (see docstring).")
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    hz, lat, lost = [], [], 0
    rows1, rows2 = [], []
    try:
        if "step1" in cells:
            for pol in STEP1_POLICIES:
                st.say(f"## step 1 (banner during deliberation), {pol}")
                for seed in seeds:
                    s_ = spec(scenario(seed, False, int(a.window_s * S)), controller(a, "step1", pol), int(a.receipt_timeout_s * S))
                    t, route, f = st.live(s_, f"live_step1_{pol}_s{seed}")
                    r = evaluate_trace(t)
                    m = s_.manifest().with_trailer(t.first("scenario_trailer"))
                    hz.append(f); lat += route.latencies_ns; lost += route.lost
                    corr = t.first("coordinate_correction")
                    rr = None
                    if not a.no_replay and r.valid_trial:
                        tr = st.replay(s_, t, f"replay_step1_{pol}_s{seed}")
                        rr = evaluate_trace(tr); pair = validate_pair(tr, t)
                    rows1.append((pol, r, rr))
                    st.say(f"- live_step1_{pol}_s{seed}: valid={r.valid_at_receipt} err={fmt(r.spatial_error_px, 1)} G={fmt(r.return_G)} "
                           f"reflow_dy={m.reflow_dy_px()} banner_at={fmt((m.measured['banner_ns'] or 0)/1e9)}s E={fmt(m.t_reveal()/1e9)}s "
                           f"translation_y={(corr or {}).get('translation_y')} counterfactual_frozen={r.proposed_valid_at_receipt} "
                           f"restarts={r.n_restarts} B→D={fmt((r.b_to_d_ns or 0)/1e9)}s firstB→D={fmt((r.first_b_to_d_ns or 0)/1e9)}s "
                           f"D→A={fmt((r.intervals_ns.get('D_A') or 0)/1e6, 1)}ms hz={fmt(f, 1)} "
                           f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                           + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}")
                           + (f" | replay: valid={rr.valid_at_receipt} G={fmt(rr.return_G)} pair={'ok' if pair.valid else 'INVALID:' + str(pair.reason)}" if rr else ""))
        if "two_step" in cells:
            for pol in TWO_STEP_POLICIES:
                st.say(f"## two-step (dialog placed from the first click, slides 2 s), {pol}")
                for seed in seeds:
                    s_ = spec(scenario(seed, True, int(a.two_step_window_s * S), int(a.slide_s * S)), controller(a, "two_step", pol), int(a.receipt_timeout_s * S))
                    t, route, f = st.live(s_, f"live_two_step_{pol}_s{seed}")
                    r = evaluate_live_two_step(t)
                    hz.append(f); lat += route.latencies_ns; lost += route.lost
                    rows2.append((pol, r))
                    st.say(f"- live_two_step_{pol}_s{seed}: completed={r.completed} s_to_complete={fmt(r.seconds_to_completion)} G={fmt(r.return_G)} "
                           f"decisions={r.n_decisions} clicks={r.n_clicks} hits={r.n_hits} lost={r.n_lost} restarts={r.n_restarts} "
                           f"step2 B→D={fmt((r.step2_b_to_d_ns or 0)/1e9)}s dialog_moved={fmt(r.dialog_moved_px, 1)}px hz={fmt(f, 1)} "
                           f"clicks=[{' '.join(f"s{c['step']}:{'hit' if c['hit'] else 'miss'}{(':' + ','.join(c['labels'])) if c['labels'] else ''}" for c in r.clicks)}] "
                           f"labels2={','.join(r.step2_labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                           + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    finally:
        browser.close()
    st.say()
    if rows1:
        st.say("| cell | policy | n | SR (valid at A) | mean G (px) | mean err (px) | Replay SR | Replay mean G | frozen counterfactual SR (same A) | restarts/trial | B→D (s) | first B→D (s) | outdated_at_commitment |")
        st.say("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for pol in STEP1_POLICIES:
            rs = [(r, rr) for (p, r, rr) in rows1 if p == pol and r.valid_trial]
            n = len(rs)
            if not n:
                continue
            cf = [r.proposed_valid_at_receipt for r, _ in rs if r.proposed_valid_at_receipt is not None]
            rp = [rr for _, rr in rs if rr is not None and rr.valid_trial]
            st.say(f"| step1 | {pol} | {n} | {sum(bool(r.valid_at_receipt) for r, _ in rs)/n:.2f} | {fmt(mean([r.return_G for r, _ in rs]), 1)} | "
                   f"{fmt(mean([r.spatial_error_px for r, _ in rs]), 1)} | {(f'{sum(bool(x.valid_at_receipt) for x in rp)/len(rp):.2f}' if rp else '-')} | "
                   f"{fmt(mean([x.return_G for x in rp]), 1) if rp else '-'} | {(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | "
                   f"{mean([r.n_restarts for r, _ in rs]):.1f} | {fmt(mean([(r.b_to_d_ns or 0)/1e9 for r, _ in rs]))} | "
                   f"{fmt(mean([(r.first_b_to_d_ns or 0)/1e9 for r, _ in rs]))} | {sum('outdated_at_commitment' in r.labels for r, _ in rs)} |")
    if rows2:
        st.say()
        st.say("| cell | policy | n | SR (completed) | mean G (−s to completion) | clicks/trial | hits/trial | step-2 B→D (s) | dialog moved at step 2 (px) | step-2 outdated_at_commitment |")
        st.say("|---|---|---|---|---|---|---|---|---|---|")
        for pol in TWO_STEP_POLICIES:
            rs = [r for (p, r) in rows2 if p == pol and r.valid_trial]
            n = len(rs)
            if not n:
                continue
            st.say(f"| two-step | {pol} | {n} | {sum(r.completed for r in rs)/n:.2f} | {fmt(mean([r.return_G for r in rs]))} | "
                   f"{mean([r.n_clicks for r in rs]):.1f} | {mean([r.n_hits for r in rs]):.1f} | {fmt(mean([(r.step2_b_to_d_ns or 0)/1e9 for r in rs]))} | "
                   f"{fmt(mean([r.dialog_moved_px for r in rs]), 1)} | {sum('outdated_at_commitment' in r.step2_labels for r in rs)} |")
    st.say()
    hz_ok = [x for x in hz if x]
    if hz_ok:
        st.say(f"Screenshot rate: scheduled {a.hz:g} Hz; achieved median {statistics.median(hz_ok):.1f} Hz (min {min(hz_ok):.1f}, max {max(hz_ok):.1f}) over {len(hz_ok)} trials.")
    if lat:
        q = sorted(lat)
        st.say(f"Receipt latency A − submit (page pointerdown minus harness submit), {len(q)} deliveries: median {q[len(q)//2]/1e6:.2f} ms, "
               f"p10 {q[len(q)//10]/1e6:.2f} ms, p90 {q[9*len(q)//10]/1e6:.2f} ms, max {q[-1]/1e6:.2f} ms; lost deliveries {lost}.")
    st.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
