"""Shared helpers for the 2026-08-24 agent studies (cards A1-sponsor-skip-agent, A2-sponsor-skip-pending,
B1-live-trading-agent, C-prl-rows, O2-remedy-matrix-agent): spec builders, the Paused competence gate, the
P/R/L triplet runner with the file-naming the analyzers expect, and a study logger."""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chronotope.__main__ import run_triplet  # noqa: E402
from chronotope.controller import CallBudget  # noqa: E402
from chronotope.evaluate import evaluate_pending_trace, evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

S = 1_000_000_000
PERIOD_NS, PUB_NS, DEL_NS = 16_666_667, 30_000_000, 20_000_000
COMPETENCE_THRESHOLD = 0.80


def agent_argparser(desc: str, seed_start: int, n_seeds: int = 10, calib_start: int = 100) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-calls", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=seed_start)
    ap.add_argument("--n-seeds", type=int, default=n_seeds, help="10 = pilot; a cited cell needs >= 100")
    ap.add_argument("--calib-seed-start", type=int, default=calib_start)
    ap.add_argument("--n-calib", type=int, default=10)
    ap.add_argument("--skip-calibration", action="store_true")
    return ap


def vlm(a, kind: str, prompt_version: int, **extra) -> dict:
    c = {"kind": kind, "server": "openai_compatible_local", "base_url": a.base_url, "model_id": a.model,
         "prompt_version": prompt_version, "image_scale": 2, "max_tokens": a.max_tokens, "temperature": a.temperature,
         "enable_thinking": bool(a.thinking), "reasoning_effort": a.reasoning_effort, "invoke_on": "frame_change"}
    c.update(extra)
    return c


def late_binding(a, prompt_version: int, policy: str = "frozen") -> dict:
    return vlm(a, "vlm_late_binding", prompt_version, at_commit_policy=policy, template_radius_px=12,
               min_match_score=0.8, restart_threshold_px=(6 if policy == "restart_if_moved" else None))


def scenario(scenario_id: str, seed: int, skin: str, window_ns: int, *, target: int = 24, w: int = 320, h: int = 240,
             event_ns: int = 500_000_000, **extra) -> dict:
    return {"scenario_id": scenario_id, "scenario_seed": seed, "canvas_w_px": w, "canvas_h_px": h,
            "target_size_px": target, "event_time_ns": event_ns, "response_window_ns": window_ns,
            "frame_period_ns": PERIOD_NS, "episode_horizon_ns": event_ns + window_ns + 500_000_000,
            "irrelevant_motion": False, "skin": skin, **extra}


def spec(sc: dict, controller: dict, *, policy: str = "live", delivery_ns: int = DEL_NS,
         route: str = "simulated_fixed_delay", pub_ns: int = PUB_NS) -> TrialSpec:
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": sc, "task_clock_policy": policy, "replay_source_live_trace_sha256": None,
        "controller": controller, "observation": {"boundary_version": 1, "publication_delay_ns": pub_ns},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "monotonic_ns",
        "delivery": {"route": route, "delivery_delay_ns": delivery_ns, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


class Study:
    """Logger + budget + output dir; RESULTS.md is rewritten after every trial so a killed run is still readable."""

    def __init__(self, a, title: str) -> None:
        self.a = a
        self.out: Path = a.out
        self.out.mkdir(parents=True, exist_ok=True)
        self.budget = CallBudget(a.max_calls)
        self.lines: list[str] = []
        self.say(f"# {title} ({a.model}, thinking={'on' if a.thinking else 'off'}, T={a.temperature}, max_tokens={a.max_tokens})")
        self.say()

    def say(self, s: str = "") -> None:
        print(s, flush=True)
        self.lines.append(s)
        (self.out / "RESULTS.md").write_text("\n".join(self.lines) + "\n")

    def mk(self, s: TrialSpec):
        return build_controller(s, budget=self.budget)

    # ---- Paused competence gate (cards/model-qualification.md, §17.7): >= 0.80 over the calibration seeds ----
    def gate(self, make_spec, seeds: list[int], label: str) -> bool:
        if self.a.skip_calibration:
            self.say(f"Calibration skipped (--skip-calibration); the gate must have been passed elsewhere.")
            self.say()
            return True
        self.say(f"## Paused competence gate — {label} (seeds {seeds[0]}..{seeds[-1]}, threshold {COMPETENCE_THRESHOLD})")
        ok = 0
        for seed in seeds:
            s = make_spec(seed)
            p = run_trial(s, self.out / f"cal_s{seed}_paused.jsonl", PayloadStore(self.out / "payloads"), controller=self.mk(s))
            r = evaluate_trace(Trace.load(p))
            ok += int(bool(r.valid_trial and r.valid_at_receipt))
            self.say(f"- seed {seed}: {'valid' if r.valid_at_receipt else 'INVALID'} (B→D {fmt((r.b_to_d_ns or 0)/1e9)} s, "
                     f"err {fmt(r.spatial_error_px, 1)} px, G {fmt(r.return_G)}, labels {','.join(r.labels) or '-'})")
        rate = ok / len(seeds)
        self.say(f"\nPaused competence: {ok}/{len(seeds)} = {rate:.2f}")
        self.say()
        if rate < COMPETENCE_THRESHOLD:
            self.say("STOP: competence below threshold; timing conditions not run (card stop condition).")
            return False
        return True

    # ---- one matched Paused / Replay / Live triplet; files prl_<cell>_s<seed>_<policy>.jsonl ----
    def triplet(self, live_spec: TrialSpec, cell: str, seed: int) -> dict:
        out = run_triplet(live_spec, self.out, tag=f"prl_{cell}_s{seed}_", controller_factory=self.mk)
        res = out["results"]
        parts = []
        for pol in ("paused", "replay", "live"):
            r = res[pol]
            v = "INVALID" if not r.valid_trial else ("1" if r.valid_at_receipt else "0")
            parts.append(f"{pol}={v}/G={fmt(r.return_G)}")
        live = res["live"]
        self.say(f"- {cell} s{seed}: {' '.join(parts)} B→D={fmt((live.b_to_d_ns or 0)/1e9)}s "
                 f"labels[live]={','.join(live.labels) or '-'} pair={'ok' if out['pair'].valid else 'INVALID:' + str(out['pair'].reason)}"
                 + (f" viol={','.join(live.contract_violations)}" if live.contract_violations else ""))
        return out

    # ---- one Live trial (single- or multi-action) ----
    def trial(self, s: TrialSpec, name: str):
        p = run_trial(s, self.out / f"{name}.jsonl", PayloadStore(self.out / "payloads"), controller=self.mk(s))
        t = Trace.load(p)
        return (evaluate_pending_trace if getattr(s.manifest(), "multi_action", False) else evaluate_trace)(t), t

    def finish(self) -> None:
        self.say(f"\nModel calls used: {self.budget.calls} of {self.a.max_calls}.")
