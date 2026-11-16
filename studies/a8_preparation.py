"""A8-PREP (cards/A8-preparation.md): preparation vs prediction vs late binding vs re-decision, by information regime.

Scripted cells (CPU, simulated clock; run now):
  PYTHONPATH=src .venv/bin/python studies/a8_preparation.py --scripted --out results/A8_preparation_scripted_100.md --n-seeds 100
Model cells (queue; the 27B model):
  PYTHONPATH=src .venv/bin/python studies/a8_preparation.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --n-seeds 100 --out runs/a8_qwen38_nothink_100 --max-calls 800

Geometry as the agent studies: 320x240, target 24 px (notification button 60x24), 60 Hz frames, publication 30 ms,
delivery 20 ms. Scripted controllers: `prepared` (content from the skin's fixed layout, released with compute 0 at first
detection) vs the four deliberating bindings at compute 5.0 s (the model's timescale): frozen = state_at_B,
late_bind = newest_at_D, restart = restart_if_changed, predicted = predicted_at_A.
  (a) notification, fixed_layout=True, windows 0.1, 0.2, 0.5, 1, 2, 4, 8 s x the five controllers; seeds 200.. (R1's).
  (b) seeded notification (window 8 s), balloon drift 8 px/s (moving_target, window 16 s), rotating code (change 2 s,
      window 16 s): prepared vs late_bind; seeds 200.. (notification) / 1000.. (balloon, code).
  (b') the change-after-D regime: balloon 8 px/s with a 1.5-s delivery route: prepared / late_bind / predicted / restart.
  (c) model: fixed-layout notification at windows 1, 2 s with E = 8 s (the one pre-event call must finish before E; the
      deliberating vlm_direct row is run at the same E so the comparison is same-E) - vlm_prepared (prompt 14) vs
      vlm_direct (prompt 4); seeded notification, window 8 s, E = 8 s: vlm_prepared vs vlm_late_binding/late_bind.
Every table leads with (SR, mean G). The regime table is written by analyze_a8.py (scripted now; model cells appended).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.runner import PayloadStore, build_controller, run_trial  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

S, MS = 1_000_000_000, 1_000_000
PERIOD, PUB, DEL = 16_666_667, 30 * MS, 20 * MS
COMPUTE = 5 * S
WINDOWS_A = [0.1, 0.2, 0.5, 1, 2, 4, 8]
CONTROLLERS = {"prepared": "prepared", "frozen": "state_at_B", "late_bind": "newest_at_D",
               "restart": "restart_if_changed", "predicted": "predicted_at_A"}


def scenario(scenario_id: str, seed: int, skin: str, window_ns: int, *, event_ns: int = 500 * MS, **extra) -> dict:
    return {"scenario_id": scenario_id, "scenario_seed": seed, "canvas_w_px": 320, "canvas_h_px": 240, "target_size_px": 24,
            "event_time_ns": event_ns, "response_window_ns": window_ns, "frame_period_ns": PERIOD,
            "episode_horizon_ns": event_ns + window_ns + 500 * MS, "irrelevant_motion": False, "skin": skin, **extra}


def scripted_spec(sc: dict, controller: str, delivery_ns: int = DEL) -> TrialSpec:
    binding = CONTROLLERS[controller]
    pred = binding == "predicted_at_A"
    return TrialSpec.from_dict({
        "schema_version": 1, "scenario": sc, "task_clock_policy": "live", "replay_source_live_trace_sha256": None,
        "controller": {"kind": "scripted_oracle", "compute_ns": 0 if controller == "prepared" else COMPUTE,
                       "coordinate_offset_px": 0, "binding": binding,
                       "expected_observation_delay_ns": PUB if pred else None,
                       "expected_delivery_delay_ns": delivery_ns if pred else None},
        "observation": {"boundary_version": 1, "publication_delay_ns": PUB},
        "action_space_version": 1, "scorer_version": 1, "timestamp_source": "simulated",
        "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": delivery_ns, "max_attempts": 1},
        "exclusion_policy": "report_invalid_separately", "trial_class": "agent_decision"})


class MemoryPayloadStore(PayloadStore):
    """Scripted cells: thousands of trials x ~1000 unique 230-KB frames each would be a terabyte on disk; the evaluator
    reads no payload and nothing is replayed, so payloads are hashed and dropped."""

    def __init__(self) -> None:
        pass

    def put(self, payload: bytes) -> str:
        from chronotope.hashing import sha256_hex
        return sha256_hex(payload)

    def get(self, payload_hash: str) -> bytes:
        raise KeyError("payloads are not kept in the scripted A8 cells")


def cell_stats(rs: list) -> dict:
    v = [r for r in rs if r.valid_trial]
    n = len(v)
    k = sum(bool(r.valid_at_receipt) for r in v)
    labs: dict[str, int] = defaultdict(int)
    for r in v:
        for lab in r.labels:
            labs[lab] += 1
        if not r.action_committed:
            labs["no_commit"] += 1
    lo, hi = (wilson(k, n)[1:] if n else (0.0, 0.0))
    ea = [r.intervals_ns["E_A"] / 1e9 for r in v if r.intervals_ns.get("E_A") is not None]
    return {"n": n, "k": k, "sr": k / n if n else None, "lo": lo, "hi": hi,
            "G": statistics.mean(r.return_G for r in v) if v else None,
            "E_A": statistics.median(ea) if ea else None, "labels": dict(labs), "invalid": len(rs) - n}


def fmt_cell(c: dict) -> str:
    if not c["n"]:
        return "-"
    return f"{c['k']}/{c['n']} ({c['sr']:.2f}), G {c['G']:.2f}"


def run_scripted(a) -> None:
    n = a.n_seeds
    seeds_n = list(range(200, 200 + n))        # notification: R1's evaluation seeds
    seeds_c = list(range(1000, 1000 + n))      # balloon / code: C1's and C4's
    root = Path(a.traces) if a.traces else Path(tempfile.mkdtemp(prefix="a8_"))
    root.mkdir(parents=True, exist_ok=True)
    store = MemoryPayloadStore()
    cells: dict[tuple, list] = defaultdict(list)

    def run(cell: tuple, sc: dict, controller: str, delivery_ns: int = DEL, seed: int = 0) -> None:
        spec = scripted_spec(sc, controller, delivery_ns)
        name = "_".join(str(x) for x in cell) + f"_s{seed}.jsonl"
        p = run_trial(spec, root / name, store)
        cells[cell].append(evaluate_trace(Trace.load(p)))

    # (a) fixed-layout notification, window sweep
    for w in WINDOWS_A:
        for ctrl in CONTROLLERS:
            for seed in seeds_n:
                run(("a", f"w{w:g}", ctrl), scenario("revealed_target", seed, "notification", int(w * S), fixed_layout=True), ctrl, seed=seed)
        print(f"(a) window {w:g} s done", flush=True)
    # (b) revealed content: prepared vs late_bind
    for ctrl in ("prepared", "late_bind"):
        for seed in seeds_n:
            run(("b", "notif_seeded_w8", ctrl), scenario("revealed_target", seed, "notification", 8 * S), ctrl, seed=seed)
        for seed in seeds_c:
            run(("b", "balloon_v8_w16", ctrl), scenario("moving_target", seed, "balloon", 16 * S, velocity_x_px_per_s=8, velocity_y_px_per_s=0), ctrl, seed=seed)
            run(("b", "code_ch2_w16", ctrl), scenario("rotating_code", seed, "notification", 16 * S, change_delay_ns=2 * S, code_digits=4), ctrl, seed=seed)
    print("(b) done", flush=True)
    # (b') change after D: balloon 8 px/s on a 1.5-s route (the O2 slow route)
    for ctrl in ("prepared", "late_bind", "predicted", "restart"):
        for seed in seeds_c:
            run(("bb", "balloon_v8_del1500", ctrl), scenario("moving_target", seed, "balloon", 16 * S, velocity_x_px_per_s=8, velocity_y_px_per_s=0),
                ctrl, delivery_ns=1500 * MS, seed=seed)
    print("(b') done", flush=True)

    L = [f"# A8-PREP scripted cells — preparation vs prediction vs late binding vs re-decision ({n} seeds per cell)", "",
         "Simulated clock. 320x240, target 24 px, 60 Hz, publication 30 ms, delivery 20 ms (1.5 s in the change-after-D "
         "cell). Controllers: prepared = content from the skin's fixed layout, released with compute 0 at first detection; "
         "frozen / late_bind / restart / predicted = the scripted bindings state_at_B / newest_at_D / restart_if_changed / "
         "predicted_at_A at compute 5.0 s (the 27B model's timescale). Every cell reports the outcome pair first: SR = "
         "valid at receipt; mean G = minus the click error at receipt in px (-400 when nothing valid arrived) on the "
         "click tasks and minus the seconds from E to a correct receipt on the code task (-16 when none).", "",
         "## (a) Fixed-layout notification: content known before E", "",
         "| window (s) | " + " | ".join(CONTROLLERS) + " |", "|---|" + "---|" * len(CONTROLLERS)]
    for w in WINDOWS_A:
        L.append(f"| {w:g} | " + " | ".join(fmt_cell(cell_stats(cells[("a", f"w{w:g}", c)])) for c in CONTROLLERS) + " |")
    L += ["", "Median E→A (s) per cell:", "", "| window (s) | " + " | ".join(CONTROLLERS) + " |", "|---|" + "---|" * len(CONTROLLERS)]
    for w in WINDOWS_A:
        L.append(f"| {w:g} | " + " | ".join((f"{cell_stats(cells[('a', f'w{w:g}', c)])['E_A']:.3f}" if cell_stats(cells[('a', f'w{w:g}', c)])['E_A'] is not None else "-") for c in CONTROLLERS) + " |")
    L += ["", "Labels per cell (window 1 s):", ""]
    for c in CONTROLLERS:
        L.append(f"- {c}: " + (", ".join(f"{k}×{v}" for k, v in sorted(cell_stats(cells[('a', 'w1', c)])['labels'].items())) or "-"))
    L += ["", "## (b) Content revealed by the event: prepared vs late_bind", "",
          "| task | prepared | late_bind | prepared labels |", "|---|---|---|---|"]
    for task, desc in (("notif_seeded_w8", "notification, seeded position, window 8 s"),
                       ("balloon_v8_w16", "balloon drift 8 px/s, window 16 s"),
                       ("code_ch2_w16", "rotating code, change at 2 s, window 16 s")):
        cp, cl = cell_stats(cells[("b", task, "prepared")]), cell_stats(cells[("b", task, "late_bind")])
        L.append(f"| {desc} | {fmt_cell(cp)} | {fmt_cell(cl)} | " + (", ".join(f"{k}×{v}" for k, v in sorted(cp['labels'].items())) or "-") + " |")
    L += ["", "## (b') Change continuing after D: balloon 8 px/s, delivery 1.5 s, window 16 s", "",
          "| prepared | late_bind | predicted | restart |", "|---|---|---|---|",
          "| " + " | ".join(fmt_cell(cell_stats(cells[("bb", "balloon_v8_del1500", c)])) for c in ("prepared", "late_bind", "predicted", "restart")) + " |",
          "", "Labels:", ""]
    for c in ("prepared", "late_bind", "predicted", "restart"):
        L.append(f"- {c}: " + (", ".join(f"{k}×{v}" for k, v in sorted(cell_stats(cells[('bb', 'balloon_v8_del1500', c)])['labels'].items())) or "-"))
    L += ["", f"Traces: {root}"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ---- model cells ---------------------------------------------------------------------------------------------------

def run_model(a) -> None:
    from agentlib import Study, late_binding, spec, vlm
    st = Study(a, "A8-PREP model cells — prepared (one pre-event call, prompt 14) vs deliberating rows")
    E8 = 8 * S
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    st.say("E = 8 s in every model cell so that the pre-event call completes before the card appears; the deliberating rows "
           "are run at the same E. Gate: prompt 4 / notification qualified in R1 (results/model_qualification.md); prompt 14 has "
           "no perception to gate — a wait on the pre-event frame is reported as 'preparation not available'.")
    for w in (1, 2):
        for ctrl_name, ctrl in (("prepared", vlm(a, "vlm_prepared", 14)), ("direct", vlm(a, "vlm_direct", 4))):
            st.say(f"### fixed layout, window {w} s, {ctrl_name}")
            for seed in seeds:
                sc = scenario("revealed_target", seed, "notification", w * S, event_ns=E8, fixed_layout=True)
                r, t = st.trial(spec(sc, ctrl), f"a8_fixed_w{w}_{ctrl_name}_s{seed}")
                inv = t.all("decision_invocation")
                st.say(f"- s{seed}: valid={r.valid_at_receipt} G={r.return_G:.1f} outcomes={[i.get('outcome') for i in inv]} "
                       f"E→A={(r.intervals_ns.get('E_A') or 0)/1e9:.3f}s labels={','.join(r.labels) or '-'}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    for ctrl_name, ctrl in (("prepared", vlm(a, "vlm_prepared", 14)), ("late_bind", late_binding(a, 4, "late_bind"))):
        st.say(f"### seeded layout, window 8 s, {ctrl_name}")
        for seed in seeds:
            sc = scenario("revealed_target", seed, "notification", 8 * S, event_ns=E8)
            r, t = st.trial(spec(sc, ctrl), f"a8_seeded_w8_{ctrl_name}_s{seed}")
            st.say(f"- s{seed}: valid={r.valid_at_receipt} G={r.return_G:.1f} err={r.spatial_error_px} labels={','.join(r.labels) or '-'}"
                   + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description="A8-PREP: preparation vs prediction vs late binding vs re-decision")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-calls", type=int, default=800)
    ap.add_argument("--out", type=Path, required=True, help="scripted: the results .md; model: the run directory")
    ap.add_argument("--traces", default=None, help="scripted: keep the traces here (default: a temporary directory)")
    ap.add_argument("--seed-start", type=int, default=200)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--calib-seed-start", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=10)
    ap.add_argument("--skip-calibration", action="store_true")
    a = ap.parse_args()
    if a.scripted:
        run_scripted(a)
        return 0
    if not (a.model and a.base_url and (a.thinking or a.no_thinking)):
        raise SystemExit("model cells need --model, --base-url and --thinking/--no-thinking (or pass --scripted)")
    run_model(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
