"""D1 — distributional gym evaluation (cards/D1-distributional-gym.md). Opt-in; touches nothing that exists.

  PYTHONPATH=src .venv/bin/python studies/d1_distributional_gym.py generate --family closing_window --n 300 --seed 1 \
      --out runs/d1/manifests/closing_window.jsonl
  ... generate --family moving_target --n 300 --seed 2 --out runs/d1/manifests/moving_target.jsonl
  ... generate --family static_control --n 300 --seed 3 --out runs/d1/manifests/static_control.jsonl
  ... generate --family late_binding_natural --n 300 --seed 4 --out runs/d1/manifests/lb_natural.jsonl
  ... generate --family late_binding_gate --n 300 --seed 5 --update-type relevant --out runs/d1/manifests/lb_gate.jsonl
  ... generate --family late_binding_gate --n 300 --seed 6 --update-type irrelevant --out runs/d1/manifests/lb_gate_irrelevant.jsonl
      (likewise duplicate, after_D, kind_change)

  ... run --manifest runs/d1/manifests/closing_window.jsonl --scripted state_at_B:5200000000 --name oracle_5.2s \
      --conditions live,replay --out runs/d1/closing_window
  ... run --manifest runs/d1/manifests/closing_window.jsonl --controller-json specs/d1_qwen_late_bind.json \
      --name qwen_late_bind --conditions live,replay --max-calls 1000 --out runs/d1/closing_window
  ... report --family closing_window --runs runs/d1/closing_window/oracle_5.2s runs/d1/closing_window/qwen_late_bind \
      --baseline oracle_5.2s --out results/D1_closing_window.md
  ... negative-controls --late late_bind --frozen frozen --strata irrelevant=runs/d1/gate_irrelevant duplicate=... \
      --out results/D1_negative_controls.md
  ... demo --out runs/d1_demo --n 12          (scripted controllers on every family; the pipeline end to end)

Every controller runs at its own realized speed on the identical frozen rows; the deadline of a row is the row's.
A --controller-json file holds the controller block of a trial spec (kind, model_id, base_url, prompt_version, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chronotope" / "python"))   # the "chronotope" controller kind

from chronotope.controller import CallBudget  # noqa: E402
from chronotope.distributional import (ControllerSpec, GenConfig, ManifestSet, build_report, dump_summary, generate,  # noqa: E402
                                       load_empirical_windows, load_outcomes, negative_controls_report, run_episode,
                                       score_trace)
from chronotope.hashing import canonical_json  # noqa: E402
from chronotope.runner import PayloadStore  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

N_MIN = 300


def scripted(binding: str, compute_ns: int) -> dict:
    return {"kind": "scripted_oracle", "compute_ns": compute_ns, "coordinate_offset_px": 0, "binding": binding,
            "expected_observation_delay_ns": None, "expected_delivery_delay_ns": None}


def cmd_generate(a) -> int:
    kw = {}
    if a.empirical_windows:
        kw["empirical_windows_s"] = load_empirical_windows(a.empirical_windows)
    if a.speed_range:
        lo, _, hi = a.speed_range.partition(",")
        kw["v_min_px_s"], kw["v_max_px_s"] = float(lo), float(hi)
    if a.deadline_range:
        lo, _, hi = a.deadline_range.partition(",")
        kw["w_min_s"], kw["w_max_s"] = float(lo), float(hi)
    if a.delay_range:
        lo, _, hi = a.delay_range.partition(",")
        kw["delay_min_s"], kw["delay_max_s"] = float(lo), float(hi)
    if a.canvas:
        w, _, h = a.canvas.partition("x")
        kw["canvas_w_px"], kw["canvas_h_px"] = int(w), int(h)
    elif a.family in ("moving_target", "static_control"):
        kw["canvas_w_px"], kw["canvas_h_px"] = 640, 480     # holds 16 px/s x 16 s of constant motion (request §6)
    cfg = GenConfig(task_family=a.family, n=a.n, master_seed=a.seed, update_type=a.update_type, seed_base=a.seed_base,
                    generous_deadline_s=a.generous_deadline_s, gate_compute_s=a.gate_compute_s, **kw)
    ms = generate(cfg)
    h = ms.freeze(a.out)
    print(f"frozen {len(ms.rows)} rows of {a.family} -> {a.out}  sha256 {h}")
    return 0


def run_controller(ms: ManifestSet, ctrl: ControllerSpec, conditions: list[str], out: Path, *, repeats: int = 1,
                   budget: CallBudget | None = None, vlm_adapter=None, quiet: bool = False) -> list:
    """Live first (Replay is derived from the controller's own Live trace of the row); outcomes appended per episode."""
    out.mkdir(parents=True, exist_ok=True)
    store = PayloadStore(out / "payloads")
    outcomes_path = out / ctrl.name / "outcomes.jsonl"
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    (out / ctrl.name / "controller.json").write_text(canonical_json(ctrl.controller) + "\n")
    (out / ctrl.name / "manifest.sha256").write_text(ms.frozen_hash + "\n")
    outs = []
    for k, row in enumerate(ms.rows):
        for rep in range(repeats):
            live_trace = None
            for cond in ("live", "replay", "paused"):
                if cond not in conditions:
                    continue
                p = run_episode(row, ms.frozen_hash, ctrl, cond, out, store, live_trace=live_trace, repeat_index=rep,
                                budget=budget, vlm_adapter=vlm_adapter)
                t = Trace.load(p)
                if cond == "live":
                    live_trace = t
                o = score_trace(row, t, ms.frozen_hash, controller=ctrl.name, repeat_index=rep)
                outs.append(o)
                with outcomes_path.open("a", encoding="utf-8") as f:
                    f.write(canonical_json(o.to_dict()) + "\n")
                if not quiet:
                    print(f"[{k + 1}/{len(ms.rows)}] {row.episode_id} r{rep} {cond}: Y={o.Y} C={o.C} R={o.R} V_A={o.V_A} "
                          f"B→D={('-' if o.T_reason is None else f'{o.T_reason / 1e9:.2f}s')} mode={o.failure_mode}", flush=True)
    return outs


def cmd_run(a) -> int:
    ms = ManifestSet.load(a.manifest)
    if not a.allow_small:
        ms.require_unique_count(N_MIN)
    if a.controller_json:
        ctrl = ControllerSpec(a.name, json.loads(Path(a.controller_json).read_text()))
    else:
        binding, _, compute = a.scripted.partition(":")
        ctrl = ControllerSpec(a.name, scripted(binding, int(compute or 0)))
    budget = CallBudget(a.max_calls) if (ctrl.is_vlm or ctrl.is_resident) else None   # a resident vlm planner counts calls too
    conds = a.conditions.split(",")
    if ms.rows and ms.rows[0].task_family == "pending_effect":
        conds = ["live"]   # multi-action tasks are defined under Live only (runner)
    outs = run_controller(ms, ctrl, conds, a.out, repeats=a.repeats, budget=budget)
    live = [o for o in outs if o.condition == "live"]
    print(f"{ctrl.name}: {sum(o.Y for o in live)}/{len(live)} Live successes on {len(ms.rows)} unique episodes")
    return 0


def _fill_returns(d: Path, outs: list) -> None:
    """Outcomes written before G_return existed: read the return from the trace once (results are not re-scored)."""
    from chronotope.evaluate import evaluate_trace
    for o in outs:
        if o.G_return is None:
            p = d / f"{o.episode_id}_r{o.repeat_index}_{o.condition}.jsonl"
            if p.exists():
                try:
                    o.G_return = evaluate_trace(Trace.load(p)).return_G
                except Exception:   # noqa: BLE001
                    pass


def load_runs(run_dirs: list[Path]) -> dict:
    by = {}
    for d in run_dirs:
        outs = load_outcomes(d / "outcomes.jsonl")
        _fill_returns(d, outs)
        name = d.name
        by[name] = {c: [o for o in outs if o.condition == c] for c in ("live", "replay", "paused")}
        by[name] = {c: v for c, v in by[name].items() if v}
    return by


def load_controllers(run_dirs: list[Path]) -> dict:
    out = {}
    for d in run_dirs:
        p = d / "controller.json"
        if p.exists():
            out[d.name] = json.loads(p.read_text())
    return out


def cmd_report(a) -> int:
    by = load_runs([Path(p) for p in a.runs])
    mh = {o.manifest_hash for d in by.values() for c in d.values() for o in c}
    ref = dict(x.split("=", 1) for x in (a.reference or []))
    dist = None
    if a.manifest:
        cfg = ManifestSet.load(a.manifest).header["config"]
        dist = {k: cfg.get(k) for k in ("w_min_s", "w_max_s", "empirical_windows_s", "delay_min_s", "delay_max_s")}
    extra = [{"w_min_s": float(x.split(",")[0]), "w_max_s": float(x.split(",")[1]), "empirical_windows_s": []} for x in (a.extra_law or [])]
    md = build_report(a.family, by, baseline=a.baseline, manifest_hash=mh.pop() if len(mh) == 1 else None, reference=ref or None,
                      deadline_dist=dist, controllers=load_controllers([Path(p) for p in a.runs]), extra_laws=extra)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md + "\n")
    Path(a.out).with_suffix(".json").write_text(dump_summary(by) + "\n")
    print(md)
    return 0


def cmd_negative(a) -> int:
    strata = {}
    for item in a.strata:
        name, _, root = item.partition("=")
        by = load_runs([Path(root) / a.late, Path(root) / a.frozen])
        strata[name] = {a.late: by[a.late]["live"], a.frozen: by[a.frozen]["live"]}
    md = negative_controls_report(strata, late=a.late, frozen=a.frozen)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(md + "\n")
    print(md)
    return 0


def cmd_demo(a) -> int:
    """Scripted controllers on every family: the pipeline end to end on a small manifest (not a cited cell)."""
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    fams = [("closing_window", {}), ("moving_target", {}), ("static_control", {}), ("late_binding_natural", {}),
            ("late_binding_gate", {"update_type": "relevant"}), ("late_binding_gate", {"update_type": "irrelevant"})]
    for i, (fam, extra) in enumerate(fams):
        tag = fam + ("" if not extra else "_" + extra["update_type"])
        if fam in ("moving_target", "static_control"):
            extra = {**extra, "generous_deadline_s": 6.0}    # the demo canvas is small; scripted oracles finish in 2 s
        ms = generate(GenConfig(task_family=fam, n=a.n, master_seed=100 + i, canvas_w_px=200, canvas_h_px=150,
                                target_size_px=16, **extra))
        ms.freeze(root / "manifests" / f"{tag}.jsonl")
        if fam == "late_binding_gate":
            comp = ms.rows[0].task_payload["gate_compute_ns"]
            ctrls = [ControllerSpec("frozen", scripted("state_at_B", comp)), ControllerSpec("late_bind", scripted("newest_at_D", comp)),
                     ControllerSpec("restart", scripted("restart_if_changed", comp))]
        elif fam == "moving_target":
            ctrls = [ControllerSpec("frozen_2s", scripted("state_at_B", 2_000_000_000)), ControllerSpec("late_bind_2s", scripted("newest_at_D", 2_000_000_000))]
        else:
            ctrls = [ControllerSpec("oracle_0.5s", scripted("state_at_B", 500_000_000)), ControllerSpec("oracle_5.2s", scripted("state_at_B", 5_200_000_000))]
        by = {}
        for c in ctrls:
            outs = run_controller(ms, c, ["live", "replay"], root / tag, quiet=True)
            by[c.name] = {k: [o for o in outs if o.condition == k] for k in ("live", "replay")}
        md = build_report(fam, by, manifest_hash=ms.frozen_hash, title=f"{tag} (demo, n = {a.n})", n_boot=400)
        (root / f"{tag}.md").write_text(md + "\n")
        print(md)
    return 0


def cmd_master(a) -> int:
    """One table over every family under runs/d1: controller | planner | detector | family | stages | SR laws | W50/v50 | hits | mean G."""
    from chronotope.distributional.metrics import deadline_curve, expected_sr_under_deadline, speed_curve, summarize_controller
    from chronotope.distributional.report import _fit, _pct3, controller_family, memory_label, planner_and_detector, runtime_label
    laws = [{"w_min_s": float(x.split(",")[0]), "w_max_s": float(x.split(",")[1]), "empirical_windows_s": []} for x in (a.laws or ["1,16", "1,64"])]
    L = ["# D1 master table — every (controller, planner, detector) row on every family", "",
         "Median E→O / O→B / B→D / D→A / E→A in ms (harness clock). SR under F_W is computed from each row's E→A samples under the stated log-uniform deadline "
         "law (closing-window families only); W50 / v50 are law-free (the deadline / speed at which the fit crosses one half); hits = observed Live successes "
         "at the manifest's drawn deadlines; mean G = the task return of Section 3.2. Rows with an emulated deliberation are excluded.", "",
         "| controller | runtime | planner | detector | memory | family | E→O | O→B | B→D | D→A | E→A p50/p95 | " + " | ".join(f"SR F_W {d['w_min_s']:g}–{d['w_max_s']:g} s" for d in laws) +
         " | W50 (s) / v50 (px/s) | hits | mean G | run |", "|" + "---|" * (16 + len(laws))]
    root = Path(a.runs_root)
    for famdir in sorted(root.iterdir()):
        if not famdir.is_dir() or famdir.name == "manifests":
            continue
        fam_family = None
        for d in sorted(famdir.iterdir()):
            if not d.is_dir() or d.name == "payloads" or "delib1000" in d.name or "delib5200" in d.name or not (d / "outcomes.jsonl").exists():
                continue
            outs = load_outcomes(d / "outcomes.jsonl"); _fill_returns(d, outs)
            live = [o for o in outs if o.condition == "live"]
            if not live or len(live) % 10:
                continue
            fam_family = live[0].task_family
            s = summarize_controller(live, [o for o in outs if o.condition == "replay"], [o for o in outs if o.condition == "paused"])
            c = json.loads((d / "controller.json").read_text()) if (d / "controller.json").exists() else {}
            pl, det = planner_and_detector(c); fc = controller_family(c); rt = runtime_label(c); mem = memory_label(c)
            p50 = lambda k: ("-" if s[k].get("p50") is None else f"{s[k]['p50'] / 1e6:.0f}")
            graded = fam_family in ("closing_window", "late_binding_natural")
            srs = [("%.2f" % expected_sr_under_deadline(live, dd)["expected_sr"]) if graded else "-" for dd in laws]
            try:
                cv = deadline_curve(live, n_boot=100) if graded else (speed_curve(live, n_boot=100) if fam_family == "moving_target" else None)
                curve = _fit(cv, "W50") if graded else (_fit(cv, "v50") if cv else "-")
            except Exception:   # noqa: BLE001
                curve = "-"
            ea = f"{s['E_to_A_ns']['p50'] / 1e6:.0f}/{s['E_to_A_ns']['p95'] / 1e6:.0f}" if s["E_to_A_ns"].get("p50") is not None else "-"
            g = "-" if s.get("mean_G") is None else f"{s['mean_G']:.1f}"
            hits_cell = f"{s['successes']}/{s['attempts']}" + (f" (dup {s['duplicates_per_episode']:.2f}/ep)" if s.get("duplicates_per_episode") is not None else "")
            L.append(f"| {fc} | {rt} | {pl} | {det} | {mem} | {famdir.name} | {p50('E_to_O_ns')} | {p50('O_to_B_ns')} | {p50('B_to_D_ns')} | {p50('D_to_A_ns')} | {ea} | "
                     + " | ".join(srs) + f" | {curve} | {hits_cell} | {g} | {d.name} |")
    Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L)); return 0


def cmd_expanded(a) -> int:
    """The expanded table (user's 9 columns, 2026-08-28): controller implementation, planner implementation, detector
    implementation, memory implementation, fully-CUDA pipeline, controller family, planner model, detector model, and
    the scores (SR under F_W with the law stated in prose, returns, raw hits), one row per run on every family."""
    from chronotope.distributional.metrics import expected_sr_under_deadline, expected_sr_under_delay, summarize_controller
    laws = [{"w_min_s": float(x.split(",")[0]), "w_max_s": float(x.split(",")[1]), "empirical_windows_s": []} for x in (a.laws or ["1,16", "1,64"])]
    delay_law = {"delay_min_s": 0.5, "delay_max_s": 8.0}

    def impl(c):
        kind, server, planner, binder, pol = c.get("kind"), c.get("server"), c.get("planner"), c.get("binder"), c.get("at_commit_policy")
        model = str(c.get("model_id") or "")
        if kind == "chronotope":
            ctrl_impl = "CUDA/C++ (separate agent process)" if server == "cpp_socket" else "CUDA/C++ (core in the referee's process, ctypes)"
            family = "Chronotope (CUDA/C++)"
            if planner == "detector":
                pl_impl, pl_model = "CUDA kernel (color detector)", "color detector (no model)"
            elif planner == "trt":
                pl_impl, pl_model = "TensorRT Edge-LLM in-process (device frame)", model
            elif server == "edge_llm":
                pl_impl, pl_model = "TensorRT Edge-LLM in-process (Python binding, PNG)", model
            elif server == "anthropic_api":
                pl_impl, pl_model = "Anthropic Messages API (Python)", model
            else:
                pl_impl, pl_model = "vLLM server (Python)", model
            if pol == "frozen":
                det_impl, det_model = "none (proposal frozen at B)", "none"
            elif binder == "cuda":
                det_impl, det_model = "CUDA kernel (NCC template tracker)", "NCC (non-LLM)"
            elif binder == "tracker_py":
                det_impl, det_model = "host Python (NCC tracker)", "NCC (non-LLM)"
            elif binder == "detector":
                det_impl, det_model = "CUDA kernel (color/feature detector -> NCC -> original)", "color detector (non-LLM)"
            elif binder == "ncc_then_detector":
                det_impl, det_model = "CUDA kernel (NCC -> color detector -> original)", "NCC + color detector (non-LLM)"
            elif binder == "grounder":
                det_impl, det_model = pl_impl + " (re-asked at D)", model + " (itself)"
            else:
                det_impl, det_model = str(binder), str(binder)
            mem = "CUDA matrix clock over E/O/B/D/A (device rings)" + (" + pending record" if c.get("memory") == "harness_pending" else "")                   + (f" + staleness gate {int(c.get('max_evidence_age_ns') or 0) / 1e9:g} s" if (c.get("max_evidence_age_ns") or 0) > 0 else "")                   + (" + ack-aware retry" if c.get("ack_aware_pending") else "")
            if c.get("memory") != "harness_pending" and not (c.get("max_evidence_age_ns") or 0):
                mem += " (recorded; no commit rule reads it here)"
            fully = "yes" if (server == "cpp_socket" and planner in ("detector", "trt") and binder != "grounder") else \
                    ("model path via Python binding" if server == "edge_llm" else ("no (planner over the API)" if server == "anthropic_api" else "core yes; model served"))
        elif server == "agents3_service":
            ctrl_impl, family = "Python (Agent S3 loop, gui_agents)", "Agent S3 (Python)"
            pl_impl, pl_model = ("vLLM server (Python)" if "edgellm" not in model.lower() else "TensorRT Edge-LLM via shim"), model.split("+")[0]
            det_impl, det_model = "its grounder call (served model)", (model.split("+")[1] if "+" in model else "its grounder")
            mem, fully = "Agent S3 prompt history (messages + screenshots + reflection)", "no"
        elif kind == "scripted_oracle":
            ctrl_impl, family, pl_impl, pl_model, det_impl, det_model, mem, fully = "host Python (scripted)", "oracle", "scripted", "none", "oracle", "none", "none", "no"
        else:
            ctrl_impl, family = "host Python (Chronotope late-binding controller)", "Chronotope (Python)"
            pl_impl = {"anthropic_api": "Anthropic Messages API (Python)", "edge_llm": "TensorRT Edge-LLM in-process (Python binding)"}.get(server, "vLLM server (Python)")
            pl_model = model
            if kind == "vlm_direct" or pol == "frozen":
                det_impl, det_model = "none (proposal frozen at B)", "none"
            elif kind == "vlm_late_binding":
                det_impl, det_model = "host Python (NCC tracker)", "NCC (non-LLM)"
            elif kind == "vlm_value_binding":
                det_impl, det_model = "host Python (value decoder)", "decoder (non-LLM)"
            else:
                det_impl, det_model = kind, kind
            m = c.get("memory", "none")
            mem = {"none": "none (memoryless)", "harness_pending": "host pending record", "model_informed": "messages history (last action in context)"}.get(m, str(m))
            fully = "no"
        return ctrl_impl, pl_impl, det_impl, mem, fully, family, pl_model, det_model

    L = ["# D1 expanded table — every run, every family (2026-08-28)", "",
         "**How to read the scores.** *Hits* is the raw count of Live episodes whose action was accepted at receipt (10 or 100 drawn episodes). "
         "*SR under F_W* is the graded score for the closing-window task: each episode's deadline W is drawn from a law F_W; the agent succeeds "
         "when its end-to-end latency E→A is shorter than W, so from an agent's measured E→A samples one computes the probability that a fresh "
         "deadline drawn from F_W would still be open — averaged over episodes. Two laws are reported: log-uniform on 1–16 s (the manifest's) "
         "and log-uniform on 1–64 s (wide enough to contain every agent's latency), because the value depends on the law while the ordering does "
         "not. For the pending-effect task the analogous score uses the delay law (log-uniform 0.5–8 s) and the agent's re-click gap. *Ḡ* is the "
         "mean task return (0 = perfect; negative = pixels of miss or seconds late, as the task defines).", "",
         "| controller impl. | planner impl. | detector impl. | memory impl. | fully CUDA/C++ pipeline | controller family | planner model | detector model | task | n | E→O | O→B | B→D | D→A | E→A p50 (ms) | SR under F_W 1–16 s | SR under F_W 1–64 s | SR under F_delay | hits | Ḡ | run |",
         "|" + "---|" * 21]
    root = Path(a.runs_root)
    for famdir in sorted(root.iterdir()):
        if not famdir.is_dir() or famdir.name == "manifests":
            continue
        for d in sorted(famdir.iterdir()):
            if not d.is_dir() or d.name == "payloads" or "delib1000" in d.name or "delib5200" in d.name or not (d / "outcomes.jsonl").exists():
                continue
            outs = load_outcomes(d / "outcomes.jsonl"); _fill_returns(d, outs)
            live = [o for o in outs if o.condition == "live"]
            if not live or len(live) % 10:
                continue
            fam = live[0].task_family
            s = summarize_controller(live, [o for o in outs if o.condition == "replay"], [o for o in outs if o.condition == "paused"])
            c = json.loads((d / "controller.json").read_text()) if (d / "controller.json").exists() else {}
            ci, pi, di, mi, fully, famc, pm, dm = impl(c)
            p50 = lambda k: ("-" if s[k].get("p50") is None else f"{s[k]['p50'] / 1e6:.0f}")
            graded = fam in ("closing_window", "late_binding_natural")
            srs = [("%.2f" % expected_sr_under_deadline(live, dd)["expected_sr"]) if graded else "-" for dd in laws]
            srd = ("%.2f" % expected_sr_under_delay(live, delay_law)["expected_sr"]) if fam == "pending_effect" else "-"
            g = "-" if s.get("mean_G") is None else f"{s['mean_G']:.1f}"
            task = {"closing_window": "card (deadline 1–16 s)", "closing_window_wide": "card (deadline 1–64 s)", "moving_target": "balloon 1–16 px/s",
                    "moving_target_fast": "balloon 50–400 px/s", "static_control": "static balloon", "pending_effect": "pending effect (delay 0.5–8 s)"}.get(famdir.name, famdir.name)
            L.append(f"| {ci} | {pi} | {di} | {mi} | {fully} | {famc} | {pm} | {dm} | {task} | {len(live)} | {p50('E_to_O_ns')} | {p50('O_to_B_ns')} | {p50('B_to_D_ns')} | {p50('D_to_A_ns')} | {p50('E_to_A_ns')} | "
                     + " | ".join(srs) + f" | {srd} | {s['successes']}/{s['attempts']} | {g} | {d.name} |")
    Path(a.out).write_text("\n".join(L) + "\n"); print(len(L) - 8, "rows ->", a.out); return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--family", required=True); g.add_argument("--n", type=int, default=N_MIN)
    g.add_argument("--seed", type=int, required=True); g.add_argument("--out", type=Path, required=True)
    g.add_argument("--update-type", default="relevant"); g.add_argument("--seed-base", type=int, default=5_000_000)
    g.add_argument("--generous-deadline-s", type=float, default=16.0); g.add_argument("--gate-compute-s", type=float, default=2.0)
    g.add_argument("--empirical-windows", type=Path, default=None, help="JSON list of real window durations (s); W = F^-1(u)")
    g.add_argument("--canvas", default=None, help="WxH (default 320x240; 640x480 for the moving families)")
    g.add_argument("--deadline-range", default=None, help="lo,hi seconds for the closing-window deadline law (default 1,16)")
    g.add_argument("--delay-range", default=None, help="lo,hi seconds for the pending-effect delay law (default 0.5,8)")
    g.add_argument("--speed-range", default=None, help="vmin,vmax px/s for the moving families (default 1,16); e.g. 50,400 to find the loop's own ceiling")
    r = sub.add_parser("run"); r.add_argument("--manifest", type=Path, required=True); r.add_argument("--name", required=True)
    x = r.add_mutually_exclusive_group(required=True)
    x.add_argument("--controller-json", type=Path); x.add_argument("--scripted", help="binding[:compute_ns]")
    r.add_argument("--conditions", default="live,replay"); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--repeats", type=int, default=1, help="runtime-jitter repeats of each row (never new episodes)")
    r.add_argument("--max-calls", type=int, default=2000); r.add_argument("--allow-small", action="store_true")
    p = sub.add_parser("report"); p.add_argument("--family", required=True); p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--baseline", default=None); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference", nargs="*", default=None, help="name=text: the same agent's fixed-grid SR, printed as a column")
    p.add_argument("--manifest", type=Path, default=None, help="the frozen manifest: its deadline law gives the graded SR column")
    p.add_argument("--extra-law", nargs="*", default=None, help="lo,hi: additional log-uniform deadline laws (s) to score the same E→A samples under, e.g. 1,64")
    n = sub.add_parser("negative-controls"); n.add_argument("--late", required=True); n.add_argument("--frozen", required=True)
    n.add_argument("--strata", nargs="+", required=True, help="name=run_root ..."); n.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", type=Path, required=True); d.add_argument("--n", type=int, default=12)
    m = sub.add_parser("master"); m.add_argument("--runs-root", default="runs/d1"); m.add_argument("--out", type=Path, required=True)
    m.add_argument("--laws", nargs="*", default=None, help="log-uniform deadline laws lo,hi (default 1,16 and 1,64)")
    x = sub.add_parser("expanded"); x.add_argument("--runs-root", default="runs/d1"); x.add_argument("--out", type=Path, required=True)
    x.add_argument("--laws", nargs="*", default=None)
    a = ap.parse_args()
    return {"generate": cmd_generate, "run": cmd_run, "report": cmd_report, "negative-controls": cmd_negative, "demo": cmd_demo, "master": cmd_master, "expanded": cmd_expanded}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
