"""Result tables for the distributional gym (request §15): one table per task family, never pooled.

  build_report(family, by_controller, *, baseline=None) -> markdown
  by_controller: {controller_name: {"live": [...], "replay": [...], "paused": [...]}} (EpisodeOutcome lists)
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from .episode import FAILURE_MODES, EpisodeOutcome, check_fairness
from .metrics import (expected_sr_under_deadline, expected_sr_under_delay, deadline_curve, negative_control_check, pairwise, slack_curve, speed_curve,
                      summarize_controller)

FAMILY_TITLES = {"pending_effect": "Pending-effect tasks (the memory axis)", "closing_window": "Closing-window tasks", "moving_target": "Moving-target tasks",
                 "static_control": "Static control (v = 0; a separate control, not pooled)",
                 "late_binding_natural": "Natural late-binding tasks (end-to-end, natural timing)",
                 "late_binding_gate": "Fixed-gate late-binding mechanism tests"}


def _ms(x: int | None) -> str:
    return "-" if x is None else f"{x / 1e6:.0f}"


def _s(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _pct3(d: dict[str, Any], keys: Sequence[str] = ("p50", "p95", "p99")) -> str:
    return "/".join(_ms(d.get(k)) for k in keys)


def _ci(v: Any) -> str:
    if not v:
        return "-"
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def _fit(cv: dict[str, Any], key: str) -> str:
    f = cv.get("fit") or {}
    v = f.get(key)
    if v is None:
        return "-"
    ci = (cv.get("ci95") or {}).get(key)
    return f"{v:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def planner_and_detector(c: dict[str, Any]) -> tuple[str, str]:
    """(planner, detector) labels of a controller block (2026-08-27 convention): the planner is the model or program
    that proposes the action at B; the detector is what resolves the click at D (a template tracker, a network, a
    small or large model), or 'none' when the proposal is committed as frozen at B."""
    kind = c.get("kind")
    model = c.get("model_id") or "-"
    if kind in ("chronotope", "kgua"):   # kgua = the kind's name before the 2026-08-27 rename (older run dirs)
        served = {"edge_llm": "Edge-LLM, in-process", "cpp_socket": "Edge-LLM in the C++ agent process, device frame",
                  "in_process": "Edge-LLM in libchronotope (ctypes), device frame"}.get(c.get("server"), "served")
        pl = {"detector": "color detector (device)", "vlm": f"{model} ({served})", "trt": f"{model} ({served})"}.get(c.get("planner"), str(c.get("planner")))
        if c.get("at_commit_policy") == "frozen":
            return pl, "none (frozen at B)"
        det = {"cuda": "NCC template (device)", "tracker_py": "NCC template (host)", "grounder": f"{model} ({served}) as grounder",
               "detector": "color/feature detector (device); NCC, then original", "ncc_then_detector": "NCC template (device); detector, then original"}.get(c.get("binder"), str(c.get("binder")))
        return pl, det
    if kind == "scripted_oracle":
        return "scripted oracle", "oracle" if c.get("binding") != "state_at_B" else "none (frozen at B)"
    if kind == "vlm_direct":
        return model, "none (frozen at B)"
    if kind == "vlm_late_binding":
        return model, "NCC template (host)" if c.get("at_commit_policy") != "frozen" else "none (frozen at B)"
    if kind == "vlm_value_binding":
        return model, "value decoder (host)" if c.get("at_commit_policy") != "frozen" else "none (frozen at B)"
    return model, str(kind)


def controller_family(c: dict[str, Any]) -> str:
    """The 'Controller' column: which agent stack ran the trial (2026-08-27, user's request): the CUDA-resident
    Chronotope core, the host Chronotope controller (the paper's late-binding controller around a served model),
    Agent S3 (its own planner/grounder loop behind studies/agents3_service.py), or the scripted oracle."""
    kind = c.get("kind")
    if kind in ("chronotope", "kgua"):   # kgua = the kind's name before the 2026-08-27 rename (older run dirs)
        return {"cpp_socket": "Chronotope (C++ process)", "in_process": "Chronotope (library, ctypes)"}.get(c.get("server"), "Chronotope (resident)")
    if c.get("server") == "agents3_service":
        return "Agent S3"
    if kind == "scripted_oracle":
        return "oracle"
    return "Chronotope (host)"


def runtime_label(c: dict[str, Any]) -> str:
    """The 'runtime' column (user's request, 2026-08-27): what executes the loop and the model. CUDA/C++ = the chronotope core
    (planner in-process as a TensorRT Edge-LLM engine, or the device color detector); 'Edge-LLM server' = a model behind
    Edge-LLM's OpenAI-compatible server (Agent S3's loop, or the host controller); 'vLLM' = the served rows."""
    kind = c.get("kind"); server = c.get("server")
    if kind in ("chronotope", "kgua"):   # kgua = the kind's name before the 2026-08-27 rename (older run dirs)
        if c.get("server") == "cpp_socket":
            return "CUDA/C++ agent process (chronotope --serve; Edge-LLM in-process, device frame)" if c.get("planner") == "trt" else "CUDA/C++ agent process (chronotope --serve)"
        return "CUDA/C++ (chronotope core; Edge-LLM via Python binding)" if c.get("planner") == "vlm" else "CUDA/C++ (chronotope core)"
    if kind == "scripted_oracle":
        return "host Python (scripted)"
    model = str(c.get("model_id") or "")
    if server == "agents3_service":
        return "Edge-LLM server (via Agent S3 service)" if ("edgellm" in model.lower() or "/engines" in model or model.count("/") == 0) else "vLLM (via Agent S3 service)"
    if server == "edge_llm":
        return "Edge-LLM in-process"
    return "vLLM (served)"


def memory_label(c: dict[str, Any]) -> str:
    """The 'memory' column (user's axis, 2026-08-27): what the agent keeps between frames and decisions and consults at
    commit. The resident core keeps a matrix clock over the five event streams (E, O, B, D, A rings in device memory:
    the event ledger and the decision trace); the pending record, the staleness gate and the ack-aware rule are the
    commit-time rules that read it. Served/host controllers: none (memoryless), a host pending record, or prompt
    history (the last actions recited to the model). Agent S3: its own prompt history and reflection."""
    kind = c.get("kind")
    if kind in ("chronotope", "kgua"):   # kgua = the kind's name before the 2026-08-27 rename (older run dirs)
        parts = ["matrix clock over E/O/B/D/A (device rings)"]
        if c.get("memory") == "harness_pending":
            parts.append("pending record")
        if (c.get("max_evidence_age_ns") or 0) > 0:
            parts.append(f"staleness gate {int(c['max_evidence_age_ns']) / 1e9:g} s")
        if c.get("ack_aware_pending"):
            parts.append("ack-aware")
        return " + ".join(parts) if len(parts) > 1 else parts[0] + "; commit rules off"
    if c.get("server") == "agents3_service":
        return "prompt history (Agent S3: last steps + reflection)"
    if kind == "scripted_oracle":
        return "none"
    m = c.get("memory", "none")
    return {"none": "none (memoryless)", "harness_pending": "host pending record", "model_informed": "prompt history (last action in context)"}.get(m, str(m))


def build_report(family: str, by_controller: dict[str, dict[str, list[EpisodeOutcome]]], *, baseline: str | None = None,
                 title: str | None = None, manifest_hash: str | None = None, n_boot: int = 2000,
                 reference: dict[str, str] | None = None, deadline_dist: dict | None = None,
                 controllers: dict[str, dict[str, Any]] | None = None, extra_laws: list[dict] | None = None) -> str:
    """reference: controller name -> text of the same agent's fixed-grid result (e.g. "0/10 · 10/10 at 1/2 s"),
    printed as a column so the distributional row can be read against the original SR."""
    names = list(by_controller)
    if not names:
        raise ValueError("no controllers")
    everything = [o for d in by_controller.values() for c in d.values() for o in c]
    check_fairness(everything)
    fams = {o.task_family for o in everything}
    if fams != {family}:
        raise ValueError(f"report for {family} received families {sorted(fams)}: families are never pooled")
    base = baseline or names[0]
    L: list[str] = [f"# {title or FAMILY_TITLES[family]}", ""]
    if manifest_hash:
        L.append(f"Frozen manifest `{manifest_hash}`; scorer `{everything[0].scorer}`; success scored at application receipt A.")
        L.append("")
    sums = {n: summarize_controller(d["live"], d.get("replay", ()), d.get("paused", ())) for n, d in by_controller.items()}

    # ---- the required per-controller table -------------------------------------------------------------------
    graded = deadline_dist is not None and family in ("closing_window", "late_binding_natural")
    pending = family == "pending_effect"
    delay_dist = deadline_dist if (pending and deadline_dist and "delay_min_s" in deadline_dist) else None
    law_name = lambda d: (f"log-uniform {d['w_min_s']:g}–{d['w_max_s']:g} s" if not d.get("empirical_windows_s") else "empirical windows")
    laws = ([deadline_dist] + list(extra_laws or [])) if graded else []
    curve_key = "W50 (s)" if family in ("closing_window", "late_binding_natural") else ("v50 (px/s)" if family == "moving_target" else None)
    curves = {}
    if curve_key:
        for n in names:
            try:
                cv = deadline_curve(by_controller[n]["live"], n_boot=max(50, n_boot // 8)) if curve_key.startswith("W") else speed_curve(by_controller[n]["live"], n_boot=max(50, n_boot // 8))
                curves[n] = _fit(cv, "W50" if curve_key.startswith("W") else "v50")
            except Exception:   # noqa: BLE001
                curves[n] = "-"
    hdr = ["run", "controller", "runtime", "planner", "detector", "memory", "E→A p50/p95/p99 (ms)"] + [f"SR under F_W ({law_name(d)})" for d in laws] + \
          (["SR under F_delay (from the re-click gap)", "duplicates / episode", "suppressed / episode"] if pending else []) + \
          ([curve_key] if curve_key else []) + ["Live SR observed", "successes/attempts", "mean G", "Wilson 95%", "semantic SR (paused/replay/live)", "Replay→Live retention",
           "Δ SR vs " + base, "Δ 95% CI", "E→O p50/p95/p99 (ms)", "O→B p50/p95/p99 (ms)", "B→D p50/p95/p99 (ms)",
           "D→A p50/p95/p99 (ms)", "slack median/p5/p95 (ms)", "x median/p95", "invalid"] + (["reference SR (fixed grid)"] if reference else [])
    if graded:
        L.append(f"Primary column: SR under F_W = mean over Live trials of P(W > E→A) with W drawn from the manifest's deadline law; a second law, when "
                 "given, scores the same E→A samples under a wider deadline law so that agents slower than the manifest's range are not scored 0 by the "
                 "choice of range; W50 (the deadline at which the agent succeeds half the time) is law-free. Law: "
                 f"({'empirical windows' if deadline_dist.get('empirical_windows_s') else f'log-uniform {deadline_dist['w_min_s']}–{deadline_dist['w_max_s']} s'}); "
                 "the observed Live SR is the same quantity at the 10 drawn deadlines (steps of 1/n).")
        L.append("")
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    pw: dict[str, dict[str, Any]] = {}
    for n in names:
        s = sums[n]
        if n != base:
            pw[n] = pairwise(by_controller[n]["live"], by_controller[base]["live"], n_boot=n_boot)
        sem = s["semantic_sr"]
        sem_s = "/".join(_s(sem.get(k)) for k in ("paused", "replay", "live"))
        ret = s.get("retention")
        ret_s = "undefined (no Replay-solvable episode)" if ("retention" in s and ret is None) else _s(ret) if ret is not None else "-"
        d = pw.get(n)
        gr = [_s(expected_sr_under_deadline(by_controller[n]["live"], d)["expected_sr"]) for d in laws]
        if pending:
            gr += [_s(expected_sr_under_delay(by_controller[n]["live"], delay_dist)["expected_sr"]) if delay_dist else "-",
                   ("-" if s.get("duplicates_per_episode") is None else f"{s['duplicates_per_episode']:.2f}"),
                   ("-" if s.get("suppressed_per_episode") is None else f"{s['suppressed_per_episode']:.2f}")]
        pl, det = planner_and_detector((controllers or {}).get(n, {}))
        fam_c = controller_family((controllers or {}).get(n, {}))
        rt = runtime_label((controllers or {}).get(n, {}))
        mem = memory_label((controllers or {}).get(n, {}))
        L.append("| " + " | ".join([
            n, fam_c, rt, pl, det, mem, _pct3(s["E_to_A_ns"])] + gr + ([curves.get(n, "-")] if curve_key else []) + [_s(s["live_sr"]), f"{s['successes']}/{s['attempts']}",
            ("-" if s.get("mean_G") is None else f"{s['mean_G']:.1f}"), _ci(s["live_sr_wilson"]), sem_s,
            ret_s, ("0 (baseline)" if n == base else f"{d['delta_sr']:+.3f}"), ("-" if n == base else _ci(d["delta_ci95"])),
            _pct3(s["E_to_O_ns"]), _pct3(s["O_to_B_ns"]), _pct3(s["B_to_D_ns"]), _pct3(s["D_to_A_ns"]),
            _pct3(s["slack_ns"], ("p50", "p5", "p95")), "/".join(_s(s["x_norm"].get(k)) for k in ("p50", "p95")),
            str(s["invalid_trials"])] + ([reference.get(n, "-")] if reference else [])) + " |")
    L.append("")

    # ---- curves -------------------------------------------------------------------------------------------------
    if family == "closing_window" or family == "late_binding_natural":
        L += ["| controller | W50 (s) [95% CI] | W90 (s) [95% CI] | α | β |", "|---|---|---|---|---|"]
        for n in names:
            cv = deadline_curve(by_controller[n]["live"], n_boot=n_boot // 4)
            L.append(f"| {n} | {_fit(cv, 'W50')} | {_fit(cv, 'W90')} | {_s(cv['alpha'])} | {_s(cv['beta'])} |")
        L.append("")
    if family == "moving_target":
        L += ["| controller | v50 (px/s) [95% CI] | α | β |", "|---|---|---|---|"]
        for n in names:
            cv = speed_curve(by_controller[n]["live"], n_boot=n_boot // 4)
            L.append(f"| {n} | {_fit(cv, 'v50')} | {_s(cv['alpha'])} | {_s(-cv['beta']) if cv['beta'] is not None else '-'} |")
        L.append("")
    if family == "late_binding_gate":
        L += ["| controller | q50 (s) [95% CI] | α | β |", "|---|---|---|---|"]
        for n in names:
            cv = slack_curve(by_controller[n]["live"], n_boot=n_boot // 4)
            L.append(f"| {n} | {_fit(cv, 'q50')} | {_s(cv['alpha'])} | {_s(cv['beta'])} |")
        L.append("")
    if family == "late_binding_natural":
        L += ["| controller | ITT Live SR | G = 1 stratum n | mechanism SR (G = 1) | updates after D |", "|---|---|---|---|---|"]
        for n in names:
            s = sums[n]
            L.append(f"| {n} | {_s(s['itt_sr'])} | {s['mechanism_stratum_n']} | {_s(s['mechanism_sr'])} | {s['update_after_D']} |")
        L.append("")

    # ---- paired outcomes and failure modes -----------------------------------------------------------------------
    if pw:
        L += [f"| pair | n | n10 | n01 | n11 | n00 | exact McNemar p | Δ SR | 95% CI |", "|---|---|---|---|---|---|---|---|---|"]
        for n, d in pw.items():
            L.append(f"| {n} − {base} | {d['n']} | {d['n10']} | {d['n01']} | {d['n11']} | {d['n00']} | {d['mcnemar_p']:.3g} | {d['delta_sr']:+.3f} | {_ci(d['delta_ci95'])} |")
        L.append("")
    L += ["| controller | " + " | ".join(FAILURE_MODES) + " |", "|---|" + "---|" * len(FAILURE_MODES)]
    for n in names:
        fm = sums[n]["failure_modes"]
        L.append(f"| {n} | " + " | ".join(str(fm[k]) for k in FAILURE_MODES) + " |")
    L.append("")
    return "\n".join(L)


def negative_controls_report(strata: dict[str, dict[str, list[EpisodeOutcome]]], *, late: str, frozen: str) -> str:
    """strata: {update_type: {controller: live outcomes}}. Late binding must not be credited on any stratum."""
    L = ["# Negative controls (fixed-gate strata)", "",
         f"| stratum | {late} SR | {frozen} SR | Δ ({late} − {frozen}) | 95% CI | late binding credited? |", "|---|---|---|---|---|---|"]
    for st, d in strata.items():
        r = negative_control_check(d[late], d[frozen])
        a = summarize_controller(d[late])["live_sr"]
        b = summarize_controller(d[frozen])["live_sr"]
        L.append(f"| {st} | {a:.2f} | {b:.2f} | {r['delta_sr']:+.3f} | {_ci(r['delta_ci95'])} | {'YES' if r['late_binding_credited'] else 'no'} |")
    L.append("")
    return "\n".join(L)


def dump_summary(by_controller: dict[str, dict[str, list[EpisodeOutcome]]]) -> str:
    return json.dumps({n: summarize_controller(d["live"], d.get("replay", ()), d.get("paused", ())) for n, d in by_controller.items()},
                      indent=2, default=str)
