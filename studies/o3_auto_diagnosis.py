"""O3 (cards/O3-auto-diagnosis.md): from a failure trace's label to an intervention, automatically.

Part 1 (CPU, now): pool the Live failures of the 27B model from the existing runs (read-only), infer each failure's
label from EVENTS ONLY (event records, observation publication times, decision begin/end, commit and receipt records,
and the scenario's declared change process: velocity / jump / change / reversal times; never the state function), map
the label to an intervention, and emit the rerun plan: each failing seed of a baseline arm under the selected
intervention AND under one wrong intervention (rotated through the other three). The precision table (U70) compares
the events-only label with the evaluator's label (which uses the manifest's state function).

  PYTHONPATH=src .venv/bin/python studies/o3_auto_diagnosis.py --plan --runs-root /home/USER/chronotope/runs \
      --out results/O3_labels_and_plan.md --plan-json results/O3_plan.json [--per-cell 25]

Part 2 (queue, the model): run the plan.
  PYTHONPATH=src .venv/bin/python studies/o3_auto_diagnosis.py --rerun --plan-json results/O3_plan.json --model Qwen/Qwen3.8-27B \
      --base-url http://127.0.0.1:8000/v1 --no-thinking --out runs/o3_qwen38_nothink_100 --max-calls 3000
  then studies/analyze_o3.py runs/o3_qwen38_nothink_100 results/O3_auto_diagnosis_100.md

Label -> intervention (the card's map):
  outdated_at_commitment, field  (a committed click/type whose field moved or changed before D)  -> late_bind
  outdated_at_commitment, kind   (a wait committed while a newer frame already showed the target) -> re_decision (validate_kind)
  invalidated_after_commitment   (valid at D, invalid at A)                                       -> earlier_delivery (D->A 20 ms -> 0)
  late                           (A after the window closed, not otherwise explained)               -> earlier_delivery
  missed_evidence                (no evidence of the event reached O in time)                        -> earlier_observation (E->O 30 ms -> 0)
Failures with no timing label (semantic_error only; dropped) are listed but not rerun.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.spec import TrialSpec  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

INTERVENTIONS = ["late_bind", "re_decision", "earlier_delivery", "earlier_observation"]
LABEL_TO_INTERVENTION = {"outdated_field": "late_bind", "outdated_kind": "re_decision",
                         "invalidated_after_commitment": "earlier_delivery", "late": "earlier_delivery",
                         "missed_evidence": "earlier_observation"}
# the pooled runs and the file patterns of their Live traces; `baseline` marks the arms whose failures are rerun
POOL = {
    "c1": ("c1_qwen38_nothink_100", re.compile(r"live_v(\d+)_(frozen|late_bind|restart_if_moved)(_irr)?_s(\d+)\.jsonl$"), {"frozen"}),
    "c2": ("c2_qwen38_nothink_100", re.compile(r"live_jump(\d+)s_(frozen|late_bind|restart_if_moved)_s(\d+)\.jsonl$"), {"frozen"}),
    "c4": ("c4_qwen38_nothink_100", re.compile(r"live_(ch2|ch14|static)_(frozen|late_bind|restart_if_changed)_s(\d+)\.jsonl$"), {"frozen"}),
    "g1": ("g1_reflow_qwen38_nothink_100", re.compile(r"live_(ch2|ch14)_(frozen|late_bind|restart_if_moved)_s(\d+)\.jsonl$"), {"frozen"}),
    "a1": ("a1_qwen38_nothink_100", re.compile(r"prl_(w\d+)_s(\d+)_live\.jsonl$"), {"direct"}),
    "b1": ("b1_qwen38_nothink_100", re.compile(r"prl_(r[0-9a-z]+)_s(\d+)_live\.jsonl$"), {"direct"}),
}


def arm_of(study: str, m: re.Match) -> tuple[str, str, int]:
    """(cell, arm, seed) from a file name."""
    if study in ("a1", "b1"):
        return m.group(1), "direct", int(m.group(2))
    if study == "c1":
        return f"v{m.group(1)}{m.group(3) or ''}", m.group(2), int(m.group(4))
    if study == "c2":
        return f"jump{m.group(1)}s", m.group(2), int(m.group(3))
    return m.group(1), m.group(2), int(m.group(3))


# ---- events-only label inference ------------------------------------------------------------------------------------

def infer_labels(t: Trace) -> tuple[list[str], dict]:
    """Labels of a Live failure from trace records only. Returns (labels, evidence)."""
    h = t.header(); sc = h["spec"]["scenario"]; h0 = h["harness_origin_ns"]
    ev = {e["kind"]: e["task_time_ns"] for e in t.all("event")}
    t_e = sc["event_time_ns"]; t_close = t_e + sc["response_window_ns"]
    obs = t.all("observation"); by_id = {o["observation_id"]: o for o in obs}
    # declared change process (scenario parameters; not the state function)
    change_t = next((ev[k] for k in ("target_jumps", "code_changes", "code_expires", "signal_reverses") if k in ev), None)
    vx = abs(sc.get("velocity_x_px_per_s", 0)) + abs(sc.get("velocity_y_px_per_s", 0))
    commits = t.all("commit"); receipts = t.all("receipt")
    decisions = t.all("decision_begin")
    inv = {i["decision_id"]: i for i in t.all("decision_invocation")}
    labels: list[str] = []; evid: dict = {}
    seen_e = [o for o in obs if o["source_event_id"] == 1 and o["publication_time_ns"] <= h0 + t_close]
    if not seen_e:
        labels.append("missed_evidence")
    if not commits:
        # a wait committed while a newer frame (published before that decision ended) already showed the event
        for d in decisions:
            i = inv.get(d["decision_id"])
            ob = by_id.get(d["observation_id"])
            if not i or i.get("outcome") != "wait" or ob is None or ob["source_time_ns"] >= t_e:
                continue
            newer = [o for o in obs if o["source_event_id"] != 0 and o["source_time_ns"] < t_close
                     and o["source_time_ns"] > ob["source_time_ns"] and o["publication_time_ns"] < i["end_time_ns"]]
            if newer:
                labels.append("outdated_kind"); evid["wait_decision"] = d["decision_id"]
                break
        if not labels:
            labels.append("no_action")
        return labels, evid
    c = commits[0]
    if not receipts:
        return ["dropped"], evid
    r = receipts[0]
    d0 = next(d for d in decisions if d["decision_id"] == c["decision_id"])
    tau = by_id[d0["observation_id"]]["source_time_ns"]        # task time of the frame the decision used
    t_d, t_a = c["task_time_at_commit_ns"], r["task_time_at_receipt_ns"]
    corr = next((x for x in t.all("coordinate_correction") if x["decision_id"] == c["decision_id"]), None)
    bound_t = by_id[corr["observation_id"]]["source_time_ns"] if corr and corr.get("matched") else tau
    evid.update({"tau": tau, "t_D": t_d, "t_A": t_a, "change_t": change_t, "vx": vx, "bound_t": bound_t})
    inside_a = r.get("click_inside_target"); visible_a = r.get("target_visible")
    if t_a >= t_close:
        labels.append("late")
        if t_d < t_close and inside_a is not False:
            labels.append("invalidated_after_commitment")        # right content, the window closed in (D, A]
        elif t_d >= t_close:
            close_seen = any(o["source_time_ns"] >= t_close and o["publication_time_ns"] < c["commitment_time_ns"] for o in obs)
            labels.append("outdated_field" if close_seen else "missed_evidence")
        return labels, evid
    # received while the window was open, yet invalid: the content was wrong for the state at A
    if change_t is not None:
        if bound_t < change_t <= t_d:
            # the change fell between the frame the action was formed on and D: outdated only if a frame depicting
            # the changed state was published before D; otherwise the evidence was missed (the evaluator's rule)
            change_seen = any(o["source_time_ns"] >= change_t and o["publication_time_ns"] < c["commitment_time_ns"] for o in obs)
            labels.append("outdated_field" if change_seen else "missed_evidence")
        elif t_d < change_t <= t_a:
            labels.append("invalidated_after_commitment")
        elif change_t <= bound_t:
            labels.append("semantic_error")
        else:
            labels.append("semantic_error")
        return labels, evid
    if vx > 0:
        moved_pre_d = vx * (t_d - bound_t) / 1e9
        moved_post_d = vx * (t_a - t_d) / 1e9
        half = sc["target_size_px"] / 2
        if moved_pre_d > half:
            labels.append("outdated_field")
        elif moved_post_d > half:
            labels.append("invalidated_after_commitment")
        else:
            labels.append("semantic_error")
        return labels, evid
    if visible_a is False:
        labels.append("invalidated_after_commitment" if t_d < t_close else "outdated_field")
    else:
        labels.append("semantic_error")
    return labels, evid


def evaluator_class(r) -> str:
    """The evaluator's labels collapsed onto the inference vocabulary."""
    labs = set(r.labels)
    if "outdated_at_commitment" in labs:
        return "outdated_kind" if not r.action_committed else "outdated_field"
    if "invalidated_after_commitment" in labs:
        return "invalidated_after_commitment"
    if any(l.startswith("missed_evidence") for l in labs):
        return "missed_evidence"
    if "late" in labs:
        return "late"
    if "dropped" in labs:
        return "dropped"
    if "semantic_error" in labs:
        return "semantic_error"
    return "no_action" if not r.action_committed else "other"


def inferred_class(labels: list[str]) -> str:
    for k in ("outdated_kind", "outdated_field", "invalidated_after_commitment", "missed_evidence", "late", "dropped", "semantic_error", "no_action"):
        if k in labels:
            return k
    return "other"


# ---- plan ---------------------------------------------------------------------------------------------------------

def rerun_spec(header: dict, intervention: str, a) -> TrialSpec:
    d = json.loads(json.dumps(header["spec"]))
    c = d["controller"]
    c.update({"base_url": a.base_url, "model_id": a.model, "enable_thinking": bool(a.thinking),
              "reasoning_effort": a.reasoning_effort, "temperature": a.temperature, "max_tokens": a.max_tokens})
    if c["kind"] == "vlm_direct":   # A1/B1: the same call as a frozen late-binding controller
        c.update({"kind": "vlm_late_binding", "at_commit_policy": "frozen", "template_radius_px": 12, "min_match_score": 0.8,
                  "restart_threshold_px": None})
    if intervention == "late_bind":
        c["at_commit_policy"] = "late_bind"
        if c["kind"] == "vlm_late_binding":
            c["restart_threshold_px"] = None
    elif intervention == "re_decision":
        c["validate_kind"] = True
    elif intervention == "earlier_delivery":
        d["delivery"]["delivery_delay_ns"] = 0
    elif intervention == "earlier_observation":
        d["observation"]["publication_delay_ns"] = 0
    else:
        raise ValueError(intervention)
    d["task_clock_policy"] = "live"; d["replay_source_live_trace_sha256"] = None
    return TrialSpec.from_dict(d)


def make_plan(a) -> None:
    root = Path(a.runs_root)
    rows = []                       # one per Live failure
    conf = Counter()                # (inferred, evaluator)
    for study, (run, rx, baselines) in POOL.items():
        rd = root / run
        if not rd.exists():
            print(f"missing {rd}", file=sys.stderr); continue
        for p in sorted(rd.glob("*.jsonl")):
            m = rx.search(p.name)
            if not m:
                continue
            cell, arm, seed = arm_of(study, m)
            t = Trace.load(p)
            if t.header()["task_clock_policy"] != "live":
                continue
            r = evaluate_trace(t)
            if not r.valid_trial or r.valid_at_receipt:
                continue
            labs, evid = infer_labels(t)
            ic, ec = inferred_class(labs), evaluator_class(r)
            conf[(ic, ec)] += 1
            rows.append({"study": study, "run": run, "cell": cell, "arm": arm, "seed": seed, "trace": str(p),
                         "inferred": ic, "evaluator": ec, "baseline": arm in baselines,
                         "intervention": LABEL_TO_INTERVENTION.get(ic)})
    # plan: baseline-arm failures with a timing label; per (study, cell) at most --per-cell seeds; rotate the wrong one
    plan = []
    per_cell = Counter(); rot = 0
    for row in rows:
        if not row["baseline"] or row["intervention"] is None:
            continue
        key = (row["study"], row["cell"])
        if per_cell[key] >= a.per_cell:
            continue
        per_cell[key] += 1
        others = [i for i in INTERVENTIONS if i != row["intervention"]]
        wrong = others[rot % len(others)]; rot += 1
        plan.append({**row, "wrong_intervention": wrong})
    Path(a.plan_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.plan_json).write_text(json.dumps(plan, indent=1))
    # report
    classes = ["outdated_field", "outdated_kind", "invalidated_after_commitment", "late", "missed_evidence", "semantic_error", "dropped", "no_action", "other"]
    L = [f"# O3 part 1 — labels inferred from events only, and the rerun plan ({len(rows)} Live failures pooled)", "",
         "Pool: the Live traces of C1 (drift 0/4/8/16 px/s), C2 (jump 2 s / 14 s), C4 (code change 2 s / 14 s / static), G1 "
         "(reflow 2 s / 14 s), A1 (player windows 1..16 s) and B1 (reversal cells) of the 27B model; a failure is a valid "
         "trial with valid_at_receipt False. The events-only inference reads event kinds and task times, observation "
         "publication times, decision begin/end, commit and receipt records, and the scenario's declared change process "
         "(velocity, jump/change/reversal time); the evaluator additionally uses the manifest's state function. Precision "
         "of an inferred label = fraction of traces carrying it whose evaluator label is the same class.", "",
         "## Failures per cell and arm (evaluator class)", "", "| study | cell | arm | n failures | classes |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for row in rows:
        by[(row["study"], row["cell"], row["arm"])].append(row)
    for k, rs in sorted(by.items()):
        cc = Counter(r["evaluator"] for r in rs)
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(rs)} | " + ", ".join(f"{c}×{n}" for c, n in sorted(cc.items())) + " |")
    L += ["", "## U70 — precision of the events-only label (rows: inferred; columns: evaluator)", "",
          "| inferred \\ evaluator | " + " | ".join(classes) + " | precision |", "|---|" + "---|" * (len(classes) + 1)]
    for ic in classes:
        row_n = sum(conf[(ic, ec)] for ec in classes)
        if not row_n:
            continue
        L.append(f"| {ic} | " + " | ".join(str(conf[(ic, ec)]) for ec in classes) + f" | {conf[(ic, ic)] / row_n:.2f} ({conf[(ic, ic)]}/{row_n}) |")
    L += ["", "Recall per evaluator class: " + "; ".join(
        f"{ec} {conf[(ec, ec)]}/{sum(conf[(ic, ec)] for ic in classes)}" for ec in classes if sum(conf[(ic, ec)] for ic in classes)), "",
        "## Label → intervention map and the plan", "",
        "| inferred label | intervention | baseline-arm failures | planned reruns (selected + one wrong) |", "|---|---|---|---|"]
    for ic, itv in LABEL_TO_INTERVENTION.items():
        n_all = sum(1 for r in rows if r["baseline"] and r["inferred"] == ic)
        n_plan = sum(1 for r in plan if r["inferred"] == ic)
        L.append(f"| {ic} | {itv} | {n_all} | {n_plan} × 2 |")
    L += ["", f"Plan: {len(plan)} failing seeds (at most {a.per_cell} per source cell, in seed order), each rerun under its selected "
          f"intervention and under one wrong intervention rotated through the other three: {2 * len(plan)} Live trials, "
          f"written to {a.plan_json}. Interventions: late_bind = at_commit_policy late_bind (value binder on C4); re_decision = "
          "validate_kind; earlier_delivery = D→A 20 ms → 0; earlier_observation = E→O 30 ms → 0. With the 27B model's 5-s "
          "deliberation the two delay interventions have at most 20 + 30 ms of headroom, so a `late` or `invalidated` failure "
          "is expected to be repaired only when D fell within that margin of the close; the plan records this expectation.", "",
          "| study | cell | seed | arm | inferred | evaluator | selected | wrong |", "|---|---|---|---|---|---|---|---|"]
    for r in plan:
        L.append(f"| {r['study']} | {r['cell']} | {r['seed']} | {r['arm']} | {r['inferred']} | {r['evaluator']} | {r['intervention']} | {r['wrong_intervention']} |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


def run_plan(a) -> None:
    from agentlib import Study
    plan = json.loads(Path(a.plan_json).read_text())
    st = Study(a, "O3 part 2 — reruns under the selected and one wrong intervention")
    for row in plan:
        hdr = Trace.load(Path(row["trace"])).header()
        for which in ("selected", "wrong"):
            itv = row["intervention"] if which == "selected" else row["wrong_intervention"]
            name = f"o3_{row['study']}_{row['cell']}_s{row['seed']}_{row['inferred']}_{which}_{itv}"
            if (a.out / f"{name}.jsonl").exists():
                continue
            try:
                spec = rerun_spec(hdr, itv, a)
            except Exception as e:  # an intervention that this controller kind cannot take (recorded, not run)
                st.say(f"- {name}: not applicable ({e})"); continue
            r, t = st.trial(spec, name)
            st.say(f"- {name}: valid={r.valid_at_receipt} G={r.return_G:.2f} labels={','.join(r.labels) or '-'}"
                   + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.finish()


def main() -> int:
    ap = argparse.ArgumentParser(description="O3 auto-diagnosis")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true"); g.add_argument("--rerun", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan-json", type=Path, default=Path("results/O3_plan.json"))
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--model"); ap.add_argument("--base-url")
    t = ap.add_mutually_exclusive_group(); t.add_argument("--thinking", action="store_true"); t.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-calls", type=int, default=3000)
    a = ap.parse_args()
    if a.plan:
        make_plan(a)
    else:
        if not (a.model and a.base_url):
            raise SystemExit("--rerun needs --model and --base-url")
        run_plan(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
