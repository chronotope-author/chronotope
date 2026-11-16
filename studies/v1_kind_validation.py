"""V1 (cards/V1-action-kind-validation.md): validate the action KIND before D. Model cells only (the 27B model); the
scripted precision checks are unit tests (tests/test_kind_validation.py).

  PYTHONPATH=src .venv/bin/python studies/v1_kind_validation.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --n-seeds 100 --out runs/v1_qwen38_nothink_100 --max-calls 3000

Cells (Live unless stated; 320x240, target 24 px, E 0.5 s, 60 Hz, publication 30 ms, delivery 20 ms):
  player   sponsor_skip on the animated player skin, seek 1 s, windows 1, 8, 16 s: Paused / Replay / Live triplets of the
           VALIDATE arm (vlm_late_binding, frozen, validate_kind, prompt 9; the same call as A1's vlm_direct). The
           RECORDED arm is A1's run (runs/a1_qwen38_nothink_100, same seeds 200..): analyze_v1.py reads it.
  code     rotating_code with kind_change (the code expires at E + 2 s: panel and field replaced by a dismiss button),
           window 16 s, prompt 13 (type / click / wait), vlm_value_binding x {late_bind, validate (late_bind +
           validate_kind), restart_if_changed}; seeds 1000.. (C4's). Gate: Paused on the static code with prompt 13.
  controls player_static window 8 s and the static notification (prompt 4) window 8 s, validate arm, Live: the
           validator must record 0 discards (precision, U90).
Files: prl_w<w>_s<seed>_<policy>.jsonl (player), live_code_kc2_<policy>_s<seed>.jsonl, live_ctrl_<skin>_s<seed>.jsonl.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, fmt, late_binding, scenario, spec, vlm  # noqa: E402

WINDOWS_S = [1, 8, 16]
SEEK_NS = 1 * S


def validate_click(a, prompt: int) -> dict:
    return {**late_binding(a, prompt, "frozen"), "validate_kind": True}


def value_ctrl(a, policy: str) -> dict:
    c = vlm(a, "vlm_value_binding", 13, at_commit_policy="late_bind" if policy == "validate" else policy)
    if policy == "validate":
        c["validate_kind"] = True
    return c


def discards(t) -> int:
    return sum(1 for e in t.all("pending_event") if e.get("event") == "proposal_discarded")


def main() -> int:
    ap = agent_argparser("V1 action-kind validation: player windows (validate arm), rotating code kind_change, controls", seed_start=200)
    ap.add_argument("--windows", default=",".join(map(str, WINDOWS_S)))
    ap.add_argument("--cells", default="player,code,controls")
    ap.add_argument("--code-seed-start", type=int, default=1000)
    a = ap.parse_args()
    st = Study(a, "V1 action-kind validation")
    cells = a.cells.split(",")
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    if "code" in cells:
        calib = list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib))
        if not st.gate(lambda seed: spec(scenario("rotating_code", seed, "notification", 16 * S, change_delay_ns=30 * S, code_digits=4),
                                         value_ctrl(a, "late_bind"), policy="paused"), calib, "static code, prompt 13 (mixed grammar)"):
            st.finish(); return 1
    if "player" in cells:
        st.say(f"## Player, validate arm (seeds {seeds[0]}..{seeds[-1]}; seek 1 s); the recorded arm is A1's run")
        for w in [int(x) for x in a.windows.split(",")]:
            for seed in seeds:
                out = st.triplet(spec(scenario("sponsor_skip", seed, "player", w * S, seek_delay_ns=SEEK_NS), validate_click(a, 9)), f"w{w}", seed)
    if "code" in cells:
        code_seeds = list(range(a.code_seed_start, a.code_seed_start + a.n_seeds))
        for pol in ("late_bind", "validate", "restart_if_changed"):
            st.say(f"## Rotating code, kind_change at 2 s, {pol}")
            for seed in code_seeds:
                sc = scenario("rotating_code", seed, "notification", 16 * S, change_delay_ns=2 * S, code_digits=4, kind_change=True)
                r, t = st.trial(spec(sc, value_ctrl(a, pol)), f"live_code_kc2_{pol}_s{seed}")
                c = t.first("commit")
                st.say(f"- s{seed}: valid={r.valid_at_receipt} G={fmt(r.return_G)} committed={(c or {}).get('action')} discards={discards(t)} "
                       f"restarts={r.n_restarts} B→D={fmt((r.b_to_d_ns or 0)/1e9)}s firstB→D={fmt((r.first_b_to_d_ns or 0)/1e9)}s "
                       f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    if "controls" in cells:
        for skin, sc_id, prompt, extra in (("player_static", "sponsor_skip", 9, {"seek_delay_ns": SEEK_NS}),
                                            ("notification", "revealed_target", 4, {})):
            st.say(f"## Control: {skin}, window 8 s, validate arm (expect 0 discards)")
            for seed in seeds:
                r, t = st.trial(spec(scenario(sc_id, seed, skin, 8 * S, **extra), validate_click(a, prompt)), f"live_ctrl_{skin}_s{seed}")
                st.say(f"- s{seed}: valid={r.valid_at_receipt} G={fmt(r.return_G)} discards={discards(t)} labels={','.join(r.labels) or '-'}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.finish()
    subprocess.run([sys.executable, "studies/analyze_v1.py", str(a.out), f"results/V1_{a.out.name.replace('v1_', '')}.md"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
