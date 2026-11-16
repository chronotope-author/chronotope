"""C4 — late binding of a NON-SPATIAL field (cards/C4-late-bound-text.md). rotating_code on the notification desktop:
a 4-digit code is shown from E and replaced by a second code at E + change_delay; the action is type(value).

  PYTHONPATH=src .venv/bin/python studies/c4_rotating_code.py --model Qwen/Qwen3.8-27B --base-url http://127.0.0.1:8000/v1 \
      --no-thinking --n-seeds 100 --out runs/c4_qwen38_nothink_100 --max-calls 1500

Predeclared: 320x240, target 24 (field 60x24), event 0.5 s, window 16 s, 60 Hz, pub 30 ms, delivery 20 ms; prompt 11;
gate: Paused on the static code (change 30 s) seeds 100..109, threshold 0.80;
cells (Live, 100 held-out seeds from --seed-start): change 2 s x {frozen, late_bind, restart_if_changed};
                                                     change 14 s (after A) x {frozen, late_bind};
                                                     no change (30 s) x {frozen, late_bind}.
Files: live_<cell>_<policy>_s<seed>.jsonl; RESULTS.md rewritten per trial; results table by analyze_c4.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agentlib import S, Study, agent_argparser, fmt, mean, scenario, spec, vlm  # noqa: E402

WINDOW = 16 * S
CELLS = [("ch2", 2 * S, ["frozen", "late_bind", "restart_if_changed"]),
         ("ch14", 14 * S, ["frozen", "late_bind"]),
         ("static", 30 * S, ["frozen", "late_bind"])]


def sc(seed: int, change_ns: int) -> dict:
    return scenario("rotating_code", seed, "notification", WINDOW, change_delay_ns=change_ns, code_digits=4)


def ctrl(a, policy: str) -> dict:
    return vlm(a, "vlm_value_binding", 11, at_commit_policy=policy)


def main() -> int:
    ap = agent_argparser("C4 rotating code: frozen / late_bind / restart_if_changed on a typed value", seed_start=1000)
    a = ap.parse_args()
    st = Study(a, "C4 rotating code (typed value, prompt 11)")
    calib = list(range(a.calib_seed_start, a.calib_seed_start + a.n_calib))
    if not st.gate(lambda seed: spec(sc(seed, 30 * S), ctrl(a, "frozen"), policy="paused"), calib, "static code, frozen"):
        st.finish(); return 1
    seeds = list(range(a.seed_start, a.seed_start + a.n_seeds))
    rows = []
    for cell, change_ns, policies in CELLS:
        for pol in policies:
            st.say(f"## {cell} (change {change_ns/S:.0f} s), {pol}")
            for seed in seeds:
                r, t = st.trial(spec(sc(seed, change_ns), ctrl(a, pol)), f"live_{cell}_{pol}_s{seed}")
                c = t.first("commit"); corr = t.first("coordinate_correction")
                rows.append((cell, pol, r))
                st.say(f"- live_{cell}_{pol}_s{seed}: valid={r.valid_at_receipt} G={fmt(r.return_G)} "
                       f"typed={(c or {}).get('action', {}).get('value')} decodedD={(corr or {}).get('decoded_value')} "
                       f"counterfactual_frozen={r.proposed_valid_at_receipt} restarts={r.n_restarts} "
                       f"B→D={fmt((r.b_to_d_ns or 0)/1e9)}s firstB→D={fmt((r.first_b_to_d_ns or 0)/1e9)}s "
                       f"labels={','.join(r.labels) or '-'} viol={','.join(r.contract_violations) or '-'}"
                       + ("" if r.valid_trial else f" INVALID:{r.invalid_reason}"))
    st.say()
    st.say("| cell | policy | n | SR (valid at A) | mean G (s) | frozen counterfactual SR (same A) | restarts/trial | B→D (s) | first B→D (s) | outdated_at_commitment | semantic_error |")
    st.say("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell, change_ns, policies in CELLS:
        for pol in policies:
            rs = [r for (c, p, r) in rows if (c, p) == (cell, pol) and r.valid_trial]
            n = len(rs)
            if not n:
                st.say(f"| {cell} | {pol} | 0 | - | - | - | - | - | - | - | - |"); continue
            cf = [r.proposed_valid_at_receipt for r in rs if r.proposed_valid_at_receipt is not None]
            st.say(f"| {cell} | {pol} | {n} | {sum(bool(r.valid_at_receipt) for r in rs)/n:.2f} | {fmt(mean([r.return_G for r in rs]))} | "
                   f"{(f'{sum(cf)/len(cf):.2f}' if cf else '-')} | {mean([r.n_restarts for r in rs]):.1f} | "
                   f"{fmt(mean([(r.b_to_d_ns or 0)/1e9 for r in rs]))} | {fmt(mean([(r.first_b_to_d_ns or 0)/1e9 for r in rs]))} | "
                   f"{sum('outdated_at_commitment' in r.labels for r in rs)} | {sum('semantic_error' in r.labels for r in rs)} |")
    st.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
