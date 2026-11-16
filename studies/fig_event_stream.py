"""One recorded episode as the five event streams of the resident-core figure, drawn from its trace.

Every mark is a stamped record: E from `event`, O from `observation` (the admitted ones from
`admission`, the dropped ones from `observation_superseded`), B from `decision_begin`, D from
`commit`, A from `receipt`. The arrows are provenance read out of the records' ids, not drawn
by hand: o <- e via source_event_id, b <- o via decision_begin.observation_id, d <- b via
decision_id, d <- o via the binding frame named in coordinate_correction, a <- d via action_id.
The intervals printed under the streams are this episode's EO(B)DA measurements, and they are
also written next to the figure as JSON so prose can cite them without retyping.

    PYTHONPATH=src .venv/bin/python studies/fig_event_stream.py [trace.jsonl ...]

With no argument it draws the paper's running example, Balloon Drift 16 px/s seed 1043, Live,
late binding, from runs/c1_qwen38_nothink_100.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.trace import Trace  # noqa: E402

OUT = ROOT / "paper_chronotope_iclr2027" / "figures"
DEFAULT = ROOT / "runs" / "c1_qwen38_nothink_100" / "live_v16_late_bind_s1043.jsonl"

# the inset's palette: ink for marks and labels, grid for the stream lines, orange for provenance
INK, MUTED, GRID, ORANGE, RED = "#1a1a1a", "#5a5a5a", "#c8c8c2", "#b45f06", "#d62828"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix"})

ROWS = [("E", "environment"), ("O", "observation"), ("B", "decision begins"),
        ("D", "commit"), ("A", "receipt")]
Y = {k: 4 - i for i, (k, _) in enumerate(ROWS)}


def _fmt(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    return f"{seconds:.3f} s"


def read_streams(trace: Trace) -> dict:
    """Pull the five streams and their provenance out of one trace, in seconds from trial start."""
    origin = trace.header()["harness_origin_ns"]
    t = lambda ns: (ns - origin) / 1e9  # noqa: E731

    events = [{"id": e["event_id"], "t": t(e["harness_time_ns"]), "kind": e["kind"]}
              for e in trace.all("event")]
    obs = {o["observation_id"]: {"t": t(o["publication_time_ns"]), "event": o["source_event_id"]}
           for o in trace.all("observation")}
    admitted = {a["observation_id"]: a["relevance"] for a in trace.all("admission")}
    superseded = {s["observation_id"] for s in trace.all("observation_superseded")}

    decisions = []
    for b in trace.all("decision_begin"):
        d = {"id": b["decision_id"], "b": t(b["begin_time_ns"]), "support": b["observation_id"]}
        inv = next((i for i in trace.all("decision_invocation") if i["decision_id"] == d["id"]), None)
        if inv is not None:
            d["model_end"] = t(inv["end_time_ns"])
            d["model"] = inv.get("model_reported")
        cor = next((c for c in trace.all("coordinate_correction") if c["decision_id"] == d["id"]), None)
        if cor is not None:
            d["bind_frame"] = cor["observation_id"]
            d["binder"] = (t(cor["binder_start_ns"]), t(cor["binder_end_ns"]))
            d["binding"] = cor["binding"]
        com = next((c for c in trace.all("commit") if c["decision_id"] == d["id"]), None)
        if com is not None:
            d["d"] = t(com["commitment_time_ns"])
            d["action_id"] = com["action_id"]
            rec = next((r for r in trace.all("receipt") if r["action_id"] == com["action_id"]), None)
            if rec is not None:
                d["a"] = t(rec["receipt_time_ns"])
                d["inside"] = rec.get("click_inside_target")
        decisions.append(d)
    return {"events": events, "obs": obs, "admitted": admitted, "superseded": superseded,
            "decisions": decisions, "end": t(trace.all("trial_end")[-1]["harness_time_ns"])
            if trace.all("trial_end") else None}


def intervals(s: dict) -> list[dict]:
    """The EO(B)DA measurements of each committed decision, every one a difference of two stamps."""
    out = []
    for d in s["decisions"]:
        if "d" not in d:
            continue
        sup = s["obs"][d["support"]]
        ev = next((e for e in s["events"] if e["id"] == sup["event"]), None)
        first_after = min((o for o in s["obs"].values() if o["event"] == sup["event"]),
                          key=lambda o: o["t"]) if ev else None
        row = {"decision": d["id"],
               "E_to_O_s": first_after["t"] - ev["t"] if ev else None,
               "O_to_B_s": d["b"] - sup["t"],
               "B_to_D_s": d["d"] - d["b"],
               "model_s": d["model_end"] - d["b"] if "model_end" in d else None,
               "binder_s": d["binder"][1] - d["binder"][0] if "binder" in d else None,
               "D_to_A_s": d["a"] - d["d"] if "a" in d else None,
               "E_to_A_s": d["a"] - ev["t"] if ev and "a" in d else None,
               "supporting_frame": d["support"], "binding_frame": d.get("bind_frame"),
               "frames_published_during_B_to_D": sum(1 for o in s["obs"].values()
                                                     if d["b"] < o["t"] <= d["d"]),
               "frames_superseded": len(s["superseded"])}
        out.append(row)
    return out


def draw(trace: Trace, out_png: Path) -> list[dict]:
    s = read_streams(trace)
    iv = intervals(s)
    t_max = (s["end"] or max(o["t"] for o in s["obs"].values())) + 0.15

    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=300)
    for k, label in ROWS:
        ax.axhline(Y[k], color=GRID, lw=1.0, zorder=1)
        ax.text(-0.02, Y[k], f"{k}  ", ha="right", va="center", fontsize=8.5, color=INK,
                weight="bold", transform=ax.get_yaxis_transform())
        ax.text(-0.02, Y[k] - 0.3, label, ha="right", va="center", fontsize=6.6, color=MUTED,
                transform=ax.get_yaxis_transform())

    # O: every published frame is a tick; dropped ones faint, admitted ones dark
    for oid, o in s["obs"].items():
        if oid in s["superseded"]:
            ax.plot([o["t"], o["t"]], [Y["O"] - 0.10, Y["O"] + 0.10], color="#d9d9d4", lw=0.5, zorder=2)
        elif oid in s["admitted"]:
            ax.plot([o["t"], o["t"]], [Y["O"] - 0.16, Y["O"] + 0.16], color=MUTED, lw=0.7, zorder=3)

    def dot(x, row, name, color=INK, side="right"):
        # labels sit beside the mark, on the side the provenance arrows do not use
        ax.plot(x, Y[row], "o", ms=4.2, color=color, zorder=6)
        dx = 0.035 * t_max
        ax.text(x + (dx if side == "right" else -dx), Y[row] + 0.16, name,
                ha="left" if side == "right" else "right", va="bottom", fontsize=7.2,
                color=color, zorder=7)

    def arrow(x0, r0, x1, r1):
        ax.annotate("", xy=(x1, Y[r1]), xytext=(x0, Y[r0]), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.1, shrinkA=3, shrinkB=3))

    for e in s["events"]:
        dot(e["t"], "E", f"e$_{{{e['id']}}}$ {e['kind'].replace('_', ' ')}")
    for d in s["decisions"]:
        sup = s["obs"][d["support"]]
        ev = next((e for e in s["events"] if e["id"] == sup["event"]), None)
        dot(sup["t"], "O", f"o$_{{{d['support']}}}$", side="left")
        if ev:
            arrow(ev["t"], "E", sup["t"], "O")
        dot(d["b"], "B", f"b$_{{{d['id']}}}$", side="left")
        arrow(sup["t"], "O", d["b"], "B")
        if "d" not in d:
            continue
        dot(d["d"], "D", f"d$_{{{d['id']}}}$")
        arrow(d["b"], "B", d["d"], "D")
        if "bind_frame" in d and d["bind_frame"] in s["obs"]:
            bf = s["obs"][d["bind_frame"]]
            dot(bf["t"], "O", f"o$_{{{d['bind_frame']}}}$ newest at D", color=ORANGE)
            arrow(bf["t"], "O", d["d"], "D")
        if "a" in d:
            dot(d["a"], "A", f"a$_{{{d['id']}}}$", color=INK if d.get("inside") else RED)
            arrow(d["d"], "D", d["a"], "A")
        if "model_end" in d:  # the deliberation, as a bar under B from b to the proposal
            ax.plot([d["b"], d["model_end"]], [Y["B"] - 0.30, Y["B"] - 0.30], color=GRID, lw=3, zorder=2)
            ax.text(d["b"] + 0.02 * t_max, Y["B"] - 0.42,
                    f"model deliberates {_fmt(d['model_end'] - d['b'])}"
                    + (f", then binds in {_fmt(d['binder'][1] - d['binder'][0])}" if "binder" in d else ""),
                    ha="left", va="top", fontsize=6.4, color=MUTED)

    ax.set_xlim(-0.05, t_max)
    ax.set_ylim(-0.75, 4.75)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=7)
    ax.set_xlabel("harness time since trial start (s)", fontsize=7.5, color=MUTED)

    n_pub = len(s["obs"])
    lines = []
    for r in iv:
        lines.append(
            f"decision {r['decision']}:  E→O {_fmt(r['E_to_O_s'])}   O→B {_fmt(r['O_to_B_s'])}   "
            f"B→D {_fmt(r['B_to_D_s'])}   D→A {_fmt(r['D_to_A_s'])}   E→A {_fmt(r['E_to_A_s'])};  "
            f"{r['frames_published_during_B_to_D']} frames published between B and D, "
            f"{n_pub} in the episode, {r['frames_superseded']} never admitted")
    fig.text(0.01, 0.01, "\n".join(lines), ha="left", va="bottom", fontsize=6.3, color=INK,
             linespacing=1.3)
    fig.text(0.01, 0.985,
             f"{trace.path.stem}: every mark is a stamped record; arrows are the ids each record "
             f"was computed from", ha="left", va="top", fontsize=7.4, color=INK, weight="bold")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.36 if len(lines) == 1 else 0.44)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)
    return iv


def main(argv: list[str]) -> int:
    paths = [Path(p) for p in argv] or [DEFAULT]
    for p in paths:
        trace = Trace.load(p)
        out_png = OUT / f"event_stream_{p.stem}.png"
        iv = draw(trace, out_png)
        (OUT / f"event_stream_{p.stem}.json").write_text(json.dumps(
            {"trace": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
             "trace_hash": trace.file_hash(), "intervals": iv}, indent=1))
        print(out_png)
        for r in iv:
            print("  " + "  ".join(f"{k}={_fmt(v) if isinstance(v, float) else v}"
                                  for k, v in r.items() if v is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
