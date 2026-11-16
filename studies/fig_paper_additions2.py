"""Figures added 2026-08-24 (second batch; all from recorded traces or final result files; PNG only).

main paper
  figures/c3_distractor.png     C3 seed 1010: frame at B and screen at A; the initial coordinate lands on a crossing
                                distractor, the late-bound click from the same decision hits the named balloon
appendix
  figures/wi001_label_phases.png oracle compute x delivery grid: which failure label the closing produces
  figures/p1_delay.png           P1 pending button, 100 seeds: duplicates/trial and exact success vs effect delay
  figures/o1_return_matrix.png   O1 remedy matrix as mean return G (px), the partial credit the success matrix hides

    .venv/bin/python studies/fig_paper_additions2.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.scenario import manifest_from_dict  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

OUT = ROOT / "paper_chronotope_iclr2027" / "figures"
AMBER, GREEN, RED, GRAY, BLUE = "#b45309", "#0f7a34", "#b91c1c", "#6b7280", "#1d4ed8"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6})


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e5e7eb", lw=0.5)
    ax.set_axisbelow(True)


def crosshair(ax, x, y, color, r=9.0, lw=1.4):
    ax.plot([x - r, x + r], [y, y], color=color, lw=lw, zorder=5)
    ax.plot([x, x], [y - r, y + r], color=color, lw=lw, zorder=5)
    ax.add_patch(Circle((x, y), r * 0.55, fill=False, edgecolor=color, lw=lw, zorder=5))


# ---------------------------------------------------------------- 1. C3 distractor hit
def fig_c3(seed=1010):
    t = Trace.load(ROOT / "runs/c3_qwen38_nothink_100" / f"live_v8_late_bind_s{seed}.jsonl")
    h = t.header(); m = manifest_from_dict(h["manifest"]); h0 = h["harness_origin_ns"]
    dec = t.committing_decision(); c = t.first("commit"); r = t.first("receipt"); corr = t.first("coordinate_correction")
    obs = {o["observation_id"]: o for o in t.all("observation")}
    tB = obs[dec["observation_id"]]["source_time_ns"]; tA = r["task_time_at_receipt_ns"]
    prop, act = corr["proposed_action"], c["action"]
    B = (dec["begin_time_ns"] - h0) / 1e9; D = (c["commitment_time_ns"] - h0) / 1e9; A = (r["receipt_time_ns"] - h0) / 1e9
    hit_frozen = m.clicked_balloon(tA, prop); hit_bound = m.clicked_balloon(tA, act)

    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.55), dpi=300)
    for ax, tt, title in ((axs[0], tB, f"frame at $B$ ({B:.1f} s): instruction “click the {m.named_color} balloon”"),
                          (axs[1], tA, f"screen at $A$ ({A:.1f} s; $D$ = {D:.1f} s)")):
        fr = np.frombuffer(m.render(tt), dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
        ax.imshow(fr, extent=[0, m.canvas_w_px, m.canvas_h_px, 0], interpolation="nearest")
        ax.set_title(title, fontsize=7.2, pad=3)
        ax.set_xlim(0, m.canvas_w_px); ax.set_ylim(m.canvas_h_px, 0); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999999")
    # left: the proposal on the frame it came from
    crosshair(axs[0], prop["x"], prop["y"], AMBER)
    axs[0].annotate("model's proposal:\non the named balloon", (prop["x"] - 16, prop["y"] + 4), color=AMBER,
                    fontsize=6.3, ha="right", va="top")
    # right: both clicks at A
    crosshair(axs[1], prop["x"], prop["y"], AMBER); crosshair(axs[1], act["x"], act["y"], GREEN)
    axs[1].annotate(f"initial coordinate:\nlands on the {hit_frozen} balloon\n(scored as a wrong object)",
                    (prop["x"] - 30, prop["y"] + 44), color=AMBER, fontsize=6.3, ha="center", va="top")
    axs[1].annotate(f"late-bound, same decision:\nhits {hit_bound}", (act["x"] + 6, act["y"] + 20), color=GREEN,
                    fontsize=6.3, ha="left", va="top")
    named_B = dict(m.balloons_at(tB))[m.named_color].center(); named_A = dict(m.balloons_at(tA))[m.named_color].center()
    axs[1].add_patch(FancyArrowPatch((named_B[0], named_B[1] - 26), (named_A[0], named_A[1] - 26), arrowstyle="-|>",
                                     mutation_scale=7, color=GRAY, lw=0.8))
    axs[1].annotate(f"{m.named_color} drifted {named_A[0]-named_B[0]} px during deliberation",
                    ((named_B[0] + named_A[0]) / 2, named_B[1] - 30), color=GRAY, fontsize=6.0, ha="center", va="bottom")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "c3_distractor.png"); plt.close(fig)
    print(f"[c3] seed {seed}: named {m.named_color}; frozen hits {hit_frozen}; bound hits {hit_bound}; B={B:.2f} D={D:.2f} A={A:.2f}")


# ---------------------------------------------------------------- 2. WI-001 label phases
def fig_wi001():
    rows = []
    for line in (ROOT / "results/WI-001_oracle_sweep.txt").read_text().splitlines():
        mm = re.match(r"compute=(\d+) delivery=(\d+)\s+.*live=(\d).*labels\[live\]=(\S+)", line)
        if mm:
            rows.append((int(mm.group(1)) / 1e6, int(mm.group(2)) / 1e6, int(mm.group(3)), mm.group(4)))
    comps = sorted({r[0] for r in rows}); dels = sorted({r[1] for r in rows})
    cls = {"-": 0, "late,invalidated_after_commitment": 1, "late,missed_evidence:window_closes": 2, "late,outdated_at_commitment": 3}
    grid = np.full((len(comps), len(dels)), np.nan)
    for c, d, ok, lab in rows:
        grid[comps.index(c), dels.index(d)] = cls.get(lab, 3)
    colors = ["#dcfce7", "#fecaca", "#fde68a", "#fca5a5"]
    cmap = matplotlib.colors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(3.9, 2.6), dpi=300)
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    names = {0: "valid", 1: "invalidated\nafter $D$", 2: "missed\nevidence", 3: "outdated\nat $D$"}
    for i in range(len(comps)):
        for j in range(len(dels)):
            v = grid[i, j]
            if not np.isnan(v):
                ax.text(j, i, names[int(v)], ha="center", va="center", fontsize=5.6, color="#1f2937")
    ax.set_xticks(range(len(dels))); ax.set_xticklabels([f"{d:g}" for d in dels], fontsize=7)
    ax.set_yticks(range(len(comps))); ax.set_yticklabels([f"{c:g}" for c in comps], fontsize=7)
    ax.set_xlabel("delivery $D\\!\\rightarrow\\!A$ (ms)", fontsize=7.5); ax.set_ylabel("compute $O\\!\\rightarrow\\!D$ (ms)", fontsize=7.5)
    ax.set_title("scripted clicker; event at 500 ms, window 400 ms, $E\\!\\rightarrow\\!O$ = 30 ms", fontsize=7, pad=4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "wi001_label_phases.png"); plt.close(fig)
    print(f"[wi001] {len(rows)} cells; classes {sorted(set(r[3] for r in rows))}")


# ---------------------------------------------------------------- 3. P1 vs effect delay (100 seeds)
def fig_p1():
    md = (ROOT / "results/P1_qwen38_nothink_100.md").read_text().split("## Paired")[0]
    cells = {}
    for l in md.splitlines():
        p = [x.strip() for x in l.split("|")[1:-1]]
        if len(p) == 12 and p[0].isdigit():
            d, drop, mem = int(p[0]), p[1] == "True", p[2]
            dup = float(p[5].split()[0]); dlo, dhi = [float(x) for x in re.findall(r"[\d.]+", p[5])[1:3]]
            ex = float(p[7].split()[0]); elo, ehi = [float(x) for x in re.findall(r"[\d.]+", p[7])[1:3]]
            cells[(d, mem, drop)] = (dup, dlo, dhi, ex, elo, ehi)
    mems = [("none", AMBER, "memoryless"), ("model_informed", BLUE, "record told to the model"), ("harness_pending", GREEN, "pending-action record")]
    ds = [1, 2, 4, 8]
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.2), dpi=300)
    for k, (ax, idx, lab) in enumerate(((axs[0], 0, "duplicate clicks per trial"), (axs[1], 3, "exact success (accepted, no duplicate)"))):
        for i, (mem, col, name) in enumerate(mems):
            xs = np.array(ds, float) * (1 + 0.04 * (i - 1))
            ys = [cells[(d, mem, False)][idx] for d in ds]
            lo = [cells[(d, mem, False)][idx + 1] for d in ds]; hi = [cells[(d, mem, False)][idx + 2] for d in ds]
            ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt="o-", color=col, ms=3, lw=1, capsize=2, label=name)
        ax.set_xscale("log", base=2); ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
        ax.axvline(5.2, color=GRAY, lw=0.7, ls=(0, (3, 2)))
        ax.text(5.35, 0.62 if k == 0 else 0.42, " deliberation\n 5.2 s", fontsize=6, color=GRAY, va="top")
        ax.set_xlabel("effect delay after the accepted click (s)", fontsize=7.5); ax.set_title(lab, fontsize=7.5)
        tidy(ax)
    axs[0].set_ylim(-0.05, 1.05); axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "p1_delay.png"); plt.close(fig)
    print("[p1] " + "; ".join(f"d{d} {mem}: dup {cells[(d,mem,False)][0]:.2f} exact {cells[(d,mem,False)][3]:.2f}" for d in ds for mem, _, _ in mems))


# ---------------------------------------------------------------- 4. O1 return matrix
def fig_o1_return():
    md = (ROOT / "results/O1_remedy_matrix.md").read_text().split("## Mean return")[1].split("## Contract")[0]
    rows = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| remedy")]
    names = [r.split("|")[1].strip() for r in rows]
    vals = np.array([[float(c.strip()) for c in r.split("|")[2:7]] for r in rows])
    cols = ["transient\nevidence", "change\nbefore $D$", "closing\nwindow", "change\nin $(D,A]$", "motion +\nclosing"]
    short = {"Baseline": "no remedy", "Observation before expiry": "observe before expiry",
             "Latest relevant pre-$D$ state": "newest state before $D$", "Earlier receipt": "earlier receipt",
             "Receipt-state prediction": "predict state at $A$", "Matched combination": "matched combination"}
    fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=300)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#fecaca", "#fde68a", "#86efac", GREEN])
    ax.imshow(vals, cmap=cmap, vmin=-120, vmax=0, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i,j]:.0f}" if vals[i, j] % 1 == 0 else f"{vals[i,j]:.1f}", ha="center", va="center",
                    fontsize=7.2, color="white" if vals[i, j] > -5 else "#374151")
    ax.set_xticks(range(5)); ax.set_xticklabels(cols, fontsize=6.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([short[n] for n in names], fontsize=7)
    ax.set_xlabel("failure condition (oracle, 100 seeds per cell)", fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("mean return $G_C=-e_{\\mathrm{click}}$ at $A$ (px)", fontsize=8, pad=4)
    fig.tight_layout(); fig.savefig(OUT / "o1_return_matrix.png"); plt.close(fig)
    print("[o1 return] " + "; ".join(f"{short[n]}: {list(v)}" for n, v in zip(names, vals)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_c3(); fig_wi001(); fig_p1(); fig_o1_return()
