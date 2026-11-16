"""Main-paper C2 figure (Balloon Jump: a discrete pre-commit change), from recorded traces only.

(a) Screen at A for one held-out Live episode (jump at 2 s, seed 1043): the balloon jumps 96 px
    during the ~5.3-s deliberation; the frozen proposal (aimed at the frame at B) and the late-bound
    click (re-found at D) are scored at the same A from the same decision.
(b) Which failure the jump produces depends on where it falls among B, D, A: oracle bindings x jump
    time (results/C2_jump_oracle_100.md; 100 seeds per cell, shared B/D/A). A velocity
    extrapolator is fooled by a discrete change; only reading the newer state at D helps before D;
    nothing but the oracle helps inside (D, A]; after A everything ties.
(c) Real agent (27B Qwen), 100 seeds per cell, Wilson 95% CI: jump during deliberation (2 s) vs the
    after-A control (14 s). Restart also succeeds here (99/100) but commits at 11.2 s instead of 5.1 s.

    .venv/bin/python studies/fig_c2_paper.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.scenario import manifest_from_dict  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

RUN = ROOT / "runs" / "c2_qwen38_nothink_100"
ORACLE_MD = ROOT / "results" / "C2_jump_oracle_100.md"
OUT = ROOT / "paper_chronotope_iclr2027" / "figures"

AMBER = "#b45309"   # frozen / state_at_B
GREEN = "#0f7a34"   # late binding / newest_at_D
RED = "#b91c1c"     # restart
BLUE = "#1d4ed8"    # predicted_at_A
GRAY = "#6b7280"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})


def crosshair(ax, x, y, color, r=9.0, lw=1.4):
    ax.add_patch(Circle((x, y), r, fill=False, edgecolor=color, linewidth=lw, zorder=5))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([x + dx * (r - 3), x + dx * (r + 5)], [y + dy * (r - 3), y + dy * (r + 5)],
                color=color, linewidth=lw, zorder=5, solid_capstyle="butt")


# ---------------------------------------------------------------- (a) screen at A
def panel_screenshot(ax, seed=1043):
    t = Trace.load(RUN / f"live_jump2s_late_bind_s{seed}.jsonl")
    h = t.header()
    m = manifest_from_dict(h["manifest"])
    h0 = h["harness_origin_ns"]
    dec = t.committing_decision()
    c = t.first("commit")
    r = t.first("receipt")
    corr = t.first("coordinate_correction")
    obs = {o["observation_id"]: o for o in t.all("observation")}

    tA = r["task_time_at_receipt_ns"]
    frame = np.frombuffer(m.render(tA), dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
    ax.imshow(frame, extent=[0, m.canvas_w_px, m.canvas_h_px, 0], interpolation="nearest", zorder=1)

    px, py = corr["proposed_action"]["x"], corr["proposed_action"]["y"]
    cx, cy = c["action"]["x"], c["action"]["y"]
    bx, by = m.target_box_at(obs[dec["observation_id"]]["source_time_ns"]).center()
    frozen_err = m.spatial_error_px(tA, corr["proposed_action"])
    bound_err = m.spatial_error_px(tA, c["action"])

    ax.add_patch(Circle((bx, by), m.target_size_px / 2, fill=False, edgecolor=GRAY,
                        linewidth=1.0, linestyle=(0, (3, 2)), zorder=4))
    ax.add_patch(FancyArrowPatch((bx + 14, by - 22), (cx - 14, cy - 22), arrowstyle="-|>",
                                 mutation_scale=8, color=GRAY, linewidth=0.9, zorder=4))
    ax.annotate("jumps 96 px at 2 s\n(during deliberation)", ((bx + cx) / 2, by - 27), color=GRAY,
                fontsize=6.5, ha="center", va="bottom", zorder=6)
    crosshair(ax, px, py, AMBER)
    crosshair(ax, cx, cy, GREEN)
    ax.annotate(f"frozen (from $B$):\nmiss by {frozen_err:.0f} px", (px, py + 16), color=AMBER,
                fontsize=6.5, ha="center", va="top", zorder=6)
    ax.annotate("late-bound (at $D$): hit", (cx - 40, cy + 38), color=GREEN,
                fontsize=6.5, ha="left", va="top", zorder=6)

    B = (dec["begin_time_ns"] - h0) / 1e9
    D = (c["commitment_time_ns"] - h0) / 1e9
    A = (r["receipt_time_ns"] - h0) / 1e9
    ax.set_title(f"screen at $A$   ($B$={B:.1f} s,  jump 2.0 s,  $D$={D:.1f} s,  $A$={A:.1f} s)",
                 fontsize=7.5, pad=3)
    ax.set_xlim(0, m.canvas_w_px); ax.set_ylim(m.canvas_h_px, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999999")
    print(f"episode seed {seed}: frozen err {frozen_err:.1f}px; bound err {bound_err:.1f}px "
          f"valid={r['valid_at_receipt']}; moved_px {corr['moved_px']:.0f}; match {corr['match_score']:.2f} "
          f"in {corr['track_ns']/1e6:.0f} ms; B={B:.2f} D={D:.2f} A={A:.2f}")


# ---------------------------------------------------------------- (b) oracle phase panel
def load_oracle():
    rows = []
    for line in ORACLE_MD.read_text().splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*([^|]+)\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([^|]*)\|", line)
        if m:
            rows.append(dict(ms=int(m.group(1)), where=m.group(2).strip(),
                             vals=[float(m.group(i)) for i in (3, 4, 5, 6)], label=m.group(7).strip()))
    return rows


def panel_phase(ax):
    rows = load_oracle()
    bindings = [("state at $B$ (frozen)", AMBER), ("predicted at $A$ (extrapolate)", BLUE),
                ("newest at $D$ (late bind)", GREEN), ("true at $A$ (oracle)", "#111111")]
    order = [0, 2, 1, 3]  # columns of the md table -> display rows
    n_col = len(rows)
    for j, row in enumerate(rows):
        for i, k in enumerate(order):
            ok = row["vals"][k] >= 0.5
            face = "#dcefe0" if ok else "#f6dada"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", linewidth=1.2, zorder=1))
            ax.text(j + 0.5, i + 0.5, "hit" if ok else "miss", ha="center", va="center", fontsize=8.8,
                    color=GREEN if ok else "#9f1c1c", zorder=2)
    # phase headers and separators: B->D | (D,A] | after A
    phases = {}
    for j, row in enumerate(rows):
        key = ("before $D$" if "before D" in row["where"] else "$(D,A]$" if "(D, A]" in row["where"] else "after $A$")
        phases.setdefault(key, []).append(j)
    labels = {"before $D$": "outdated at\ncommitment", "$(D,A]$": "invalidated\nafter commit", "after $A$": "no label"}
    for key, cols in phases.items():
        x0, x1 = cols[0], cols[-1] + 1
        ax.plot([x0 + 0.05, x1 - 0.05], [-0.12, -0.12], color="#333333", linewidth=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, -0.20, f"jump {key}", ha="center", va="bottom", fontsize=9.5, clip_on=False)
        ax.text((x0 + x1) / 2, len(bindings) + 0.55, "label: " + labels[key],
                ha="center", va="top", fontsize=8.5, color="#333333", style="italic", clip_on=False)
        if x1 < n_col:
            ax.plot([x1, x1], [0, len(bindings)], color="#333333", linewidth=0.9, zorder=3)
    ax.set_xlim(0, n_col); ax.set_ylim(len(bindings), 0)
    ax.set_xticks([j + 0.5 for j in range(n_col)], [f"{r['ms']} ms" for r in rows], fontsize=8.5)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_yticks([i + 0.5 for i in range(len(bindings))], [b for b, _ in bindings], fontsize=8.8)
    for lab, (_, col) in zip(ax.get_yticklabels(), bindings):
        lab.set_color(col)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- (c) real-agent bars
def cell(jump_s: int, pol: str):
    k = n = 0; first = []
    for p in sorted(RUN.glob(f"live_jump{jump_s}s_{pol}_s*.jsonl")):
        r = evaluate_trace(Trace.load(p))
        if not r.valid_trial:
            continue
        n += 1; k += bool(r.valid_at_receipt)
        if r.first_b_to_d_ns is not None:
            first.append(r.first_b_to_d_ns / 1e9)
    return k, n, (statistics.mean(first) if first else float("nan"))


def panel_bars(ax):
    groups = [(2, "jump at 2 s\n(before $D$)"), (14, "jump at 14 s\n(after $A$, control)")]
    series = [("frozen", "frozen", AMBER), ("late_bind", "late binding", GREEN), ("restart_if_moved", "restart-on-update", RED)]
    w = 0.26
    for gi, (js, gname) in enumerate(groups):
        for si, (pol, label, color) in enumerate(series):
            k, n, first = cell(js, pol)
            if n == 0:
                continue
            p, lo, hi = wilson(k, n)
            x = gi + (si - 1) * w
            ax.bar(x, p, width=w * 0.92, color=color, zorder=3, label=label if gi == 0 else None)
            ax.errorbar(x, p, yerr=[[max(0.0, p - lo)], [max(0.0, hi - p)]], color="#222222", capsize=2, elinewidth=0.8, capthick=0.8, zorder=4)
            ax.text(x, max(p, 0.02) + 0.03, f"{k}/{n}", ha="center", va="bottom", fontsize=5.4)
            print(f"jump={js}s {pol}: {k}/{n} [{lo:.2f},{hi:.2f}] first B->D {first:.2f} s")
            if pol == "restart_if_moved":
                ax.text(x, 0.5, f"commits at {first:.1f} s\n(late bind 5.1 s)", color="white", fontsize=6.0,
                        ha="center", va="center", rotation=90, zorder=5)
    ax.set_xticks([0, 1], [g for _, g in groups], fontsize=7)
    ax.set_ylabel("clicks valid at $A$", fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.5, 1.5)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color="#e2e2de", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=7.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5, handlelength=1.2,
              ncol=3, borderaxespad=0.0, columnspacing=0.8, handletextpad=0.4)
    ax.set_title("27B Qwen, 100 seeds per cell, Wilson 95% CI", fontsize=7.5, pad=14)


def main():
    # Main-paper figure: (a) one episode, (b) the model result. The oracle jump sweep is a separate appendix figure
    # (c2_oracle.png) so that the main figure stays legible at half width (round-3 critique, Figure 3 too small).
    fig = plt.figure(figsize=(5.5, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], left=0.005, right=0.995, top=0.84, bottom=0.14, wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    panel_screenshot(ax_a)
    panel_bars(ax_c)
    for ax, tag, dx, dy in ((ax_a, "(a)", -0.01, -0.06), (ax_c, "(b)", -0.22, -0.06)):
        ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="top")
    fig.savefig(OUT / "c2_main.png", dpi=300)
    print("wrote", OUT / "c2_main.png")
    # Round-8 critique: bars alone for the main paper (larger), the episode screenshot for the appendix.
    figb = plt.figure(figsize=(3.2, 2.5))
    gsb = figb.add_gridspec(1, 1, left=0.2, right=0.98, top=0.82, bottom=0.14)
    panel_bars(figb.add_subplot(gsb[0, 0]))
    figb.savefig(OUT / "c2_bars.png", dpi=300)
    print("wrote", OUT / "c2_bars.png")
    fige = plt.figure(figsize=(3.4, 2.5))
    gse = fige.add_gridspec(1, 1, left=0.005, right=0.995, top=0.84, bottom=0.02)
    panel_screenshot(fige.add_subplot(gse[0, 0]))
    fige.savefig(OUT / "c2_episode.png", dpi=300)
    print("wrote", OUT / "c2_episode.png")
    fig2 = plt.figure(figsize=(5.2, 2.3))
    gs2 = fig2.add_gridspec(1, 1, left=0.27, right=0.995, top=0.86, bottom=0.27)
    ax_b = fig2.add_subplot(gs2[0, 0])
    panel_phase(ax_b)
    fig2.savefig(OUT / "c2_oracle.png", dpi=300)
    print("wrote", OUT / "c2_oracle.png")


if __name__ == "__main__":
    main()
