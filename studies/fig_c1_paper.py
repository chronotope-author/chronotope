"""Main-paper C1 figure: one screenshot + one stats panel, from recorded traces only.

(a) Screen at A for one held-out live episode (v=8, seed 1005): the frozen proposal
    (aimed at the frame at B) and the late-bound click (re-anchored at D), scored at
    the same A from the same decision.
(b) Valid-at-A vs speed for frozen / restart_if_moved / late_bind, 100 seeds per cell,
    Wilson 95% CIs, recomputed from the run traces (matches results/C1_qwen38_nothink_100.md).

    .venv/bin/python studies/fig_c1_paper.py
"""
from __future__ import annotations

import json
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
from chronotope.stats import wilson  # noqa: E402
from chronotope.trace import Trace  # noqa: E402

RUN = ROOT / "runs" / "c1_qwen38_nothink_100"
OUT = ROOT / "paper_chronotope_iclr2027" / "figures"

AMBER = "#b45309"   # frozen proposal
GREEN = "#0f7a34"   # late-bound click (hit)
RED = "#b91c1c"     # restart
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


def panel_screenshot(ax, seed=1043, v=16):
    t = Trace.load(RUN / f"live_v{v}_late_bind_s{seed}.jsonl")
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
    frozen_ok = m.action_valid_at(tA, corr["proposed_action"])

    # where the balloon was on the frame the model deliberated on (B)
    ax.add_patch(Circle((bx, by), m.target_size_px / 2, fill=False, edgecolor=GRAY,
                        linewidth=1.0, linestyle=(0, (3, 2)), zorder=4))
    ax.add_patch(FancyArrowPatch((bx + 16, by - 22), (cx - 4, cy - 22), arrowstyle="-|>",
                                 mutation_scale=8, color=GRAY, linewidth=0.9, zorder=4))
    ax.annotate("drift during\ndeliberation", ((bx + cx) / 2 + 5, by - 27), color=GRAY,
                fontsize=6.5, ha="center", va="bottom", zorder=6)

    crosshair(ax, px, py, AMBER)
    crosshair(ax, cx, cy, GREEN)
    ax.annotate(f"frozen (from $B$):\nmiss by {frozen_err:.0f} px", (px, py + 16), color=AMBER,
                fontsize=6.5, ha="center", va="top", zorder=6)
    ax.annotate("late-bound (at $D$):\nhit", (cx + 16, cy + 4), color=GREEN,
                fontsize=6.5, ha="left", va="top", zorder=6)

    B = (dec["begin_time_ns"] - h0) / 1e9
    D = (c["commitment_time_ns"] - h0) / 1e9
    A = (r["receipt_time_ns"] - h0) / 1e9
    ax.set_title(f"screen at $A$   ($B$={B:.1f} s,  $D$={D:.1f} s,  $A$={A:.1f} s)", fontsize=7.5, pad=3)

    ax.set_xlim(0, m.canvas_w_px)
    ax.set_ylim(m.canvas_h_px, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999999")
    print(f"episode seed {seed} v={v}: frozen err {frozen_err:.1f}px valid={frozen_ok}; "
          f"bound err {bound_err:.1f}px valid={r['valid_at_receipt']}; moved_px {corr['moved_px']:.0f}; "
          f"match {corr['match_score']:.2f} in {corr['track_ns']/1e6:.0f} ms; B={B:.2f} D={D:.2f} A={A:.2f}")


def cell_valid(v: int, pol: str):
    k = n = 0
    for p in sorted(RUN.glob(f"live_v{v}_{pol}_s*.jsonl")):
        rec = None
        for line in p.open():
            row = json.loads(line)
            if row.get("record") == "receipt":
                rec = row
        n += 1
        k += bool(rec and rec["valid_at_receipt"])
    return k, n


def panel_stats(ax):
    speeds = [0, 4, 8, 16]
    series = [("frozen", "frozen", AMBER, "s", -0.13),
              ("restart_if_moved", "restart-on-update", RED, "D", 0.0),
              ("late_bind", "late binding", GREEN, "o", 0.13)]
    xs = np.arange(len(speeds))
    for pol, label, color, marker, off in series:
        ys, lo, hi = [], [], []
        for v in speeds:
            k, n = cell_valid(v, pol)
            p, l, h = wilson(k, n)
            ys.append(p); lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
            print(f"{pol} v={v}: {k}/{n} [{l:.2f},{h:.2f}]")
        ax.errorbar(xs + off, ys, yerr=[lo, hi], color=color, marker=marker, markersize=3.5,
                    linewidth=1.1, capsize=2, capthick=0.8, elinewidth=0.8, label=label, zorder=4)
    ax.annotate("restart also 3.4$\\times$ slower to commit\n(first commit 17.6 s vs 5.1 s)",
                (1.85, 0.11), color=RED, fontsize=6.5, ha="center", va="bottom")
    ax.set_xticks(xs, [str(v) for v in speeds])
    ax.set_xlabel("balloon speed (px/s)", fontsize=8)
    ax.set_ylabel("clicks valid at $A$", fontsize=8)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(-0.45, len(speeds) - 0.55)
    ax.grid(axis="y", color="#e2e2de", linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=7.5, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(0.30, 0.52), frameon=False, fontsize=7,
              handlelength=1.6, borderaxespad=0.2)
    ax.set_title("100 seeds per cell, Wilson 95% CI", fontsize=7.5, pad=3)


def main():
    fig = plt.figure(figsize=(5.5, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], left=0.005, right=0.995,
                          top=0.90, bottom=0.16, wspace=0.16)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_screenshot(ax_a)
    panel_stats(ax_b)
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.01, -0.10, tag, transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="left", va="top")
    fig.savefig(OUT / "c1_main.pdf")
    fig.savefig(OUT / "c1_main.png", dpi=300)
    print("wrote", OUT / "c1_main.pdf")


if __name__ == "__main__":
    main()
