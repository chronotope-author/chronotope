"""Main-paper C1 figures, told for a newcomer. Two figures, from recorded traces only.

Figure 1 (c1_task):    what the task is — two moments of one real episode under a
                       timeline: the frame the model starts thinking on (B), and the
                       screen when its click lands (A). The balloon moved meanwhile.
Figure 2 (c1_results): what we found — (a) the three strategies as schematic cards,
                       (b) hit rate vs balloon speed (100 seeds/cell, Wilson 95% CI),
                       (c) how long each strategy takes to commit its click.

    .venv/bin/python studies/fig_c1_story.py
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
SEED, V = 1043, 16

AMBER = "#b45309"   # frozen
RED = "#b91c1c"     # restart
GREEN = "#0f7a34"   # late binding
GRAY = "#6b7280"
SKY = "#cfe6f5"
BALLOON = "#d62828"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
})


def crosshair(ax, x, y, color, r=9.0, lw=1.4):
    ax.add_patch(Circle((x, y), r, fill=False, edgecolor=color, linewidth=lw, zorder=6))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([x + dx * (r - 3), x + dx * (r + 5)], [y + dy * (r - 3), y + dy * (r + 5)],
                color=color, linewidth=lw, zorder=6, solid_capstyle="butt")


def load_episode():
    t = Trace.load(RUN / f"live_v{V}_late_bind_s{SEED}.jsonl")
    h = t.header()
    m = manifest_from_dict(h["manifest"])
    h0 = h["harness_origin_ns"]
    dec = t.committing_decision()
    c = t.first("commit")
    r = t.first("receipt")
    corr = t.first("coordinate_correction")
    obs = {o["observation_id"]: o for o in t.all("observation")}
    ob_b = obs[dec["observation_id"]]
    frame_b = np.frombuffer((RUN / "payloads" / f"{ob_b['payload_hash']}.bin").read_bytes(),
                            dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
    tA = r["task_time_at_receipt_ns"]
    frame_a = np.frombuffer(m.render(tA), dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
    return dict(m=m, h0=h0, dec=dec, c=c, r=r, corr=corr, ob_b=ob_b,
                frame_b=frame_b, frame_a=frame_a, tA=tA)


def frame_axis(ax, frame, m):
    ax.imshow(frame, extent=[0, m.canvas_w_px, m.canvas_h_px, 0], interpolation="nearest", zorder=1)
    ax.set_xlim(0, m.canvas_w_px)
    ax.set_ylim(m.canvas_h_px, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999999")


# ------------------------------------------------------------------ figure 1: the task
def fig_task():
    e = load_episode()
    m, h0 = e["m"], e["h0"]
    px, py = e["corr"]["proposed_action"]["x"], e["corr"]["proposed_action"]["y"]
    E = m.event_time_ns / 1e9
    O = (e["ob_b"]["publication_time_ns"] - h0) / 1e9
    D = (e["c"]["commitment_time_ns"] - h0) / 1e9
    A = (e["r"]["receipt_time_ns"] - h0) / 1e9
    miss = m.spatial_error_px(e["tA"], e["corr"]["proposed_action"])
    bx, by = m.target_box_at(e["ob_b"]["source_time_ns"]).center()
    ax_, ay_ = m.target_box_at(e["tA"]).center()
    # newest frame at commitment: the one the late binder re-anchored on
    obs = {o["observation_id"]: o for o in
           Trace.load(RUN / f"live_v{V}_late_bind_s{SEED}.jsonl").all("observation")}
    ob_d = obs[e["corr"]["observation_id"]]
    frame_d = np.frombuffer((RUN / "payloads" / f"{ob_d['payload_hash']}.bin").read_bytes(),
                            dtype=np.uint8).reshape(m.canvas_h_px, m.canvas_w_px, 3)
    dx_, dy_ = m.target_box_at(ob_d["source_time_ns"]).center()

    fig = plt.figure(figsize=(5.5, 2.45))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], left=0.012, right=0.988,
                          top=0.97, bottom=0.09, hspace=0.30, wspace=0.08)

    # timeline: environment-caused timings above the bar, agent-caused below.
    # t is task time; t = 0 is the start of the episode (first frame published).
    tl = fig.add_subplot(gs[0, :])
    tl.set_xlim(0, 10)
    tl.set_ylim(0, 1)
    tl.axis("off")
    y = 0.52
    x0, xe, xo, xd, xa = 1.05, 2.85, 4.1, 7.75, 9.4
    tl.annotate("", (9.9, y), (0.95, y), arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    tl.text(0.02, 0.88, "environment", ha="left", va="center", fontsize=6,
            color=GRAY, style="italic")
    tl.text(0.02, 0.16, "agent", ha="left", va="center", fontsize=6,
            color="black", style="italic")
    # origin
    tl.plot([x0, x0], [y - 0.06, y + 0.06], color="black", lw=1.0)
    tl.text(x0, y + 0.13, "$t = 0$: episode begins", ha="center", va="bottom", fontsize=5.2,
            color=GRAY)
    # environment side (above)
    tl.plot([xe], [y], "o", color=BALLOON, markersize=4, zorder=5)
    tl.text(xe, y + 0.13, f"$E$: balloon starts moving\n($t_E$ = {E:.2f} s)",
            ha="center", va="bottom", fontsize=6, color=BALLOON, linespacing=1.15)
    tl.text((xo + xd) / 2 + 0.35, y + 0.16, "the balloon keeps drifting $\\rightarrow$",
            ha="center", va="bottom", fontsize=6, color=BALLOON)
    # agent side (below)
    for x, lab in ((xo, f"$O$: frame reaches model;\nthinking starts ($t_O$ = {O:.2f} s)"),
                   (xd, f"$D$: answer sealed\n($t_D$ = {D:.2f} s)"),
                   (xa, f"$A$: click lands\n($t_A$ = {A:.2f} s)")):
        tl.plot([x], [y], "o", color="black", markersize=4, zorder=5)
        tl.text(x, y - 0.13, lab, ha="center", va="top", fontsize=6, linespacing=1.15)
    tl.annotate("", (xd - 0.85, y - 0.42), (xo + 1.0, y - 0.42),
                arrowprops=dict(arrowstyle="<->", color=GRAY, lw=0.7))
    tl.text((xo + xd) / 2 + 0.1, y - 0.47, f"thinking: {D - O:.1f} s",
            ha="center", va="top", fontsize=6, color=GRAY)
    tl.text(9.9, 0.02, "(spacing not to scale)", ha="right", va="bottom",
            fontsize=5, color=GRAY)

    # 1: frame at O — what the model sees
    a1 = fig.add_subplot(gs[1, 0])
    frame_axis(a1, e["frame_b"], m)
    crosshair(a1, px, py, AMBER)
    a1.annotate("model: “click\nthe balloon,\nright here”", (px + 15, py + 4),
                color=AMBER, fontsize=6.5, ha="left", va="center", zorder=7)
    a1.set_title(f"1.  what the model sees ($t_O$ = {O:.2f} s)", fontsize=7, pad=3)

    # 2: newest frame at D — balloon moved while thinking; answer unchanged
    a2 = fig.add_subplot(gs[1, 1])
    frame_axis(a2, frame_d, m)
    a2.add_patch(Circle((bx, by), m.target_size_px / 2, fill=False, edgecolor=GRAY,
                        linewidth=1.0, linestyle=(0, (3, 2)), zorder=5))
    a2.annotate("was", (bx - 18, by), color=GRAY, fontsize=6, ha="right", va="center")
    a2.add_patch(FancyArrowPatch((bx + 14, by - 20), (dx_ - 4, dy_ - 20), arrowstyle="-|>",
                                 mutation_scale=7, color=GRAY, linewidth=0.9, zorder=5))
    crosshair(a2, px, py, AMBER)
    a2.annotate("the answer still\npoints here", (px, py + 38), color=AMBER, fontsize=6.5,
                ha="center", va="top", zorder=7)
    a2.set_title(f"2.  balloon has moved ($t_D$ = {D:.2f} s)", fontsize=7, pad=3)

    # 3: screen at A — the click lands and misses
    a3 = fig.add_subplot(gs[1, 2])
    frame_axis(a3, e["frame_a"], m)
    crosshair(a3, px, py, AMBER)
    a3.annotate(f"click lands:\nmiss by {miss:.0f} px", (px, py + 38), color=AMBER,
                fontsize=6.5, ha="center", va="top", zorder=7)
    a3.set_title(f"3.  the click lands ($t_A$ = {A:.2f} s)", fontsize=7, pad=3)

    fig.savefig(OUT / "c1_task.png", dpi=300)
    print(f"c1_task: E={E:.2f} O={O:.2f} D={D:.2f} A={A:.2f} miss={miss:.0f}px "
          f"balloon O=({bx:.0f},{by:.0f}) D=({dx_:.0f},{dy_:.0f}) A=({ax_:.0f},{ay_:.0f})")


# ------------------------------------------------------------------ figure 2: findings
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


def strategy_card(ax, title, color, mode):
    """Tiny schematic: dashed circle = balloon when thinking began, solid = balloon now."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(SKY)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#bbbbbb")
    was, now, r = (20, 12), (66, 12), 6.0
    ax.add_patch(Circle(was, r, fill=False, edgecolor=GRAY, linewidth=1.0, linestyle=(0, (3, 2))))
    ax.text(was[0], 23.0, "was", color=GRAY, fontsize=6, ha="center", va="top")
    ax.add_patch(Circle(now, r, facecolor=BALLOON, edgecolor="none"))
    ax.text(now[0], 23.0, "now", color=BALLOON, fontsize=6, ha="center", va="top")
    def ch(x, y):
        crosshair(ax, x, y, color, r=3.6, lw=1.2)
    if mode == "frozen":
        ch(*was)
    elif mode == "restart":
        ch(43, 12)
        ax.text(87, 12, "always one\nthink behind", color=color, fontsize=5.5,
                ha="center", va="center", linespacing=1.1, clip_on=True)
    else:
        ch(*now)
    ax.set_title(title, fontsize=7.3, color=color, pad=2, loc="left")


def fig_results():
    fig = plt.figure(figsize=(5.5, 2.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[0.85, 1.0], left=0.055, right=0.985,
                          top=0.90, bottom=0.15, hspace=1.25, wspace=0.28)

    cards = [("Frozen: click where it was", AMBER, "frozen"),
             ("Restart: throw it away, think again (+5 s)", RED, "restart"),
             ("Late binding: keep the answer, slide the click (60 ms)", GREEN, "late_bind")]
    for i, (title, color, mode) in enumerate(cards):
        strategy_card(fig.add_subplot(gs[i, 0]), title, color, mode)

    # (b) hit rate vs speed
    axb = fig.add_subplot(gs[0:2, 1])
    speeds = [0, 4, 8, 16]
    xs = np.arange(len(speeds))
    series = [("frozen", AMBER, "s", -0.13), ("restart_if_moved", RED, "D", 0.0),
              ("late_bind", GREEN, "o", 0.13)]
    for pol, color, marker, off in series:
        ys, lo, hi = [], [], []
        for v in speeds:
            k, n = cell_valid(v, pol)
            p, l, h = wilson(k, n)
            ys.append(p)
            lo.append(max(0.0, p - l))
            hi.append(max(0.0, h - p))
        axb.errorbar(xs + off, ys, yerr=[lo, hi], color=color, marker=marker, markersize=3,
                     linewidth=1.1, capsize=1.8, capthick=0.8, elinewidth=0.8, zorder=4)
    axb.text(1.55, 0.90, "late binding: 100/100", color=GREEN, fontsize=7, ha="center")
    axb.text(1.9, 0.10, "frozen and restart: 0/100", color=RED, fontsize=7, ha="center")
    axb.text(-0.38, 0.62, "$v$ = 0:\nall hit", color=GRAY, fontsize=6, ha="left", va="center")
    axb.set_xticks(xs, [str(v) for v in speeds])
    axb.set_xlabel("balloon speed (px/s)", fontsize=7.5, labelpad=1)
    axb.set_ylabel("clicks that hit", fontsize=7.5)
    axb.set_ylim(-0.05, 1.07)
    axb.set_xlim(-0.45, len(speeds) - 0.55)
    axb.grid(axis="y", color="#e2e2de", linewidth=0.6, zorder=0)
    axb.tick_params(labelsize=7, width=0.6)
    for side in ("top", "right"):
        axb.spines[side].set_visible(False)
    axb.set_title("hit rate (100 tries per point, 95% CI)", fontsize=7.5, pad=2)

    # (c) time to commit
    axc = fig.add_subplot(gs[2, 1])
    names = ["frozen", "late binding", "restart"]
    times = [5.07, 5.12, 17.57]
    colors = [AMBER, GREEN, RED]
    ypos = np.arange(len(names))[::-1]
    axc.barh(ypos, times, color=colors, height=0.62, zorder=3)
    for y, t_, name, c_ in zip(ypos, times, names, colors):
        axc.text(t_ + 0.3, y, f"{t_:.1f} s", va="center", fontsize=6.5, color=c_)
    axc.text(17.57 / 2, ypos[2], "three full thinks — still 0/100", color="white",
             fontsize=6, ha="center", va="center")
    axc.set_yticks(ypos, names, fontsize=7)
    axc.set_xlim(0, 21.5)
    axc.set_xlabel("seconds from first look to committed click (moving balloon)", fontsize=7, labelpad=1)
    axc.tick_params(labelsize=6.5, width=0.6)
    axc.grid(axis="x", color="#e2e2de", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        axc.spines[side].set_visible(False)

    for ax, tag, dy in ((fig.axes[0], "(a)", 0.5), (axb, "(b)", 0.22), (axc, "(c)", 0.55)):
        ax.text(-0.06, 1.0 + dy, tag, transform=ax.transAxes, fontsize=8, fontweight="bold",
                ha="right", va="top")

    fig.savefig(OUT / "c1_results.png", dpi=300)
    print("wrote c1_results")


if __name__ == "__main__":
    fig_task()
    fig_results()
