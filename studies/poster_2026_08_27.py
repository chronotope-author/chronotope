"""A0 landscape poster from the 2026-08-27 D1 pilots (PDF + PNG). Numbers from results/D1_*_pilot.md and the outcomes files."""
import json, os, sys, textwrap
NO_RUNNING = os.environ.get("POSTER_NO_RUNNING") == "1"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

def live(path):
    return [json.loads(l) for l in open(path) if json.loads(l)["condition"] == "live"]

INK, SUB, ACC, ACC2, GRID = "#1a1a1a", "#555555", "#1f77b4", "#d62728", "#dddddd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.edgecolor": SUB, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "text.color": INK})
fig = plt.figure(figsize=(46.8, 33.1))
gs = GridSpec(100, 100, figure=fig, left=0.02, right=0.98, top=0.97, bottom=0.02)

def text_ax(spec, title, body, title_size=30, body_size=21, fc="#f7f7f7", width=150):
    ax = fig.add_subplot(spec); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fc=fc, ec=GRID, lw=2))
    ax.text(0.015, 0.95, title, fontsize=title_size, fontweight="bold", va="top", transform=ax.transAxes)
    paras = [textwrap.fill(par, width=width) for par in body.split("\n\n")]
    ax.text(0.015, 0.82, "\n\n".join(paras), fontsize=body_size, va="top", transform=ax.transAxes, linespacing=1.35)
    return ax

def _numeric(x):
    return x.replace(" *", "").replace(" †", "").replace(".", "").replace("/", "").replace("–", "").replace("-", "").strip().isdigit() or x in ("–", "-", "")

def table_ax(spec, title, columns, rows, col_w=None, fs=20, bold_rows=(), top=0.86):
    """Booktabs-style table: top/mid/bottom rules, no vertical rules, text left, numbers right, bold_rows in bold."""
    ax = fig.add_subplot(spec); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 1.0, title, fontsize=30, fontweight="bold", va="top", transform=ax.transAxes)
    n_c = len(columns); col_w = col_w or [1 / n_c] * n_c
    x0 = np.concatenate([[0], np.cumsum(col_w)]); x0 = x0 / x0[-1]
    rh = min(top / (len(rows) + 1), 0.032 / ax.get_position().height); pad = 0.008
    num_col = [all(_numeric(r[c]) for r in rows) for c in range(n_c)]
    def put(y, cells, weight):
        for c, txt in enumerate(cells):
            if num_col[c]: ax.text(x0[c + 1] - pad, y, txt, fontsize=fs, ha="right", va="center", fontweight=weight)
            else: ax.text(x0[c] + pad, y, txt, fontsize=fs, ha="left", va="center", fontweight=weight)
    y = top - rh / 2
    ax.plot([0, 1], [top, top], color=INK, lw=2.5, clip_on=False)
    put(y, columns, "bold"); y -= rh
    ax.plot([0, 1], [top - rh, top - rh], color=INK, lw=1.2, clip_on=False)
    for i, r in enumerate(rows):
        if i in bold_rows: ax.add_patch(plt.Rectangle((0, y - rh / 2), 1, rh, fc="#fff4e0", ec="none", zorder=0))
        put(y, r, "bold" if i in bold_rows else "normal"); y -= rh
    ax.plot([0, 1], [y + rh / 2, y + rh / 2], color=INK, lw=2.5, clip_on=False)
    return ax

# ---- title band ------------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0:8, :]); ax.axis("off")
ax.text(0.0, 0.85, "A GUI agent's usable speed is its end-to-end latency E→A, not its tokens per second", fontsize=50, fontweight="bold", va="top")
ax.text(0.0, 0.42, textwrap.fill("Five timestamps per action (E event · O observation · B begin · D decision · A applied), deadlines drawn per episode, and one CUDA-resident controller "
                   "whose rows differ only in the planner (proposes kind + x,y at B) and the detector (resolves x,y at D). 10 episodes per cell; Chronotope rows run their models in-process (TensorRT Edge-LLM), Agent S3 serves its two models locally.", 210),
        fontsize=26, va="top", color=SUB)

# ---- Table 1: the five stages ----------------------------------------------------------------------------------------
rows1 = [["Chronotope", "color detector (device)", "NCC template", "30", "5", "8", "20", "63", "1.00"],
         ["Chronotope", "Qwen3.5-0.8B", "detector → NCC → original", "30", "1", "259", "20", "312", "0.94"],
         ["Chronotope", "Qwen3.5-0.8B", "NCC template only", "30", "1", "254", "20", "306", "0.00"],
         ["Chronotope", "Qwen3.8-27B", "NCC template", "30", "1", "5490", "20", "5541", "0.34"],
         ["Chronotope", "Qwen3.8-27B", "Qwen3.8-27B", "30", "1", "10543", "20", "10594", "0.18"],
         *([] if NO_RUNNING else [["Chronotope", "UI-TARS-1.5-7B", "NCC template", "running", "", "", "", "", ""]]),
         ["Agent S3", "EvoCUA-8B", "UI-TARS-1.5-7B", "30", "1", "19391", "20", "19441", "0.01 †"]]
table_ax(gs[10:34, 0:47], "1. Where the time goes — closing-window task (deadline W ~ log-uniform 1–16 s); ms, p50",
         ["Agent controller", "Planner (at B)", "Detector (at D)", "E→O", "O→B", "B→D", "D→A", "E→A", "SR"], rows1,
         col_w=[0.15, 0.17, 0.22, 0.07, 0.07, 0.08, 0.07, 0.08, 0.06], bold_rows=(1,))
text_ax(gs[35:47, 0:47], "Reading table 1",
        "In a timed task, a fast small model plus a last-moment detector (Qwen3.5-0.8B + detector, 0.94) beats a big slow model (Qwen3.8-27B, 0.34) and beats "
        "an off-the-shelf agent framework (Agent S3, 0.01), because being on time matters more than being precise, and precision can be added cheaply at the "
        "last moment while speed cannot be added afterwards. SR: chance the click lands before a deadline drawn between 1 and 16 s.  † Same task, fixed 16 s "
        "deadline, 100 episodes." + ("" if NO_RUNNING else "  UI-TARS rows queued."), body_size=22, width=140)

# ---- Table 2: planner x detector grid ----------------------------------------------------------------------------------
rows2 = [["color detector (63 ms)", "1.00", "1.00", "1.00 *", "–", "–"],
         ["Qwen3.5-0.8B (312 ms)", "0.00", "0.00", "0.94", "0.76", "0.00"],
         ["Qwen3.8-27B (5.5 s)", "0.34", "0.34", "0.34", "0.35", "0.18"],
         *([] if NO_RUNNING else [["UI-TARS-1.5-7B", "running", "running", "running", "running", "running"]])]
table_ax(gs[49:69, 0:47], "2. Planner × detector — who answers \"where\" decides the outcome (SR under F_W)",
         ["Planner ↓ / Detector →", "None (frozen)", "NCC template", "Detector-first", "NCC-first", "Planner re-asked at D"], rows2,
         col_w=[0.26, 0.13, 0.14, 0.15, 0.13, 0.19], bold_rows=(1,))
text_ax(gs[70:84, 0:47], "Reading table 2",
        "The 0.8B row isolates the detector: nothing else changes between 0.00 and 0.94. Its own coordinates are 30–140 px off; a detector at commit "
        "supplies the location and the same 245 ms planner becomes 10/10. The 27B row is flat across detectors — its coordinates are exact (≤ 1 px), "
        "its ceiling is time. Re-asking the planner at D is the worst detector (doubles the delay, adds nothing). "
        "Rule order matters for a weak planner: trusting the template first lets a confidently-wrong template through.  * measured on the moving target.", body_size=22, width=140)

# ---- plot A: E->A vs deadline (closing window) ------------------------------------------------------------------------
axA = fig.add_subplot(gs[12:44, 55:75])
W = np.logspace(0, np.log10(16), 200)
axA.fill_between(W, 0.01, W, color="#e6f2e6", zorder=0, label="action lands before the deadline (E→A < W)")
for name, lab, col, mk in (("chronotope_late_bind_delib0", "color detector + NCC", "#2ca02c", "o"),
                           ("edgellm_qwen35_0p8b_detector", "0.8B + detector", ACC, "s"),
                           ("edgellm_qwen38_27b_late_bind", "27B + NCC", ACC2, "^")):
    rows = live(f"runs/d1/closing_window/{name}/outcomes.jsonl")
    x = [o["deadline_s"] for o in rows]; y = [max(o["L_EA"] or 1e9, 1e6) / 1e9 for o in rows]; hit = [o["Y"] for o in rows]
    axA.scatter([xi for xi, h in zip(x, hit) if h], [yi for yi, h in zip(y, hit) if h], s=260, marker=mk, color=col, label=f"{lab} (hit)", zorder=3)
    axA.scatter([xi for xi, h in zip(x, hit) if not h], [yi for yi, h in zip(y, hit) if not h], s=260, marker=mk, facecolors="none", edgecolors=col, linewidths=3, label=f"{lab} (miss)", zorder=3)
axA.plot(W, W, color=SUB, lw=2, ls="--"); axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlim(1, 16); axA.set_ylim(0.03, 100)
axA.set_xlabel("deadline W of the episode (s)"); axA.set_ylabel("E→A of the agent on that episode (s)")
axA.set_title("A. Success is E→A < W (one point per episode)", fontsize=26, fontweight="bold", loc="left", pad=14)
axA.grid(True, color=GRID); axA.legend(fontsize=17, loc="upper left", ncol=1, framealpha=0.95)

# ---- plot B: error vs speed (moving target fast) ----------------------------------------------------------------------
axB = fig.add_subplot(gs[12:44, 78:100])
v = np.linspace(40, 420, 100)
axB.plot(v, v * 0.225, color=ACC2, lw=2, ls="--", label="frozen: e ≈ v × 225 ms (whole E→A)")
axB.plot(v, v * 0.08, color=ACC, lw=2, ls="--", label="late-bound: e ≈ v × 80 ms (evidence age + delivery)")
axB.axhline(12, color=SUB, lw=2, ls=":", label="hit tolerance 12 px")
for name, lab, col, mk in (("chronotope_frozen_delib0", "frozen at B", ACC2, "^"), ("chronotope_late_bind_delib0", "late-bound (NCC at D)", ACC, "o"),
                           ("chronotope_detector_detector", "detector-first at D", "#2ca02c", "s")):
    rows = live(f"runs/d1/moving_target_fast/{name}/outcomes.jsonl")
    axB.scatter([o["speed_px_s"] for o in rows], [o["e_px"] for o in rows], s=260, marker=mk, color=col, label=lab, zorder=3)
axB.set_xlabel("target speed drawn for the episode (px/s)"); axB.set_ylabel("receipt error (px)")
axB.set_title("B. The controller against itself: v50 = 170 px/s", fontsize=26, fontweight="bold", loc="left", pad=14)
axB.set_xlim(40, 420); axB.set_ylim(0, 95); axB.grid(True, color=GRID); axB.legend(fontsize=17, loc="upper left", framealpha=0.95)

# ---- Table 3b ---------------------------------------------------------------------------------------------------------
rows3 = [["none (frozen at B)", "30", "176", "0", "20", "227", "0/10", "– (all miss)"],
         ["NCC template (late-bound)", "30", "183", "5", "20", "242", "5/10", "170 [113, 204]"],
         ["detector → NCC → original", "30", "175", "5", "20", "229", "5/10", "170 [113, 204]"]]
table_ax(gs[47:63, 51:100], "3. The controller's own ceiling — moving target, speed ~ 50–400 px/s, deadline 1 s, 640×480",
         ["Detector at D (planner: color detector)", "E→O", "O→B", "B→D", "D→A", "E→A", "Live SR", "v50 px/s [95 % CI]"], rows3,
         col_w=[0.34, 0.07, 0.07, 0.07, 0.07, 0.08, 0.1, 0.2], bold_rows=(1, 2))
text_ax(gs[64:84, 51:100], "Reading plot B and table 3",
        "Pushed past its own latency the resident controller is not 100 % or 0 %: the receipt error grows linearly with speed and the boundary is "
        "predictable from two numbers — how old the evidence is at commit, and the hit tolerance. A frozen click carries the target's travel over the "
        "whole E→A (225 ms) and already misses at 59 px/s; binding at D cuts the exposure to ~80 ms and moves the boundary to ~170 px/s, a factor of ~3.\n\n"
        "Harness caveat, stated: on the 640×480 software track each frame's render + publish + upload costs ~22 ms against a 16.7 ms period, so admissions "
        "lag publication by a median 94 ms and the first frame showing the target is admitted ~180 ms late; the controller's own work (O→B + B→D + bind) "
        "is ~10 ms. The 320×240 track has no such lag (5 ms). A faster frame path moves v50 up; the exposure argument is unchanged.", body_size=22, width=138)

# ---- bottom band: why drawn deadlines + method -----------------------------------------------------------------------
text_ax(gs[86:100, 0:47], "Why a drawn deadline instead of a fixed grid",
        "On a fixed grid {1, 2, 4, 8, 16 s} every agent is 0 % or 100 % at each window with the step at its own E→A (this morning's pilots: resident loop 10/10 "
        "everywhere; a served two-model framework with E→A 19.5 s: 0/100 at ≤ 8 s, 29/100 at 16 s, then 20 → 89 % at 15.5 → 30.9 s). The number carries no "
        "more information than E→A itself. Drawing W per episode turns the same fact into a graded score that slower agents can score on, and the stage "
        "columns show why.", body_size=21, fc="#fbfbfb", width=138)
text_ax(gs[86:100, 51:100], "Method",
        "Five timestamps per trial on one monotonic clock, reported as p50/p95/p99. Identical frozen episodes for every controller (manifest hash on every "
        "record); Live success scored at application receipt A; Paused and Replay conditions run on the same episodes (semantic SR and Replay→Live retention "
        "in the full tables). Models are TensorRT Edge-LLM engines loaded once in the controller's process (0.8B: 3 s; 27B: 60 s). Rows with an emulated "
        "deliberation are excluded; every row is a real planner at its real speed. 10 episodes per cell (pilot); Wilson 95 % intervals in results/D1_*_pilot.md. "
        "Physical capture (camera + HID) is a separate paper.", body_size=21, fc="#fbfbfb", width=140)

stem = "results/poster/poster_2026-08-27" + ("_no_running" if NO_RUNNING else "")
fig.savefig(stem + ".pdf")
fig.savefig(stem + ".png", dpi=60)
print("saved")
