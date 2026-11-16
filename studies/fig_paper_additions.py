"""Figures added 2026-08-24 (all from recorded traces or final result files; PNG only).

main paper
  figures/c1_error_law.png      miss distance vs. balloon travel during the decision (v * I_A), 600 paired trials
  figures/o1_remedy_matrix.png  oracle remedy matrix heatmap (results/O1_remedy_matrix.md)
  figures/b1_jitter.png         same order, jittered delivery: D->A histogram coloured by outcome (B1 jitter route,
                                regenerated deterministically from the seeded spec)
appendix
  figures/b1_regimes.png        schematic: where the reversal falls relative to D and A decides the utility
  figures/r1_window_curve.png   success vs response window, 100 seeds per cell, with the deliberation distribution
  figures/wi004_think_strip.png thinking-on subset: per-trial deliberation vs outcome

    .venv/bin/python studies/fig_paper_additions.py
"""
from __future__ import annotations

import re
import shutil
import statistics
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "studies"))
from chronotope.evaluate import evaluate_trace  # noqa: E402
from chronotope.stats import wilson  # noqa: E402
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


# ---------------------------------------------------------------- 1. C1 error law
def fig_c1_error_law():
    cells = {1: ROOT / "runs/c1_qwen38_lb_v12_100", 2: ROOT / "runs/c1_qwen38_lb_v12_100",
             4: ROOT / "runs/c1_qwen38_nothink_100", 8: ROOT / "runs/c1_qwen38_nothink_100",
             16: ROOT / "runs/c1_qwen38_nothink_100"}
    xs, yf, yl, vs = [], [], [], []
    for v, run in cells.items():
        for p in sorted(run.glob(f"live_v{v}_late_bind_s*.jsonl")):
            r = evaluate_trace(Trace.load(p))
            if not r.valid_trial or r.proposed_spatial_error_px is None or r.intervals_ns.get("I_A") is None:
                continue
            xs.append(v * r.intervals_ns["I_A"] / 1e9)
            yf.append(r.proposed_spatial_error_px)
            yl.append(r.spatial_error_px)
            vs.append(v)
    xs, yf, yl, vs = map(np.array, (xs, yf, yl, vs))
    resid = yf - xs
    print(f"[c1_error_law] n={len(xs)} frozen-minus-travel: median {np.median(resid):+.2f} px, "
          f"95% within [{np.percentile(resid,2.5):+.2f}, {np.percentile(resid,97.5):+.2f}] px; "
          f"late-bind err median {np.median(yl):.1f} px, max {yl.max():.1f} px")
    fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)
    lim = 100
    ax.plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls="--", zorder=1)
    ax.axhspan(0, 12, color="#dcfce7", zorder=0, lw=0)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.6, 0.6, len(xs))
    ax.scatter(xs + jit, yf, s=9, facecolor="none", edgecolor=AMBER, lw=0.6, zorder=3, label="initial coordinate (same decision)")
    ax.scatter(xs + jit, yl, s=9, marker="^", facecolor="none", edgecolor=GREEN, lw=0.6, zorder=3, label="late-bound coordinate")
    ax.text(lim * 0.50, lim * 0.60, "miss = travel\n(prediction)", color=GRAY, fontsize=7, ha="left", rotation=41)
    ax.text(lim * 0.98, 13.5, "balloon half-width 12 px: below this line the click hits", color=GREEN, fontsize=6.2, ha="right")
    for v in sorted(set(vs)):
        m = vs == v
        ax.annotate(f"{v} px/s", (np.median(xs[m]), np.median(yf[m])), xytext=(0, 6), textcoords="offset points",
                    fontsize=6.5, ha="center", color=AMBER)
    ax.set_xlim(-2, lim); ax.set_ylim(-2, lim)
    ax.set_xlabel("balloon travel while the model decided,  $v\\,(t_A-\\tau)$  (px)")
    ax.set_ylabel("distance from click to balloon at $A$ (px)")
    ax.legend(loc="upper left", frameon=False, fontsize=6.8, handletextpad=0.3)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "c1_error_law.png"); plt.close(fig)


# ---------------------------------------------------------------- 2. O1 remedy matrix
def fig_o1_matrix():
    md = (ROOT / "results/O1_remedy_matrix.md").read_text().split("## Mean return")[0]
    rows = [l for l in md.splitlines() if l.startswith("| ") and "/100" in l]
    names = [r.split("|")[1].strip() for r in rows]
    vals = np.array([[int(c.strip().split("/")[0]) for c in r.split("|")[2:7]] for r in rows])
    cols = ["transient\nevidence", "change\nbefore $D$", "closing\nwindow", "change\nin $(D,A]$", "motion +\nclosing"]
    short = {"Baseline": "no remedy", "Observation before expiry": "observe before expiry",
             "Latest relevant pre-$D$ state": "newest state before $D$", "Earlier receipt": "earlier receipt",
             "Receipt-state prediction": "predict state at $A$", "Matched combination": "matched combination"}
    fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=300)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#f3f4f6", "#86efac", GREEN])
    ax.imshow(vals, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            ax.text(j, i, f"{vals[i,j]}", ha="center", va="center", fontsize=7.5,
                    color="white" if vals[i, j] >= 60 else "#374151")
    ax.set_xticks(range(5)); ax.set_xticklabels(cols, fontsize=6.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([short[n] for n in names], fontsize=7)
    ax.set_xlabel("failure condition (oracle, 100 seeds per cell)", fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("successful clicks out of 100", fontsize=8, pad=4)
    fig.tight_layout()
    fig.savefig(OUT / "o1_remedy_matrix.png"); plt.close(fig)


# ---------------------------------------------------------------- 3. B1 jitter histogram
def fig_b1_jitter():
    import b1_live_trading as b1  # noqa: E402  (module-level: constants + spec(); main() not run)
    from chronotope.runner import PayloadStore, run_trial  # noqa: E402
    tmp = Path("/tmp/chronotope-scratch")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    das, ok = [], []
    for s in b1.SEEDS:
        p = run_trial(b1.spec(250_000_000, (0, 40_000_000), s), tmp / f"jit_s{s}.jsonl", PayloadStore(tmp / "p"))
        r = evaluate_trace(Trace.load(p))
        das.append(r.intervals_ns["D_A"] / 1e6); ok.append(bool(r.valid_at_receipt))
    das, ok = np.array(das), np.array(ok)
    print(f"[b1_jitter] valid {ok.sum()}/100; D->A min {das.min():.1f} median {np.median(das):.1f} max {das.max():.1f} ms; "
          f"all valid < 20 ms: {bool((das[ok] < 20).all())}, all invalid >= 20 ms: {bool((das[~ok] >= 20).all())}")
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    bins = np.arange(0, 41, 2)
    ax.hist(das[ok], bins=bins, color=GREEN, alpha=0.9, label=f"order valid at $A$ (+1): {ok.sum()}/100")
    ax.hist(das[~ok], bins=bins, color=RED, alpha=0.9, label=f"order invalid at $A$ ($-$1): {(~ok).sum()}/100")
    ax.axvline(20, color="#111827", lw=1.0, ls="--")
    ax.text(19.4, ax.get_ylim()[1] * 0.93, "signal reverses\n20 ms after $D$", fontsize=6.8, va="top", ha="right")
    ax.set_xlabel("delivery delay $D\\!\\rightarrow\\!A$ (ms), same decision in every trial")
    ax.set_ylabel("trials")
    ax.legend(frameon=False, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "b1_jitter.png"); plt.close(fig)
    shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 4. B1 regimes schematic
def fig_b1_regimes():
    fig, ax = plt.subplots(figsize=(5.2, 1.9), dpi=300)
    t = {"E": 500, "O": 530, "D": 730, "A": 790}
    ax.plot([400, 1000], [0, 0], color="#111827", lw=1.0)
    for k, x in t.items():
        ax.plot([x, x], [-0.12, 0.12], color="#111827", lw=1.0)
        ax.text(x, -0.3, f"${k}$", ha="center", va="top", fontsize=9)
    ax.text(t["E"], -0.62, "signal\nappears", ha="center", va="top", fontsize=6.5, color=GRAY)
    ax.text(t["O"] + 8, -0.62, "first frame\nreaches agent", ha="left", va="top", fontsize=6.5, color=GRAY)
    ax.text(t["D"], -0.62, "order\ncommitted", ha="center", va="top", fontsize=6.5, color=GRAY)
    ax.text(t["A"], -0.62, "order\nreceived", ha="center", va="top", fontsize=6.5, color=GRAY)
    cases = [(600, RED, "reversal at 600 ms\n(before $D$, and seen)\n$G=-1$ · outdated at commitment"),
             (750, RED, "reversal at 750 ms\n(between $D$ and $A$)\n$G=-1$ · invalidated after commitment"),
             (900, GREEN, "reversal at 900 ms\n(after $A$)\n$G=+1$ · no failure")]
    for i, (x, c, lab) in enumerate(cases):
        y = 0.55 + 0.55 * i
        ax.add_patch(FancyArrowPatch((x, y + 0.22), (x, 0.08), arrowstyle="-|>", mutation_scale=7, color=c, lw=1.0))
        ax.text(x - 6 if i == 0 else x + 6, y + 0.16, lab, fontsize=6.3, va="center", ha="right" if i == 0 else "left", color=c, linespacing=1.15)
    ax.text(402, 1.7, "same signal, same frames, same decision, same order — only the reversal time moves",
            fontsize=6.8, va="center", style="italic")
    ax.set_xlim(395, 1005); ax.set_ylim(-1.15, 1.95)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "b1_regimes.png"); plt.close(fig)


# ---------------------------------------------------------------- 5. R1 window curve
def fig_r1_window_curve():
    run = ROOT / "runs/r1_qwen38_notification_100"
    res = {}
    bds = []
    for p in sorted(run.glob("eval_w*_s*_live.jsonl")):
        w = int(re.search(r"eval_w(\d+)_s\d+_live", p.name).group(1)) / 1e9
        r = evaluate_trace(Trace.load(p))
        if not r.valid_trial:
            continue
        res.setdefault(w, []).append(bool(r.valid_at_receipt))
        if r.b_to_d_ns:
            bds.append(r.b_to_d_ns / 1e9)
    ws = sorted(res)
    ks = [sum(res[w]) for w in ws]; ns = [len(res[w]) for w in ws]
    print("[r1_window_curve] " + ", ".join(f"w={w:g}s {k}/{n}" for w, k, n in zip(ws, ks, ns)) +
          f"; B->D median {statistics.median(bds):.2f} s, min {min(bds):.2f}, max {max(bds):.2f} (n={len(bds)})")
    fig, ax = plt.subplots(figsize=(3.4, 2.3), dpi=300)
    p = [k / n for k, n in zip(ks, ns)]
    lo = [wilson(k, n)[1] for k, n in zip(ks, ns)]; hi = [wilson(k, n)[2] for k, n in zip(ks, ns)]
    ax.errorbar(ws, p, yerr=[np.clip(np.subtract(p, lo), 0, None), np.clip(np.subtract(hi, p), 0, None)], fmt="o",
                color=BLUE, ms=4, capsize=2, lw=0.8, zorder=3)
    ax.step([0.7] + ws + [20], [0, 0] + p[1:] + [p[-1]], where="post", color=BLUE, lw=0.8, alpha=0.5)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.text(0.75, 1.03, "Paused and Replay: 100/100 at every window", fontsize=6.5, color=GRAY, va="bottom")
    md = statistics.median(bds)
    ax.axvline(md, color="#111827", lw=0.9, ls="--")
    ax.text(md * 1.04, 0.5, f"deliberation\n{min(bds):.1f}–{max(bds):.1f} s\n(500 Live trials)", fontsize=6.5, va="center")
    for w, k, n in zip(ws, ks, ns):
        ax.annotate(f"{k}/{n}", (w, k / n), xytext=(0, -11 if k else 7), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set_xscale("log"); ax.set_xticks(ws); ax.set_xticklabels([f"{w:g}" for w in ws]); ax.minorticks_off()
    ax.set_ylim(-0.08, 1.15); ax.set_ylabel("Live success rate (Wilson 95%)")
    ax.set_xlabel("response window (s), log scale")
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "r1_window_curve.png"); plt.close(fig)


# ---------------------------------------------------------------- 6. thinking-on strip
def fig_think_strip():
    run = ROOT / "runs/wi004_qwen38_think"
    pts = []
    for p in sorted(run.glob("live_v*_late_bind_s*.jsonl")):
        v = int(re.search(r"live_v(\d+)_", p.name).group(1))
        r = evaluate_trace(Trace.load(p))
        if r.valid_trial and r.b_to_d_ns:
            pts.append((v, r.b_to_d_ns / 1e9, bool(r.valid_at_receipt), "late" in r.labels, r.spatial_error_px))
    n_in = sum(1 for v, bd, ok, late, e in pts if bd < 32); n_in_ok = sum(1 for v, bd, ok, late, e in pts if bd < 32 and ok)
    n_out = sum(1 for v, bd, ok, late, e in pts if bd >= 32); n_out_late = sum(1 for v, bd, ok, late, e in pts if bd >= 32 and late)
    print(f"[think_strip] n={len(pts)}; D inside 32-s window: {n_in_ok}/{n_in} hit; D after window: {n_out_late}/{n_out} labelled late; "
          f"deliberation min {min(x[1] for x in pts):.0f} max {max(x[1] for x in pts):.0f} s")
    fig, ax = plt.subplots(figsize=(3.4, 1.9), dpi=300)
    ax.axvspan(0, 32, color="#dcfce7", lw=0)
    ax.axvline(32, color="#111827", lw=0.9, ls="--"); ax.text(32.8, -0.42, "window closes (32 s)", fontsize=6.5, va="center")
    rng = np.random.default_rng(1)
    for v, bd, ok, late, e in pts:
        y = {4: 1, 8: 0}[v] + rng.uniform(-0.18, 0.18)
        if ok:
            ax.scatter(bd, y, s=14, marker="^", facecolor="none", edgecolor=GREEN, lw=0.7, zorder=3)
        else:
            ax.scatter(bd, y, s=14, marker="x", color=RED, lw=0.7, zorder=3)
    ax.scatter([], [], s=14, marker="^", facecolor="none", edgecolor=GREEN, label="late-bound click hit")
    ax.scatter([], [], s=14, marker="x", color=RED, label="miss, labelled late")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["8 px/s", "4 px/s"]); ax.set_ylim(-0.6, 1.5)
    ax.set_xlabel("deliberation $B\\!\\rightarrow\\!D$ with thinking on (s)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.set_xlim(0, 100)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "wi004_think_strip.png"); plt.close(fig)


if __name__ == "__main__":
    fig_c1_error_law()
    fig_o1_matrix()
    fig_b1_jitter()
    fig_b1_regimes()
    fig_r1_window_curve()
    fig_think_strip()
    print("done")
