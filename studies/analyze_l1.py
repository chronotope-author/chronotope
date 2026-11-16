"""L1 step (3): the normalized outcome law from the assembled trial table (cards/L1-normalized-law.md).

  PYTHONPATH=src .venv/bin/python studies/analyze_l1.py [--trials results/L1_trials.csv] [--out results/L1_law.md]
      [--fig paper_chronotope_iclr2027/figures/l1_collapse.png]

(i)  Two collapse panels (PNG): success vs the normalized state change x = v (t_A - tau) / r (generalized to
     |displacement(A) - displacement(tau_eff)| / r) pooled over agents and moving-target tasks; success vs the
     receipt slack S_A = t_close - t_A pooled over windowed tasks and agents.
(ii) U62, held-out prediction: a logistic model on the four-event features {S_A, x, update_before_D, D->A}
     versus the same model on total latency {R = t_A - t_E} alone, both fitted ONCE on {notification 27B (R1),
     C1 27B} and evaluated without refitting on the held-out sets. Also the zero-parameter rule
     success <=> (S_A > 0 and x < 1). Log-loss (natural log, mean per trial) and accuracy per set and pooled.
(iii) The transition window per agent on the notification task (R1, R2, M3 rows): the bracket between the largest
     window with SR < 0.5 and the smallest with SR >= 0.5, divided by the agent's own median E->A.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chronotope.stats import wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOWED = ("revealed_target", "sponsor_skip", "live_trading")
MOVING = ("moving_target", "jump_target", "balloon_select", "reflow_page")
TRAIN_RUNS = ("r1_qwen38_notification_100", "c1_qwen38_nothink_100")
HELD_OUT = [  # (label, predicate on row)
    ("player (A1)", lambda r: r["run"] == "a1_qwen38_nothink_100"),
    ("C2 jump", lambda r: r["run"] == "c2_qwen38_nothink_100"),
    ("C3 select", lambda r: r["run"] == "c3_qwen38_nothink_100"),
    ("UI-TARS C1", lambda r: r["run"] == "c1_uitars_native_100"),
    ("G1 Chromium page", lambda r: r["run"] == "g1_reflow_qwen38_nothink_100"),
    ("Task D typed code (C4)", lambda r: r["run"] == "c4_qwen38_nothink_100"),
    ("M3 agents (notification)", lambda r: r["run"].startswith("m3_")),
    ("B trading (B1)", lambda r: r["run"] == "b1_qwen38_nothink_100"),
    ("O2 remedy matrix", lambda r: r["run"] == "o2_qwen38_nothink_100"),
    ("player, static (A1)", lambda r: r["run"] == "a1_static_qwen38_nothink_100"),
    ("R2 fine window sweep", lambda r: r["run"] == "r2_window_sweep_100c"),
    ("C P/R/L rows (Live)", lambda r: r["run"] == "c_prl_qwen38_nothink_100"),
]
REQUIRED = {"player (A1)", "C2 jump", "C3 select", "UI-TARS C1", "G1 Chromium page", "Task D typed code (C4)", "M3 agents (notification)"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # fixed categorical order
AGENT_ORDER = ["Qwen3.8-27B", "UI-TARS-1.5-7B", "Qwen3.8-27B-NVFP4", "Gemma-4-31B-NVFP4"]


def fnum(s: str):
    return None if s in ("", None) else float(s)


def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("t_E", "t_O", "tau", "tau_eff", "B", "D", "A", "A_task", "t_close", "v", "r", "x_norm", "D_A", "S_A", "R", "delivery_ms", "G"):
                r[k] = fnum(r[k])
            r["seed"] = int(r["seed"])
            r["success"] = {"True": True, "False": False}.get(r["success"])
            r["update_before_D"] = r["update_before_D"] == "True"
            r["valid_trial"] = r["valid_trial"] == "True"
            rows.append(r)
    return rows


# ---- (ii) logistic regression, fitted once ----------------------------------------------------------------------
def features4(r) -> list[float]:
    return [r["S_A"], r["x_norm"], float(r["update_before_D"]), r["D_A"]]


def features_lat(r) -> list[float]:
    return [r["R"]]


class Logit:
    def __init__(self, l2: float = 1e-2) -> None:
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logit":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        w = np.zeros(Z.shape[1]); reg = self.l2 * np.eye(Z.shape[1]); reg[0, 0] = 0
        for _ in range(100):   # Newton's method; the ridge keeps separable data finite
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + reg @ w
            H = Z.T @ (Z * (p * (1 - p))[:, None]) + reg
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def prob(self, X: np.ndarray) -> np.ndarray:
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-Z @ self.w))


def scores(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(ll), float(np.mean((p >= 0.5) == (y == 1)))


def rule(X4: np.ndarray) -> np.ndarray:
    return ((X4[:, 0] > 0) & (X4[:, 1] < 1)).astype(float)


# ---- (i) figure ----------------------------------------------------------------------------------------------------
def binned(xs, ys, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if len(sel) >= 5:   # a bin with fewer trials is shown by its points only
            k, n = sum(sel), len(sel)
            _, a, b = wilson(k, n)
            out.append(((lo + hi) / 2, k / n, a, b, n))
    return out


def figure(rows: list[dict], fig_path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    live = [r for r in rows if r["policy"] == "live" and r["valid_trial"] and r["A"] is not None]
    mov = [r for r in live if r["task"] in MOVING and r["x_norm"] is not None and r["S_A"] is not None and r["S_A"] > 0]
    win = [r for r in live if r["task"] in WINDOWED and r["S_A"] is not None]
    agents = [a for a in AGENT_ORDER if any(r["agent"] == a for r in mov + win)]
    agents += sorted({r["agent"] for r in mov + win} - set(agents))
    col = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(agents)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    rng = np.random.default_rng(0)
    # (a) moving targets: success vs x
    edges = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2, 3, 5, 8, 12, 20]
    for a in agents:
        rs = [r for r in mov if r["agent"] == a]
        if not rs:
            continue
        xs = [min(r["x_norm"], 19.9) for r in rs]; ys = [int(r["success"]) for r in rs]
        ax1.scatter([x + 0.02 * rng.standard_normal() for x in xs], [y + 0.04 * rng.standard_normal() for y in ys],
                    s=6, alpha=0.12, color=col[a], linewidths=0)
        b = binned(xs, ys, edges)
        ax1.errorbar([c for c, *_ in b], [p for _, p, *_ in b], yerr=[[max(0.0, p - lo) for _, p, lo, *_ in b], [max(0.0, hi - p) for _, p, lo, hi, _ in b]],
                     fmt="o", ms=5, lw=1.5, color=col[a], label=f"{a} (n={len(rs)})", capsize=2)
    ax1.axvline(1.0, color="#555", lw=1, ls="--")
    ax1.set_xscale("symlog", linthresh=1.0); ax1.set_xlim(-0.05, 20); ax1.set_ylim(-0.08, 1.08)
    ax1.set_xlabel(r"state change since the supporting frame, $|\Delta|/r$ ($v\,(t_A-\tau)/r$ for drift)")
    ax1.set_ylabel("success at receipt"); ax1.set_title("(a) moving targets: C1, C2, C3, Chromium page, O2", fontsize=10)
    ax1.legend(fontsize=8, loc="center right", frameon=False)
    # (b) windowed tasks: success vs S_A
    edges2 = [-12, -6, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 6, 12]
    for a in agents:
        rs = [r for r in win if r["agent"] == a]
        if not rs:
            continue
        xs = [max(-11.9, min(r["S_A"], 11.9)) for r in rs]; ys = [int(r["success"]) for r in rs]
        ax2.scatter([x + 0.03 * rng.standard_normal() for x in xs], [y + 0.04 * rng.standard_normal() for y in ys],
                    s=6, alpha=0.12, color=col[a], linewidths=0)
        b = binned(xs, ys, edges2)
        ax2.errorbar([c for c, *_ in b], [p for _, p, *_ in b], yerr=[[max(0.0, p - lo) for _, p, lo, *_ in b], [max(0.0, hi - p) for _, p, lo, hi, _ in b]],
                     fmt="o", ms=5, lw=1.5, color=col[a], label=f"{a} (n={len(rs)})", capsize=2)
    ax2.axvline(0.0, color="#555", lw=1, ls="--")
    ax2.set_xscale("symlog", linthresh=1.0); ax2.set_xlim(-12, 12); ax2.set_ylim(-0.08, 1.08)
    ax2.set_xlabel(r"receipt slack $S_A = t_{\rm close} - t_A$ (s)")
    ax2.set_title("(b) windowed tasks: notification, player, trading", fontsize=10)
    ax2.legend(fontsize=8, loc="center right", frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#ddd", lw=0.6)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    # the step checks behind each panel
    def split(rs, key, thr):
        lo = [int(r["success"]) for r in rs if key(r) < thr]; hi = [int(r["success"]) for r in rs if key(r) >= thr]
        return (sum(lo), len(lo)), (sum(hi), len(hi))
    return {"mov_n": len(mov), "win_n": len(win),
            "mov_split": {a: split([r for r in mov if r["agent"] == a], lambda r: r["x_norm"], 1.0) for a in agents},
            "win_split": {a: split([r for r in win if r["agent"] == a], lambda r: r["S_A"], 0.0) for a in agents}}


# ---- (iii) transition table -----------------------------------------------------------------------------------------
def transition_table(rows: list[dict]) -> list[str]:
    L = ["| agent | run | windows (s): Live SR | median Live E→A (s) | transition bracket (s) | bracket / median E→A | S_A > 0 → success | S_A ≤ 0 → success |",
         "|---|---|---|---|---|---|---|---|"]
    runs = sorted({r["run"] for r in rows if r["task"] == "revealed_target" and r["policy"] == "live"})
    for run in runs:
        rs = [r for r in rows if r["run"] == run and r["policy"] == "live" and r["valid_trial"]]
        agent = rs[0]["agent"]
        by_w = defaultdict(list)
        for r in rs:
            by_w[r["t_close"] - r["t_E"]].append(r)
        ws = sorted(by_w)
        sr = {w: sum(bool(r["success"]) for r in by_w[w]) / len(by_w[w]) for w in ws}
        ea = [r["R"] for r in by_w[ws[-1]] if r["R"] is not None]
        med = statistics.median(ea) if ea else None
        below = [w for w in ws if sr[w] < 0.5]; above = [w for w in ws if sr[w] >= 0.5]
        lo = max(below) if below else None; hi = min(above) if above else None
        br = f"({lo:g}, {hi:g}]" if lo is not None and hi is not None else "-"
        nb = f"({lo/med:.2f}, {hi/med:.2f}]" if (med and lo is not None and hi is not None) else "-"
        pos = [int(r["success"]) for r in rs if r["S_A"] is not None and r["S_A"] > 0]
        neg = [int(r["success"]) for r in rs if r["S_A"] is not None and r["S_A"] <= 0]
        L.append(f"| {agent} | {run} | " + ", ".join(f"{w:g}: {sr[w]:.2f}" for w in ws) + f" | {med:.2f} | {br} | {nb} | "
                 f"{sum(pos)}/{len(pos)} | {sum(neg)}/{len(neg)} |")
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=Path, default=ROOT / "results" / "L1_trials.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "L1_law.md")
    ap.add_argument("--fig", type=Path, default=ROOT / "paper_chronotope_iclr2027" / "figures" / "l1_collapse.png")
    a = ap.parse_args()
    rows = load(a.trials)
    live = [r for r in rows if r["policy"] == "live" and r["valid_trial"] and r["policy"] == "live"]
    usable = [r for r in live if r["A"] is not None and r["S_A"] is not None and r["x_norm"] is not None and r["D_A"] is not None
              and r["task"] != "pending_effect"]
    no_receipt = [r for r in live if r["A"] is None and r["task"] != "pending_effect"]

    train = [r for r in usable if r["run"] in TRAIN_RUNS]
    X4 = np.array([features4(r) for r in train]); XL = np.array([features_lat(r) for r in train])
    y = np.array([int(r["success"]) for r in train], dtype=float)
    m4 = Logit().fit(X4, y); mL = Logit().fit(XL, y)

    T = ["| held-out set | n | SR | log-loss: 4 features | log-loss: latency R | accuracy: 4 features | accuracy: latency R | accuracy: rule S_A>0 ∧ x<1 |",
         "|---|---|---|---|---|---|---|---|"]
    pooled_req: list[dict] = []; pooled_all: list[dict] = []
    for label, pred in HELD_OUT:
        rs = [r for r in usable if pred(r)]
        if not rs:
            T.append(f"| {label} | 0 | - | - | - | - | - | - |"); continue
        Xa = np.array([features4(r) for r in rs]); Xl = np.array([features_lat(r) for r in rs]); ya = np.array([int(r["success"]) for r in rs], float)
        l4, a4 = scores(m4.prob(Xa), ya); ll, al = scores(mL.prob(Xl), ya); _, ar = scores(rule(Xa), ya)
        T.append(f"| {label} | {len(rs)} | {int(ya.sum())}/{len(rs)} | {l4:.3f} | {ll:.3f} | {a4:.3f} | {al:.3f} | {ar:.3f} |")
        pooled_all += rs
        if label in REQUIRED:
            pooled_req += rs
    for label, rs in (("**pooled, card's held-out sets**", pooled_req), ("**pooled, all held-out sets**", pooled_all)):
        Xa = np.array([features4(r) for r in rs]); Xl = np.array([features_lat(r) for r in rs]); ya = np.array([int(r["success"]) for r in rs], float)
        l4, a4 = scores(m4.prob(Xa), ya); ll, al = scores(mL.prob(Xl), ya); _, ar = scores(rule(Xa), ya)
        T.append(f"| {label} | {len(rs)} | {int(ya.sum())}/{len(rs)} | {l4:.3f} | {ll:.3f} | {a4:.3f} | {al:.3f} | {ar:.3f} |")
    l4t, a4t = scores(m4.prob(X4), y); llt, alt = scores(mL.prob(XL), y); _, art = scores(rule(X4), y)
    T.append(f"| training set (R1 notification 27B + C1 27B; fit) | {len(train)} | {int(y.sum())}/{len(train)} | {l4t:.3f} | {llt:.3f} | {a4t:.3f} | {alt:.3f} | {art:.3f} |")

    fig_stats = figure(rows, a.fig)
    agents = sorted({r["agent"] for r in live})
    L = ["# L1 — a normalized outcome law across agents, tasks, and controllers", "",
         f"Source: `{a.trials.relative_to(ROOT) if a.trials.is_relative_to(ROOT) else a.trials}` ({len(rows)} rows; {len(live)} valid Live rows; "
         f"{len(usable)} Live rows with a receipt enter the predictor; {len(no_receipt)} Live rows with no receipt (wait / no action / dropped) "
         f"and the {sum(r['task'] == 'pending_effect' for r in live)} multi-action P1 rows are excluded from it). Agents: {', '.join(agents)}.", "",
         "## U62 — held-out prediction (fitted once on R1 notification 27B + C1 27B; no refitting on test)", "",
         "Four-event features: S_A = t_close − t_A, x = |Δ(t_A) − Δ(τ_eff)| / r (v (t_A − τ)/r for drift; τ_eff = τ_D under late binding), "
         "update-before-D flag, D→A. Latency baseline: R = t_A − t_E alone. Logistic regression (ridge 0.01 on standardized features, "
         "Newton), log-loss in nats per trial, accuracy at 0.5. The rule column is the zero-parameter law success ⇔ S_A > 0 ∧ x < 1.", ""]
    L += T
    L += ["", f"Fitted 4-feature weights (standardized): intercept {m4.w[0]:.2f}, S_A {m4.w[1]:.2f}, x {m4.w[2]:.2f}, update_before_D {m4.w[3]:.2f}, D→A {m4.w[4]:.2f}. "
          f"Latency-only: intercept {mL.w[0]:.2f}, R {mL.w[1]:.2f}.", ""]
    L += ["## Collapse (figure: `" + str(a.fig.relative_to(ROOT) if a.fig.is_relative_to(ROOT) else a.fig) + "`)", "",
          f"(a) moving targets, Live rows with a receipt and S_A > 0 (n = {fig_stats['mov_n']}): success by agent on either side of x = 1:", "",
          "| agent | x < 1 | x ≥ 1 |", "|---|---|---|"]
    for ag, ((k0, n0), (k1, n1)) in fig_stats["mov_split"].items():
        if n0 + n1:
            L.append(f"| {ag} | {k0}/{n0} | {k1}/{n1} |")
    L += ["", f"(b) windowed tasks, Live rows with a receipt (n = {fig_stats['win_n']}): success by agent on either side of S_A = 0:", "",
          "| agent | S_A < 0 | S_A ≥ 0 |", "|---|---|---|"]
    for ag, ((k0, n0), (k1, n1)) in fig_stats["win_split"].items():
        if n0 + n1:
            L.append(f"| {ag} | {k0}/{n0} | {k1}/{n1} |")
    L += ["", "## Transition window per agent (notification task; R1, R2, M3 rows)", ""] + transition_table(rows)
    L += ["", "## Per-cell (SR, mean G) of the rows entering the predictor, Live", "",
          "| run | cell | controller | n | SR | mean G | median x | median S_A (s) | median D→A (s) |", "|---|---|---|---|---|---|---|---|---|"]
    groups = defaultdict(list)
    for r in usable:
        groups[(r["run"], r["cell"], r["controller"])].append(r)
    for k in sorted(groups):
        g = groups[k]
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(g)} | {sum(bool(r['success']) for r in g)}/{len(g)} | "
                 f"{statistics.mean(r['G'] for r in g):.2f} | {statistics.median(r['x_norm'] for r in g):.2f} | "
                 f"{statistics.median(r['S_A'] for r in g):.2f} | {statistics.median(r['D_A'] for r in g):.3f} |")
    a.out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(T) + 10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
