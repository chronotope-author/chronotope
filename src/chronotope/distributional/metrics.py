"""Aggregates for the distributional gym (request §9–§14). Episode-level everywhere: intervals are resampled over
episode ids (whole clusters when an episode was repeated); repeated runs never count as new episodes.

  summarize_controller(outcomes)      counts, SRs, Wilson, retention, interval percentiles, failure modes
  pairwise(a, b)                      Delta SR, n10/n01/n11/n00, exact McNemar, cluster bootstrap CI
  deadline_curve / speed_curve / slack_curve   ridge-penalised logistic fits with W50/W90, v50, q50 and
                                      episode-bootstrap intervals
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

import numpy as np

from ..stats import paired_exact, wilson
from .episode import FAILURE_MODES, EpisodeOutcome

Z = 1.959963984540054


# ---- helpers --------------------------------------------------------------------------------------------------
def percentiles(xs: Sequence[float], ps: Sequence[float] = (50, 95, 99)) -> dict[str, float | None]:
    """Nearest-rank percentiles; None when there is nothing to rank."""
    v = sorted(x for x in xs if x is not None)
    if not v:
        return {f"p{p:g}": None for p in ps}
    return {f"p{p:g}": v[min(len(v) - 1, max(0, math.ceil(p / 100 * len(v)) - 1))] for p in ps}


def _clusters(outcomes: Sequence[EpisodeOutcome]) -> dict[str, list[EpisodeOutcome]]:
    d: dict[str, list[EpisodeOutcome]] = {}
    for o in outcomes:
        d.setdefault(o.episode_id, []).append(o)
    return d


def episode_success(outcomes: Sequence[EpisodeOutcome]) -> dict[str, float]:
    """Y per unique episode; a repeated episode contributes the mean of its runs (one cluster, one weight)."""
    return {eid: sum(o.Y for o in runs) / len(runs) for eid, runs in _clusters(outcomes).items()}


def wilson_interval(x: int, n: int) -> tuple[float, float, float]:
    """(p, lo, hi) — request §11 (identical to stats.wilson, restated with its z)."""
    return wilson(x, n, Z)


# ---- one controller --------------------------------------------------------------------------------------------
def summarize_controller(live: Sequence[EpisodeOutcome], replay: Sequence[EpisodeOutcome] = (),
                         paused: Sequence[EpisodeOutcome] = ()) -> dict[str, Any]:
    if not live:
        raise ValueError("no Live outcomes")
    fam = {o.task_family for o in live}
    if len(fam) != 1:
        raise ValueError("do not pool task families")
    eps = _clusters(live)
    n = len(eps)
    ylive = episode_success(live)
    x = sum(1 for v in ylive.values() if v >= 1.0)
    p, lo, hi = wilson_interval(x, n)
    out: dict[str, Any] = {
        "task_family": fam.pop(), "controller": live[0].controller, "unique_episodes": n,
        "attempts": len(live), "successes": x, "live_sr": p, "live_sr_wilson": [lo, hi],
        "semantic_sr": {"live": sum(o.C for o in live) / len(live)},
        "invalid_trials": sum(o.invalid_trial for o in live),
        "failure_modes": {k: sum(1 for o in live if o.failure_mode == k) for k in FAILURE_MODES},
        "E_to_O_ns": percentiles([o.L_EO for o in live]), "O_to_B_ns": percentiles([o.L_OB for o in live]),
        "E_to_A_ns": percentiles([o.L_EA for o in live]), "B_to_D_ns": percentiles([o.L_BD for o in live]),
        "D_to_A_ns": percentiles([o.L_DA for o in live]),
        "slack_ns": percentiles([o.S_A_ns for o in live], (50, 5, 95)),
        "x_norm": percentiles([o.x_norm for o in live], (50, 95)),
        "restarts_per_episode": sum(o.n_restarts for o in live) / len(live),
        "duplicates_per_episode": (sum(o.duplicates or 0 for o in live) / len(live)) if any(o.duplicates is not None for o in live) else None,
        "suppressed_per_episode": (sum(o.n_suppressed or 0 for o in live) / len(live)) if any(o.n_suppressed is not None for o in live) else None,
        "mean_G": (sum(o.G_return for o in live if o.G_return is not None) / max(1, sum(1 for o in live if o.G_return is not None)))
                  if any(o.G_return is not None for o in live) else None,
    }
    if replay:
        rc = {o.episode_id: o.C for o in replay}
        out["semantic_sr"]["replay"] = sum(o.C for o in replay) / len(replay)
        den = sum(1 for eid in eps if rc.get(eid) == 1)
        num = sum(ylive[eid] for eid in eps if rc.get(eid) == 1)
        out["retention"] = (num / den) if den else None       # undefined, never zero, when nothing was Replay-solvable
        out["retention_denominator"] = den
        out["temporal_loss"] = (1 - num / den) if den else None
    if paused:
        out["semantic_sr"]["paused"] = sum(o.C for o in paused) / len(paused)
    if any(o.G is not None for o in live):
        g1 = [o for o in live if o.G == 1]
        out["itt_sr"] = p
        out["mechanism_stratum_n"] = len(_clusters(g1))
        out["mechanism_sr"] = (sum(episode_success(g1).values()) / len(_clusters(g1))) if g1 else None
        out["update_after_D"] = sum(1 for o in live if o.G == 0 and o.U is not None and o.D is not None and o.U >= o.D)
    return out


# ---- pairwise (request §10) -------------------------------------------------------------------------------------
def pairwise(a: Sequence[EpisodeOutcome], b: Sequence[EpisodeOutcome], *, n_boot: int = 10_000, seed: int = 0) -> dict[str, Any]:
    """Controller a minus controller b on identical episodes: Delta SR, discordant counts, exact McNemar and a
    paired bootstrap over episode ids (clusters of repeats are resampled whole)."""
    ya, yb = episode_success(a), episode_success(b)
    ids = sorted(set(ya) & set(yb))
    if not ids:
        raise ValueError("no common episodes")
    if set(ya) != set(yb):
        raise ValueError("controllers were not evaluated on identical episodes")
    xa = [ya[i] >= 1.0 for i in ids]
    xb = [yb[i] >= 1.0 for i in ids]
    n11 = sum(1 for p, q in zip(xa, xb) if p and q)
    n10 = sum(1 for p, q in zip(xa, xb) if p and not q)
    n01 = sum(1 for p, q in zip(xa, xb) if (not p) and q)
    n00 = len(ids) - n11 - n10 - n01
    mc = paired_exact(xa, xb)
    diffs = np.array([ya[i] - yb[i] for i in ids])
    rng = np.random.default_rng(seed)
    n = len(ids)
    boots = np.sort(np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)]))
    return {"a": a[0].controller, "b": b[0].controller, "n": n, "delta_sr": float(diffs.mean()),
            "n10": n10, "n01": n01, "n11": n11, "n00": n00, "mcnemar_p": mc["p_two_sided"],
            "delta_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


# ---- curves (request §12–§14) -----------------------------------------------------------------------------------
def fit_logistic(x: Sequence[float], y: Sequence[float], *, lam: float = 0.01, iters: int = 100) -> tuple[float, float] | None:
    """P(y=1|x) = sigma(alpha + beta x) by Newton's method with an L2 penalty lam on beta (a penalised fit: finite
    under complete separation). None when y is constant (no curve to fit)."""
    xv, yv = np.asarray(x, float), np.asarray(y, float)
    if len(xv) < 2 or yv.min() == yv.max():
        return None
    w = np.zeros(2)
    X = np.column_stack([np.ones_like(xv), xv])
    P = np.diag([0.0, lam])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - yv) + P @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + P + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def _curve(xs: Sequence[float], ys: Sequence[float], thresholds: Callable[[float, float], dict[str, float | None]],
           *, n_boot: int, seed: int) -> dict[str, Any]:
    fit = fit_logistic(xs, ys)
    out: dict[str, Any] = {"n": len(xs), "alpha": None, "beta": None, "fit": None}
    if fit is None:
        out["note"] = "outcome constant across episodes; no curve"
        return out
    a, b = fit
    out.update(alpha=a, beta=b, fit=thresholds(a, b))
    rng = random.Random(seed)
    n = len(xs)
    boots: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        f = fit_logistic([xs[i] for i in idx], [ys[i] for i in idx])
        if f is None:
            continue
        for k, v in thresholds(*f).items():
            if v is not None and math.isfinite(v):
                boots.setdefault(k, []).append(v)
    out["ci95"] = {k: [sorted(v)[int(0.025 * len(v))], sorted(v)[min(len(v) - 1, int(0.975 * len(v)))]]
                   for k, v in boots.items() if len(v) >= 20}
    out["bootstrap_fits"] = {k: len(v) for k, v in boots.items()}
    return out


def _safe_exp(z: float) -> float | None:
    try:
        v = math.exp(z)
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


def deadline_curve(live: Sequence[EpisodeOutcome], *, n_boot: int = 500, seed: int = 0) -> dict[str, Any]:
    """P(Y=1 | W) = sigma(alpha + beta log W); W_p = exp((logit p - alpha) / beta). Lower is better (§12)."""
    ys = episode_success(live)
    ws = {o.episode_id: o.deadline_s for o in live}
    ids = sorted(ys)
    xs = [math.log(ws[i]) for i in ids]
    yv = [ys[i] for i in ids]

    def thr(a: float, b: float) -> dict[str, float | None]:
        if b == 0:
            return {"W50": None, "W90": None}
        return {"W50": _safe_exp(-a / b), "W90": _safe_exp((math.log(9) - a) / b)}
    return _curve(xs, yv, thr, n_boot=n_boot, seed=seed)


def speed_curve(live: Sequence[EpisodeOutcome], *, n_boot: int = 500, seed: int = 0) -> dict[str, Any]:
    """P(Y=1 | v) = sigma(alpha - beta log v), beta > 0; v_p = exp((alpha - logit p) / beta). Higher v50 is better
    (§13). Fitted as sigma(alpha + b log v) with beta = -b; a fit with beta <= 0 has no finite v50 in the model."""
    ys = episode_success(live)
    vs = {o.episode_id: o.speed_px_s for o in live}
    if any(v is None or v <= 0 for v in vs.values()):
        raise ValueError("speed_curve is defined on dynamic episodes (v > 0); the static control is a separate family")
    ids = sorted(ys)
    xs = [math.log(vs[i]) for i in ids]
    yv = [ys[i] for i in ids]

    def thr(a: float, b: float) -> dict[str, float | None]:
        beta = -b
        return {"v50": (_safe_exp(a / beta) if beta > 0 else None)}
    return _curve(xs, yv, thr, n_boot=n_boot, seed=seed)


def slack_curve(live: Sequence[EpisodeOutcome], *, n_boot: int = 500, seed: int = 0) -> dict[str, Any]:
    """P(Y_bind=1 | q) = sigma(alpha + beta log q); q_p = exp((logit p - alpha) / beta). Lower q50 is better (§14)."""
    ys = episode_success(live)
    qs = {o.episode_id: o.q_s for o in live}
    if any(q is None for q in qs.values()):
        raise ValueError("slack_curve needs fixed-gate outcomes (q_s on every episode)")
    ids = sorted(ys)
    xs = [math.log(qs[i]) for i in ids]
    yv = [ys[i] for i in ids]

    def thr(a: float, b: float) -> dict[str, float | None]:
        return {"q50": (_safe_exp(-a / b) if b != 0 else None)}
    return _curve(xs, yv, thr, n_boot=n_boot, seed=seed)


def negative_control_check(late: Sequence[EpisodeOutcome], frozen: Sequence[EpisodeOutcome]) -> dict[str, Any]:
    """On a negative-control stratum late binding must not beat the frozen baseline: reports the paired delta and
    whether its 95% interval excludes zero on the positive side."""
    pw = pairwise(late, frozen)
    pw["late_binding_credited"] = pw["delta_ci95"][0] > 0
    return pw


# ---- graded success under the deadline distribution (2026-08-27) --------------------------------------------------
def deadline_survival(t_s: float, dist: dict) -> float:
    """P(W > t) under the manifest's deadline law: log-uniform on [w_min, w_max] (the generator's default) or the
    empirical window sample (W = F^-1(u)). This is the probability that a deadline drawn from the SAME law the
    manifest used would still be open at t."""
    emp = dist.get("empirical_windows_s") or ()
    if emp:
        return sum(1 for w in emp if w > t_s) / len(emp)
    lo, hi = float(dist["w_min_s"]), float(dist["w_max_s"])
    if t_s <= lo:
        return 1.0
    if t_s >= hi:
        return 0.0
    return 1.0 - math.log(t_s / lo) / math.log(hi / lo)


def expected_sr_under_deadline(live: Sequence[EpisodeOutcome], dist: dict) -> dict[str, Any]:
    """For closing-window tasks success is E->A < W, so the graded score of an agent is the E->A distribution read
    against F_W: mean over Live trials of P(W > E->A_i) among trials whose action was semantically correct and
    received (C = R = 1); a trial with no committed action or a wrong action contributes 0. Smooth in the agent's
    speed even with 10 episodes, unlike the observed SR (steps of 1/n)."""
    vals = []
    for o in live:
        if o.L_EA is None or o.C != 1 or o.R != 1:
            vals.append(0.0)
        else:
            vals.append(deadline_survival(o.L_EA / 1e9, dist))
    return {"expected_sr": (sum(vals) / len(vals)) if vals else None, "n": len(vals)}


def expected_sr_under_delay(live: Sequence[EpisodeOutcome], dist: dict) -> dict[str, Any]:
    """pending_effect (the memory axis): a trial that accepted its first click and then re-clicked after a gap g would
    have produced a duplicate for every effect delay longer than g, so its score under a delay law F_D is P(D <= g)
    (1 when the agent never re-clicked); a trial with no accepted click scores 0. Log-uniform law on [d_min, d_max]."""
    lo, hi = float(dist["delay_min_s"]), float(dist["delay_max_s"])
    def cdf(g):
        if g is None: return 1.0
        if g <= lo: return 0.0
        if g >= hi: return 1.0
        return math.log(g / lo) / math.log(hi / lo)
    vals = [0.0 if not o.C else cdf(o.reclick_gap_s) for o in live]
    return {"expected_sr": (sum(vals) / len(vals)) if vals else None, "n": len(vals)}
