"""Statistical protocol (paper §4.7): paired binary outcomes use exact paired tests; confidence
intervals are bootstrapped over task templates (seeds), never over frames. Pure stdlib.

  paired_exact(a, b)         exact McNemar (binomial on discordant pairs), two-sided p
  bootstrap_ci(values, stat) percentile CI over task templates
  wilson(k, n)               Wilson score interval for one proportion
"""
from __future__ import annotations

import math
import random
from typing import Callable, Sequence


def _binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, k + 1))


def paired_exact(a: Sequence[bool], b: Sequence[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (same seeds, same order).
    Returns counts of discordant pairs and the two-sided exact p-value (sign test on discordants)."""
    if len(a) != len(b):
        raise ValueError("paired sequences must have equal length")
    b_only = sum(1 for x, y in zip(a, b) if (not x) and y)
    a_only = sum(1 for x, y in zip(a, b) if x and (not y))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2.0 * _binom_cdf(k, n))
    return {"n_pairs": len(a), "a_only": a_only, "b_only": b_only, "discordant": n, "p_two_sided": p,
            "rate_a": sum(map(bool, a)) / len(a) if a else None, "rate_b": sum(map(bool, b)) / len(b) if b else None}


def bootstrap_ci(values: Sequence, stat: Callable[[Sequence], float], *, n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over task templates: resample whole templates (the items of `values`)
    with replacement; `stat` maps a sample to a number. Returns (point, lo, hi). Explicit seed."""
    if not values:
        raise ValueError("no values")
    rng = random.Random(seed)
    n = len(values)
    point = stat(values)
    boots = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int(math.floor(alpha / 2 * n_boot))]
    hi = boots[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return point, lo, hi


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if n == 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_difference_ci(a: Sequence[bool], b: Sequence[bool], **kw) -> tuple[float, float, float]:
    """Bootstrap CI (over templates) of the paired success-rate difference b - a."""
    pairs = list(zip(map(bool, a), map(bool, b)))
    return bootstrap_ci(pairs, lambda s: sum(y for _, y in s) / len(s) - sum(x for x, _ in s) / len(s), **kw)


def success_by_deadline(outcomes: Sequence[tuple[bool, int | None]], deadlines_ns: Sequence[int]) -> list[float]:
    """Success-by-deadline curve F(w) (gym-outcome-contract card): the fraction of episodes whose
    valid-at-receipt action arrived before E + w (strict, matching the evaluator's `late` rule
    t_A >= t_close). `outcomes` is (valid_at_receipt, E->A ns, or None if
    nothing was received). Computed post hoc from ONE generous-window run, the curve is exact only
    for controllers insensitive to the rendered window (the oracle timing-isolation track); an
    agent that reads the countdown must be swept window by window, and those sweep points ARE the
    curve. Nondecreasing in w by construction."""
    if not outcomes:
        raise ValueError("no outcomes")
    n = len(outcomes)
    return [sum(1 for ok, ea in outcomes if ok and ea is not None and ea < w) / n for w in deadlines_ns]
