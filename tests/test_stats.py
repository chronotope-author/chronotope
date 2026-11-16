from chronotope.stats import bootstrap_ci, paired_difference_ci, paired_exact, wilson


def test_paired_exact_known_values():
    a = [True] * 10
    b = [True] * 10
    assert paired_exact(a, b)["p_two_sided"] == 1.0
    a = [False] * 6 + [True] * 4
    b = [True] * 10            # 6 discordant pairs, all in b's favour: p = 2 * 0.5^6 = 0.03125
    r = paired_exact(a, b)
    assert r["discordant"] == 6 and r["b_only"] == 6 and abs(r["p_two_sided"] - 0.03125) < 1e-12
    a = [False] * 5 + [True] * 5
    b = [True] * 5 + [False] * 5   # 10 discordant, 5 each way: p = 1
    assert paired_exact(a, b)["p_two_sided"] == 1.0


def test_bootstrap_ci_is_deterministic_and_brackets_point():
    vals = [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    p1 = bootstrap_ci(vals, lambda s: sum(s) / len(s), n_boot=2000, seed=1)
    p2 = bootstrap_ci(vals, lambda s: sum(s) / len(s), n_boot=2000, seed=1)
    assert p1 == p2 and p1[1] <= p1[0] <= p1[2] and p1[0] == 0.7
    d = paired_difference_ci([False] * 8 + [True] * 2, [True] * 10, n_boot=2000, seed=2)
    assert d[0] == 0.8 and d[1] > 0


def test_wilson():
    p, lo, hi = wilson(8, 10)
    assert p == 0.8 and 0.49 < lo < 0.50 and 0.94 < hi < 0.95
