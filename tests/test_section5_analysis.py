"""Scientific invariants for deadline re-scoring and the observed frontier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies"))
from section5_results import boundary, deadline_count
from section5_frontier import frontier


def test_deadline_is_strict_and_missing_receipts_remain_failures():
    rows = [dict(success=True, E_A_ns=10), dict(success=False, E_A_ns=5),
            dict(success=False, E_A_ns=None)]
    assert deadline_count(rows, 10) == 0
    assert deadline_count(rows, 11) == 1
    assert boundary(rows, .5) is None
    assert boundary(rows, 1 / 3) == 11


def test_frontier_preserves_tradeoffs_and_removes_dominated_points():
    points = [dict(latency_p50_s=x, successes=y) for x, y in
              [(5, 27), (4, 39), (4, 20), (9, 9), (6, 80), (7, 80)]]
    assert [(p['latency_p50_s'], p['successes']) for p in frontier(points)] == [(4, 39), (6, 80)]
