"""U4: does the wall-clock mode (lazy publication, sleeping oracle) reproduce the simulated
results within sleep jitter? Uses the same closed-form prediction as the simulated tests."""
from __future__ import annotations

import pytest

from chronotope.__main__ import run_triplet
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict
from test_oracle_predictions import predict

JITTER_NS = 15_000_000  # 15 ms: generous for time.sleep on Linux

# keep the cells short so the test is quick; windows 400 ms, event at 500 ms as in the spec
CELLS = [
    dict(controller__compute_ns=100_000_000, delivery__delivery_delay_ns=20_000_000),   # live succeeds
    dict(controller__compute_ns=355_000_000, delivery__delivery_delay_ns=60_000_000),   # invalidated after D
    dict(controller__compute_ns=600_000_000, delivery__delivery_delay_ns=0),            # outdated at D
]


@pytest.mark.parametrize("over", CELLS)
def test_wallclock_matches_closed_form_within_jitter(ws, over):
    d = base_spec_dict(timestamp_source="monotonic_ns", **over)
    pred = predict(d)
    out = run_triplet(TrialSpec.from_dict(d), ws)
    res, pair = out["results"], out["pair"]
    assert pair.valid, pair.reason
    for p in ("paused", "replay", "live"):
        assert res[p].valid_trial and not res[p].contract_violations, res[p]
        assert bool(res[p].valid_at_receipt) == pred[p], (p, res[p].to_dict())
    live = res["live"]
    assert set(live.labels) == pred["live_labels"]
    for k in ("E_O", "O_D", "D_A", "E_A", "S_A"):
        assert abs(live.intervals_ns[k] - pred[k]) <= JITTER_NS, (k, live.intervals_ns[k], pred[k])
    # O stamps are due times, so frames published lazily during B->D are still ordered before D
    t = Trace.load(ws / "live.jsonl")
    c = t.first("commit")
    during = [o for o in t.all("observation") if o["publication_time_ns"] <= c["commitment_time_ns"]]
    assert all(a["publication_time_ns"] < b["publication_time_ns"] for a, b in zip(during, during[1:]))
