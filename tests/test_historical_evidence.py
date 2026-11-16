"""Check that the selected original traces can be read by the current scorers."""
from pathlib import Path

import pytest

from chronotope.evaluate import evaluate_pending_trace, evaluate_trace
from chronotope.trace import Trace


EVIDENCE = Path(__file__).resolve().parents[1] / "evidence" / "selected_raw_events"


def test_selected_original_events_rescore_without_rewriting():
    if not EVIDENCE.exists():
        pytest.skip("the source-only installation does not include the supplementary traces")
    paths = sorted(EVIDENCE.rglob("*.jsonl"))
    assert len(paths) == 294
    legacy = 0
    for path in paths:
        original_bytes = path.read_bytes()
        trace = Trace.load(path)
        header = trace.header()
        if "skin" not in header["manifest"]:
            legacy += 1
        if header["manifest"]["scenario_id"] == "pending_effect":
            result = evaluate_pending_trace(trace)
        else:
            result = evaluate_trace(trace)
            receipt = trace.first("receipt")
            if receipt is not None:
                assert result.valid_at_receipt == receipt["valid_at_receipt"], path
        assert result.valid_trial, (path, result.invalid_reason)
        assert not result.contract_violations, (path, result.contract_violations)
        assert path.read_bytes() == original_bytes
    assert legacy == 105
