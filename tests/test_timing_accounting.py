"""Distinguish scheduled publication from frames rendered after inference."""
from pathlib import Path

import pytest

from chronotope.runner import PayloadStore, run_trial
from chronotope.spec import TrialSpec
from chronotope.trace import Trace
from conftest import base_spec_dict


def test_wall_accounted_mode_records_actual_frame_readiness(tmp_path: Path):
    spec = TrialSpec.from_dict(base_spec_dict(
        timestamp_source="monotonic_ns", controller__compute_ns=180_000_000,
        delivery__delivery_delay_ns=10_000_000))
    trace_path = run_trial(spec, tmp_path / "trace.jsonl", PayloadStore(tmp_path / "payloads"),
                           timing_accounting="wall_accounted")
    t = Trace.load(trace_path)
    assert t.header()["timing_accounting"] == "wall_accounted"
    assert t.header()["receipt_source"] == "simulated"
    obs = t.all("observation")
    assert obs
    assert all(o["publication_time_ns"] >= o["scheduled_publication_time_ns"] for o in obs)
    assert all(o["publication_time_ns"] >= o["runner_frame_ready_time_ns"] for o in obs)
    # The observation may have been scheduled during a blocking model call,
    # but it must not appear to be available before the delayed rendering.
    assert any(o["publication_time_ns"] > o["scheduled_publication_time_ns"] + 20_000_000 for o in obs)
    c = t.first("commit")
    assert c is not None
    assert abs(c["commit_observed_time_ns"] - c["commitment_time_ns"]) < 5_000_000


def test_wall_accounting_cannot_be_used_with_simulated_time(tmp_path: Path):
    spec = TrialSpec.from_dict(base_spec_dict())
    with pytest.raises(ValueError, match="requires timestamp_source"):
        run_trial(spec, tmp_path / "bad.jsonl", PayloadStore(tmp_path / "payloads"),
                  timing_accounting="wall_accounted")


def test_environment_frame_source_renders_only_once_per_frame():
    """Checking a future deadline and popping a peeked frame must not rerender it."""
    from types import SimpleNamespace
    from chronotope.runner import EnvFrameSource

    class CountingManifest:
        frame_period_ns = 20
        episode_horizon_ns = 100

        def __init__(self):
            self.rendered = []

        def state_at(self, t):
            return SimpleNamespace(latest_event_id=1)

        def render(self, t):
            self.rendered.append(t)
            return str(t).encode()

    manifest = CountingManifest()
    source = EnvFrameSource(manifest, None, 5, 100)
    assert source.next_due_ns() == 105
    assert source.peek_due(104) is None
    assert manifest.rendered == []
    first = source.peek_due(105)
    assert first is source.peek()
    assert source.pop() is first
    assert manifest.rendered == [0]
    assert source.next_due_ns() == 125
    assert source.peek_due(124) is None
    assert manifest.rendered == [0]
    second = source.pop()
    assert second.source_time_ns == 20
    assert manifest.rendered == [0, 20]


def test_environment_frame_cache_invalidates_on_accepted_action():
    from types import SimpleNamespace
    from chronotope.runner import EnvFrameSource

    class StateManifest:
        frame_period_ns = 20
        episode_horizon_ns = 100

        def __init__(self):
            self.rendered = []

        def state_env(self, t, accepted):
            return SimpleNamespace(latest_event_id=accepted or 0)

        def render_env(self, t, accepted):
            self.rendered.append(accepted)
            return str(accepted).encode()

    state = {"accepted": None}
    manifest = StateManifest()
    source = EnvFrameSource(manifest, None, 0, 0, env_state=lambda: state["accepted"])
    assert source.peek().payload == b"None"
    state["accepted"] = 12
    assert source.pop().payload == b"12"
    assert manifest.rendered == [None, 12]


def test_wall_accounted_catches_up_frames_due_during_deferred_rendering(tmp_path: Path):
    """The commit cutoff must cover frames due while earlier frames are rendered."""
    import time
    from chronotope.runner import EnvFrameSource

    raw = base_spec_dict(timestamp_source="monotonic_ns",
                         controller__compute_ns=120_000_000,
                         controller__binding="newest_at_D",
                         scenario__frame_period_ns=10_000_000,
                         scenario__response_window_ns=1_400_000_000,
                         observation__publication_delay_ns=5_000_000)
    spec = TrialSpec.from_dict(raw)
    manifest = spec.manifest()
    original_render = manifest.render
    rendered = []

    def delayed_render(t):
        time.sleep(0.003)
        rendered.append(t)
        return original_render(t)

    object.__setattr__(manifest, "render", delayed_render)

    # A source factory receives the same manifest and trial clock as the runner.
    def source(h0, tc):
        return EnvFrameSource(manifest, tc, 5_000_000, h0)

    trace_path = run_trial(spec, tmp_path / "caught_up.jsonl", PayloadStore(tmp_path / "payloads"),
                           manifest=manifest, source=source, timing_accounting="wall_accounted")
    trace = Trace.load(trace_path)
    commit = trace.first("commit")
    assert commit is not None
    cutoff = commit["commitment_time_ns"]
    h0 = trace.header()["harness_origin_ns"]
    due_sources = list(range(0, cutoff - h0 - 5_000_000 + 1, 10_000_000))
    actual_sources = [o["source_time_ns"] for o in trace.all("observation")
                      if o["scheduled_publication_time_ns"] <= cutoff]
    assert actual_sources == due_sources
    assert rendered == [o["source_time_ns"] for o in trace.all("observation")]
    assert len(rendered) == len(set(rendered))
    invocation_end = trace.first("decision_invocation")["end_time_ns"]
    assert sum(invocation_end < o["scheduled_publication_time_ns"] <= cutoff
               for o in trace.all("observation")) >= 2
    assert cutoff - h0 > 620_000_000  # the event appears at 500 ms, then the model thinks
    corrections = trace.all("coordinate_correction")
    assert corrections
    assert corrections[-1]["observation_id"] == max(
        o["observation_id"] for o in trace.all("observation")
        if o["publication_time_ns"] <= cutoff)
