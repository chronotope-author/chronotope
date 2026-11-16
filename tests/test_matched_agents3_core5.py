import copy
import json

import pytest

from studies.prepare_matched_agents3_core5 import CORE_TASKS, restrict_manifest
from studies.run_matched_agents3_serial import ancestor_runs


def test_restriction_preserves_allowed_jobs_and_recomputes_totals():
    tasks = sorted(CORE_TASKS) + ["jump", "select"]
    source = {"jobs": [{"task": task, "id": task, "modes": ["live", "replay", "paused"],
                         "spec": {"unchanged": task}} for task in tasks],
              "templates": {task: {} for task in tasks}, "planned_trials": 21,
              "motion_override": {"variants": ["drift", "select"], "canvas_w_px": 640}}
    original = copy.deepcopy(source)
    result = restrict_manifest(source)
    assert source == original
    assert result["jobs"] == original["jobs"][:5]
    assert set(result["templates"]) == CORE_TASKS
    assert result["planned_trials"] == 15
    assert result["motion_override"] == {"variants": ["drift"], "canvas_w_px": 640}


def test_ancestor_locks_are_acquired_oldest_first(tmp_path):
    first, second, current = [tmp_path / name for name in ("first", "second", "current")]
    for path in (first, second, current):
        path.mkdir()
    (first / "manifest.json").write_text("{}")
    (second / "manifest.json").write_text(json.dumps({"parent_run": str(first)}))
    assert ancestor_runs(current, {"parent_run": str(second)}) == [first, second]
    (first / "manifest.json").write_text(json.dumps({"parent_run": str(current)}))
    with pytest.raises(ValueError, match="Cycle"):
        ancestor_runs(current, {"parent_run": str(second)})
