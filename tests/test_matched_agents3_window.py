import copy
import hashlib
import json

import pytest

from studies.prepare_matched_agents3_window import prepare, uniform_manifest


def source_manifest():
    scenario = {"event_time_ns": 500_000_000, "response_window_ns": 16_000_000_000,
                "episode_horizon_ns": 17_000_000_000, "change_delay_ns": 2_000_000_000}
    jobs = []
    for phase, modes in (("qualification", ["paused"]),
                         ("prl", ["live", "replay", "paused"]),
                         ("policy", ["live"]), ("window", ["live"])):
        for implementation in ("host", "chronotope"):
            jobs.append({"id": f"{phase}_{implementation}", "phase": phase,
                         "task": "button", "implementation": implementation,
                         "policy": "update" if phase == "policy" else "keep",
                         "seed": 100 if phase == "qualification" else 1000,
                         "modes": modes, "spec": {"scenario": copy.deepcopy(scenario)}})
    return {"templates": {"button": {"spec": {"scenario": scenario}}}, "jobs": jobs,
            "n_evaluation_seeds": 100, "model_pair": "same models", "planned_trials": 12}


def test_uniform_windows_preserve_comparison_and_task_events():
    source = source_manifest()
    original = copy.deepcopy(source)
    result = uniform_manifest(source, 64)
    assert source == original
    assert result["planned_trials"] == 10
    assert len(result["jobs"]) == 6
    assert len({job["id"] for job in result["jobs"]}) == 6
    assert result["model_pair"] == source["model_pair"]
    for actual, expected in zip(result["jobs"], source["jobs"][:6]):
        assert actual["spec"]["scenario"]["response_window_ns"] == 64_000_000_000
        assert actual["spec"]["scenario"]["episode_horizon_ns"] == 65_000_000_000
        assert actual["spec"]["scenario"]["change_delay_ns"] == 2_000_000_000
        assert "_w64000000000_" in actual["id"]
        for field in ("phase", "task", "implementation", "policy", "seed", "modes"):
            assert actual[field] == expected[field]


@pytest.mark.parametrize("window", [0, -1, float("nan"), float("inf"), 1e-12])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(ValueError):
        uniform_manifest(source_manifest(), window)


def test_duplicate_cells_are_rejected():
    source = source_manifest()
    source["jobs"].append(copy.deepcopy(source["jobs"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        uniform_manifest(source, 64)


@pytest.mark.parametrize("override,field,value", [
    ({"motion_speed_px_s": 2}, "velocity_x_px_per_s", 2),
    ({"motion_canvas_w_px": 640}, "canvas_w_px", 640),
])
def test_motion_override_is_explicit_and_limited_to_moving_variants(override, field, value):
    source = source_manifest()
    for job in source["jobs"][:2]:
        job["spec"]["scenario"].update(scenario_id="moving_target", velocity_x_px_per_s=8,
                                      canvas_w_px=320)
    result = uniform_manifest(source, 64, **override)
    assert all(job["spec"]["scenario"][field] == value for job in result["jobs"][:2])
    assert all(field not in job["spec"]["scenario"] for job in result["jobs"][2:])
    assert source["jobs"][0]["spec"]["scenario"][field] != value


def test_prepare_preserves_every_hashed_file_including_bytecode(tmp_path):
    source = tmp_path / "source"
    (source / "code/__pycache__").mkdir(parents=True)
    (source / "services").mkdir()
    payload = b"frozen bytecode fixture"
    (source / "code/__pycache__/service.pyc").write_bytes(payload)
    manifest = source_manifest()
    manifest["code_sha256"] = {"__pycache__/service.pyc": hashlib.sha256(payload).hexdigest()}
    (source / "manifest.json").write_text(json.dumps(manifest))
    (source / "environment.json").write_text(json.dumps({"vllm": {"packages": {"vllm": "test"}}}))
    services = {role: {"model": role, "port": port, "snapshot": "fixture"}
                for role, port in (("planner", 8010), ("grounder", 8011))}
    (source / "services/processes.json").write_text(json.dumps(services))
    destination = tmp_path / "uniform"
    result = prepare(source, destination, 64)
    assert (destination / "code/__pycache__/service.pyc").read_bytes() == payload
    assert result["code_sha256"] == manifest["code_sha256"]
