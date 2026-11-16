"""The in-process TensorRT Edge-LLM planner (chronotope --planner trt) on the synthetic target. Needs engines: set
CHRONOTOPE_TRT_ENGINES=<dir with engines/llm and engines/visual> (the quick-start Qwen3.5-0.8B engines from
studies/queue_t1_20260826.sh) and a free GPU; skipped otherwise."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
BIN = ROOT / "chronotope" / "build" / "chronotope"
ENGINES = os.environ.get("CHRONOTOPE_TRT_ENGINES")
pytestmark = pytest.mark.skipif(not BIN.exists() or not ENGINES or not (pathlib.Path(ENGINES) / "llm").exists(),
                                reason="chronotope binary or CHRONOTOPE_TRT_ENGINES (engines/llm + engines/visual) not available")


def test_trt_planner_answers_and_commits(tmp_path):
    trace = tmp_path / "trt.jsonl"
    out = subprocess.run([str(BIN), "--source", "synthetic:v=0", "--planner", "trt", "--engine-dir", f"{ENGINES}/llm",
                          "--multimodal-engine-dir", str(ENGINES), "--prompt-version", "5", "--image-scale", "2",
                          "--policy", "late_bind", "--hid", "fake", "--max-actions", "1", "--duration-s", "60", "--seed", "7",
                          "--trace", str(trace)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    recs = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    inv = next(r for r in recs if r["record"] == "decision_invocation")
    assert inv["ok"] is True and inv["planner"] == "tensorrt_edgellm" and inv["planner_host_steps"] >= 1
    assert inv["prefill_ms"] > 0 and inv["deliberation_ms"] > 0
    audit = next(r for r in recs if r["record"] == "audit")
    assert audit["planner_d2h_pixels"] == 0 and audit["residency_ok"] is True and audit["planner_requests"] >= 1
    commit = next((r for r in recs if r["record"] == "commit"), None)
    if inv["click"]:
        assert commit is not None and commit["outcome"] == 1
        receipt = next(r for r in recs if r["record"] == "receipt")
        print("click", (commit["x"], commit["y"]), "target", (receipt["target_cx"], receipt["target_cy"]), "error", receipt["error_px"],
              "deliberation ms", inv["deliberation_ms"], "raw", inv["raw_text"])
    else:
        print("the model answered wait / unparseable:", inv["raw_text"])
