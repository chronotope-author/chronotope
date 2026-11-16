"""The standalone loop (chronotope/build/chronotope): headless runs on the synthetic target with the detector planner, and the
VLM planner against a fake OpenAI-compatible server (request format, PNG, structured answer, coordinate mapping)."""
from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
BIN = ROOT / "chronotope" / "build" / "chronotope"
pytestmark = pytest.mark.skipif(not BIN.exists(), reason="chronotope binary not built")


def synth_center(seed: int, w=320, h=240, size=24) -> tuple[int, int]:
    """Synthetic::place_for_seed, ported."""
    s = (seed * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    margin = size + 8
    span_x, span_y = max(1, w - 2 * margin - size), max(1, h - 2 * margin - size - h // 10)
    x0 = margin + ((s >> 17) % span_x)
    y0 = margin + ((s >> 41) % span_y)
    return x0 + size // 2, y0 + size // 2


def run_bin(args: list[str], trace: pathlib.Path) -> tuple[str, list[dict]]:
    out = subprocess.run([str(BIN), *args, "--trace", str(trace)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    recs = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    return out.stdout, recs


@pytest.mark.parametrize("policy,expect_hit", [("frozen", False), ("late_bind", True)])
def test_headless_loop_detector(tmp_path, policy, expect_hit):
    stdout, recs = run_bin(["--source", "synthetic:v=40", "--planner", "detector", "--deliberation-ms", "500", "--policy", policy,
                            "--hid", "fake", "--max-actions", "1", "--duration-s", "6", "--seed", "3"], tmp_path / "t.jsonl")
    kinds = [r["record"] for r in recs]
    assert kinds[0] == "trial_header" and "decision_begin" in kinds and "commit" in kinds and "delivery" in kinds and "receipt" in kinds
    commit = next(r for r in recs if r["record"] == "commit")
    receipt = next(r for r in recs if r["record"] == "receipt")
    audit = next(r for r in recs if r["record"] == "audit")
    assert commit["outcome"] == 1 and audit["residency_ok"] is True and audit["d2h_state_in_window"] == 0
    assert receipt["valid_at_receipt"] is expect_hit, stdout
    if policy == "frozen":
        assert commit["binding"] == 0 and 15 < receipt["error_px"] < 26      # 40 px/s x (0.5 s + 20 ms + a frame)
    else:
        assert commit["binding"] == 1 and commit["matched"] is True and receipt["error_px"] < 3
    assert "residency OK" in stdout


class _Fake(BaseHTTPRequestHandler):
    seen: list[dict] = []
    answer = {"action": "click", "x": 0, "y": 0}

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n))
        _Fake.seen.append(body)
        resp = {"id": "chatcmpl-fake", "model": "fake-vlm", "choices": [{"message": {"role": "assistant", "content": json.dumps(_Fake.answer)},
                                                                        "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 7}}
        data = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_vlm_planner_against_fake_server(tmp_path):
    cx, cy = synth_center(7)
    _Fake.answer = {"action": "click", "x": cx * 1000 // 320, "y": cy * 1000 // 240}   # prompt 5: the 0-1000 grid
    _Fake.seen = []
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    try:
        stdout, recs = run_bin(["--source", "synthetic:v=0", "--planner", "vlm", "--base-url", f"http://127.0.0.1:{port}/v1", "--model", "fake-vlm",
                                "--prompt-version", "5", "--policy", "late_bind", "--hid", "fake", "--max-actions", "1", "--duration-s", "8", "--seed", "7"],
                               tmp_path / "v.jsonl")
    finally:
        srv.shutdown()
    assert _Fake.seen, "no request reached the fake server"
    req = _Fake.seen[-1]
    assert req["model"] == "fake-vlm" and req["response_format"]["type"] == "json_schema"
    assert req["response_format"]["json_schema"]["schema"]["properties"]["action"]["enum"] == ["click", "wait"]
    assert req["chat_template_kwargs"] == {"enable_thinking": False}
    assert req["messages"][0]["role"] == "system" and "red balloon" in req["messages"][0]["content"]
    img = req["messages"][1]["content"][0]["image_url"]["url"]
    assert img.startswith("data:image/png;base64,")
    png = base64.b64decode(img.split(",", 1)[1])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    assert (w, h) == (640, 480)                                                # image_scale 2
    inv = next(r for r in recs if r["record"] == "decision_invocation")
    assert inv["ok"] is True and inv["click"] is True and inv["invocation_id"] == "chatcmpl-fake" and inv["input_tokens"] == 12
    commit = next(r for r in recs if r["record"] == "commit")
    receipt = next(r for r in recs if r["record"] == "receipt")
    assert commit["outcome"] == 1 and abs(commit["x"] - cx) <= 1 and abs(commit["y"] - cy) <= 1
    assert receipt["valid_at_receipt"] is True, stdout
