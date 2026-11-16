"""Prompt 17 / server agents3_service: Agent S3 behind studies/agents3_service.py (cards/M4-cross-agent-prl.md U63)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from chronotope.controller import (PROMPT_COORDS, SYSTEM_PROMPTS, AgentS3Adapter, CallBudget, VLMDirectController,
                                   parse_agents3)
from chronotope.spec import PROMPT_VERSIONS

CLICK = "import pyautogui; import pyautogui; pyautogui.click(322, 250, clicks=1, button='left'); "


def test_parse_agents3():
    assert parse_agents3([CLICK]) == {"action": "click", "x": 322, "y": 250}
    assert parse_agents3(CLICK) == {"action": "click", "x": 322, "y": 250}
    assert parse_agents3(["import time; time.sleep(3.0)"]) == {"action": "wait"}
    assert parse_agents3(["WAIT"]) == {"action": "wait"}
    assert parse_agents3(["DONE"]) is None and parse_agents3(["FAIL"]) is None
    assert parse_agents3(["import pyautogui; pyautogui.typewrite('x')"]) is None
    assert parse_agents3(None) is None and parse_agents3([]) is None
    # a click anywhere in the list wins over a sleep
    assert parse_agents3(["import time; time.sleep(1)", CLICK]) == {"action": "click", "x": 322, "y": 250}


def test_prompt17_is_instruction_only_and_pixel_space():
    assert PROMPT_COORDS[17] == "image_px" and 17 in PROMPT_VERSIONS and SYSTEM_PROMPTS[17] == "{instruction}"


class _Svc(BaseHTTPRequestHandler):
    seen = []
    def log_message(self, *a): pass
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        _Svc.seen.append(req)
        body = json.dumps({"exec_code": [CLICK], "plan_code": "agent.click('blue button')", "model_reported": "P+G",
                           "elapsed_ns": 123, "grounding_wh": [644, 476], "screen_wh": [640, 480]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


@pytest.fixture
def svc():
    srv = HTTPServer(("127.0.0.1", 0), _Svc)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_adapter_posts_instruction_resets_once_and_maps_pixels(svc):
    _Svc.seen.clear()
    ad = AgentS3Adapter(base_url=svc, model_id="P+G")
    c = VLMDirectController(adapter=ad, budget=CallBudget(5), prompt_version=17, image_scale=2, invoke_on="frame_change",
                            canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert c.system.startswith("At some point a notification card")
    r1 = c._invoke(b"\x89PNG", "ignored"); r2 = c._invoke(b"\x89PNG", "ignored")
    assert [s["reset"] for s in _Svc.seen] == [True, False] and _Svc.seen[0]["instruction"] == c.system
    assert r1.parsed == {"action": "click", "x": 322, "y": 250} and r2.detail["model_reported"] == "P+G"
    assert r1.detail["agent_elapsed_ns"] == 123 and r1.detail["plan_code"].startswith("agent.click")
    assert c._to_canvas(322, 250) == {"kind": "click", "x": 161, "y": 125}   # image px -> canvas px (scale 2)
    assert ad.spec_fields() == {"server": "agents3_service", "model_id": "P+G"}


def test_adapter_reports_service_error_as_unparseable(svc, monkeypatch):
    ad = AgentS3Adapter(base_url=svc + "/nowhere", model_id="P+G", timeout_s=5)
    import urllib.error
    def boom(req, timeout): raise urllib.error.HTTPError(req.full_url, 500, "err", {}, __import__("io").BytesIO(b'{"error": "TypeError: x"}'))
    monkeypatch.setattr("urllib.request.urlopen", boom)
    r = ad.invoke(b"png", "instr", "u")
    assert r.parsed is None and r.detail["finish_reason"] == "error" and "TypeError" in r.detail["raw_text"]


def test_spec_accepts_agents3_server():
    from chronotope.spec import TrialSpec
    import importlib, sys
    sys.path.insert(0, "studies")
    r1 = importlib.import_module("r1_revealed_target")
    r1.SERVER = "agents3_service"
    try:
        spec = r1.make_spec("P+G", "http://127.0.0.1:8100", False, None, 0.0, 64, 200, 4_000_000_000, "notification")
        spec.raw["controller"]["prompt_version"] = 17
        TrialSpec.from_dict(spec.raw)
    finally:
        r1.SERVER = "openai_compatible_local"
