"""Agent S3 as a local HTTP service (cards/M4-cross-agent-prl.md phase 3, U63). Runs in ~/agents3-venv (gui-agents
0.3.x), NOT in the chronotope venv; the chronotope adapter (controller.AgentS3Adapter) posts one screenshot per
decision and receives the pyautogui code Agent S3 would execute. Agent S3 is used as shipped (Worker + reflection +
code agent, max_trajectory_length 8); only its two model endpoints are local vLLM servers so every clock is measured.

  ~/agents3-venv/bin/python studies/agents3_service.py --port 8100 \
      --planner meituan/EvoCUA-32B-20260105 --planner-url http://127.0.0.1:8000/v1 \
      --grounder ByteDance-Seed/UI-TARS-1.5-7B --grounder-url http://127.0.0.1:8001/v1

POST /predict  {"png_b64": ..., "instruction": ..., "reset": bool}
  -> {"exec_code": [...], "plan_code": str, "model_reported": "<planner>+<grounder>", "elapsed_ns": int,
      "grounding_wh": [w, h], "screen_wh": [w, h]}   or {"error": ...}
GET  /models   -> {"data": [{"id": "<planner>+<grounder>"}]}   (liveness, same shape the queue scripts poll)
reset=True builds a fresh AgentS3 (a new trial); the grounding width/height are the UI-TARS smart-resize of the
screenshot (factor 28), so its absolute coordinates map back onto the screenshot pixels.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056):
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Session:
    def __init__(self, a, width: int, height: int):
        from gui_agents.s3.agents.agent_s import AgentS3
        from gui_agents.s3.agents.grounding import OSWorldACI
        gh, gw = smart_resize(height, width)
        gen = {"engine_type": "vllm", "model": a.planner, "base_url": a.planner_url, "api_key": "dummy"}
        ground = {"engine_type": "vllm", "model": a.grounder, "base_url": a.grounder_url, "api_key": "dummy",
                  "grounding_width": gw, "grounding_height": gh}
        self.aci = OSWorldACI(env=None, platform="linux", engine_params_for_generation=gen,
                              engine_params_for_grounding=ground, width=width, height=height)
        self.agent = AgentS3(gen, self.aci, platform="linux")
        self.grounding_wh, self.screen_wh = (gw, gh), (width, height)


class State:
    def __init__(self, a):
        self.a, self.session, self.lock = a, None, threading.Lock()
        self.model_reported = f"{a.planner}+{a.grounder}"


def make_handler(state: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(200, {"data": [{"id": state.model_reported}]})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/predict"):
                return self._send(404, {"error": "unknown path"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n).decode())
                png = base64.b64decode(req["png_b64"])
                w, h = Image.open(io.BytesIO(png)).size
                with state.lock:
                    if req.get("reset") or state.session is None or state.session.screen_wh != (w, h):
                        state.session = Session(state.a, w, h)
                    s = state.session
                    t0 = time.monotonic_ns()
                    info, actions = s.agent.predict(instruction=req["instruction"], observation={"screenshot": png})
                    el = time.monotonic_ns() - t0
                self._send(200, {"exec_code": list(actions), "plan_code": (info or {}).get("plan_code"),
                                 "model_reported": state.model_reported, "elapsed_ns": el,
                                 "grounding_wh": list(s.grounding_wh), "screen_wh": list(s.screen_wh)})
            except Exception as e:  # the adapter records the error text as an unparseable answer
                self._send(500, {"error": f"{type(e).__name__}: {e}", "model_reported": state.model_reported})
    return H


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--planner", required=True); ap.add_argument("--planner-url", required=True)
    ap.add_argument("--grounder", required=True); ap.add_argument("--grounder-url", required=True)
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(State(a)))
    print(f"agents3_service on {a.host}:{a.port} planner={a.planner} grounder={a.grounder}", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
