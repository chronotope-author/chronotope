"""A minimal OpenAI-compatible /v1/chat/completions front for an in-process TensorRT Edge-LLM engine (2026-08-27).

Why: Agent S3's client speaks the OpenAI chat format with `image_url` data URLs; Edge-LLM's experimental server rejects
that content type and its image path is unverified. This shim loads the engine ONCE through the repository's `LLM` class
(the same in-process path as chronotope/python/chronotope_core/edgellm.py) and answers chat requests, writing each data-URL image to a
temporary PNG the runtime can read. No vLLM. Stdlib only; single-threaded (the runtime holds one slot anyway).

    PYTHONPATH=src:chronotope/python .venv/bin/python studies/edgellm_openai_shim.py --engines ~/tensorrt-edgellm-workspace/Qwen3.8-27B/engines \
        --model-id Qwen3.8-27B --port 8002
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chronotope", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from chronotope_core.edgellm import EDGELLM_REPO, _load_llm  # noqa: E402

TMP = tempfile.mkdtemp(prefix="edgellm_shim_")
N = 0


def _materialize(content):
    """OpenAI content (str or list of parts) -> Edge-LLM content (str or list with {'type': 'image', 'image': path})."""
    global N
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        t = part.get("type")
        if t == "text":
            out.append({"type": "text", "text": part.get("text", "")})
        elif t in ("image_url", "image"):
            ref = part["image_url"]["url"] if t == "image_url" else part.get("image", "")
            if isinstance(ref, str) and ref.startswith("data:"):
                N += 1
                path = os.path.join(TMP, f"img_{N}.png")
                with open(path, "wb") as f:
                    f.write(base64.b64decode(ref.split(",", 1)[1]))
                ref = path
            out.append({"type": "image", "image": ref})
    return out


class Handler(BaseHTTPRequestHandler):
    llm = None
    model_id = "edgellm"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models") or self.path == "/models":
            return self._json(200, {"object": "list", "data": [{"id": self.model_id, "object": "model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        from experimental.server.engine import SamplingParams
        messages = [{"role": m.get("role", "user"), "content": _materialize(m.get("content", ""))} for m in body.get("messages", [])]
        sp = SamplingParams(temperature=float(body.get("temperature", 0.0) or 0.0), max_tokens=int(body.get("max_tokens") or 256))
        t0 = time.perf_counter()
        try:
            out = Handler.llm.chat(messages, sp)
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": f"{type(e).__name__}: {e}"[:500]})
        text = out.text or ""
        self._json(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()), "model": self.model_id,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": out.finish_reason or "stop"}],
                         "usage": {"prompt_tokens": 0, "completion_tokens": len(out.token_ids or []), "total_tokens": len(out.token_ids or []),
                                   "runtime_ms": round((time.perf_counter() - t0) * 1000)}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True); ap.add_argument("--model-id", required=True)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8002)
    a = ap.parse_args()
    if EDGELLM_REPO not in sys.path:
        sys.path.insert(0, EDGELLM_REPO)
    t0 = time.perf_counter()
    Handler.llm = _load_llm(os.path.expanduser(a.engines)); Handler.model_id = a.model_id
    print(f"edgellm shim: {a.model_id} loaded in {time.perf_counter() - t0:.1f} s; serving on {a.host}:{a.port}", flush=True)
    HTTPServer((a.host, a.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
