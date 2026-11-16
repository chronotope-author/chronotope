"""In-process TensorRT Edge-LLM planner adapter for the chronotope controller (2026-08-27; no vLLM, no HTTP).

The engines built by ~/TensorRT-Edge-LLM (llm_build / visual_build) are loaded ONCE per process through the
repository's pybind runtime (experimental/server/engine.py `LLM`), and every planner call is a direct call into that
runtime. Same `invoke(png, system, user_text, structured, schema, parser)` contract as chronotope.controller's
LocalOpenAIAdapter, so the chronotope `vlm` planner and the `grounder` binder use it unchanged.

Spec fields (glue.CHRONOTOPE_SCHEMA): server "edge_llm"; base_url = the engines root (…/engines: llm/ + the visual engine);
model_id = the display name (e.g. "Qwen3.5-0.8B"). Environment: LD_LIBRARY_PATH must hold ~/trt-prefix's TensorRT
libraries and EDGELLM_PLUGIN_PATH the plugin .so (chronotope/patches/build_edgellm.sh); both are set by the study scripts.

Residency note (K1 §5.4): the frame crosses host memory as a PNG file here (the runtime loads images by path), so the
planner counts as host steps in the audit — the letter of §5.4 is not met by this path either (see T1 ledger entry,
Finding 4). What it removes is the serving stack: no server process, no socket, one process on the device.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from typing import Any

from chronotope.controller import RAW_TEXT_KEEP, RawInvocation, parse_uitars

EDGELLM_REPO = os.environ.get("EDGELLM_REPO", os.path.expanduser("~/TensorRT-Edge-LLM"))
_LLMS: dict[str, Any] = {}


def _load_llm(engines_root: str):
    if engines_root in _LLMS:
        return _LLMS[engines_root]
    if EDGELLM_REPO not in sys.path:
        sys.path.insert(0, EDGELLM_REPO)
    from experimental.server.engine import LLM   # the pybind runtime behind a vLLM-style class
    llm_dir = os.path.join(engines_root, "llm")
    if not os.path.isfile(os.path.join(llm_dir, "config.json")):
        raise FileNotFoundError(f"no engine at {llm_dir}/config.json (build it with llm_build; see studies/t1a_engines.sh)")
    t0 = time.perf_counter()
    llm = LLM(engine_dir=llm_dir, multimodal_engine_dir=engines_root)
    _LLMS[engines_root] = llm
    _LLMS[engines_root + "#load_s"] = time.perf_counter() - t0
    return llm


class EdgeLLMAdapter:
    """Chat call into the in-process Edge-LLM runtime; returns RawInvocation like LocalOpenAIAdapter."""

    def __init__(self, *, engines_root: str, model_id: str, max_tokens: int, temperature: float,
                 enable_thinking: bool = False, reasoning_effort: str | None = None) -> None:
        self.engines_root = os.path.expanduser(engines_root)
        self.model_id, self.max_tokens, self.temperature = model_id, max_tokens, temperature
        self.enable_thinking, self.reasoning_effort = enable_thinking, reasoning_effort
        self._n = 0
        self._tmp = tempfile.mkdtemp(prefix="chronotope_edgellm_")

    def spec_fields(self) -> dict[str, Any]:
        return {"server": "edge_llm", "engines_root": self.engines_root, "model_id": self.model_id,
                "max_tokens": self.max_tokens, "temperature": self.temperature, "enable_thinking": self.enable_thinking}

    def invoke(self, png: bytes, system: str, user_text: str, structured: bool = True,
               schema: dict | None = None, parser=None) -> RawInvocation:
        llm = _load_llm(self.engines_root)
        from experimental.server.engine import SamplingParams
        self._n += 1
        path = os.path.join(self._tmp, f"frame_{self._n}.png")
        with open(path, "wb") as f:
            f.write(png)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": [{"type": "image", "image": path}, {"type": "text", "text": user_text}]}]
        t0 = time.perf_counter_ns()
        out = llm.chat(messages, SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens))
        dt = time.perf_counter_ns() - t0
        text = out.text or ""
        if structured:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = _first_json_object(text)
            parsed = _normalize_action(parsed)
        else:
            parsed = (parser or parse_uitars)(text)
        try:
            os.unlink(path)
        except OSError:
            pass
        return RawInvocation(parsed=parsed, invocation_id=f"edgellm-{self._n}",
                             detail={"model_reported": self.model_id, "finish_reason": out.finish_reason,
                                     "input_tokens": None, "output_tokens": len(out.token_ids) or None,
                                     "reasoning_tokens": None, "raw_text": text[:RAW_TEXT_KEEP], "runtime_ns": dt,
                                     "engine_load_s": _LLMS.get(self.engines_root + "#load_s")})


def _normalize_action(obj: Any) -> dict | None:
    """The paper's gui_action object {action, x, y}: a bare list is its first element; "coordinate": [x, y] (the
    Qwen computer-use habit) becomes x, y. Anything else is left for the controller's own validation."""
    if isinstance(obj, list):
        obj = obj[0] if obj and isinstance(obj[0], dict) else None
    if not isinstance(obj, dict):
        return None
    c = obj.get("coordinate")
    if "x" not in obj and isinstance(c, (list, tuple)) and len(c) == 2 and all(isinstance(v, (int, float)) for v in c):
        obj = {**obj, "x": int(round(c[0])), "y": int(round(c[1]))}
    return obj


def _first_json_object(text: str) -> dict | None:
    """The first {...} in a free-text answer (the small models rarely emit the object alone)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None
