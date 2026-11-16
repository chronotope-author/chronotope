"""Claude models as planners (2026-08-28): the paper's single-frame click prompt through the Anthropic Messages API.

Same `invoke(png, system, user_text, structured, schema, parser)` contract as LocalOpenAIAdapter (controller.py), so
every controller kind (vlm_direct, vlm_late_binding, the resident core's vlm planner) can use a Claude model as the
planner without change. The frame goes in as a base64 PNG image block; the answer is the gui_action JSON object
(action, x, y on the prompt's coordinate space), enforced by structured outputs (output_config.format).

Credentials: ANTHROPIC_API_KEY (or ~/.anthropic_key), and for identity-linked keys ANTHROPIC_WORKSPACE_ID (sent as
the anthropic-workspace-id header). The key is never written to traces; the trace records the model id, the request
id and the usage counts.
"""
from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

from .controller import ACTION_JSON_SCHEMA, RawInvocation, parse_uitars

DEFAULT_MODEL = "claude-opus-5"


def _api_key() -> str | None:
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    p = os.path.expanduser("~/.anthropic_key")
    return open(p).read().strip() if os.path.exists(p) else None


class AnthropicAdapter:
    def __init__(self, *, model_id: str = DEFAULT_MODEL, max_tokens: int = 256, temperature: float = 0.0,
                 effort: str | None = "low", workspace_id: str | None = None, timeout_s: float = 120.0) -> None:
        import anthropic
        ws = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": ws} if ws else None
        self.model_id, self.max_tokens, self.temperature, self.effort, self.workspace_id = model_id, max_tokens, temperature, effort, ws
        self.client = anthropic.Anthropic(api_key=_api_key(), default_headers=headers, timeout=timeout_s, max_retries=2)

    def spec_fields(self) -> dict[str, Any]:
        return {"server": "anthropic_api", "model_id": self.model_id, "max_tokens": self.max_tokens, "temperature": self.temperature, "effort": self.effort}

    def invoke(self, png: bytes, system: str, user_text: str, structured: bool = True,
               schema: dict | None = None, parser=None) -> RawInvocation:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode("ascii")}},
                   {"type": "text", "text": user_text}]
        kw: dict[str, Any] = {"model": self.model_id, "max_tokens": self.max_tokens, "system": system, "messages": [{"role": "user", "content": content}]}
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        if structured:
            kw.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": schema or ACTION_JSON_SCHEMA}
        t0 = time.perf_counter_ns()
        resp = self.client.messages.create(**kw)
        dt = time.perf_counter_ns() - t0
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        if resp.stop_reason == "refusal":
            parsed = None
        elif structured:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        else:
            parsed = (parser or parse_uitars)(text)
        u = resp.usage
        return RawInvocation(parsed=parsed, invocation_id=str(getattr(resp, "id", "") or "anthropic"),
                             detail={"model_reported": resp.model, "finish_reason": resp.stop_reason,
                                     "input_tokens": getattr(u, "input_tokens", None), "output_tokens": getattr(u, "output_tokens", None),
                                     "reasoning_tokens": None, "raw_text": text[:500], "runtime_ns": dt, "api": "anthropic_messages"})
