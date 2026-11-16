"""Prompt 16: the EvoCUA native S2 grammar (cards/M4-cross-agent-prl.md)."""
import json

from chronotope.controller import (EVOCUA_TOOLS_DEF, PROMPT_COORDS, SYSTEM_PROMPTS, LocalOpenAIAdapter, VLMDirectController,
                                   parse_evocua)
from chronotope.spec import PROMPT_VERSIONS

ANSWER = ('<think>\nThe blue button sits near the centre.\n</think>\n\nAction: Click the blue button on the notification card.\n'
          '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [512, 310]}}\n</tool_call>')


def test_parse_evocua_variants():
    assert parse_evocua(ANSWER) == {"action": "click", "x": 512, "y": 310}
    assert parse_evocua('Action: wait.\n<tool_call>{"name": "computer_use", "arguments": {"action": "wait", "time": 2}}</tool_call>') == {"action": "wait"}
    assert parse_evocua('<tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [10.4, 20.6]}}</tool_call>') == {"action": "click", "x": 10, "y": 21}
    # a different action kind is not a click and not a wait
    assert parse_evocua('<tool_call>{"name": "computer_use", "arguments": {"action": "type", "text": "hi"}}</tool_call>') is None
    assert parse_evocua('<tool_call>{"name": "computer_use", "arguments": {"action": "left_click"}}</tool_call>') == {"action": "click"}
    assert parse_evocua("<think>still thinking") is None
    assert parse_evocua("") is None and parse_evocua(None) is None
    # only the part after the last </think> counts
    assert parse_evocua('<think>\n<tool_call>{"name": "computer_use", "arguments": {"action": "wait"}}</tool_call>\n</think>\nAction: none') is None


def test_prompt16_system_is_evocua_verbatim_layout():
    s = SYSTEM_PROMPTS[16]
    assert s.startswith("# Tools") and "<tools>\n" + json.dumps(EVOCUA_TOOLS_DEF) + "\n</tools>" in s
    assert "The screen's resolution is 1000x1000." in json.loads(s.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0])["function"]["description"]
    assert PROMPT_COORDS[16] == "norm999" and 16 in PROMPT_VERSIONS


class _Adapter:
    def __init__(self): self.calls = []
    def spec_fields(self): return {"server": "fake"}
    def invoke(self, png, system, user_text, structured=True, schema=None, parser=None):
        self.calls.append({"system": system, "user_text": user_text, "structured": structured, "parser": parser})
        from chronotope.controller import RawInvocation
        return RawInvocation(parsed=parser(ANSWER), invocation_id="i1", detail={"raw_text": ANSWER[-200:]})


def test_prompt16_controller_maps_999_grid_to_canvas_and_puts_instruction_in_user_turn():
    from chronotope.controller import CallBudget
    ad = _Adapter()
    c = VLMDirectController(adapter=ad, budget=CallBudget(5), prompt_version=16, image_scale=2, invoke_on="frame_change",
                            canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert c.system == SYSTEM_PROMPTS[16]
    assert "Instruction: At some point a notification card" in c.user_text and c.user_text.endswith("Previous actions:\nNone")
    assert c._to_canvas(512, 310) == {"kind": "click", "x": round(512 * 320 / 999), "y": round(310 * 240 / 999)}
    assert c._to_canvas(999, 999) == {"kind": "click", "x": 320, "y": 240}
    r = c._invoke(b"png", "ignored default text")
    assert ad.calls[-1]["structured"] is False and ad.calls[-1]["parser"] is parse_evocua
    assert ad.calls[-1]["user_text"] == c.user_text
    assert r.parsed == {"action": "click", "x": 512, "y": 310}
    h16 = c.spec_hash()
    c7 = VLMDirectController(adapter=ad, budget=CallBudget(5), prompt_version=7, image_scale=2, invoke_on="frame_change",
                             canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert h16 != c7.spec_hash()


def test_adapter_parser_default_is_uitars(monkeypatch):
    """The unstructured path still parses UI-TARS answers when no parser is passed."""
    from chronotope import controller as m
    seen = {}
    monkeypatch.setattr(m, "parse_uitars", lambda t: seen.setdefault("t", t) and {"action": "wait"})
    class Resp:
        headers = {}
        def read(self): return json.dumps({"choices": [{"message": {"content": "Action: wait()"}}], "usage": {}}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(m.urllib.request if hasattr(m, "urllib") else __import__("urllib.request").request, "urlopen", lambda req, timeout: Resp())
    ad = LocalOpenAIAdapter(base_url="http://x/v1", model_id="m", max_tokens=8, temperature=0.0, enable_thinking=False, reasoning_effort=None)
    r = ad.invoke(b"png", "sys", "user", structured=False)
    assert seen["t"] == "Action: wait()" and r.parsed == {"action": "wait"}


def test_nowait_variants_share_grammar_and_add_one_sentence():
    from chronotope.controller import NOWAIT_SENTENCE, UITARS_INSTRUCTION, UITARS_INSTRUCTION_NOWAIT, CallBudget
    for k in UITARS_INSTRUCTION:
        assert UITARS_INSTRUCTION_NOWAIT[k] == UITARS_INSTRUCTION[k] + " " + NOWAIT_SENTENCE
    assert PROMPT_COORDS[18] == "norm999" and PROMPT_COORDS[19] == "image_px" and PROMPT_COORDS[20] == "uitars_abs"
    assert all(v in PROMPT_VERSIONS for v in (18, 19, 20))
    assert SYSTEM_PROMPTS[18] == SYSTEM_PROMPTS[16] and SYSTEM_PROMPTS[19] == SYSTEM_PROMPTS[17] and SYSTEM_PROMPTS[20] == SYSTEM_PROMPTS[7]
    c16 = VLMDirectController(adapter=LocalOpenAIAdapter(model_id="m", base_url="http://x", max_tokens=8, temperature=0.0, enable_thinking=True, reasoning_effort=None),
                              budget=CallBudget(10), prompt_version=16, image_scale=2, invoke_on="frame_change",
                              canvas_w_px=320, canvas_h_px=240, skin="notification")
    c18 = VLMDirectController(adapter=LocalOpenAIAdapter(model_id="m", base_url="http://x", max_tokens=8, temperature=0.0, enable_thinking=True, reasoning_effort=None),
                              budget=CallBudget(10), prompt_version=18, image_scale=2, invoke_on="frame_change",
                              canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert c18.system == c16.system and c18.user_text == c16.user_text.replace(UITARS_INSTRUCTION["notification"], UITARS_INSTRUCTION_NOWAIT["notification"])
    assert NOWAIT_SENTENCE in c18.user_text and NOWAIT_SENTENCE not in c16.user_text
    c20 = VLMDirectController(adapter=LocalOpenAIAdapter(model_id="m", base_url="http://x", max_tokens=8, temperature=0.0, enable_thinking=True, reasoning_effort=None),
                              budget=CallBudget(10), prompt_version=20, image_scale=2, invoke_on="frame_change",
                              canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert NOWAIT_SENTENCE in c20.system and c20.user_text is None


def test_prompt21_fara_native_grammar():
    from chronotope.controller import FARA_USER_TEMPLATE, CallBudget
    assert PROMPT_COORDS[21] == "norm1000" and 21 in PROMPT_VERSIONS
    assert SYSTEM_PROMPTS[21].startswith("You are Fara, a computer use agent") and "computer_use" in SYSTEM_PROMPTS[21] and "<tool_call>" in SYSTEM_PROMPTS[21]
    c = VLMDirectController(adapter=LocalOpenAIAdapter(model_id="m", base_url="http://x", max_tokens=8, temperature=0.0, enable_thinking=False, reasoning_effort=None),
                            budget=CallBudget(10), prompt_version=21, image_scale=2, invoke_on="frame_change", canvas_w_px=320, canvas_h_px=240, skin="notification")
    assert c.system == SYSTEM_PROMPTS[21] and "notification card" in c.user_text and "do not wait" in c.user_text
    assert parse_evocua('I see the card.\n<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [500, 250]}}\n</tool_call>') == {"action": "click", "x": 500, "y": 250}
    assert c._to_canvas(500, 250) == {"kind": "click", "x": 160, "y": 60}
