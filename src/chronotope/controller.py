"""Controllers for the Paused, Replay, and Live task harness.

Controllers accept published observations and return a concrete action or an
abstention. Binding controllers can refresh an action argument at commitment
without changing the selected action kind.
"""
from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .hashing import hash_obj
from .png import encode_png, to_rgb

INVOCATION_FIXED_INPUT = "fixed_input"  # the only invocation kind that exists before Gate 5


@dataclass(frozen=True)
class Observation:
    observation_id: int
    source_event_id: int
    source_time_ns: int
    publication_time_ns: int  # O, harness time
    payload_hash: str
    observation_version: int
    payload: bytes


@dataclass(frozen=True)
class ControlUpdate:
    """What admitting one observation did to the controller's control state (§8)."""
    relevance: str  # "action" | "none"
    controller_state_hash: str


@dataclass(frozen=True)
class Decision:
    action: dict | None              # complete concrete action, or None ("no action this decision")
    declared_compute_ns: int | None  # consumed by the SIMULATED clock only; wall clock measures
    invocation_kind: str = INVOCATION_FIXED_INPUT
    invocation_id: str = "local"     # model request id for VLMs
    invocation_detail: dict[str, Any] = field(default_factory=dict)  # reported model, usage, stop reason


class Controller(Protocol):
    def spec_hash(self) -> str: ...
    def admit(self, obs: Observation) -> ControlUpdate: ...
    def ready_to_decide(self) -> bool: ...
    def decide(self) -> Decision: ...


def _find_target_box(payload: bytes, w: int, h: int, color: tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    """Bounding box of all pixels exactly equal to `color` in an RGB payload (numpy; 2026-08-26: the pure-Python loop
    took ~30 ms per 320x240 frame, and the V1 validator and A8 prepared controllers call this on every admitted frame)."""
    import numpy as np
    payload = to_rgb(payload, w, h)
    a = np.frombuffer(payload, dtype=np.uint8)
    if a.size != w * h * 3:
        raise ValueError(f"payload size {a.size} is not {w}x{h}x3")
    hit = np.all(a.reshape(h, w, 3) == np.asarray(color, dtype=np.uint8), axis=2)
    ys, xs = np.nonzero(hit)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class Correction:
    """A separately identified coordinate correction applied at commitment (§9.4). NOT a
    same-decision update: the decision is fixed-input; only its coordinates are re-bound."""
    action: dict
    binding: str
    observation_id: int | None
    detail: dict[str, Any]


class ScriptedOracleController:
    """Control state = perceived target box (or None). Its version advances only when that box
    changes, so an irrelevant-motion frame does not advance it (§7.1).

    bindings (S3.2): state_at_B    — click the center seen at B (the direct controller).
                     newest_at_D   — at D, re-bind to the newest admitted frame's center.
                     predicted_at_A— at D, extrapolate the newest center to the expected receipt
                                     time using a velocity estimate (first seen -> newest) and
                                     the declared expected observation and delivery delays.
                     true_at_A     — oracle upper bound: the harness supplies the true box at A.
                     prepared      — A8-PREP: the click is prepared BEFORE the event from the skin's fixed layout
                                     (prepared_xy, supplied by the builder from public skin geometry, never from
                                     the manifest) and released with compute 0 at the first frame in which the
                                     target is detected. On a seeded layout the prepared content is wrong.
                     restart_if_changed — at D, if the perceived box differs from the one at B, discard the
                                     proposal and start a new decision (the scripted re-decision baseline).
    action_kind "type" (rotating_code): the perceived state is the code decoded from the panel (decoder.py);
                     state_at_B types the value read at B, newest_at_D re-decodes at D, prepared has nothing to
                     prepare and never acts.
    """

    kind = "scripted_oracle"

    def __init__(self, compute_ns: int, coordinate_offset_px: int, canvas_w_px: int, canvas_h_px: int,
                 wall_clock: bool = False, binding: str = "state_at_B",
                 expected_observation_delay_ns: int | None = None, expected_delivery_delay_ns: int | None = None,
                 true_box_at_receipt=None, target_color: tuple[int, int, int] = (255, 255, 255),
                 prepared_xy: tuple[int, int] | None = None, action_kind: str = "click", code_digits: int = 4) -> None:
        self.target_color = tuple(target_color)
        self.compute_ns = compute_ns
        self.coordinate_offset_px = coordinate_offset_px
        self.w, self.h = canvas_w_px, canvas_h_px
        self.wall_clock = wall_clock
        self.binding = binding
        self.exp_obs, self.exp_del = expected_observation_delay_ns, expected_delivery_delay_ns
        self.true_box_at_receipt = true_box_at_receipt
        self.prepared_xy = prepared_xy
        self.action_kind, self.code_digits = action_kind, code_digits
        if binding == "prepared" and action_kind == "click" and prepared_xy is None:
            raise ValueError("binding prepared needs prepared_xy (the skin's fixed layout)")
        self._box: tuple[int, int, int, int] | None = None
        self._value: str | None = None                                    # action_kind type: decoded code
        self._first: tuple[Observation, tuple[int, int]] | None = None   # first target sighting
        self._newest: tuple[Observation, tuple[int, int]] | None = None  # newest target sighting
        self._newest_value: tuple[Observation, str] | None = None
        self._box_at_b: tuple[int, int, int, int] | None = None
        self._decided = False

    def spec_hash(self) -> str:
        d = {"kind": self.kind, "compute_ns": self.compute_ns, "binding": self.binding,
             "coordinate_offset_px": self.coordinate_offset_px,
             "expected_observation_delay_ns": self.exp_obs, "expected_delivery_delay_ns": self.exp_del}
        if self.action_kind != "click":   # added for rotating_code (A8/V1); omitted at default so recorded traces keep their hash
            d["action_kind"] = self.action_kind
        return hash_obj(d)

    def admit(self, obs: Observation) -> ControlUpdate:
        if self.action_kind == "type":
            from .decoder import decode
            v = decode(obs.payload, self.w, self.h, self.code_digits).value
            relevance = "action" if v != self._value else "none"
            self._value = v
            if v is not None:
                self._newest_value = (obs, v)
            return ControlUpdate(relevance=relevance, controller_state_hash=hash_obj({"value": v}))
        box = _find_target_box(obs.payload, self.w, self.h, self.target_color)
        relevance = "action" if box != self._box else "none"
        self._box = box
        if box is not None:
            c = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            if self._first is None:
                self._first = (obs, c)
            self._newest = (obs, c)
        return ControlUpdate(relevance=relevance, controller_state_hash=hash_obj({"box": box}))

    def ready_to_decide(self) -> bool:
        if self.action_kind == "type":
            return self.binding != "prepared" and self._value is not None and not self._decided
        return self._box is not None and not self._decided

    def _clamp(self, x: int, y: int) -> dict:
        return {"kind": "click", "x": max(0, min(self.w - 1, x)), "y": max(0, min(self.h - 1, y))}

    def decide(self) -> Decision:
        self._decided = True
        if self.action_kind == "type":
            assert self._value is not None
            if self.wall_clock and self.compute_ns > 0:
                time.sleep(self.compute_ns / 1e9)
            return Decision(action={"kind": "type", "value": self._value}, declared_compute_ns=self.compute_ns)
        assert self._box is not None
        if self.binding == "prepared":
            # the complete action existed before the event; detection releases it with no compute
            px, py = self.prepared_xy
            return Decision(action=self._clamp(px + self.coordinate_offset_px, py), declared_compute_ns=0,
                            invocation_id="prepared", invocation_detail={"outcome": "release_prepared"})
        if self.wall_clock and self.compute_ns > 0:
            time.sleep(self.compute_ns / 1e9)  # consume real time
        self._box_at_b = self._box
        x0, y0, x1, y1 = self._box
        return Decision(action=self._clamp((x0 + x1) // 2 + self.coordinate_offset_px, (y0 + y1) // 2),
                        declared_compute_ns=self.compute_ns)

    def restart(self) -> None:
        """restart_if_changed: the proposal is discarded; the next decision starts from the newest frame."""
        self._decided, self._box_at_b = False, None

    def correct_at_commit(self, commit_time_ns: int) -> "Correction | Restart | None":
        """Called by the runner at D after frames from B->D have been coalesced and admitted."""
        if self.binding in ("state_at_B", "prepared"):
            return None
        if self.action_kind == "type":
            if self.binding != "newest_at_D" or self._newest_value is None:
                return None
            obs, v = self._newest_value
            return Correction({"kind": "type", "value": v}, "decoded_at_D", obs.observation_id,
                              {"match_score": 1.0, "matched": True, "moved_px": 0.0, "track_ns": 0, "decoded_value": v,
                               "source_time_ns": obs.source_time_ns})
        if self._newest is None:
            return None
        obs, (cx, cy) = self._newest
        if self.binding == "restart_if_changed":
            if self._box_at_b is not None and self._box != self._box_at_b:
                return Restart("box_changed_since_B", obs.observation_id,
                               {"box_at_B": list(self._box_at_b), "box_at_D": list(self._box) if self._box else None})
            return None
        if self.binding == "newest_at_D":
            return Correction(self._clamp(cx + self.coordinate_offset_px, cy), self.binding, obs.observation_id,
                              {"source_time_ns": obs.source_time_ns})
        if self.binding == "predicted_at_A":
            assert self._first is not None and self.exp_obs is not None and self.exp_del is not None
            f_obs, (fx, fy) = self._first
            dt = obs.source_time_ns - f_obs.source_time_ns
            vx = (cx - fx) / dt if dt > 0 else 0.0   # px per ns
            vy = (cy - fy) / dt if dt > 0 else 0.0
            horizon = (commit_time_ns - obs.publication_time_ns) + self.exp_obs + self.exp_del
            px, py = round(cx + vx * horizon), round(cy + vy * horizon)
            return Correction(self._clamp(px + self.coordinate_offset_px, py), self.binding, obs.observation_id,
                              {"velocity_px_per_s": [vx * 1e9, vy * 1e9], "horizon_ns": horizon,
                               "frames_for_velocity": [f_obs.observation_id, obs.observation_id]})
        if self.binding == "true_at_A":
            assert self.true_box_at_receipt is not None
            b = self.true_box_at_receipt(commit_time_ns)
            tx, ty = b.center()
            return Correction(self._clamp(tx + self.coordinate_offset_px, ty), self.binding, None,
                              {"oracle": "harness supplied the true box at receipt"})
        raise ValueError(self.binding)


# ---- VLM direct controller -----------------------------------------------------------------

SYSTEM_PROMPTS = {
    1: ("You control a graphical interface by clicking. You are shown one screenshot of a "
        "{w}x{h}-pixel screen (origin top-left, x to the right, y down). A bright white square "
        "target appears at an unknown location. If the target is visible, answer with "
        "action \"click\" and the integer pixel coordinates of its center. If no target is visible, "
        "answer with action \"wait\". Answer with the JSON object only."),
    2: ("You control a desktop by clicking. You are shown one screenshot of a {w}x{h}-pixel screen "
        "(origin top-left, x to the right, y down). At some point a notification card with a blue "
        "button appears at an unknown position and expires shortly after. If the card is visible, "
        "answer with action \"click\" and the integer pixel coordinates of the center of its blue "
        "button. If no card is visible, answer with action \"wait\". Answer with the JSON object only."),
    3: ("You control a game by clicking. You are shown one screenshot of a {w}x{h}-pixel screen "
        "(origin top-left, x to the right, y down). A red balloon appears at an unknown position and "
        "may drift; it disappears after a short time. If the balloon is visible, answer with action "
        "\"click\" and the integer pixel coordinates of the center of the red balloon. If no balloon is "
        "visible, answer with action \"wait\". Answer with the JSON object only."),
    # Same tasks as 2 and 3, but coordinates on the 0-1000 normalized grid that Qwen-VL models use
    # natively (x: 0 = left edge, 1000 = right edge; y: 0 = top, 1000 = bottom).
    4: ("You control a desktop by clicking. You are shown one screenshot. At some point a notification "
        "card with a blue button appears at an unknown position and expires shortly after. If the card "
        "is visible, answer with action \"click\" and the coordinates of the center of its blue button on "
        "a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no "
        "card is visible, answer with action \"wait\". Answer with the JSON object only."),
    5: ("You control a game by clicking. You are shown one screenshot. A red balloon appears at an "
        "unknown position and may drift; it disappears after a short time. If the balloon is visible, "
        "answer with action \"click\" and the coordinates of the center of the red balloon on a 0-1000 "
        "grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no balloon "
        "is visible, answer with action \"wait\". Answer with the JSON object only."),
    6: ("You control a graphical interface by clicking. You are shown one screenshot. A bright white "
        "square target appears at an unknown location. If the target is visible, answer with action "
        "\"click\" and the coordinates of its center on a 0-1000 grid (x: 0 = left edge, 1000 = right "
        "edge; y: 0 = top edge, 1000 = bottom edge). If no target is visible, answer with action "
        "\"wait\". Answer with the JSON object only."),
}
# 7: UI-TARS-1.5 native grammar (cards/WI-007 U21: the one declared alternative after the shared JSON
# grammar failed its competence gate). Output "Thought: ... Action: click(point='<point>x y</point>')" or
# "Action: wait()"; coordinates are ABSOLUTE pixels of the image after the Qwen2.5-VL smart resize
# (UI-TARS README_coordinates.md); the controller maps them back to the canvas. No structured output.
UITARS_INSTRUCTION = {
    "notification": ("At some point a notification card with a blue button appears on the desktop and expires shortly "
                     "after. If the card is visible, click the center of its blue button. If no card is visible, wait."),
    "balloon": ("A red balloon appears at an unknown position and may drift; it disappears after a short time. If the "
                "balloon is visible, click the center of the red balloon. If no balloon is visible, wait."),
    "plain": ("A bright white square target appears at an unknown location. If the target is visible, click its center. "
              "If no target is visible, wait."),
}
SYSTEM_PROMPTS[7] = (
    "You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the "
    "next action to complete the task.\n\n## Output Format\n```\nThought: ...\nAction: ...\n```\n\n## Action Space\n\n"
    "click(point='<point>x1 y1</point>')\nwait() #Sleep for 5s and take a screenshot to check for any changes.\n\n"
    "## Note\n- Use English in `Thought` part.\n- Write a small plan and finally summarize your next action (with its "
    "target element) in one sentence in `Thought` part.\n\n## User Instruction\n{instruction}")

# 16: EvoCUA native S2 grammar (cards/M4-cross-agent-prl.md U60). The system prompt, tool definition, and
# message layout are those of meituan/EvoCUA mm_agents/evocua/prompts.py + evocua_agent.py (S2, coordinate_type
# "relative"): the model answers "Action: ...\n<tool_call>{"name": "computer_use", "arguments": {...}}</tool_call>"
# with coordinates on a 0-999 grid of the screenshot ("The screen's resolution is 1000x1000"); EvoCUA's harness maps
# x -> x * width / 999. The task instruction goes in the USER turn, as in EvoCUA's first-turn message. The chat
# template opens a <think> block unconditionally, so the answer is parsed after the last </think>. Its image
# preprocessing (smart_resize, factor 32) leaves a 640x480 frame unchanged, so the model sees our frame as is.
EVOCUA_ACTION_DESCRIPTION = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `key_down`: Press and HOLD the specified key(s) down in order (no release). Use this for stateful holds like holding Shift while clicking.
* `key_up`: Release the specified key(s) in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
"""
EVOCUA_DESCRIPTION_PROMPT = """Use a mouse and keyboard to interact with a computer, and take screenshots.
* This is an interface to a desktop GUI. You must click on desktop icons to start applications.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.
* The screen's resolution is 1000x1000.
* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked."""
EVOCUA_TOOLS_DEF = {
    "type": "function",
    "function": {
        "name_for_human": "computer_use",
        "name": "computer_use",
        "description": EVOCUA_DESCRIPTION_PROMPT,
        "parameters": {
            "properties": {
                "action": {"description": EVOCUA_ACTION_DESCRIPTION,
                           "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag",
                                    "right_click", "middle_click", "double_click", "triple_click", "scroll",
                                    "wait", "terminate", "key_down", "key_up"],
                           "type": "string"},
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"},
                "pixels": {"description": "The amount of scrolling.", "type": "number"},
                "time": {"description": "The seconds to wait.", "type": "number"},
                "status": {"description": "The status of the task.", "type": "string", "enum": ["success", "failure"]},
            },
            "required": ["action"],
            "type": "object",
        },
        "args_format": "Format the arguments as a JSON object.",
    },
}
SYSTEM_PROMPTS[16] = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with "
                      "function signatures within <tools></tools> XML tags:\n<tools>\n" + json.dumps(EVOCUA_TOOLS_DEF) +
                      "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                      "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                      "</tool_call>\n\n# Response format\n\nResponse format for every step:\n1) Action: a short imperative "
                      "describing what to do in the UI.\n2) A single <tool_call>...</tool_call> block containing only the JSON: "
                      "{\"name\": <function-name>, \"arguments\": <args-json-object>}.\n\nRules:\n- Output exactly in the order: "
                      "Action, <tool_call>.\n- Be brief: one sentence for Action.\n- Do not output anything else outside those "
                      "parts.\n- If finishing, use action=terminate in the tool call.")
EVOCUA_USER_TEMPLATE = ("\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
                        "Instruction: {instruction}\n\nPrevious actions:\nNone")

# Characters of a model answer kept in the decision record. 500 cut every Fara1.5-9B answer that exceeded it
# (17% of its decisions, 2026-08-28); the parser always sees the whole answer.
RAW_TEXT_KEEP = 2000

_EVOCUA_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_BARE_TOOL_CALL = re.compile(r"\{\s*\"name\"")


def _bare_tool_call(text: str) -> dict | None:
    """Fallback for answers that carry the tool call as a bare JSON object without the <tool_call> tags (Fara1.5's
    trained habit, 2026-08-28: every Fara decision was well formed and every one was discarded). The last object
    starting with {"name" that decodes and names computer_use is the call."""
    dec = json.JSONDecoder()
    for m in reversed(list(_BARE_TOOL_CALL.finditer(text))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("name") == "computer_use":
            return obj
    return None


def parse_evocua(text: str) -> dict | None:
    """EvoCUA S2 answer -> {action: click, x, y} (0-999 grid) | {action: wait} | None. Text after the last </think>
    is parsed; a tool call whose action is neither left_click nor wait is unparseable (a different action kind).
    A tagged <tool_call> wins; without one, a bare {"name": "computer_use", ...} object is accepted (2026-09-02)."""
    if text is None:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = _EVOCUA_TOOL_CALL.search(text)
    if m:
        try:
            call = json.loads(m.group(1))
        except ValueError:
            return None
    else:
        call = _bare_tool_call(text)
        if call is None:
            return None
    args = call.get("arguments") if isinstance(call, dict) else None
    if call.get("name") != "computer_use" or not isinstance(args, dict):
        return None
    act = args.get("action")
    if act == "wait":
        return {"action": "wait"}
    if act == "left_click":
        c = args.get("coordinate")
        if isinstance(c, (list, tuple)) and len(c) == 2 and all(isinstance(v, (int, float)) for v in c):
            return {"action": "click", "x": int(round(c[0])), "y": int(round(c[1]))}
        return {"action": "click"}
    return None


# 17: Agent S3 (cards/M4-cross-agent-prl.md U63). The "prompt" is the task instruction handed to AgentS3.predict();
# the framework's own planner/grounder prompts are untouched. The service (studies/agents3_service.py) returns the
# pyautogui code Agent S3 would execute; a click's coordinates are absolute pixels of the screenshot it saw.
SYSTEM_PROMPTS[17] = "{instruction}"
# 21: Fara1.5 native grammar (Microsoft, 2026-08-28). The system prompt is the one Fara's own harness renders
# (github.com/microsoft/fara, agents/fara/_prompts.py get_computer_use_system_prompt with display_size=1000: identity,
# critical points, the computer_use tool schema, the <tool_call> format), stored verbatim in
# specs/prompts/fara15_system_prompt_display1000.txt. The task instruction is the first user turn's text, after the
# screenshot, as in the harness. Answers: reasoning + <tool_call>{"name":"computer_use","arguments":{"action":
# "left_click","coordinate":[x,y]}}</tool_call> with coordinates on the 1000x1000 display (proc_coords in the
# harness maps them to the viewport); parsed with parse_evocua (same tool-call shape), mapped as norm1000.
import pathlib as _pl
_FARA_PROMPT = _pl.Path(__file__).resolve().parents[2] / "specs" / "prompts" / "fara15_system_prompt_display1000.txt"
SYSTEM_PROMPTS[21] = _FARA_PROMPT.read_text() if _FARA_PROMPT.exists() else "You are Fara, a computer use agent (CUA) specialized for web browsers."
FARA_USER_TEMPLATE = "{instruction}"

# 18/19/20: the no-wait variants of 16/17/7 (same grammar, same coordinate space; only the task text differs)
SYSTEM_PROMPTS[18] = SYSTEM_PROMPTS[16]
SYSTEM_PROMPTS[19] = SYSTEM_PROMPTS[17]
SYSTEM_PROMPTS[20] = SYSTEM_PROMPTS[7]

_AGENTS3_CLICK = re.compile(r"pyautogui\.click\(\s*(-?\d+)\s*,\s*(-?\d+)")
_AGENTS3_WAIT = re.compile(r"time\.sleep\(|pyautogui\.sleep\(")


def parse_agents3(exec_code: list[str] | str | None) -> dict | None:
    """Agent S3 exec code -> {action: click, x, y} (screenshot pixels) | {action: wait} | None (done/fail/other)."""
    if exec_code is None:
        return None
    codes = [exec_code] if isinstance(exec_code, str) else list(exec_code)
    for code in codes:
        if not isinstance(code, str):
            continue
        m = _AGENTS3_CLICK.search(code)
        if m:
            return {"action": "click", "x": int(m.group(1)), "y": int(m.group(2))}
    for code in codes:
        if isinstance(code, str) and (_AGENTS3_WAIT.search(code) or code.strip() == "WAIT"):
            return {"action": "wait"}
    return None


class AgentS3Adapter:
    """Adapter for the Agent S3 service (server "agents3_service"); same .invoke/.spec_fields surface as
    LocalOpenAIAdapter. model_id names the pair "planner+grounder" and must equal what the service reports."""

    def __init__(self, *, base_url: str, model_id: str, timeout_s: float = 900.0) -> None:
        self.base_url, self.model_id, self.timeout_s = base_url.rstrip("/"), model_id, timeout_s
        self._fresh = True   # the first invocation of a controller instance starts a new Agent S3 trajectory

    def spec_fields(self) -> dict[str, Any]:
        return {"server": "agents3_service", "model_id": self.model_id}

    def invoke(self, png: bytes, system: str, user_text: str, structured: bool = False,
               schema: dict | None = None, parser=None) -> RawInvocation:
        import urllib.error
        import urllib.request
        body = {"png_b64": base64.standard_b64encode(png).decode("ascii"), "instruction": system, "reset": self._fresh}
        self._fresh = False
        req = urllib.request.Request(self.base_url + "/predict", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            try:
                raw = json.loads(e.read().decode())
            except ValueError:
                raw = {"error": f"HTTP {e.code}"}
        codes = raw.get("exec_code")
        text = "\n".join(codes) if isinstance(codes, list) else (raw.get("error") or "")
        return RawInvocation(parsed=parse_agents3(codes), invocation_id=f"agents3-{time.monotonic_ns()}",
                             detail={"model_reported": raw.get("model_reported"), "finish_reason": "error" if "error" in raw else "stop",
                                     "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                                     "agent_elapsed_ns": raw.get("elapsed_ns"), "plan_code": (raw.get("plan_code") or "")[:500],
                                     "grounding_wh": raw.get("grounding_wh"), "raw_text": text[:RAW_TEXT_KEEP]})


# 8: balloon_select (C3), 0-1000 grid: several balloons, the instruction names one by colour
SYSTEM_PROMPTS[8] = ("You control a game by clicking. You are shown one screenshot. Three balloons of different colors "
                     "(red, blue, yellow) appear at unknown positions and may drift; they disappear after a short time. "
                     "Your task: if the {color} balloon is visible, answer with action \"click\" and the coordinates of the "
                     "center of the {color} balloon on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, "
                     "1000 = bottom edge). Do not click any other balloon. If no balloons are visible, answer with action "
                     "\"wait\". Answer with the JSON object only.")
UITARS_INSTRUCTION["balloon_select"] = ("Three balloons of different colors (red, blue, yellow) appear at unknown positions and may "
                                        "drift; they disappear after a short time. If the {color} balloon is visible, click the center "
                                        "of the {color} balloon and no other balloon. If no balloons are visible, wait.")
# No-wait variant (M4c, 2026-08-27): EvoCUA's trained first step on a fresh screen is wait(0.1 s) + re-screenshot,
# which a frame-change-invoked harness never answers (0/10 at the M4 gate with the card in view). One sentence in the
# TASK text (the frameworks' own prompts stay verbatim) tells every native-grammar agent the same thing the JSON
# prompts already imply: act in this step. Prompts 18 (EvoCUA), 19 (Agent S3), 20 (UI-TARS) = 16, 17, 7 + this sentence.
NOWAIT_SENTENCE = ("The target expires within seconds. When it is visible, click it in this same step: do not wait, do not "
                   "take another screenshot and do not check the interface first.")
UITARS_INSTRUCTION_NOWAIT = {k: v + " " + NOWAIT_SENTENCE for k, v in UITARS_INSTRUCTION.items()}

# 9: player skin (A1/A2 sponsor skip, 0-1000 grid): prompt 4 with the nouns changed (cards/A1-sponsor-skip-agent.md)
SYSTEM_PROMPTS[9] = ("You control a video player by clicking. You are shown one screenshot. At some point a sponsored "
                     "segment starts: a yellow banner appears at the bottom-left of the video and an orange button with "
                     "a white border appears somewhere over the video; the segment ends shortly after. If the orange "
                     "button is visible, answer with action \"click\" and the coordinates of the center of the orange "
                     "button on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom "
                     "edge). If no orange button is visible, answer with action \"wait\". Answer with the JSON object only.")
# 10: chart skin (B1 live trading, 0-1000 grid): the signal IS the order button turning green (cards/B1-live-trading-agent.md)
SYSTEM_PROMPTS[10] = ("You control a trading terminal by clicking. You are shown one screenshot: a price chart on the "
                      "left and an order panel on the right. When a buy signal is active, the order button in the "
                      "right-hand panel turns green; otherwise it is grey. If the green order button is visible, answer "
                      "with action \"click\" and the coordinates of the center of the green button on a 0-1000 grid "
                      "(x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no green button "
                      "is visible, answer with action \"wait\". Answer with the JSON object only.")

# 11: rotating_code (C4, cards/C4-late-bound-text.md): the action is type(value); no coordinates
SYSTEM_PROMPTS[11] = ("You control a desktop by typing. You are shown one screenshot. At some point a card appears with a "
                      "yellow panel showing a {n}-digit code and, below it, an empty text field; the code may be replaced "
                      "by a different one while the card is shown. Your task is to type the code that is currently shown "
                      "into the field. If the card is visible, answer with action \"type\" and the {n} digits currently "
                      "shown as a string in \"value\". If no card is visible, answer with action \"wait\". Answer with "
                      "the JSON object only.")

# 12: browser page (G1, cards/G1-realistic-reflow.md), 0-1000 grid
SYSTEM_PROMPTS[12] = ("You control a web browser by clicking. You are shown one screenshot of a web page. The page has a "
                      "purple \"Continue\" button; a notice may be inserted above it, moving it down. If the Continue button "
                      "is visible, answer with action \"click\" and the coordinates of the center of the Continue button on a "
                      "0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no Continue "
                      "button is visible, answer with action \"wait\". Answer with the JSON object only.")

# 13: rotating_code with kind_change (V1, cards/V1-action-kind-validation.md): the code may expire, in which case the
# panel and field are replaced by a blue dismiss button; the grammar allows type, click, and wait (MIXED_JSON_SCHEMA)
SYSTEM_PROMPTS[13] = ("You control a desktop by typing and clicking. You are shown one screenshot. At some point a card "
                      "appears with a yellow panel showing a {n}-digit code and, below it, an empty text field; the code may "
                      "be replaced by a different one while the card is shown, or it may expire, in which case the panel "
                      "and the field are replaced by a blue button. If the code panel is visible, answer with action "
                      "\"type\" and the {n} digits currently shown as a string in \"value\". If a blue button is visible "
                      "instead, answer with action \"click\" and the coordinates of the center of the blue button on a "
                      "0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top edge, 1000 = bottom edge). If no card "
                      "is visible, answer with action \"wait\". Answer with the JSON object only.")

# 14: the prepared instruction (A8-PREP, cards/A8-preparation.md). ONE call on the pre-event frame: the layout is known in
# advance (the notification always appears at the same place; the builder supplies that place from the skin's fixed
# geometry, never from the manifest), and the model is asked to produce the complete click BEFORE the card appears.
# The call therefore tests whether the model will act on prior layout knowledge, not its perception of the card.
SYSTEM_PROMPTS[14] = ("You control a desktop by clicking. You are shown one screenshot. A notification card with a blue "
                      "button is about to appear; on this desktop the card always appears at the same place, and the "
                      "center of its blue button is at approximately ({x}, {y}) on a 0-1000 grid (x: 0 = left edge, "
                      "1000 = right edge; y: 0 = top edge, 1000 = bottom edge). Prepare the click now, before the card "
                      "appears: answer with action \"click\" and the coordinates of the button's center on that grid. "
                      "Answer with the JSON object only.")

# 15: the live two-step page (G2, cards/G2-live-page.md): prompt 12 plus the confirm dialog of step 2
SYSTEM_PROMPTS[15] = ("You control a web browser by clicking. You are shown one screenshot of a web page. The page has a "
                      "purple \"Continue\" button; a notice may be inserted above it, moving it down. After Continue is "
                      "clicked, a small dialog with a white \"Confirm\" button outlined in green appears. If the dialog is visible, answer "
                      "with action \"click\" and the coordinates of the center of the Confirm button; otherwise, if the "
                      "Continue button is visible, answer with action \"click\" and the coordinates of the center of the "
                      "Continue button. Coordinates are on a 0-1000 grid (x: 0 = left edge, 1000 = right edge; y: 0 = top "
                      "edge, 1000 = bottom edge). If neither button is visible, answer with action \"wait\". Answer with "
                      "the JSON object only.")

# coordinate space each prompt asks for; the controller maps it to canvas pixels
PROMPT_COORDS = {1: "image_px", 2: "image_px", 3: "image_px", 4: "norm1000", 5: "norm1000", 6: "norm1000", 7: "uitars_abs",
                 8: "norm1000", 9: "norm1000", 10: "norm1000", 11: "none", 12: "norm1000", 13: "norm1000", 14: "norm1000", 15: "norm1000", 16: "norm999", 17: "image_px",
                 18: "norm999", 19: "image_px", 20: "uitars_abs", 21: "norm1000"}

_UITARS_POINT = re.compile(r"click\(\s*(?:point|start_box)\s*=\s*['\"]\s*(?:<point>)?\s*\(?\s*(\d+)\s*[, ]\s*(\d+)\s*\)?\s*(?:</point>)?\s*['\"]", re.S)
_UITARS_WAIT = re.compile(r"Action:\s*wait\s*\(", re.S)


def parse_uitars(text: str) -> dict | None:
    """UI-TARS action text -> the same dict shape as the JSON grammar ({action, x, y} | {action: wait})."""
    if text is None:
        return None
    m = _UITARS_POINT.search(text)
    if m:
        return {"action": "click", "x": int(m.group(1)), "y": int(m.group(2))}
    if _UITARS_WAIT.search(text) or re.search(r"\bwait\(\)", text):
        return {"action": "wait"}
    return None


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056) -> tuple[int, int]:
    """Qwen2.5-VL image preprocessing (as in UI-TARS README_coordinates.md): the model's absolute coordinates
    live in this resized frame. Returns (h_bar, w_bar)."""
    import math
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

# The concrete action contract, enforced by the decoder (structured output).
ACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["click", "wait"]},
                   "x": {"type": "integer"}, "y": {"type": "integer"}},
    "required": ["action"], "additionalProperties": False,
}
# C4: the typed-value grammar (prompt 11); the value is a digit string
VALUE_JSON_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["type", "wait"]},
                   "value": {"type": "string", "pattern": "^[0-9]+$"}},
    "required": ["action"], "additionalProperties": False,
}
# V1 (prompt 13): the action kind may change from type to click while the card is shown
MIXED_JSON_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ["type", "click", "wait"]},
                   "value": {"type": "string", "pattern": "^[0-9]+$"},
                   "x": {"type": "integer"}, "y": {"type": "integer"}},
    "required": ["action"], "additionalProperties": False,
}


@dataclass(frozen=True)
class RawInvocation:
    """What one model invocation returned, adapter-agnostic."""
    parsed: dict | None          # the JSON object the model produced (None if unparseable)
    invocation_id: str
    detail: dict[str, Any]       # reported model, usage, finish reason, raw text


class LocalOpenAIAdapter:
    """Chat-completions call to a LOCAL OpenAI-compatible server (vLLM) over stdlib urllib.
    No API keys. Structured output via response_format json_schema. One request per decision."""

    def __init__(self, *, base_url: str, model_id: str, max_tokens: int, temperature: float,
                 enable_thinking: bool, reasoning_effort: str | None, timeout_s: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id, self.max_tokens, self.temperature = model_id, max_tokens, temperature
        self.enable_thinking, self.reasoning_effort, self.timeout_s = enable_thinking, reasoning_effort, timeout_s

    def spec_fields(self) -> dict[str, Any]:
        return {"server": "openai_compatible_local", "model_id": self.model_id, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "enable_thinking": self.enable_thinking,
                "reasoning_effort": self.reasoning_effort, "schema": ACTION_JSON_SCHEMA}

    def invoke(self, png: bytes, system: str, user_text: str, structured: bool = True,
               schema: dict | None = None, parser=None) -> RawInvocation:
        """parser: text -> dict for an unstructured (native-grammar) answer; default parse_uitars."""
        import json
        import urllib.request
        body: dict[str, Any] = {
            "model": self.model_id, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": "data:image/png;base64,"
                                                                  + base64.standard_b64encode(png).decode("ascii")}},
                             {"type": "text", "text": user_text}]}],
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if structured:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "gui_action", "strict": True,
                                                                              "schema": schema or ACTION_JSON_SCHEMA}}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = json.loads(resp.read().decode())
            req_id = resp.headers.get("x-request-id") or raw.get("id") or "unknown"
        choice = raw["choices"][0]
        text = choice["message"].get("content") or ""
        if structured:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        else:
            parsed = (parser or parse_uitars)(text)
        usage = raw.get("usage") or {}
        return RawInvocation(parsed=parsed, invocation_id=str(req_id),
                             detail={"model_reported": raw.get("model"), "finish_reason": choice.get("finish_reason"),
                                     "input_tokens": usage.get("prompt_tokens"),
                                     "output_tokens": usage.get("completion_tokens"),
                                     "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                                     "raw_text": text[:RAW_TEXT_KEEP]})


class CallBudget:
    """Hard cap on model invocations for one study run."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls, self.calls = max_calls, 0

    def take(self) -> None:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"call budget of {self.max_calls} exhausted")
        self.calls += 1


def kw_code_digits(ctrl) -> int:
    return getattr(ctrl, "code_digits", 4)


class VLMDirectController:
    """Invokes the model on the first frame whose content differs from the last frame it was
    invoked on (the very first frame is the baseline). One fixed-input invocation per decision.

    `adapter` has .invoke(png, system, user_text) -> RawInvocation and .spec_fields(); tests pass a
    mock. Control state = hash of the newest admitted frame (any pixel change counts as
    action-relevant because this controller has no semantic model of its own).
    """

    kind = "vlm_direct"

    def __init__(self, *, adapter: Any, budget: CallBudget, prompt_version: int, image_scale: int,
                 invoke_on: str, canvas_w_px: int, canvas_h_px: int, skin: str = "plain",
                 instruction_color: str | None = None, target_color: tuple[int, int, int] | None = None,
                 prepared_xy: tuple[int, int] | None = None) -> None:
        if invoke_on != "frame_change":
            raise ValueError("only invoke_on=frame_change is implemented")
        self.adapter, self.budget = adapter, budget
        self.prompt_version, self.scale, self.invoke_on, self.skin = prompt_version, image_scale, invoke_on, skin
        self.instruction_color = instruction_color      # balloon_select: the colour the instruction names
        self.target_color = tuple(target_color) if target_color is not None else None   # evaluator-only detector (V1, A8)
        self.prepared_xy = prepared_xy                  # A8-PREP: the skin's fixed layout, canvas pixels
        self.w, self.h = canvas_w_px, canvas_h_px
        self.clock_ns = None                            # harness clock reader, bound by the runner (bind_clock)
        self._events: list[dict[str, Any]] = []         # proposal_discarded records, drained by the runner
        self.user_text: str | None = None                 # prompt 16: the instruction travels in the user turn
        if prompt_version in (7, 16, 17, 18, 19, 20, 21):
            key = "balloon_select" if instruction_color else skin
            table = UITARS_INSTRUCTION_NOWAIT if prompt_version >= 18 else UITARS_INSTRUCTION
            instruction = table[key].format(color=instruction_color or "")
            if prompt_version in (16, 18):
                self.system = SYSTEM_PROMPTS[prompt_version]
                self.user_text = EVOCUA_USER_TEMPLATE.format(instruction=instruction)
            elif prompt_version == 21:
                self.system = SYSTEM_PROMPTS[21]
                self.user_text = FARA_USER_TEMPLATE.format(instruction=instruction)
            else:
                self.system = SYSTEM_PROMPTS[prompt_version].format(instruction=instruction)
        elif prompt_version == 8:
            if not instruction_color:
                raise ValueError("prompt 8 needs the colour the instruction names (balloon_select)")
            self.system = SYSTEM_PROMPTS[8].format(color=instruction_color)
        elif prompt_version in (11, 13):
            self.system = SYSTEM_PROMPTS[prompt_version].format(n=kw_code_digits(self))
        elif prompt_version == 14:
            if prepared_xy is None:
                raise ValueError("prompt 14 needs prepared_xy (the skin's fixed layout)")
            self.system = SYSTEM_PROMPTS[14].format(x=prepared_xy[0] * 1000 // canvas_w_px, y=prepared_xy[1] * 1000 // canvas_h_px)
        else:
            self.system = SYSTEM_PROMPTS[prompt_version].format(w=canvas_w_px * image_scale, h=canvas_h_px * image_scale)
        self._newest: Observation | None = None
        self._baseline_hash: str | None = None   # hash of the frame last invoked on (or the first frame)
        self._done = False

    # --- shared by the direct, late-binding and pending controllers ---------------------------
    def _invoke(self, png: bytes, user_text: str) -> "RawInvocation":
        if PROMPT_COORDS[self.prompt_version] == "uitars_abs":
            return self.adapter.invoke(png, self.system, user_text, structured=False)
        if self.prompt_version in (17, 19):
            return self.adapter.invoke(png, self.system, user_text, structured=False, parser=parse_agents3)
        if PROMPT_COORDS[self.prompt_version] == "norm999" or self.prompt_version == 21:
            return self.adapter.invoke(png, self.system, self.user_text or user_text, structured=False, parser=parse_evocua)
        return self.adapter.invoke(png, self.system, user_text)

    def _to_canvas(self, x: int, y: int) -> dict:
        space = PROMPT_COORDS[self.prompt_version]
        if space == "norm1000":
            return {"kind": "click", "x": int(x * self.w // 1000), "y": int(y * self.h // 1000)}
        if space == "norm999":   # EvoCUA relative grid: x_scale = width / 999 (evocua_agent.py)
            return {"kind": "click", "x": int(round(x * self.w / 999)), "y": int(round(y * self.h / 999))}
        if space == "uitars_abs":
            W, H = self.w * self.scale, self.h * self.scale
            hr, wr = smart_resize(H, W)
            return {"kind": "click", "x": int(round(x * W / wr)) // self.scale, "y": int(round(y * H / hr)) // self.scale}
        return {"kind": "click", "x": x // self.scale, "y": y // self.scale}

    def spec_hash(self) -> str:
        return hash_obj({"kind": self.kind, "prompt_version": self.prompt_version, "prompt_hash": hash_obj(self.system),
                         **({"user_text_hash": hash_obj(self.user_text)} if getattr(self, "user_text", None) else {}),
                         "coordinate_space": PROMPT_COORDS[self.prompt_version],
                         "image_scale": self.scale, "invoke_on": self.invoke_on, **self.adapter.spec_fields()})

    def bind_clock(self, clock_ns) -> None:
        self.clock_ns = clock_ns

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self._events = self._events, []
        return ev

    def _now(self) -> int | None:
        return self.clock_ns() if self.clock_ns is not None else None

    def admit(self, obs: Observation) -> ControlUpdate:
        changed = self._newest is None or obs.payload_hash != self._newest.payload_hash
        self._newest = obs
        if self._baseline_hash is None:
            self._baseline_hash = obs.payload_hash
        return ControlUpdate(relevance="action" if changed else "none",
                             controller_state_hash=hash_obj({"frame": obs.payload_hash}))

    def ready_to_decide(self) -> bool:
        return (not self._done and self._newest is not None and self._baseline_hash is not None
                and self._newest.payload_hash != self._baseline_hash)

    def decide(self) -> Decision:
        assert self._newest is not None
        obs = self._newest
        self._baseline_hash = obs.payload_hash
        self.budget.take()
        png = encode_png(obs.payload, self.w, self.h, self.scale)
        r = self._invoke(png, "Screenshot attached. Click the target's center or wait.")
        detail = {**r.detail, "observation_id": obs.observation_id}
        p = r.parsed
        if not isinstance(p, dict) or p.get("action") not in ("click", "wait"):
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "unparseable"})
        if p["action"] == "wait":
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "wait"})
        if type(p.get("x")) is not int or type(p.get("y")) is not int:
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "click_without_coordinates"})
        self._done = True
        x, y = p["x"], p["y"]
        action = self._to_canvas(x, y)
        return Decision(action=action, declared_compute_ns=None, invocation_id=r.invocation_id,
                        invocation_detail={**detail, "outcome": "click", "image_x": x, "image_y": y})


# ---- action-kind validation (V1, cards/V1-action-kind-validation.md) -----------------------------

class _KindValidator:
    """Mixed into the binding controllers. The evaluator-only detector the binders use (the target box by its
    colour; the code panel by the decoder) says which action KIND the newest frame calls for: click when the
    target box is present, type when the code panel is present, wait when neither. A proposal whose kind
    contradicts the newest frame is discarded: a click/type at D returns Restart("kind") (a NEW decision begins,
    as after any restart); a wait is discarded when a frame published before that decision's end (i.e. available
    at its D) already shows the target/panel, and the next decision starts from that frame. Every discard is
    written to the trace as a pending_event record {event: proposal_discarded, reason: kind, ...} through the
    runner's drain_events hook (the runner is not modified). Nothing is discarded when the newest frame agrees
    with the proposal, so on a static target the validator is inert (precision test).
    """

    validate_kind = False
    _wait_end_ns: int | None = None

    def expected_kind(self, payload: bytes) -> str:
        if self.target_color is not None and _find_target_box(payload, self.w, self.h, self.target_color) is not None:
            return "click"
        if self.kind == "vlm_value_binding":
            from .decoder import decode
            if decode(payload, self.w, self.h, self.code_digits).panel is not None:
                return "type"
        return "wait"

    def _discard(self, proposal_kind: str, expected: str, obs: Observation) -> dict[str, Any]:
        detail = {"event": "proposal_discarded", "reason": "kind", "proposal_kind": proposal_kind,
                  "expected_kind": expected, "observation_id": obs.observation_id, "harness_time_ns": self._now()}
        self._events.append(detail)
        return detail

    def _note_wait(self) -> None:
        """Called when the model answered wait: remember that decision's end for the admit-time check."""
        self._wait_end_ns = self._now() if self.validate_kind else None

    def _validate_wait_on_admit(self, obs: Observation) -> None:
        if self._wait_end_ns is None or obs.publication_time_ns > self._wait_end_ns:
            return
        exp = self.expected_kind(obs.payload)
        if exp != "wait":
            self._discard("wait", exp, obs)
            self._wait_end_ns = None
            self._baseline_hash = "<restart>"   # the next decision starts from this frame

    def _validate_proposal_at_commit(self, proposal_kind: str) -> "Restart | None":
        if not self.validate_kind:
            return None
        newest = self._newest
        exp = self.expected_kind(newest.payload)
        if exp == proposal_kind:
            return None
        self._discard(proposal_kind, exp, newest)
        return Restart("kind", newest.observation_id, {"proposal_kind": proposal_kind, "expected_kind": exp})


# ---- VLM late-binding controller (WI-004, Gate 5) ------------------------------------------

@dataclass(frozen=True)
class Restart:
    """Returned at commit time when the controller discards its proposal and begins a NEW decision
    (restart-after-update baseline). Logged as `decision_restarted`; never as the same decision."""
    reason: str
    observation_id: int
    detail: dict[str, Any]


AT_COMMIT_POLICIES = ("frozen", "restart_if_moved", "late_bind", "late_bind_or_restart")


class VLMLateBindingController(_KindValidator, VLMDirectController):
    """Same deliberative call as VLMDirectController (it proposes click(x, y) from the frame at B).
    ONE typed updateable state, ProvisionalClick = (x, y, template cut around (x, y) from the frame at
    B), and ONE update operation at D: track the template in the newest admitted frame.

      frozen           commit the proposal (state at B)                     — baseline
      restart_if_moved track; if the target moved > threshold, discard the proposal and start a new
                       decision (new id; D later) — baseline
      late_bind        track; commit the matched click under the SAME decision id (coordinate
                       correction by a separately identified component, §9.4); if the template is not
                       found, commit the proposal unchanged (recorded matched=False)
      late_bind_or_restart  as late_bind, but if the template is not found the proposal is discarded and
                       a NEW decision begins (never converts the click into another action kind)
    """

    kind = "vlm_late_binding"

    def __init__(self, *, at_commit_policy: str, template_radius_px: int, min_match_score: float,
                 restart_threshold_px: int | None, validate_kind: bool = False, **kw: Any) -> None:
        super().__init__(**kw)
        if at_commit_policy not in AT_COMMIT_POLICIES:
            raise ValueError(at_commit_policy)
        self.policy, self.radius, self.min_score, self.restart_px = (
            at_commit_policy, template_radius_px, min_match_score, restart_threshold_px)
        self.validate_kind = validate_kind
        if validate_kind and self.target_color is None:
            raise ValueError("validate_kind needs the target colour (the evaluator-only detector)")
        self._proposal: tuple[int, int, Observation] | None = None   # ProvisionalClick
        self._template = None

    def spec_hash(self) -> str:
        from .tracker import BINDER_VERSION
        return hash_obj({"base": super().spec_hash(), "kind": self.kind, "at_commit_policy": self.policy,
                         "template_radius_px": self.radius, "min_match_score": self.min_score,
                         "restart_threshold_px": self.restart_px, "binder_version": BINDER_VERSION,
                         **({"validate_kind": True} if self.validate_kind else {})})

    def admit(self, obs: Observation) -> ControlUpdate:
        upd = super().admit(obs)
        self._validate_wait_on_admit(obs)
        return upd

    def decide(self) -> Decision:
        from .tracker import cut_template
        obs = self._newest
        d = super().decide()
        if d.action is not None:
            self._proposal = (d.action["x"], d.action["y"], obs)
            self._template = cut_template(obs.payload, self.w, self.h, d.action["x"], d.action["y"], self.radius,
                                          rgb=bool(self.instruction_color))   # colour matters when colour defines the target
        elif d.invocation_detail.get("outcome") == "wait":
            self._note_wait()
        return d

    def _clamp(self, x: int, y: int) -> dict:
        return {"kind": "click", "x": min(max(0, x), self.w - 1), "y": min(max(0, y), self.h - 1)}

    def restart(self) -> None:
        """Discard the proposal; the next admitted frame (even an identical one) starts a new decision."""
        self._proposal, self._template, self._done = None, None, False
        self._baseline_hash = "<restart>"

    def correct_at_commit(self, commit_time_ns: int) -> "Correction | Restart | None":
        if self._proposal is None:
            return None
        rs = self._validate_proposal_at_commit("click")   # V1: the kind first, then the field
        if rs is not None:
            return rs
        if self.policy == "frozen":
            return None
        from .tracker import track
        px, py, obs_b = self._proposal
        newest = self._newest
        t0 = time.perf_counter_ns()
        m = track(self._template, newest.payload, self.w, self.h)
        track_ns = time.perf_counter_ns() - t0
        from .tracker import AMBIGUITY_MARGIN, BINDER_VERSION, PLATEAU_MAX
        ambiguous = (m.second >= m.score - AMBIGUITY_MARGIN    # the template re-occurs (nearly) as well elsewhere,
                     or m.plateau > PLATEAU_MAX)               # or slides inside a uniform region: keep the proposal
        matched = m.score >= self.min_score and not ambiguous
        moved = math.hypot(m.x - px, m.y - py) if matched else 0.0
        detail = {"match_score": round(m.score, 4), "matched": matched, "moved_px": round(moved, 2),
                  "track_ns": track_ns, "template_radius_px": self.radius, "min_match_score": self.min_score,
                  "second_best_score": round(m.second, 4), "plateau": m.plateau, "ambiguous": ambiguous,
                  "binder_version": BINDER_VERSION,
                  "proposal_observation_id": obs_b.observation_id, "binder_channels": self._template.channels}
        if self.policy == "restart_if_moved":
            if matched and moved > self.restart_px:
                return Restart("moved_since_B", newest.observation_id, {**detail, "restart_threshold_px": self.restart_px})
            return None   # commit the proposal unchanged
        if self.policy == "late_bind_or_restart" and not matched:
            # the provisional field cannot be resolved from the newest frame (target gone / changed):
            # do NOT commit the stale click and do NOT convert it into another action kind — a new decision
            # begins (§9.5: commitment fails explicitly when the relevant state changed)
            return Restart("target_not_found_at_D", newest.observation_id, detail)
        # late_bind: translate the proposal by the template's displacement (the model's own offset is kept)
        x, y = (m.x, m.y) if matched else (px, py)
        return Correction(self._clamp(x, y), "tracked_at_D", newest.observation_id, detail)


# ---- VLM value-binding controller (C4, cards/C4-late-bound-text.md) ----------------------------------------

VALUE_POLICIES = ("frozen", "restart_if_changed", "late_bind")


class VLMValueBindingController(_KindValidator, VLMDirectController):
    """The same deliberative call as VLMDirectController, but the model proposes type(value) (prompt 11).
    ONE typed updateable state, ProvisionalValue = (value, frame at B), and ONE update operation at D: decode the
    code panel in the newest admitted frame (decoder.py: pixels + the font, never the manifest).

      frozen              commit the proposal (the value read from the frame at B)            — baseline
      restart_if_changed  decode; if the value at D differs from the value decoded at B, discard the proposal
                          and start a new decision (new id; D later)                          — baseline
      late_bind           decode; commit type(decoded value) under the SAME decision id; if the panel cannot be
                          decoded, commit the proposal unchanged (recorded matched=False)

    With prompt 13 (V1, rotating_code kind_change) the grammar also allows click(x, y): a click proposal is committed
    as proposed (the dismiss button does not move; no binder applies); validate_kind checks its kind at D like a
    type proposal's. restart_if_changed keeps its C4 meaning (the VALUE changed), so a vanished panel does not
    restart it; only the validator handles a change of kind.
    """

    kind = "vlm_value_binding"

    def __init__(self, *, at_commit_policy: str, code_digits: int = 4, validate_kind: bool = False, **kw: Any) -> None:
        self.code_digits = code_digits
        super().__init__(**kw)
        if at_commit_policy not in VALUE_POLICIES:
            raise ValueError(at_commit_policy)
        self.policy = at_commit_policy
        self.validate_kind = validate_kind
        self._proposal: tuple[dict, Observation, str | None] | None = None   # (action, frame at B, decoded at B)

    def spec_hash(self) -> str:
        from .decoder import DECODER_VERSION
        return hash_obj({"base": super().spec_hash(), "kind": self.kind, "at_commit_policy": self.policy,
                         "code_digits": self.code_digits, "decoder_version": DECODER_VERSION,
                         **({"validate_kind": True} if self.validate_kind else {})})

    def _invoke(self, png: bytes, user_text: str) -> "RawInvocation":
        schema = MIXED_JSON_SCHEMA if self.prompt_version == 13 else VALUE_JSON_SCHEMA
        return self.adapter.invoke(png, self.system, user_text, schema=schema)

    def admit(self, obs: Observation) -> ControlUpdate:
        upd = super().admit(obs)
        self._validate_wait_on_admit(obs)
        return upd

    def decide(self) -> Decision:
        from .decoder import decode
        assert self._newest is not None
        obs = self._newest
        self._baseline_hash = obs.payload_hash
        self.budget.take()
        png = encode_png(obs.payload, self.w, self.h, self.scale)
        mixed = self.prompt_version == 13
        r = self._invoke(png, "Screenshot attached. Type the code currently shown, click the button if one is shown, or wait."
                         if mixed else "Screenshot attached. Type the code currently shown, or wait.")
        detail = {**r.detail, "observation_id": obs.observation_id}
        p = r.parsed
        kinds = ("type", "click", "wait") if mixed else ("type", "wait")
        if not isinstance(p, dict) or p.get("action") not in kinds:
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "unparseable"})
        if p["action"] == "wait":
            self._note_wait()
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "wait"})
        if p["action"] == "click":
            if type(p.get("x")) is not int or type(p.get("y")) is not int:
                return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                                invocation_detail={**detail, "outcome": "click_without_coordinates"})
            self._done = True
            action = self._to_canvas(p["x"], p["y"])
            self._proposal = (action, obs, None)
            return Decision(action=action, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "click", "image_x": p["x"], "image_y": p["y"]})
        v = p.get("value")
        if type(v) is not str or len(v) != self.code_digits or not v.isdigit():
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "type_without_value"})
        self._done = True
        dec_b = decode(obs.payload, self.w, self.h, self.code_digits)
        self._proposal = ({"kind": "type", "value": v}, obs, dec_b.value)
        return Decision(action={"kind": "type", "value": v}, declared_compute_ns=None, invocation_id=r.invocation_id,
                        invocation_detail={**detail, "outcome": "type", "value": v, "decoded_at_B": dec_b.value})

    def restart(self) -> None:
        self._proposal, self._done = None, False
        self._baseline_hash = "<restart>"

    def correct_at_commit(self, commit_time_ns: int) -> "Correction | Restart | None":
        if self._proposal is None:
            return None
        act, obs_b, dec_b = self._proposal
        rs = self._validate_proposal_at_commit(act["kind"])   # V1: the kind first, then the value
        if rs is not None:
            return rs
        if self.policy == "frozen" or act["kind"] == "click":
            return None
        v_b = act["value"]
        from .decoder import DECODER_VERSION, decode
        newest = self._newest
        t0 = time.perf_counter_ns()
        d = decode(newest.payload, self.w, self.h, self.code_digits)
        track_ns = time.perf_counter_ns() - t0
        matched = d.value is not None
        changed = matched and dec_b is not None and d.value != dec_b
        detail = {"match_score": 1.0 if matched else 0.0, "matched": matched, "moved_px": 1.0 if changed else 0.0,
                  "track_ns": track_ns, "decoded_value": d.value, "decoded_at_B": dec_b, "proposal_value": v_b,
                  "decode_reason": d.reason, "hamming": d.hamming, "decoder_version": DECODER_VERSION,
                  "proposal_observation_id": obs_b.observation_id}
        if self.policy == "restart_if_changed":
            if changed:
                return Restart("value_changed_since_B", newest.observation_id, detail)
            return None
        value = d.value if matched else v_b
        return Correction({"kind": "type", "value": value}, "decoded_at_D", newest.observation_id, detail)


# ---- pending-action record (WI-006) ----------------------------------------------------------

PENDING_MEMORIES = ("none", "harness_pending", "model_informed")


class PendingMemory:
    """The one piece of controller state WI-006 tests: a record of the last committed click whose
    visible effect has not yet been observed. Resolution = the pixel patch around the pending click
    (radius pending_radius_px, cut from the frame the click was decided on) differs in the newest
    admitted frame; expiry = pending_timeout_ns after commitment. It sees only admitted pixels.
    """

    def __init__(self, *, memory: str, pending_timeout_ns: int, pending_radius_px: int, w: int, h: int) -> None:
        if memory not in PENDING_MEMORIES:
            raise ValueError(memory)
        self.memory, self.timeout, self.radius, self.w, self.h = memory, pending_timeout_ns, pending_radius_px, w, h
        self.pending: dict[str, Any] | None = None   # {action, decision_id, decided_ns, patch, observation_id}
        self.events: list[dict[str, Any]] = []        # created / resolved / expired, drained by the runner

    def _patch(self, payload: bytes, x: int, y: int) -> bytes:
        r, w = self.radius, self.w
        payload = to_rgb(payload, self.w, self.h)
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(self.w, x + r + 1), min(self.h, y + r + 1)
        bpp = 3 if len(payload) == self.w * self.h * 3 else 1
        return b"".join(payload[(yy * w + x0) * bpp:(yy * w + x1) * bpp] for yy in range(y0, y1))

    def observe(self, obs: Observation, now_ns: int) -> None:
        """Called on every admitted frame: resolve or expire the pending record."""
        p = self.pending
        if p is None:
            return
        if now_ns - p["decided_ns"] >= self.timeout:
            self.events.append({"event": "pending_expired", "pending_decision_id": p["decision_id"], "harness_time_ns": now_ns})
            self.pending = None
            return
        if self._patch(obs.payload, p["action"]["x"], p["action"]["y"]) != p["patch"]:
            self.events.append({"event": "pending_resolved", "pending_decision_id": p["decision_id"],
                                "observation_id": obs.observation_id, "harness_time_ns": now_ns})
            self.pending = None

    def note_click(self, action: dict, decision_id: int, obs: Observation, now_ns: int) -> None:
        self.pending = {"action": action, "decision_id": decision_id, "decided_ns": now_ns,
                        "patch": self._patch(obs.payload, action["x"], action["y"]), "observation_id": obs.observation_id}
        self.events.append({"event": "pending_created", "pending_decision_id": decision_id,
                            "observation_id": obs.observation_id, "harness_time_ns": now_ns})

    def would_duplicate(self, action: dict) -> bool:
        p = self.pending
        return p is not None and math.hypot(action["x"] - p["action"]["x"], action["y"] - p["action"]["y"]) <= self.radius

    def prompt_text(self, now_ns: int) -> str:
        """For model_informed: the record in words (the intervention); empty if nothing is pending."""
        p = self.pending
        if p is None or self.memory != "model_informed":
            return ""
        age = (now_ns - p["decided_ns"]) / 1e9
        return (f" Note: {age:.1f} s ago you already clicked at canvas pixel ({p['action']['x']}, {p['action']['y']}) and the "
                f"effect of that click has not appeared yet; the application may still be processing it. "
                f"Only click again if you are sure the earlier click was lost.")


class VLMPendingController(VLMDirectController):
    """vlm_direct that keeps acting for the whole episode (multi-action tasks) with a PendingMemory:
    none (memoryless baseline), harness_pending (suppress a proposal that duplicates the pending
    click), model_informed (the prompt carries the record; no suppression)."""

    kind = "vlm_pending"

    def __init__(self, *, memory: str, pending_timeout_ns: int, pending_radius_px: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.mem = PendingMemory(memory=memory, pending_timeout_ns=pending_timeout_ns,
                                 pending_radius_px=pending_radius_px, w=self.w, h=self.h)
        self.clock_ns = None              # harness clock reader, bound by the runner (bind_clock)
        self._decision_id = 0

    def bind_clock(self, clock_ns) -> None:
        self.clock_ns = clock_ns

    def spec_hash(self) -> str:
        return hash_obj({"base": super().spec_hash(), "kind": self.kind, "memory": self.mem.memory,
                         "pending_timeout_ns": self.mem.timeout, "pending_radius_px": self.mem.radius})

    def admit(self, obs: Observation) -> ControlUpdate:
        upd = super().admit(obs)
        self.mem.observe(obs, self.clock_ns())
        return upd

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self.mem.events = self.mem.events + self._events, []
        self._events = []
        return ev

    def decide(self) -> Decision:
        obs = self._newest
        self._decision_id += 1
        note = self.mem.prompt_text(self.clock_ns())
        # same deliberative call as vlm_direct; only the user text may carry the pending note
        self._done = False
        self._baseline_hash = obs.payload_hash
        self.budget.take()
        png = encode_png(obs.payload, self.w, self.h, self.scale)
        r = self._invoke(png, "Screenshot attached. Click the target's center or wait." + note)
        detail = {**r.detail, "observation_id": obs.observation_id, "pending_note": bool(note)}
        p = r.parsed
        if not isinstance(p, dict) or p.get("action") not in ("click", "wait"):
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "unparseable"})
        if p["action"] == "wait":
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "wait"})
        if type(p.get("x")) is not int or type(p.get("y")) is not int:
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "click_without_coordinates"})
        x, y = p["x"], p["y"]
        action = self._to_canvas(x, y)
        if self.mem.memory == "harness_pending" and self.mem.would_duplicate(action):
            return Decision(None, None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "suppressed_pending", "image_x": x, "image_y": y,
                                               "suppressed_action": action,
                                               "pending_decision_id": self.mem.pending["decision_id"]})
        self.mem.note_click(action, self._decision_id, obs, self.clock_ns())
        return Decision(action, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "click", "image_x": x, "image_y": y})


class VLMChronotopeController(VLMLateBindingController):
    """The complete controller as ONE agent configuration (cards/A2-sponsor-skip-pending.md): the ProvisionalClick
    + binder of VLMLateBindingController (policy late_bind) AND the PendingMemory of VLMPendingController, around
    the same model call. Keeps acting for the whole episode on multi-action tasks; on a one-response task it is
    vlm_late_binding/late_bind with a record that never fires. Each component is inert where irrelevant: the
    binder returns translation 0 on a static target; the record suppresses nothing when no click is pending."""

    kind = "vlm_chronotope"

    def __init__(self, *, memory: str, pending_timeout_ns: int, pending_radius_px: int, **kw: Any) -> None:
        super().__init__(at_commit_policy="late_bind", restart_threshold_px=None, **kw)
        self.mem = PendingMemory(memory=memory, pending_timeout_ns=pending_timeout_ns,
                                 pending_radius_px=pending_radius_px, w=self.w, h=self.h)
        self.clock_ns = None
        self._decision_id = 0

    def bind_clock(self, clock_ns) -> None:
        self.clock_ns = clock_ns

    def spec_hash(self) -> str:
        return hash_obj({"base": super().spec_hash(), "kind": self.kind, "memory": self.mem.memory,
                         "pending_timeout_ns": self.mem.timeout, "pending_radius_px": self.mem.radius})

    def admit(self, obs: Observation) -> ControlUpdate:
        upd = super().admit(obs)
        self.mem.observe(obs, self.clock_ns())
        return upd

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self.mem.events = self.mem.events + self._events, []
        self._events = []
        return ev

    def decide(self) -> Decision:
        from .tracker import cut_template
        obs = self._newest
        self._decision_id += 1
        note = self.mem.prompt_text(self.clock_ns())
        self._done = False                      # multi-action: the next changed frame may start another decision
        self._baseline_hash = obs.payload_hash
        self._proposal, self._template = None, None
        self.budget.take()
        png = encode_png(obs.payload, self.w, self.h, self.scale)
        r = self._invoke(png, "Screenshot attached. Click the target's center or wait." + note)
        detail = {**r.detail, "observation_id": obs.observation_id, "pending_note": bool(note)}
        p = r.parsed
        if not isinstance(p, dict) or p.get("action") not in ("click", "wait"):
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "unparseable"})
        if p["action"] == "wait":
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "wait"})
        if type(p.get("x")) is not int or type(p.get("y")) is not int:
            return Decision(None, None, invocation_id=r.invocation_id, invocation_detail={**detail, "outcome": "click_without_coordinates"})
        x, y = p["x"], p["y"]
        action = self._to_canvas(x, y)
        if self.mem.memory == "harness_pending" and self.mem.would_duplicate(action):
            return Decision(None, None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "suppressed_pending", "image_x": x, "image_y": y,
                                               "suppressed_action": action,
                                               "pending_decision_id": self.mem.pending["decision_id"]})
        self.mem.note_click(action, self._decision_id, obs, self.clock_ns())
        # the ProvisionalClick: the coordinate stays open until the binder resolves it at D (same decision id)
        self._proposal = (action["x"], action["y"], obs)
        self._template = cut_template(obs.payload, self.w, self.h, action["x"], action["y"], self.radius,
                                      rgb=bool(self.instruction_color))
        return Decision(action, None, invocation_id=r.invocation_id,
                        invocation_detail={**detail, "outcome": "click", "image_x": x, "image_y": y})

    def correct_at_commit(self, commit_time_ns: int) -> "Correction | Restart | None":
        """The pending record refers to the click AS COMMITTED: after the binder moves the coordinate, the record's
        action and resolution patch move with it (cut from the frame the binder resolved on). Found on the live page
        (G2): with the record left at the proposal, a reflow between B and D changed the patch there and resolved the
        record on the next frame, so the duplicate proposal was never suppressed. On a static target the correction's
        translation is 0 and the record is unchanged (A2 results unaffected)."""
        corr = super().correct_at_commit(commit_time_ns)
        p = self.mem.pending
        if isinstance(corr, Correction) and p is not None and p["decision_id"] == self._decision_id:
            p["action"] = dict(corr.action)
            p["patch"] = self.mem._patch(self._newest.payload, corr.action["x"], corr.action["y"])
            p["observation_id"] = self._newest.observation_id
        return corr


class VLMPreparedController(VLMDirectController):
    """A8-PREP (cards/A8-preparation.md): preparation with a model. Decision 1 is ONE call on the first admitted
    (pre-event) frame with the prepared instruction (prompt 14: the layout is known; give the click now); its
    answer is stored, not committed (decision_no_action, outcome "prepared"). Decision 2 releases the stored click
    with no model call at the first admitted frame in which the target is detected (the evaluator-only colour
    detector; outcome "release_prepared"), so D is that frame's admission plus nothing. If the model answers wait
    on the pre-event frame, preparation is "not available" (outcome "prepare_unavailable") and the controller never
    acts; the analyzer reports that separately from timing failures. If the target is already visible when the
    pre-event call returns (E earlier than the call), the release follows immediately and D records that."""

    kind = "vlm_prepared"

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        if self.target_color is None:
            raise ValueError("vlm_prepared needs the target colour (the evaluator-only detector)")
        self._prepared: dict | None = None
        self._prepared_obs: int | None = None
        self._phase = "prepare"          # prepare -> armed | unavailable -> released
        self._detected = False

    def spec_hash(self) -> str:
        return hash_obj({"base": super().spec_hash(), "kind": self.kind, "prepared_xy": list(self.prepared_xy)})

    def admit(self, obs: Observation) -> ControlUpdate:
        upd = super().admit(obs)
        if _find_target_box(obs.payload, self.w, self.h, self.target_color) is not None:
            self._detected = True
        return upd

    def ready_to_decide(self) -> bool:
        if self._newest is None:
            return False
        if self._phase == "prepare":
            return True
        return self._phase == "armed" and self._detected

    def decide(self) -> Decision:
        obs = self._newest
        if self._phase == "armed":
            self._phase = "released"
            return Decision(action=dict(self._prepared), declared_compute_ns=0, invocation_id="prepared",
                            invocation_detail={"outcome": "release_prepared", "observation_id": obs.observation_id,
                                               "prepared_on_observation_id": self._prepared_obs})
        self.budget.take()
        png = encode_png(obs.payload, self.w, self.h, self.scale)
        r = self._invoke(png, "Screenshot attached. The card has not appeared yet. Give the prepared click now.")
        detail = {**r.detail, "observation_id": obs.observation_id}
        p = r.parsed
        if isinstance(p, dict) and p.get("action") == "click" and type(p.get("x")) is int and type(p.get("y")) is int:
            self._prepared, self._prepared_obs, self._phase = self._to_canvas(p["x"], p["y"]), obs.observation_id, "armed"
            return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                            invocation_detail={**detail, "outcome": "prepared", "prepared_action": self._prepared,
                                               "image_x": p["x"], "image_y": p["y"]})
        self._phase = "unavailable"
        why = "wait" if isinstance(p, dict) and p.get("action") == "wait" else "unparseable"
        return Decision(action=None, declared_compute_ns=None, invocation_id=r.invocation_id,
                        invocation_detail={**detail, "outcome": "prepare_unavailable", "model_answer": why})


class ScriptedPendingController:
    """Test double for WI-006 (simulated clock): clicks the target centre (found by colour) on every
    admitted frame that differs from the one it last decided on, with declared compute time, through
    the same PendingMemory. Lets the pending mechanism be contract-tested without a model."""

    kind = "scripted_pending"

    def __init__(self, *, compute_ns: int, memory: str, pending_timeout_ns: int, pending_radius_px: int,
                 canvas_w_px: int, canvas_h_px: int, target_color: tuple[int, int, int]) -> None:
        self.compute_ns, self.w, self.h, self.color, self.clock_ns = compute_ns, canvas_w_px, canvas_h_px, target_color, None
        self.mem = PendingMemory(memory=memory, pending_timeout_ns=pending_timeout_ns, pending_radius_px=pending_radius_px,
                                 w=canvas_w_px, h=canvas_h_px)
        self._newest: Observation | None = None
        self._last_decided: str | None = None
        self._decision_id = 0
        self._events: list[dict[str, Any]] = []

    def bind_clock(self, clock_ns) -> None:
        self.clock_ns = clock_ns

    def spec_hash(self) -> str:
        return hash_obj({"kind": self.kind, "compute_ns": self.compute_ns, "memory": self.mem.memory,
                         "pending_timeout_ns": self.mem.timeout, "pending_radius_px": self.mem.radius})

    def admit(self, obs: Observation) -> ControlUpdate:
        changed = self._newest is None or obs.payload_hash != self._newest.payload_hash
        self._newest = obs
        self.mem.observe(obs, self.clock_ns())
        return ControlUpdate(relevance="action" if changed else "none", controller_state_hash=hash_obj({"frame": obs.payload_hash}))

    def drain_events(self) -> list[dict[str, Any]]:
        ev, self.mem.events = self.mem.events + self._events, []
        self._events = []
        return ev

    def ready_to_decide(self) -> bool:
        return self._newest is not None and self._newest.payload_hash != self._last_decided

    def decide(self) -> Decision:
        obs = self._newest
        self._last_decided = obs.payload_hash
        self._decision_id += 1
        box = _find_target_box(obs.payload, self.w, self.h, self.color)
        if box is None:
            return Decision(None, self.compute_ns, invocation_detail={"outcome": "wait"})
        x0, y0, x1, y1 = box
        action = {"kind": "click", "x": (x0 + x1) // 2, "y": (y0 + y1) // 2}
        if self.mem.memory == "harness_pending" and self.mem.would_duplicate(action):
            return Decision(None, self.compute_ns, invocation_detail={
                "outcome": "suppressed_pending", "suppressed_action": action, "pending_decision_id": self.mem.pending["decision_id"]})
        # the record is created at decision time (commit follows immediately); decided_ns = B + compute
        self.mem.note_click(action, self._decision_id, obs, self.clock_ns() + self.compute_ns)
        return Decision(action, self.compute_ns, invocation_detail={"outcome": "click"})
