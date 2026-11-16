"""One-response scenarios.

  revealed_target — a square appears at a seed-determined (x, y) at task time t_E and stays put,
                    valid for [t_E, t_E + W).
  moving_target   — the same, but the square moves at a constant, explicit velocity after it
                    appears (predictable post-commit change; S3.1 / S3.2).

The only valid concrete action is click(x, y) inside the target AT THE TIME OF RECEIPT.
Environment state is a pure function of (manifest, task_time_ns). Nothing here may read a wall
clock or know the policy/condition/controller (experiments.md §6.1).
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from dataclasses import asdict, dataclass

from .hashing import hash_obj, sha256_hex

ACTION_SPACE_VERSION = 1  # {click(x, y)} with 0 <= x < canvas_w_px, 0 <= y < canvas_h_px

# Frames are RGB, 3 bytes per pixel, row-major.
SKINS = ("plain", "notification", "balloon", "calendar", "player", "player_static", "chart", "browser")
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "skin_calendar"
# The exact colour of the clickable target per skin (the oracle perceives by this colour; no other
# pixel in the skin uses it). Everything else in a skin is inert decoration.
TARGET_COLOR = {"plain": (255, 255, 255), "notification": (37, 99, 235), "balloon": (220, 38, 38),
                "calendar": (37, 99, 235), "player": (234, 88, 12), "player_static": (234, 88, 12), "chart": (22, 163, 74),
                "browser": (79, 70, 229)}
TARGET_ASPECT = {"plain": 1.0, "notification": 2.5, "balloon": 1.0, "calendar": 2.5,
                 "player": 2.5, "player_static": 2.5, "chart": 2.5, "browser": 104 / 37}   # width / height of the target box
DISTRACTOR_COLOR = (128, 128, 128)
DISTRACTOR_SIZE_PX = 2  # top-left corner, blinks every frame when irrelevant_motion is set

EVENT_TARGET_APPEARS = 1
EVENT_WINDOW_CLOSES = 2
RESPONSE_TRIGGERING_EVENT_ID = EVENT_TARGET_APPEARS
# scenario options that may be absent from a spec (then they take these values; see spec.py and manifest_hash)
OPTIONAL_SCENARIO_FIELDS = {"fixed_layout": False, "kind_change": False}


def _fill(buf: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int, c: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    row = bytes(c) * max(0, x1 - x0)
    for yy in range(y0, y1):
        o = (yy * w + x0) * 3
        buf[o:o + len(row)] = row


@dataclass(frozen=True)
class Event:
    event_id: int
    kind: str
    task_time_ns: int


@dataclass(frozen=True)
class TargetBox:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def center(self) -> tuple[int, int]:
        return (self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2

    def shifted(self, dx: int, dy: int) -> "TargetBox":
        return TargetBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)


@dataclass(frozen=True)
class State:
    task_time_ns: int
    target_visible: bool
    window_open: bool
    latest_event_id: int  # 0 before any event


@dataclass(frozen=True)
class Manifest:
    """revealed_target (static)."""
    scenario_id = "revealed_target"
    multi_action = False      # one response per episode unless a subclass says otherwise

    scenario_seed: int
    canvas_w_px: int
    canvas_h_px: int
    target_size_px: int
    event_time_ns: int
    response_window_ns: int
    frame_period_ns: int
    episode_horizon_ns: int
    irrelevant_motion: bool
    skin: str
    # A8-PREP (cards/A8-preparation.md): when True the target is placed at the skin's fixed position (the midpoint of
    # the seeded placement range) for every seed, so its content is known before E; the generator still draws it.
    fixed_layout: bool = False

    def __post_init__(self) -> None:
        if self.skin not in SKINS:
            raise ValueError(f"skin must be one of {SKINS}")
        if self.target_size_px <= 0 or self.frame_period_ns <= 0 or self.response_window_ns <= 0:
            raise ValueError("target_size_px, frame_period_ns, response_window_ns must be positive")
        if self.canvas_w_px < self.target_w_px() + 2 * DISTRACTOR_SIZE_PX + 2:
            raise ValueError("canvas too small for target and distractor region")
        if self.canvas_h_px < self.target_size_px + 2 * DISTRACTOR_SIZE_PX + 2:
            raise ValueError("canvas too small for target and distractor region")
        if self.episode_horizon_ns < self.event_time_ns + self.response_window_ns:
            raise ValueError("episode_horizon_ns must cover the response window")
        self.target_box()  # validates placement

    # --- identity ---------------------------------------------------------------------
    def manifest_hash(self) -> str:
        # Options added after the first cited runs (OPTIONAL_SCENARIO_FIELDS) are omitted from the hash while at
        # their default, so a manifest that does not use them keeps the hash its traces were recorded with.
        d = {k: v for k, v in asdict(self).items() if not (k in OPTIONAL_SCENARIO_FIELDS and v == OPTIONAL_SCENARIO_FIELDS[k])}
        return hash_obj({"scenario_id": self.scenario_id, **d})

    # --- displacement since appearance (static: none) -----------------------------------
    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        return (0, 0)

    def target_w_px(self) -> int:
        return max(1, round(self.target_size_px * TARGET_ASPECT[self.skin]))

    # --- derived from seed (explicit generator, no process-wide random state) ----------
    def target_box(self) -> TargetBox:
        """Box at the moment of appearance (the clickable region)."""
        rng = random.Random(self.scenario_seed)
        dx_end, dy_end = self.displacement_at(self.window_close_ns() - 1)
        lo = DISTRACTOR_SIZE_PX + 1  # keep clear of the distractor corner
        tw, th = self.target_w_px(), self.target_size_px
        # skins draw decoration around the target (a card above a button, a string under a balloon,
        # player chrome above and a sponsor banner + progress bar below)
        margin_top = th * 2 if self.skin in ("notification", "calendar") else 0
        margin_bottom = th if self.skin == "balloon" else 0
        if self.skin in ("player", "player_static"):
            margin_top = self.canvas_h_px // 12 + 1
            margin_bottom = self.canvas_h_px // 5
        x_lo, x_hi = lo - min(0, dx_end), self.canvas_w_px - tw - 1 - max(0, dx_end)
        y_lo, y_hi = lo + margin_top - min(0, dy_end), self.canvas_h_px - th - 1 - margin_bottom - max(0, dy_end)
        if x_lo > x_hi or y_lo > y_hi:
            raise ValueError("target would leave the canvas within the response window")
        if self.fixed_layout:
            x0, y0 = (x_lo + x_hi) // 2, (y_lo + y_hi) // 2   # the skin's usual place: seed-independent
        else:
            x0 = rng.randint(x_lo, x_hi)
            y0 = rng.randint(y_lo, y_hi)
        return TargetBox(x0, y0, x0 + tw, y0 + th)

    def target_box_at(self, task_time_ns: int) -> TargetBox:
        dx, dy = self.displacement_at(task_time_ns)
        return self.target_box().shifted(dx, dy)

    # --- events -----------------------------------------------------------------------
    def events(self) -> tuple[Event, ...]:
        return (
            Event(EVENT_TARGET_APPEARS, "target_appears", self.event_time_ns),
            Event(EVENT_WINDOW_CLOSES, "window_closes", self.event_time_ns + self.response_window_ns),
        )

    def window_close_ns(self) -> int:
        return self.event_time_ns + self.response_window_ns

    # --- pure state function ----------------------------------------------------------
    def state_at(self, task_time_ns: int) -> State:
        t_e, t_c = self.event_time_ns, self.window_close_ns()
        if task_time_ns < t_e:
            return State(task_time_ns, False, False, 0)
        if task_time_ns < t_c:
            return State(task_time_ns, True, True, EVENT_TARGET_APPEARS)
        return State(task_time_ns, False, False, EVENT_WINDOW_CLOSES)


    # --- calendar skin: pre-baked assets (cards/P1-reschedule.md; site/make_skin_assets.py).
    # Single-response tasks use plain/toast; pending_effect adds the done state.
    def _asset(self, state: str) -> bytes:
        p = ASSET_DIR / f"{self.canvas_w_px}x{self.canvas_h_px}_s{self.scenario_seed}_{state}.bin"
        if not p.exists():
            raise FileNotFoundError(f"baked calendar asset missing: {p} — run site/make_skin_assets.py")
        data = p.read_bytes()
        if len(data) != self.canvas_w_px * self.canvas_h_px * 3:
            raise ValueError(f"asset {p} has wrong size")
        return data

    def _render_calendar(self, task_time_ns: int, effect_visible: bool) -> bytes:
        st = self.state_at(task_time_ns)
        if effect_visible:
            return self._asset("done")
        if not st.target_visible:
            return self._asset("plain")
        buf = bytearray(self._asset("toast"))
        w, b, th = self.canvas_w_px, self.target_box(), self.target_size_px
        cx0, cy1 = b.x0 - th // 2, b.y1 + th // 2
        rem = max(0.0, min(1.0, (self.window_close_ns() - task_time_ns) / self.response_window_ns))
        bar_w = int((b.x1 + th // 2 - cx0 - th // 2) * rem)
        row = bytes((245, 158, 11)) * max(0, bar_w)
        for yy in range(max(0, cy1 - th // 4), min(self.canvas_h_px, cy1 - th // 8)):
            o = (yy * w + max(0, cx0 + th // 4)) * 3
            buf[o:o + len(row)] = row
        return bytes(buf)

    def render(self, task_time_ns: int) -> bytes:
        """Row-major RGB canvas depicting state_at(task_time_ns), drawn in the manifest's skin."""
        w, h = self.canvas_w_px, self.canvas_h_px
        st = self.state_at(task_time_ns)
        buf = bytearray(w * h * 3)

        def fill(x0: int, y0: int, x1: int, y1: int, c: tuple[int, int, int]) -> None:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            row = bytes(c) * max(0, x1 - x0)
            for yy in range(y0, y1):
                o = (yy * w + x0) * 3
                buf[o:o + len(row)] = row

        def disc(cx: float, cy: float, r: float, c: tuple[int, int, int]) -> None:
            for yy in range(max(0, int(cy - r)), min(h, int(cy + r) + 1)):
                for xx in range(max(0, int(cx - r)), min(w, int(cx + r) + 1)):
                    if (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r:
                        buf[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes(c)

        if self.skin == "calendar":
            return self._render_calendar(task_time_ns, effect_visible=False)
        if self.skin == "plain":
            pass  # black
        elif self.skin == "notification":
            fill(0, 0, w, h, (233, 236, 239))                       # desktop
            fill(0, 0, w, h // 12, (52, 58, 64))                     # title bar
            fill(w // 20, h // 6, w // 2 - w // 40, h - h // 8, (255, 255, 255))   # inert panel
            fill(w // 2 + w // 40, h // 6, w - w // 20, h // 2, (255, 255, 255))   # inert panel
            for i in range(4):                                        # placeholder text bars
                fill(w // 20 + w // 40, h // 6 + h // 20 + i * (h // 14), w // 2 - w // 12, h // 6 + h // 20 + i * (h // 14) + h // 40, (206, 212, 218))
        elif self.skin == "balloon":
            fill(0, 0, w, h, (186, 222, 246))                         # sky
            for cx, cy, r in ((w * 0.2, h * 0.22, h * 0.09), (w * 0.3, h * 0.2, h * 0.07), (w * 0.75, h * 0.3, h * 0.08), (w * 0.84, h * 0.33, h * 0.06)):
                disc(cx, cy, r, (248, 250, 252))                      # inert clouds
            fill(0, h - h // 10, w, h, (134, 190, 120))               # ground
        if self.irrelevant_motion and (task_time_ns // self.frame_period_ns) % 2 == 0:
            fill(0, 0, DISTRACTOR_SIZE_PX, DISTRACTOR_SIZE_PX, DISTRACTOR_COLOR)
        if st.target_visible:
            b = self.target_box_at(task_time_ns)
            tc = TARGET_COLOR[self.skin]
            if self.skin == "plain":
                fill(b.x0, b.y0, b.x1, b.y1, tc)
            elif self.skin == "notification":
                th = self.target_size_px
                cx0, cy0, cx1, cy1 = b.x0 - th // 2, b.y0 - 2 * th, b.x1 + th // 2, b.y1 + th // 2
                fill(cx0 - 1, cy0 - 1, cx1 + 1, cy1 + 1, (173, 181, 189))   # card border
                fill(cx0, cy0, cx1, cy1, (255, 255, 255))                   # card
                fill(cx0, cy0, cx0 + max(1, th // 6), cy1, (245, 158, 11))  # accent stripe
                fill(cx0 + th // 2, cy0 + th // 3, cx1 - th // 2, cy0 + th // 3 + th // 4, (73, 80, 87))   # title bar text
                fill(cx0 + th // 2, cy0 + th, cx1 - th, cy0 + th + th // 5, (173, 181, 189))                # body text
                fill(b.x0, b.y0, b.x1, b.y1, tc)                             # the button = the target
                # countdown bar: remaining fraction of the response window
                rem = max(0.0, min(1.0, (self.window_close_ns() - task_time_ns) / self.response_window_ns))
                bar_w = int((cx1 - cx0 - th // 2) * rem)
                fill(cx0 + th // 4, cy1 - th // 4, cx0 + th // 4 + bar_w, cy1 - th // 8, (245, 158, 11))
            elif self.skin == "balloon":
                r = self.target_size_px / 2
                cx, cy = b.x0 + r, b.y0 + r
                fill(int(cx) , b.y1, int(cx) + 1, b.y1 + self.target_size_px, (90, 90, 90))   # string
                disc(cx, cy, r, tc)
                disc(cx - r * 0.35, cy - r * 0.35, r * 0.22, (252, 165, 165))                 # highlight
        return bytes(buf)

    # --- action contract --------------------------------------------------------------
    def validate_action(self, action: dict) -> str | None:
        """Return None if `action` satisfies the concrete action contract, else a reason."""
        if set(action.keys()) != {"kind", "x", "y"}:
            return f"action fields must be exactly kind,x,y; got {sorted(action.keys())}"
        if action["kind"] != "click":
            return f"action kind {action['kind']!r} not in action space {{click}}"
        for k in ("x", "y"):
            v = action[k]
            if type(v) is not int:
                return f"{k} must be int, got {type(v).__name__}"
        if not (0 <= action["x"] < self.canvas_w_px and 0 <= action["y"] < self.canvas_h_px):
            return "click outside canvas"
        return None

    def action_valid_at(self, task_time_ns: int, action: dict) -> bool:
        """Application-time validity: target visible at task_time_ns and click inside it."""
        if self.validate_action(action) is not None:
            return False
        st = self.state_at(task_time_ns)
        return st.target_visible and self.target_box_at(task_time_ns).contains(action["x"], action["y"])

    def target_color(self) -> tuple[int, int, int]:
        return TARGET_COLOR[self.skin]

    def spatial_error_px(self, task_time_ns: int, action: dict) -> float:
        """Distance from the click to the target center at task_time_ns (a natural task value)."""
        cx, cy = self.target_box_at(task_time_ns).center()
        return math.hypot(action["x"] - cx, action["y"] - cy)

    # --- outcome contract (gym-outcome-contract card): the episodic return G ------------------
    # G is task value in the state present at application receipt (logic.md "Outcome Reporting
    # Contract"). It must exist for EVERY valid terminal trace: -e_max (the display diagonal) when no
    # click was received AND when a click was received but the target is absent at receipt (window
    # closed, balloon gone) — the state at receipt then contains nothing to be near; a click received
    # while the target is present is graded by its receipt-time distance. (2026-08-24: the earlier
    # reading "late click, perfect geometry -> G = 0" made the return blind to the threshold
    # failure — R1's Live return equalled Paused's at every window while SR went 100 -> 0 — and
    # contradicted the O1 card, studies/paper_tables.py and the paper's return-matrix figure.)
    def e_max_px(self) -> float:
        return math.hypot(self.canvas_w_px, self.canvas_h_px)   # display diagonal

    def return_on_receipt(self, task_time_ns: int, action: dict) -> float:
        if self.validate_action(action) is not None or not self.state_at(task_time_ns).target_visible:
            return -self.e_max_px()
        return -min(self.spatial_error_px(task_time_ns, action), self.e_max_px())

    def return_no_receipt(self) -> float:
        return -self.e_max_px()


@dataclass(frozen=True)
class MovingTargetManifest(Manifest):
    """moving_target: constant velocity (px/s) after appearance; displacement floors to pixels."""
    scenario_id = "moving_target"

    velocity_x_px_per_s: int = 0
    velocity_y_px_per_s: int = 0

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        dt = max(0, task_time_ns - self.event_time_ns)
        return ((self.velocity_x_px_per_s * dt) // 1_000_000_000,
                (self.velocity_y_px_per_s * dt) // 1_000_000_000)


SCENARIOS: dict[str, type[Manifest]] = {"revealed_target": Manifest, "moving_target": MovingTargetManifest}


def manifest_from_dict(scenario: dict) -> Manifest:
    if scenario["scenario_id"] == "live_page" and "live_page" not in SCENARIOS:
        from . import live_page  # noqa: F401  (registers LivePageManifest; imported lazily to avoid a cycle)
    cls = SCENARIOS[scenario["scenario_id"]]
    return cls(**{k: v for k, v in scenario.items() if k != "scenario_id"})


def payload_hash(payload: bytes) -> str:
    return sha256_hex(payload)


# ---- pending_effect (WI-006): the effect of an accepted click becomes visible only later ----------

EVENT_EFFECT_VISIBLE = 3


@dataclass(frozen=True)
class PendingEffectManifest(Manifest):
    """Notification card with a button. The FIRST click received inside the button (while visible and
    the window open) is accepted at application time; its visible effect (the card turns into a
    'Done' card without a button) is rendered only effect_delay_ns after that receipt. Until then the
    screen is unchanged except the draining countdown bar. Further clicks inside the button before
    the effect is visible are duplicates. The state therefore depends on env state (accepted_at_ns),
    which the runner owns and the trace records."""
    scenario_id = "pending_effect"
    multi_action = True
    POST_EFFECT_NS = 2_000_000_000   # the episode ends this long after the effect became visible

    effect_delay_ns: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.skin not in ("notification", "calendar"):
            raise ValueError("pending_effect is drawn in the notification or calendar skin")
        if self.effect_delay_ns <= 0:
            raise ValueError("effect_delay_ns must be positive")

    def effect_visible_ns(self, accepted_at_ns: int | None) -> int | None:
        return None if accepted_at_ns is None else accepted_at_ns + self.effect_delay_ns

    def render(self, task_time_ns: int) -> bytes:
        if self.skin == "calendar":
            return self._render_calendar(task_time_ns, effect_visible=False)
        return super().render(task_time_ns)

    def state_env(self, task_time_ns: int, accepted_at_ns: int | None) -> State:
        ev = self.effect_visible_ns(accepted_at_ns)
        if ev is not None and task_time_ns >= ev:
            return State(task_time_ns, False, False, EVENT_EFFECT_VISIBLE)
        return self.state_at(task_time_ns)

    def render_env(self, task_time_ns: int, accepted_at_ns: int | None) -> bytes:
        st = self.state_env(task_time_ns, accepted_at_ns)
        if self.skin == "calendar":
            return self._render_calendar(task_time_ns, st.latest_event_id == EVENT_EFFECT_VISIBLE)
        if st.latest_event_id != EVENT_EFFECT_VISIBLE:
            return self.render(task_time_ns)
        # 'Done' card: same card geometry, green check bar instead of the button, no countdown
        w, h = self.canvas_w_px, self.canvas_h_px
        buf = bytearray(self.render(self.event_time_ns - 1))           # desktop without the card
        # the base render already drew the blinking distractor for THAT time; redraw for this one
        if self.irrelevant_motion:
            on = (task_time_ns // self.frame_period_ns) % 2 == 0
            c = DISTRACTOR_COLOR if on else (233, 236, 239)
            for yy in range(DISTRACTOR_SIZE_PX):
                for xx in range(DISTRACTOR_SIZE_PX):
                    buf[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes(c)
        b = self.target_box()
        th = self.target_size_px

        def fill(x0, y0, x1, y1, c):
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            row = bytes(c) * max(0, x1 - x0)
            for yy in range(y0, y1):
                o = (yy * w + x0) * 3
                buf[o:o + len(row)] = row
        cx0, cy0, cx1, cy1 = b.x0 - th // 2, b.y0 - 2 * th, b.x1 + th // 2, b.y1 + th // 2
        fill(cx0 - 1, cy0 - 1, cx1 + 1, cy1 + 1, (173, 181, 189))
        fill(cx0, cy0, cx1, cy1, (255, 255, 255))
        fill(cx0, cy0, cx0 + max(1, th // 6), cy1, (22, 163, 74))        # green stripe
        fill(cx0 + th // 2, cy0 + th // 3, cx1 - th // 2, cy0 + th // 3 + th // 4, (73, 80, 87))
        fill(b.x0, b.y0 + th // 3, b.x1, b.y1 - th // 3, (22, 163, 74))   # "done" bar where the button was
        return bytes(buf)

    def receipt_outcome(self, task_time_ns: int, action: dict, accepted_at_ns: int | None) -> str:
        """accepted | duplicate | invalid — what the application does with a click at this time."""
        if self.validate_action(action) is not None:
            return "invalid"
        st = self.state_env(task_time_ns, accepted_at_ns)
        inside = self.target_box().contains(action["x"], action["y"])
        if st.latest_event_id == EVENT_EFFECT_VISIBLE or not st.target_visible or not inside:
            return "invalid"
        return "accepted" if accepted_at_ns is None else "duplicate"

    def return_pending(self, accepted_at_ns: int | None) -> float:
        """Outcome contract for the pending-effect task (gym-outcome-contract, extended 2026-08-24): the return is
        the time until the accepted click's effect is visible, in seconds, negative — the same shape as Task A's
        G_A (the notification is handled when its effect shows); the full window is charged when no click was
        ever accepted (wait-only, no action, dropped, all clicks invalid, or the card expired). Duplicates are not
        charged here; exact success counts them."""
        if accepted_at_ns is None:
            return -self.response_window_ns / 1e9
        return -(min(accepted_at_ns + self.effect_delay_ns, self.window_close_ns()) - self.event_time_ns) / 1e9


SCENARIOS["pending_effect"] = PendingEffectManifest


# ---- jump_target (WI-009): the target relocates discretely before/after commitment -------------

EVENT_TARGET_JUMPS = 4


@dataclass(frozen=True)
class JumpTargetManifest(Manifest):
    """revealed_target whose target relocates by (jump_dx_px, jump_dy_px) at event_time + jump_delay_ns
    (the paper's 'pre-commit change' family when the jump precedes D, 'post-commit change' when it
    falls in (D, A]). The appearance is unchanged; only the position changes."""
    scenario_id = "jump_target"

    jump_delay_ns: int = 0
    jump_dx_px: int = 0
    jump_dy_px: int = 0

    def __post_init__(self) -> None:
        if self.jump_delay_ns <= 0:
            raise ValueError("jump_delay_ns must be positive")
        if max(abs(self.jump_dx_px), abs(self.jump_dy_px)) < max(1, round(self.target_size_px * TARGET_ASPECT[self.skin])):
            raise ValueError("the jump must move the target by at least one target width")
        super().__post_init__()   # validates that the post-jump box stays on canvas (displacement_at)

    def jump_time_ns(self) -> int:
        return self.event_time_ns + self.jump_delay_ns

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        return (self.jump_dx_px, self.jump_dy_px) if task_time_ns >= self.jump_time_ns() else (0, 0)

    def events(self) -> tuple[Event, ...]:
        base = super().events()
        if self.jump_time_ns() < self.window_close_ns():
            return (base[0], Event(EVENT_TARGET_JUMPS, "target_jumps", self.jump_time_ns()), base[1])
        return base

    def state_at(self, task_time_ns: int) -> State:
        st = super().state_at(task_time_ns)
        if st.target_visible and task_time_ns >= self.jump_time_ns():
            return State(task_time_ns, True, True, EVENT_TARGET_JUMPS)
        return st


SCENARIOS["jump_target"] = JumpTargetManifest


# ---- balloon_select (C3): several balloons drift; the instruction names one by colour ----------------

BALLOON_COLORS = {"red": (220, 38, 38), "blue": (37, 99, 235), "yellow": (234, 179, 8)}
BALLOON_ORDER = ("red", "blue", "yellow")


@dataclass(frozen=True)
class BalloonSelectManifest(Manifest):
    """Three balloons (red, blue, yellow) appear at event_time_ns at seeded, non-overlapping positions and
    drift horizontally; the instruction names ONE colour (target_color). The named balloon moves at
    velocity_x_px_per_s; the two distractors move at seeded velocities of similar magnitude (one of them
    opposite in sign, so crossings happen). A click is valid iff it lands inside the NAMED balloon's box
    while visible and the window is open; a click on a distractor is a semantic error.

    Separates the model's job (choose the right balloon) from the binder's job (keep its coordinate current):
    a wrong choice stays wrong after binding; a right choice whose position aged is recovered."""
    scenario_id = "balloon_select"

    named_color: str = "red"
    velocity_x_px_per_s: int = 0

    def __post_init__(self) -> None:
        if self.skin != "balloon":
            raise ValueError("balloon_select is drawn in the balloon skin")
        if self.named_color not in BALLOON_COLORS:
            raise ValueError(f"named_color must be one of {BALLOON_ORDER}")
        super().__post_init__()
        self.balloons_at(self.event_time_ns)   # validates the seeded layout

    # --- layout: seeded start positions and distractor velocities -----------------------------
    def _layout(self) -> list[tuple[str, int, int, int]]:
        """[(colour, x0, y0, vx)] at appearance; the named balloon uses target_box()/velocity_x_px_per_s."""
        rng = random.Random(self.scenario_seed * 7919 + 17)
        th = self.target_size_px; tw = self.target_w_px()
        tb = self.target_box()
        out = [(self.named_color, tb.x0, tb.y0, self.velocity_x_px_per_s)]
        others = [c for c in BALLOON_ORDER if c != self.named_color]
        mag = max(4, abs(self.velocity_x_px_per_s))
        signs = [-1, 1] if self.velocity_x_px_per_s >= 0 else [1, -1]
        for c, sgn in zip(others, signs):
            for _ in range(200):
                x0 = rng.randint(DISTRACTOR_SIZE_PX + 1, self.canvas_w_px - tw - 1)
                y0 = rng.randint(DISTRACTOR_SIZE_PX + 1, self.canvas_h_px - th - 1 - th)
                if all(abs(x0 - ox) > tw + 4 or abs(y0 - oy) > th + 4 for _, ox, oy, _ in out):
                    break
            else:
                raise ValueError("could not place distractor balloons without overlap")
            out.append((c, x0, y0, sgn * (mag + rng.randint(0, 4))))
        return out

    def balloons_at(self, task_time_ns: int) -> list[tuple[str, TargetBox]]:
        dt = max(0, task_time_ns - self.event_time_ns)
        res = []
        for c, x0, y0, vx in self._layout():
            dx = (vx * dt) // 1_000_000_000
            x = min(max(DISTRACTOR_SIZE_PX + 1, x0 + dx), self.canvas_w_px - self.target_w_px() - 1)   # distractors stop at the edge
            if c == self.named_color:
                res.append((c, self.target_box_at(task_time_ns)))
            else:
                res.append((c, TargetBox(x, y0, x + self.target_w_px(), y0 + self.target_size_px)))
        return res

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        dt = max(0, task_time_ns - self.event_time_ns)
        return ((self.velocity_x_px_per_s * dt) // 1_000_000_000, 0)

    def target_color(self) -> tuple[int, int, int]:
        return BALLOON_COLORS[self.named_color]

    def instruction_color(self) -> str:
        return self.named_color

    def render(self, task_time_ns: int) -> bytes:
        w, h = self.canvas_w_px, self.canvas_h_px
        st = self.state_at(task_time_ns)
        buf = bytearray(super().render(self.event_time_ns - 1))   # sky, clouds, ground; no balloon
        if self.irrelevant_motion and (task_time_ns // self.frame_period_ns) % 2 == 0:
            for yy in range(DISTRACTOR_SIZE_PX):
                for xx in range(DISTRACTOR_SIZE_PX):
                    buf[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes(DISTRACTOR_COLOR)
        if not st.target_visible:
            return bytes(buf)

        def disc(cx, cy, r, c):
            for yy in range(max(0, int(cy - r)), min(h, int(cy + r) + 1)):
                for xx in range(max(0, int(cx - r)), min(w, int(cx + r) + 1)):
                    if (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r:
                        buf[(yy * w + xx) * 3:(yy * w + xx) * 3 + 3] = bytes(c)
        r = self.target_size_px / 2
        for c, b in sorted(self.balloons_at(task_time_ns), key=lambda x: x[0] != self.named_color):  # target drawn last (on top)
            cx, cy = b.x0 + r, b.y0 + r
            for yy in range(b.y1, min(h, b.y1 + self.target_size_px)):
                o = (yy * w + int(cx)) * 3
                buf[o:o + 3] = bytes((90, 90, 90))
            disc(cx, cy, r, BALLOON_COLORS[c])
            disc(cx - r * 0.35, cy - r * 0.35, r * 0.22, (252, 220, 220) if c != "yellow" else (255, 245, 200))
        return bytes(buf)

    def clicked_balloon(self, task_time_ns: int, action: dict) -> str | None:
        """Which balloon (colour) the click lands on at task_time_ns, or None."""
        if self.validate_action(action) is not None or not self.state_at(task_time_ns).target_visible:
            return None
        for c, b in self.balloons_at(task_time_ns):
            if b.contains(action["x"], action["y"]):
                return c
        return None


SCENARIOS["balloon_select"] = BalloonSelectManifest


# ---- sponsor_skip (A1, paper Task A): fixed known action, closing opportunity, graded exposure ----

@dataclass(frozen=True)
class SponsorSkipManifest(Manifest):
    """A sponsor segment plays over [event_time_ns, event_time_ns + response_window_ns); the
    response window IS the sponsor segment. The skip button (the target box) is visible while the
    sponsor plays. A valid skip received at task time t removes the sponsor at t + seek_delay_ns
    (the seek delay is watched time). The return is graded exposure in SECONDS:
        valid skip  ->  -(min(t + seek_delay, t_close) - t_E) / 1e9
        otherwise   ->  -response_window / 1e9   (the whole sponsor was watched)
    One-response episodes only (the repeated-skip pending-effect study is P1's mechanics)."""
    scenario_id = "sponsor_skip"

    seek_delay_ns: int = 0

    def __post_init__(self) -> None:
        if self.skin not in ("player", "player_static"):
            raise ValueError("sponsor_skip is drawn in the player or player_static skin")
        if self.seek_delay_ns < 0:
            raise ValueError("seek_delay_ns must be non-negative")
        super().__post_init__()

    def events(self) -> tuple[Event, ...]:
        return (Event(EVENT_TARGET_APPEARS, "sponsor_starts", self.event_time_ns),
                Event(EVENT_WINDOW_CLOSES, "window_closes", self.window_close_ns()))

    # --- outcome contract ----------------------------------------------------------------
    def watched_ns(self, task_time_ns: int, action: dict) -> int:
        """Sponsor exposure implied by a click received at task_time_ns."""
        if self.action_valid_at(task_time_ns, action):
            return min(task_time_ns + self.seek_delay_ns, self.window_close_ns()) - self.event_time_ns
        return self.response_window_ns

    def return_on_receipt(self, task_time_ns: int, action: dict) -> float:
        return -self.watched_ns(task_time_ns, action) / 1e9

    def return_no_receipt(self) -> float:
        return -self.response_window_ns / 1e9

    # --- player skin -----------------------------------------------------------------------
    def render(self, task_time_ns: int) -> bytes:
        return self._render_player(task_time_ns, self.state_at(task_time_ns).target_visible)

    def _render_player(self, task_time_ns: int, sponsor_visible: bool) -> bytes:
        w, h = self.canvas_w_px, self.canvas_h_px
        buf = bytearray(w * h * 3)
        _fill(buf, w, h, 0, 0, w, h, (24, 24, 27))                       # player chrome
        chrome_h = h // 12
        _fill(buf, w, h, 0, 0, w, chrome_h, (39, 39, 42))                # title bar
        _fill(buf, w, h, 0, chrome_h, w, h - h // 8, (16, 16, 18))       # video area
        # "content playing": a light block that pans with task time (never the target colour); in the
        # player_static control skin (Q3, 2026-08-24) the block and the progress bar are frozen, so no frame
        # changes before the sponsor appears — the paired control for the pre-event-animation cost
        static = self.skin == "player_static"
        step = max(1, w // 8)
        t_draw = self.event_time_ns if static else task_time_ns
        cx0 = (t_draw // max(1, self.frame_period_ns) * 2) % max(1, w - step)
        _fill(buf, w, h, int(cx0), h // 3, int(cx0) + step, h // 2, (63, 63, 70))
        # progress bar: elapsed fraction of the episode (frozen at the event-time fraction in the static skin)
        frac = max(0.0, min(1.0, t_draw / self.episode_horizon_ns))
        _fill(buf, w, h, 0, h - 4, w, h - 2, (63, 63, 70))
        _fill(buf, w, h, 0, h - 4, int(w * frac), h - 2, (161, 161, 170))
        if self.irrelevant_motion and (task_time_ns // self.frame_period_ns) % 2 == 0:
            _fill(buf, w, h, 0, 0, DISTRACTOR_SIZE_PX, DISTRACTOR_SIZE_PX, DISTRACTOR_COLOR)
        if sponsor_visible:
            # sponsor banner (bottom-left) with a countdown of the remaining sponsor time
            _fill(buf, w, h, 2, h - h // 5 + 1, w // 3, h - h // 9, (250, 204, 21))
            rem = max(0.0, min(1.0, (self.window_close_ns() - task_time_ns) / self.response_window_ns))
            _fill(buf, w, h, 2, h - h // 9 + 1, 2 + int((w // 3 - 2) * rem), h - h // 12, (245, 158, 11))
            b = self.target_box_at(task_time_ns)                          # the skip button
            _fill(buf, w, h, b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1, (250, 250, 250))
            _fill(buf, w, h, b.x0, b.y0, b.x1, b.y1, TARGET_COLOR["player"])
        return bytes(buf)


SCENARIOS["sponsor_skip"] = SponsorSkipManifest


# ---- sponsor_skip_pending (A2): Task A with repeated skips — the seek delay IS a pending effect ----

@dataclass(frozen=True)
class SponsorSkipPendingManifest(SponsorSkipManifest):
    """sponsor_skip, multi-action (cards/A2-sponsor-skip-pending.md). The FIRST click received inside the skip
    button while the sponsor plays is accepted at application time; the sponsor (banner + button) stays on screen
    until accepted_at + seek_delay (the effect: the player has seeked past the sponsor). Clicks inside the button
    before the effect are duplicates (the player ignores them); the episode ends POST_EFFECT_NS after the effect
    or at the horizon. G_A is sponsor_skip's graded exposure computed from the ACCEPTED receipt; duplicates are
    not charged in G_A (they are counted in exact success)."""
    scenario_id = "sponsor_skip_pending"
    multi_action = True
    POST_EFFECT_NS = 2_000_000_000

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.seek_delay_ns <= 0:
            raise ValueError("sponsor_skip_pending needs a positive seek_delay_ns (the pending effect)")

    def effect_visible_ns(self, accepted_at_ns: int | None) -> int | None:
        return None if accepted_at_ns is None else accepted_at_ns + self.seek_delay_ns

    def state_env(self, task_time_ns: int, accepted_at_ns: int | None) -> State:
        ev = self.effect_visible_ns(accepted_at_ns)
        if ev is not None and task_time_ns >= ev:
            return State(task_time_ns, False, False, EVENT_EFFECT_VISIBLE)
        return self.state_at(task_time_ns)

    def render_env(self, task_time_ns: int, accepted_at_ns: int | None) -> bytes:
        return self._render_player(task_time_ns, self.state_env(task_time_ns, accepted_at_ns).target_visible)

    def receipt_outcome(self, task_time_ns: int, action: dict, accepted_at_ns: int | None) -> str:
        """accepted | duplicate | invalid — what the player does with a click received at this time."""
        if self.validate_action(action) is not None:
            return "invalid"
        st = self.state_env(task_time_ns, accepted_at_ns)
        inside = self.target_box().contains(action["x"], action["y"])
        if st.latest_event_id == EVENT_EFFECT_VISIBLE or not st.target_visible or not inside:
            return "invalid"
        return "accepted" if accepted_at_ns is None else "duplicate"

    def return_pending(self, accepted_at_ns: int | None) -> float:
        """G_A for a multi-action terminal trace: seconds of sponsor watched (full window if never skipped)."""
        if accepted_at_ns is None:
            return -self.response_window_ns / 1e9
        return -(min(accepted_at_ns + self.seek_delay_ns, self.window_close_ns()) - self.event_time_ns) / 1e9


SCENARIOS["sponsor_skip_pending"] = SponsorSkipPendingManifest


# ---- live_trading (B1, paper Task B): semantic validity change; predeclared utilities at receipt ----

EVENT_SIGNAL_REVERSES = 5


@dataclass(frozen=True)
class LiveTradingManifest(Manifest):
    """A seeded synthetic price crosses the marked threshold at event_time_ns (the signal). The
    order button (right-hand panel) is armed while the signal is valid; validity ends at the
    reversal (reversal_delay_ns > 0) or at window close, whichever is earlier. The utilities are
    predeclared in the spec and applied to the state at receipt:
        order received while valid   -> u_valid_order
        order received while invalid -> u_invalid_order   (reversed or expired)
        no order received            -> u_no_order        (wait, no action, dropped delivery)
    Not a forecasting task: the rule, event times, and validity are evaluator-defined."""
    scenario_id = "live_trading"

    reversal_delay_ns: int = 0
    u_valid_order: float = 1.0
    u_invalid_order: float = -1.0
    u_no_order: float = 0.0

    def __post_init__(self) -> None:
        if self.skin != "chart":
            raise ValueError("live_trading is drawn in the chart skin")
        if self.reversal_delay_ns < 0:
            raise ValueError("reversal_delay_ns must be non-negative (0 = no reversal)")
        if 0 < self.reversal_delay_ns < self.frame_period_ns:
            raise ValueError("a reversal must be at least one frame after the signal")
        if self.reversal_delay_ns >= self.response_window_ns > 0:
            raise ValueError("a reversal at or after expiry is meaningless; use reversal_delay_ns = 0")
        super().__post_init__()

    # --- signal timeline -------------------------------------------------------------------
    def reversal_time_ns(self) -> int | None:
        return self.event_time_ns + self.reversal_delay_ns if self.reversal_delay_ns > 0 else None

    def valid_until_ns(self) -> int:
        rv = self.reversal_time_ns()
        return rv if rv is not None else self.window_close_ns()

    def events(self) -> tuple[Event, ...]:
        base = (Event(EVENT_TARGET_APPEARS, "signal_appears", self.event_time_ns),
                Event(EVENT_WINDOW_CLOSES, "window_closes", self.window_close_ns()))
        rv = self.reversal_time_ns()
        if rv is not None:
            return (base[0], Event(EVENT_SIGNAL_REVERSES, "signal_reverses", rv), base[1])
        return base

    def state_at(self, task_time_ns: int) -> State:
        if task_time_ns < self.event_time_ns:
            return State(task_time_ns, False, False, 0)
        if task_time_ns < self.valid_until_ns():
            return State(task_time_ns, True, True, EVENT_TARGET_APPEARS)
        if task_time_ns < self.window_close_ns():
            return State(task_time_ns, False, False, EVENT_SIGNAL_REVERSES)
        return State(task_time_ns, False, False, EVENT_WINDOW_CLOSES)

    # --- geometry: static order button in the right-hand panel ------------------------------
    def target_box(self) -> TargetBox:
        rng = random.Random(self.scenario_seed)
        tw, th = self.target_w_px(), self.target_size_px
        x_lo = (2 * self.canvas_w_px) // 3 + 2
        x_hi = self.canvas_w_px - tw - 3
        y_lo = self.canvas_h_px // 6
        y_hi = self.canvas_h_px - th - 3
        if x_lo > x_hi or y_lo > y_hi:
            raise ValueError("canvas too small for the order panel and button")
        x0, y0 = rng.randint(x_lo, x_hi), rng.randint(y_lo, y_hi)
        return TargetBox(x0, y0, x0 + tw, y0 + th)

    # --- outcome contract ----------------------------------------------------------------
    def return_on_receipt(self, task_time_ns: int, action: dict) -> float:
        return self.u_valid_order if self.action_valid_at(task_time_ns, action) else self.u_invalid_order

    def return_no_receipt(self) -> float:
        return self.u_no_order

    # --- price path: pure function of task time; noise scaled away from the threshold so the
    # pixels never contradict the state function -------------------------------------------
    def price_frac(self, task_time_ns: int) -> float:
        """Signed distance from the threshold in [-1, 1]; crosses 0 exactly at event_time_ns."""
        e, close = self.event_time_ns, self.window_close_ns()
        ramp = max(1, self.response_window_ns // 6)
        rv = self.reversal_time_ns()
        if task_time_ns < e:
            return -max(0.05, min(1.0, (e - task_time_ns) / max(1, e)))
        if rv is not None and task_time_ns >= rv:
            return -min(1.0, 0.05 + (task_time_ns - rv) / ramp)
        if rv is None and task_time_ns >= close:
            return -min(1.0, 0.05 + (task_time_ns - close) / ramp)
        return min(1.0, 0.05 + (task_time_ns - e) / ramp)

    def _noise(self, column: int, frac: float) -> float:
        rng = random.Random(self.scenario_seed * 1_000_003 + column * 7919)
        return (rng.random() - 0.5) * 0.4 * min(1.0, abs(frac) / 0.3) * (1.0 - abs(frac) * 0.5)

    def render(self, task_time_ns: int) -> bytes:
        w, h = self.canvas_w_px, self.canvas_h_px
        st = self.state_at(task_time_ns)
        buf = bytearray(w * h * 3)
        _fill(buf, w, h, 0, 0, w, h, (248, 250, 252))
        chart_x1 = (2 * w) // 3 - 2
        y_mid, amp = h // 2, h // 2 - 3
        for x in range(2, chart_x1, 4):                                   # dashed threshold line
            _fill(buf, w, h, x, y_mid, x + 2, y_mid + 1, (148, 163, 184))
        for x in range(2, chart_x1):                                      # price up to current time
            t_col = (x - 2) * self.episode_horizon_ns // max(1, chart_x1 - 3)
            if t_col > task_time_ns:
                break
            frac = self.price_frac(t_col)
            y = y_mid - round((frac + self._noise(x, frac)) * amp)
            _fill(buf, w, h, x, max(1, y - 1), x + 1, min(h - 1, y + 1), (37, 99, 235))
        # order panel
        _fill(buf, w, h, chart_x1 + 2, 2, w - 2, h - 2, (255, 255, 255))
        _fill(buf, w, h, chart_x1 + 2, 2, chart_x1 + 3, h - 2, (203, 213, 225))
        if self.irrelevant_motion and (task_time_ns // self.frame_period_ns) % 2 == 0:
            _fill(buf, w, h, 0, 0, DISTRACTOR_SIZE_PX, DISTRACTOR_SIZE_PX, DISTRACTOR_COLOR)
        b = self.target_box()                                             # the order button:
        if st.target_visible:                                             # armed only while valid
            _fill(buf, w, h, b.x0, b.y0, b.x1, b.y1, TARGET_COLOR["chart"])
        else:
            _fill(buf, w, h, b.x0, b.y0, b.x1, b.y1, (203, 213, 225))
        return bytes(buf)


SCENARIOS["live_trading"] = LiveTradingManifest

# ---- rotating_code (C4): a NON-SPATIAL late-bound field (cards/C4-late-bound-text.md) --------------------------

EVENT_CODE_CHANGES = 6


@dataclass(frozen=True)
class RotatingCodeManifest(Manifest):
    """A card (notification skin) with a text field (the target box) and a code panel above it. The panel shows a
    seeded 4-digit code from E; at E + change_delay_ns the code changes to a second seeded value and then holds
    (change_delay_ns >= response_window_ns: it never changes). The only valid concrete action is
    type(value) with value equal to the code CURRENTLY shown at the time of receipt, while the window is open.
    Return: minus the seconds from E to the receipt of a correct value; the window length when none arrives.

    kind_change=True (cards/V1-action-kind-validation.md): at E + change_delay_ns the code panel and the field are
    replaced by a blue "code expired - dismiss" button drawn over the field's box; from then on the only valid action
    is click(x, y) inside that button. A type(value) received after the change is invalid (the field is gone). The
    action space then contains both shapes, {type(value), click(x, y)}; the return keeps its definition."""
    scenario_id = "rotating_code"

    change_delay_ns: int = 0
    code_digits: int = 4
    kind_change: bool = False

    def __post_init__(self) -> None:
        if self.skin != "notification":
            raise ValueError("rotating_code is drawn in the notification skin")
        if self.change_delay_ns <= 0:
            raise ValueError("change_delay_ns must be positive (>= response_window_ns means the code never changes)")
        if not (1 <= self.code_digits <= 8):
            raise ValueError("code_digits must be in 1..8")
        super().__post_init__()

    def codes(self) -> tuple[str, str]:
        rng = random.Random(self.scenario_seed * 7919 + 17)
        a = "".join(rng.choice("0123456789") for _ in range(self.code_digits))
        while True:
            b = "".join(rng.choice("0123456789") for _ in range(self.code_digits))
            if b != a:
                return a, b

    def change_time_ns(self) -> int:
        return self.event_time_ns + self.change_delay_ns

    def code_at(self, task_time_ns: int) -> str | None:
        st = self.state_at(task_time_ns)
        if not st.target_visible:
            return None
        if self.dismiss_visible_at(task_time_ns):
            return None   # the code panel is gone
        a, b = self.codes()
        return b if task_time_ns >= self.change_time_ns() else a

    def dismiss_visible_at(self, task_time_ns: int) -> bool:
        """kind_change only: the dismiss button is shown from the change until the window closes."""
        return (self.kind_change and self.state_at(task_time_ns).target_visible
                and task_time_ns >= self.change_time_ns())

    def dismiss_box(self) -> TargetBox:
        """The dismiss button: the field's row, widened to the card's content width."""
        b, p = self.target_box(), self.panel_box()
        return TargetBox(min(b.x0, p.x0), b.y0, max(b.x1, p.x1), b.y1)

    def target_box_at(self, task_time_ns: int) -> TargetBox:
        return self.dismiss_box() if self.dismiss_visible_at(task_time_ns) else self.target_box()

    def target_color(self) -> tuple[int, int, int]:
        return TARGET_COLOR["notification"]   # the dismiss button (kind_change); no other pixel of this skin uses it

    def events(self) -> tuple[Event, ...]:
        base = super().events()
        if self.change_time_ns() < self.window_close_ns():
            kind = "code_expires" if self.kind_change else "code_changes"
            return (base[0], Event(EVENT_CODE_CHANGES, kind, self.change_time_ns()), base[1])
        return base

    def state_at(self, task_time_ns: int) -> State:
        st = super().state_at(task_time_ns)
        if st.target_visible and task_time_ns >= self.change_time_ns():
            return State(task_time_ns, True, True, EVENT_CODE_CHANGES)
        return st

    # --- action contract: type(value) ---
    def validate_action(self, action: dict) -> str | None:
        if self.kind_change and action.get("kind") == "click":
            return Manifest.validate_action(self, action)
        if set(action.keys()) != {"kind", "value"}:
            return f"action fields must be exactly kind,value; got {sorted(action.keys())}"
        if action["kind"] != "type":
            space = "{type, click}" if self.kind_change else "{type}"
            return f"action kind {action['kind']!r} not in action space {space}"
        v = action["value"]
        if type(v) is not str or len(v) != self.code_digits or not v.isdigit():
            return f"value must be a {self.code_digits}-digit string"
        return None

    def action_valid_at(self, task_time_ns: int, action: dict) -> bool:
        if self.validate_action(action) is not None:
            return False
        if action["kind"] == "click":
            return self.dismiss_visible_at(task_time_ns) and self.dismiss_box().contains(action["x"], action["y"])
        return self.code_at(task_time_ns) == action["value"]

    def spatial_error_px(self, task_time_ns: int, action: dict) -> float | None:   # type: ignore[override]
        if action.get("kind") == "click" and self.dismiss_visible_at(task_time_ns):
            return Manifest.spatial_error_px(self, task_time_ns, action)
        return None   # a value has no distance

    def return_on_receipt(self, task_time_ns: int, action: dict) -> float:
        if not self.action_valid_at(task_time_ns, action):
            return -self.response_window_ns / 1e9
        return -(task_time_ns - self.event_time_ns) / 1e9

    def return_no_receipt(self) -> float:
        return -self.response_window_ns / 1e9

    def panel_box(self) -> TargetBox:
        from .glyphs import panel_size
        pw, ph = panel_size(self.code_digits)
        b = self.target_box()
        x0 = min(b.x0, self.canvas_w_px - pw - 1)          # the panel is wider than the field; keep it on canvas
        return TargetBox(x0, b.y0 - 4 - ph, x0 + pw, b.y0 - 4)

    def render(self, task_time_ns: int) -> bytes:
        from .glyphs import PAD, PANEL_COLOR, INK_COLOR, draw_digits
        w, h = self.canvas_w_px, self.canvas_h_px
        st = self.state_at(task_time_ns)
        buf = bytearray(w * h * 3)
        _fill(buf, w, h, 0, 0, w, h, (30, 41, 59))                                 # desktop
        _fill(buf, w, h, 0, h - h // 12, w, h, (15, 23, 42))                       # taskbar
        if self.irrelevant_motion and (task_time_ns // max(1, self.frame_period_ns)) % 2 == 0:
            _fill(buf, w, h, 0, 0, DISTRACTOR_SIZE_PX, DISTRACTOR_SIZE_PX, DISTRACTOR_COLOR)
        if st.target_visible:
            b, p = self.target_box(), self.panel_box()
            th = self.target_size_px
            _fill(buf, w, h, min(b.x0, p.x0) - th // 2, p.y0 - th // 2, max(b.x1, p.x1) + th // 2, b.y1 + th // 2, (241, 245, 249))   # card
            if self.dismiss_visible_at(task_time_ns):
                # "code expired - dismiss": a grey notice line where the panel was, a blue button over the field's row
                _fill(buf, w, h, p.x0, p.y0 + (p.y1 - p.y0) // 2 - 2, p.x1, p.y0 + (p.y1 - p.y0) // 2 + 2, (148, 163, 184))
                d = self.dismiss_box()
                _fill(buf, w, h, d.x0, d.y0, d.x1, d.y1, TARGET_COLOR["notification"])
                _fill(buf, w, h, d.x0 + th // 2, d.y0 + th // 2 - 2, d.x1 - th // 2, d.y0 + th // 2 + 2, (219, 234, 254))   # label bar
                return bytes(buf)
            _fill(buf, w, h, p.x0, p.y0, p.x1, p.y1, PANEL_COLOR)                    # code panel
            draw_digits(buf, w, h, p.x0 + PAD, p.y0 + PAD, self.code_at(task_time_ns) or "", INK_COLOR)
            _fill(buf, w, h, b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1, (148, 163, 184))   # field border
            _fill(buf, w, h, b.x0, b.y0, b.x1, b.y1, (255, 255, 255))   # field
        return bytes(buf)


SCENARIOS["rotating_code"] = RotatingCodeManifest


# ---- reflow_page (G1): a Chromium-rendered page whose button moves when a banner is inserted -------------------
# cards/G1-realistic-reflow.md. Frames are baked by site/make_reflow_assets.py (real browser rendering of a seeded
# page, before and after the asynchronous notice banner reflows the layout); the button geometry comes from the DOM.
# The manifest stays a pure function of task time: before-frame until E + change_delay, after-frame from then on.

BROWSER_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "skin_browser"


@dataclass(frozen=True)
class ReflowPageManifest(Manifest):
    scenario_id = "reflow_page"

    change_delay_ns: int = 0

    def __post_init__(self) -> None:
        if self.skin != "browser":
            raise ValueError("reflow_page is drawn in the browser skin")
        if self.change_delay_ns <= 0:
            raise ValueError("change_delay_ns must be positive (>= response_window_ns means no reflow)")
        meta = self._meta()
        if (meta["w"], meta["h"]) != (self.canvas_w_px, self.canvas_h_px):
            raise ValueError("canvas must match the baked page size")
        if self.target_size_px != round(meta["button_before"]["height"]):
            raise ValueError(f"target_size_px must equal the baked button height {round(meta['button_before']['height'])}")
        super().__post_init__()

    def _meta(self) -> dict:
        p = BROWSER_ASSET_DIR / f"{self.canvas_w_px}x{self.canvas_h_px}_s{self.scenario_seed}.json"
        if not p.exists():
            raise FileNotFoundError(f"baked browser asset missing: {p} — run site/make_reflow_assets.py")
        import json
        return json.loads(p.read_text())

    def _asset(self, state: str) -> bytes:
        p = BROWSER_ASSET_DIR / f"{self.canvas_w_px}x{self.canvas_h_px}_s{self.scenario_seed}_{state}.bin"
        data = p.read_bytes()
        if len(data) != self.canvas_w_px * self.canvas_h_px * 3:
            raise ValueError(f"asset {p} has wrong size")
        return data

    def change_time_ns(self) -> int:
        return self.event_time_ns + self.change_delay_ns

    def reflow_dy_px(self) -> int:
        return int(self._meta()["dy"])

    def target_box(self) -> TargetBox:
        b = self._meta()["button_before"]
        x0, y0 = int(round(b["x"])), int(round(b["y"]))
        return TargetBox(x0, y0, x0 + int(round(b["width"])), y0 + int(round(b["height"])))

    def displacement_at(self, task_time_ns: int) -> tuple[int, int]:
        return (0, self.reflow_dy_px()) if task_time_ns >= self.change_time_ns() else (0, 0)

    def events(self) -> tuple[Event, ...]:
        base = super().events()
        if self.change_time_ns() < self.window_close_ns():
            return (base[0], Event(EVENT_TARGET_JUMPS, "target_jumps", self.change_time_ns()), base[1])
        return base

    def state_at(self, task_time_ns: int) -> State:
        st = super().state_at(task_time_ns)
        if st.target_visible and task_time_ns >= self.change_time_ns():
            return State(task_time_ns, True, True, EVENT_TARGET_JUMPS)
        return st

    def render(self, task_time_ns: int) -> bytes:
        st = self.state_at(task_time_ns)
        if not st.target_visible:
            # before the page has loaded / after the window: a blank tab (the header bar only), no button
            buf = bytearray(self._asset("before"))
            _fill(buf, self.canvas_w_px, self.canvas_h_px, 0, 48, self.canvas_w_px, self.canvas_h_px, (243, 244, 246))
            return bytes(buf)
        return self._asset("after" if task_time_ns >= self.change_time_ns() else "before")


SCENARIOS["reflow_page"] = ReflowPageManifest
