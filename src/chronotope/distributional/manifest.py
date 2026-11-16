"""Episode manifests for the distributional gym (cards/D1-distributional-gym.md, request §3, §4, §6, §7, §8).

One immutable row per episode. Every model and controller receives the identical row; nothing in a row depends
on any model. The deadline W_i is drawn from the row's stimulus seed, never from a latency (W_im = W_i for all m).

  generate(cfg)                 -> ManifestSet (rows in memory; not yet frozen)
  ManifestSet.freeze(path)      writes <path> (JSONL: header line + rows) and <path>.sha256; returns the hash
  ManifestSet.load(path)        reads a frozen manifest and verifies its hash
  build_manifest(row)           -> the scenario object (pure function of task time) for one row
  proxy_scenario(row)           -> a scenario dict the strict TrialSpec loader accepts for that row (the runtime
                                   drives the trial with build_manifest(row); the proxy exists for the spec)

Seeds: environment_seed places the target (the scenario seed); stimulus_seed draws W, v, theta, rho or q.
Held-out episodes: environment seeds start at 5_000_000 by default, disjoint from every existing study
(100.., 300.., 1000..). Repeated executions of one row are runtime-jitter estimates and carry a repeat index at
run time; they are never new rows.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..hashing import canonical_json, hash_obj, sha256_hex
from ..scenario import EVENT_TARGET_APPEARS, Manifest, payload_hash
from .scenarios import UPDATE_TYPES, DistMovingTargetManifest, DistRotatingCodeManifest

MANIFEST_SCHEMA = "distributional-manifest-1"
TASK_FAMILIES = ("closing_window", "moving_target", "static_control", "late_binding_natural", "late_binding_gate", "pending_effect")
ROW_FIELDS = ("episode_id", "environment_seed", "stimulus_seed", "task_family", "task_payload", "deadline_s",
              "target_speed_px_s", "target_direction", "target_start_position", "target_half_width_px",
              "update_time_s", "update_type", "route_condition", "expected_event_hash", "expected_observation_hash")

S = 1_000_000_000
PERIOD_NS, PUB_NS, DEL_NS = 16_666_667, 30_000_000, 20_000_000
EVENT_NS = 500_000_000
POST_WINDOW_NS = 500_000_000            # horizon margin after the window closes
FRAME_SAMPLES = 12                      # rendered frames folded into expected_observation_hash (plus the key frames)
DEFAULT_SEED_BASE = 5_000_000


@dataclass(frozen=True)
class GenConfig:
    task_family: str
    n: int
    master_seed: int
    seed_base: int = DEFAULT_SEED_BASE
    # closing window (request §4) and natural late binding (§7): the synthetic fallback range
    w_min_s: float = 1.0
    w_max_s: float = 16.0
    empirical_windows_s: tuple[float, ...] = ()     # real GUI-window durations; when given, W = F^-1(u) (§4)
    # moving target (§6)
    v_min_px_s: float = 1.0
    v_max_px_s: float = 16.0
    generous_deadline_s: float = 16.0
    # pending effect (the memory axis, 2026-08-27): the accepted click's effect appears after a delay drawn per episode
    delay_min_s: float = 0.5
    delay_max_s: float = 8.0
    # fixed-gate mechanism study (§8)
    q_min_s: float = 1 / 60
    q_max_s: float = 1.0
    gate_compute_s: float = 2.0
    update_type: str = "relevant"                    # late_binding_gate strata; late_binding_natural is relevant
    # geometry and route (common to every controller on the family)
    canvas_w_px: int = 320
    canvas_h_px: int = 240
    target_size_px: int = 24
    code_digits: int = 4
    frame_period_ns: int = PERIOD_NS
    publication_delay_ns: int = PUB_NS
    delivery_delay_ns: int = DEL_NS
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if self.task_family not in TASK_FAMILIES:
            raise ValueError(f"task_family must be one of {TASK_FAMILIES}")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.update_type not in UPDATE_TYPES or self.update_type == "none":
            raise ValueError(f"update_type must be one of {UPDATE_TYPES[1:]}")
        if not (0 < self.w_min_s <= self.w_max_s and 0 < self.v_min_px_s <= self.v_max_px_s and 0 < self.q_min_s <= self.q_max_s):
            raise ValueError("range bounds must satisfy 0 < min <= max")
        if self.gate_compute_s <= self.q_max_s:
            raise ValueError("gate_compute_s must exceed q_max_s so that B* < U < D*")
        if self.task_family == "moving_target":
            # constant motion for the whole window must fit the canvas in EVERY direction (request §6): otherwise
            # rows would have to be redrawn, which biases the (v, theta) distribution. Balloon skin margins.
            travel = self.v_max_px_s * self.generous_deadline_s
            span_x = self.canvas_w_px - self.target_size_px - 1 - 3
            span_y = self.canvas_h_px - self.target_size_px - 1 - 3 - self.target_size_px
            if travel > min(span_x, span_y):
                raise ValueError(f"v_max * deadline = {travel:.0f} px exceeds the usable canvas span "
                                 f"{min(span_x, span_y)} px; enlarge the canvas (640x480 holds 16 px/s x 16 s) or shorten the deadline")


@dataclass(frozen=True)
class EpisodeRow:
    episode_id: str
    environment_seed: int
    stimulus_seed: int
    task_family: str
    task_payload: dict[str, Any]
    deadline_s: float
    target_speed_px_s: float
    target_direction: float
    target_start_position: list[int]
    target_half_width_px: float
    update_time_s: float | None
    update_type: str
    route_condition: str
    expected_event_hash: str
    expected_observation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ROW_FIELDS}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeRow":
        if set(d) != set(ROW_FIELDS):
            raise ValueError(f"row fields must be exactly {ROW_FIELDS}; got {sorted(d)}")
        return cls(**d)

    @property
    def window_ns(self) -> int:
        return int(round(self.deadline_s * S))

    @property
    def event_ns(self) -> int:
        return int(self.task_payload["event_time_ns"])

    @property
    def close_ns(self) -> int:
        """T_close = E + W (task time)."""
        return self.event_ns + self.window_ns

    @property
    def update_ns(self) -> int | None:
        return None if self.update_time_s is None else int(round(self.update_time_s * S))

    @property
    def delivery(self) -> dict[str, Any]:
        return dict(self.task_payload["delivery"])

    @property
    def publication_delay_ns(self) -> int:
        return int(self.task_payload["publication_delay_ns"])


# ---- sampling (explicit generators only) ----------------------------------------------------------------------
def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    """lo (hi/lo)^u, u ~ U(0,1)  ==  log X ~ U(log lo, log hi)."""
    u = rng.random()
    return lo * (hi / lo) ** u


def empirical_quantile(samples: tuple[float, ...], u: float) -> float:
    """F^-1(u) of the empirical distribution (inverse of the step CDF)."""
    s = sorted(samples)
    k = min(len(s) - 1, max(0, math.ceil(u * len(s)) - 1))
    return s[k]


def first_target_frame_ns(event_ns: int, period_ns: int) -> int:
    return math.ceil(event_ns / period_ns) * period_ns


def fixed_gates_ns(cfg: GenConfig) -> tuple[int, int, int]:
    """B*, D*, A* on the simulated clock (h0 = 0): B* = O of the first frame that depicts the target."""
    b = first_target_frame_ns(EVENT_NS, cfg.frame_period_ns) + cfg.publication_delay_ns
    d = b + int(round(cfg.gate_compute_s * S))
    a = d + cfg.delivery_delay_ns
    return b, d, a


# ---- scenario objects from rows ------------------------------------------------------------------------------
def _common(row: EpisodeRow) -> dict[str, Any]:
    p = row.task_payload
    return dict(scenario_seed=row.environment_seed, canvas_w_px=p["canvas_w_px"], canvas_h_px=p["canvas_h_px"],
                target_size_px=p["target_size_px"], event_time_ns=p["event_time_ns"], response_window_ns=row.window_ns,
                frame_period_ns=p["frame_period_ns"], episode_horizon_ns=p["episode_horizon_ns"],
                irrelevant_motion=False, skin=p["skin"])


def build_manifest(row: EpisodeRow) -> Manifest:
    c = _common(row)
    fam = row.task_family
    if fam == "closing_window":
        return Manifest(**c)
    if fam in ("moving_target", "static_control"):
        return DistMovingTargetManifest(**c, speed_px_s=row.target_speed_px_s, direction_rad=row.target_direction)
    if fam in ("late_binding_natural", "late_binding_gate"):
        return DistRotatingCodeManifest(**c, change_delay_ns=row.update_ns - row.event_ns, code_digits=row.task_payload["code_digits"],
                                        kind_change=(row.update_type == "kind_change"), update_type=row.update_type)
    if fam == "pending_effect":
        from ..scenario import PendingEffectManifest
        return PendingEffectManifest(**c, effect_delay_ns=int(round(row.task_payload["effect_delay_s"] * S)))
    raise ValueError(fam)


def proxy_scenario(row: EpisodeRow) -> dict[str, Any]:
    """A scenario dict for the strict spec loader. The runtime uses build_manifest(row); the proxy only has to be a
    valid registered scenario with the same canvas, skin and timing (the trace header records the row's own hash)."""
    c = {"scenario_id": "revealed_target", **_common(row)}
    fam = row.task_family
    if fam in ("moving_target", "static_control"):
        vx = int(round(row.target_speed_px_s * math.cos(row.target_direction)))
        vy = int(round(row.target_speed_px_s * math.sin(row.target_direction)))
        return {**c, "scenario_id": "moving_target", "velocity_x_px_per_s": vx, "velocity_y_px_per_s": vy}
    if fam in ("late_binding_natural", "late_binding_gate"):
        return {**c, "scenario_id": "rotating_code", "change_delay_ns": row.update_ns - row.event_ns,
                "code_digits": row.task_payload["code_digits"], "kind_change": row.update_type == "kind_change"}
    if fam == "pending_effect":
        return {**c, "scenario_id": "pending_effect", "effect_delay_ns": int(round(row.task_payload["effect_delay_s"] * S))}
    return c


def expected_event_hash(m: Manifest) -> str:
    return hash_obj([[e.event_id, e.kind, e.task_time_ns] for e in m.events()])


def sample_frame_times(m: Manifest, update_ns: int | None) -> list[int]:
    """Key frames (first target frame, the frames around the update, the last open frame) plus FRAME_SAMPLES evenly
    spaced frames across the window; frame times are multiples of the period."""
    per = m.frame_period_ns
    t1 = first_target_frame_ns(m.event_time_ns, per)
    last = ((m.window_close_ns() - 1) // per) * per
    ts = {t1, last, 0}
    if update_ns is not None:
        ts.update({(update_ns // per) * per, math.ceil(update_ns / per) * per})
    for k in range(FRAME_SAMPLES):
        ts.add(t1 + ((last - t1) * k // max(1, FRAME_SAMPLES - 1)) // per * per)
    return sorted(t for t in ts if 0 <= t <= m.episode_horizon_ns)


def expected_observation_hash(m: Manifest, update_ns: int | None) -> str:
    return hash_obj([[t, payload_hash(m.render(t))] for t in sample_frame_times(m, update_ns)])


# ---- generation ----------------------------------------------------------------------------------------------
def _route_condition(cfg: GenConfig) -> str:
    return f"simulated_fixed_delay:{cfg.delivery_delay_ns // 1_000_000}ms:pub{cfg.publication_delay_ns // 1_000_000}ms"


def _payload(cfg: GenConfig, window_ns: int, skin: str, **extra: Any) -> dict[str, Any]:
    return {"canvas_w_px": cfg.canvas_w_px, "canvas_h_px": cfg.canvas_h_px, "target_size_px": cfg.target_size_px,
            "skin": skin, "event_time_ns": EVENT_NS, "frame_period_ns": cfg.frame_period_ns,
            "publication_delay_ns": cfg.publication_delay_ns, "episode_horizon_ns": EVENT_NS + window_ns + POST_WINDOW_NS,
            "delivery": {"route": "simulated_fixed_delay", "delivery_delay_ns": cfg.delivery_delay_ns,
                         "max_attempts": cfg.max_attempts}, **extra}


def _draw_window_s(cfg: GenConfig, rng: random.Random) -> float:
    if cfg.empirical_windows_s:
        return empirical_quantile(cfg.empirical_windows_s, rng.random())
    return log_uniform(rng, cfg.w_min_s, cfg.w_max_s)


def generate_row(cfg: GenConfig, index: int, stimulus_seed: int) -> EpisodeRow:
    env_seed = cfg.seed_base + index
    rng = random.Random(stimulus_seed)
    fam = cfg.task_family
    eid = f"{fam}-{env_seed}"
    speed, direction, update_s, update_type = 0.0, 0.0, None, "none"
    if fam == "closing_window":
        w_s = _draw_window_s(cfg, rng)
        window_ns = int(round(w_s * S))
        payload = _payload(cfg, window_ns, "notification")
    elif fam in ("moving_target", "static_control"):
        w_s = cfg.generous_deadline_s
        window_ns = int(round(w_s * S))
        if fam == "moving_target":
            speed = log_uniform(rng, cfg.v_min_px_s, cfg.v_max_px_s)
            direction = rng.random() * 2 * math.pi
        payload = _payload(cfg, window_ns, "balloon")
    elif fam == "pending_effect":
        w_s = cfg.generous_deadline_s
        window_ns = int(round(w_s * S))
        delay_s = log_uniform(rng, cfg.delay_min_s, cfg.delay_max_s)
        payload = _payload(cfg, window_ns, "notification", effect_delay_s=delay_s)
    elif fam == "late_binding_natural":
        w_s = _draw_window_s(cfg, rng)
        window_ns = int(round(w_s * S))
        rho = rng.random()
        update_s = (EVENT_NS + rho * window_ns) / S           # U = E + rho W, absolute task time
        update_type = cfg.update_type
        payload = _payload(cfg, window_ns, "notification", code_digits=cfg.code_digits, rho=rho)
    elif fam == "late_binding_gate":
        w_s = cfg.generous_deadline_s
        window_ns = int(round(w_s * S))
        q = log_uniform(rng, cfg.q_min_s, cfg.q_max_s)
        b, d, a = fixed_gates_ns(cfg)
        update_type = cfg.update_type
        u_ns = (a + int(round(q * S))) if update_type == "after_D" else (d - int(round(q * S)))
        if not (b < u_ns < d < a) and update_type != "after_D":
            raise ValueError("gate order B* < U < D* < A* violated")
        update_s = u_ns / S
        payload = _payload(cfg, window_ns, "notification", code_digits=cfg.code_digits, q_s=q,
                           gate_compute_ns=int(round(cfg.gate_compute_s * S)), B_star_ns=b, D_star_ns=d, A_star_ns=a)
    else:
        raise ValueError(fam)
    update_ns = None if update_s is None else int(round(update_s * S))
    if update_ns is not None and update_ns <= EVENT_NS:
        update_ns = EVENT_NS + 1                                  # the update is after the event by definition
        update_s = update_ns / S
    partial = EpisodeRow(eid, env_seed, stimulus_seed, fam, payload, window_ns / S, speed, direction, [0, 0],
                         cfg.target_size_px / 2, update_s, update_type, _route_condition(cfg), "", "")
    m = build_manifest(partial)
    b = m.target_box_at(EVENT_NS)
    payload = {**payload, "target_box_at_E": [b.x0, b.y0, b.x1, b.y1]}
    if fam in ("late_binding_natural", "late_binding_gate"):
        payload["codes"] = list(m.codes())
    row = EpisodeRow(eid, env_seed, stimulus_seed, fam, payload, window_ns / S, speed, direction,
                     list(m.target_box_at(EVENT_NS).center()), cfg.target_size_px / 2, update_s, update_type,
                     _route_condition(cfg), expected_event_hash(m), expected_observation_hash(m, update_ns))
    return row


def generate(cfg: GenConfig) -> "ManifestSet":
    master = random.Random(cfg.master_seed)
    stim = [master.randrange(1, 2**31) for _ in range(cfg.n)]
    rows = [generate_row(cfg, i, stim[i]) for i in range(cfg.n)]
    ids = [r.episode_id for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("episode ids are not unique")
    return ManifestSet(header={"schema": MANIFEST_SCHEMA, "config": asdict(cfg), "n": cfg.n}, rows=rows)


# ---- the frozen file ----------------------------------------------------------------------------------------
@dataclass
class ManifestSet:
    header: dict[str, Any]
    rows: list[EpisodeRow]
    path: Path | None = None
    frozen_hash: str | None = None
    _index: dict[str, EpisodeRow] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {r.episode_id: r for r in self.rows}
        if len(self._index) != len(self.rows):
            raise ValueError("duplicate episode ids")

    def manifest_hash(self) -> str:
        return hash_obj({"header": self.header, "rows": [r.to_dict() for r in self.rows]})

    def text(self) -> str:
        return "\n".join([canonical_json({"record": "manifest_header", **self.header})]
                         + [canonical_json({"record": "episode", **r.to_dict()}) for r in self.rows]) + "\n"

    def freeze(self, path: Path) -> str:
        """Write the manifest and its hash sidecar. A frozen manifest is never rewritten."""
        path = Path(path)
        side = path.with_suffix(path.suffix + ".sha256")
        if path.exists() or side.exists():
            raise FileExistsError(f"manifest already frozen: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text(), encoding="utf-8")
        h = self.manifest_hash()
        side.write_text(h + "\n", encoding="utf-8")
        self.path, self.frozen_hash = path, h
        return h

    @classmethod
    def load(cls, path: Path) -> "ManifestSet":
        path = Path(path)
        side = path.with_suffix(path.suffix + ".sha256")
        if not side.exists():
            raise FileNotFoundError(f"manifest is not frozen (no {side.name}); run generate/freeze first")
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
        if not lines or lines[0].get("record") != "manifest_header":
            raise ValueError("manifest file has no header line")
        header = {k: v for k, v in lines[0].items() if k != "record"}
        rows = [EpisodeRow.from_dict({k: v for k, v in l.items() if k != "record"}) for l in lines[1:]]
        ms = cls(header=header, rows=rows, path=path)
        want = side.read_text(encoding="utf-8").strip()
        got = ms.manifest_hash()
        if got != want:
            raise ValueError(f"manifest hash mismatch: file {got} != sidecar {want} (the frozen manifest was edited)")
        ms.frozen_hash = got
        return ms

    def row(self, episode_id: str) -> EpisodeRow:
        return self._index[episode_id]

    def family(self) -> str:
        fams = {r.task_family for r in self.rows}
        if len(fams) != 1:
            raise ValueError(f"a manifest holds one task family; got {sorted(fams)}")
        return fams.pop()

    def require_unique_count(self, n_min: int = 300) -> None:
        if len(self.rows) < n_min:
            raise ValueError(f"{len(self.rows)} unique episodes < the required {n_min} (request §3)")


def verify_row_against_manifest(row: EpisodeRow, m: Manifest) -> None:
    """Same seed, payload, deadline, event sequence and rendered evidence as the frozen row (request §16)."""
    if m.scenario_seed != row.environment_seed:
        raise AssertionError(f"{row.episode_id}: scenario seed {m.scenario_seed} != environment_seed {row.environment_seed}")
    if m.response_window_ns != row.window_ns:
        raise AssertionError(f"{row.episode_id}: manifest window {m.response_window_ns} != row deadline {row.window_ns}")
    if expected_event_hash(m) != row.expected_event_hash:
        raise AssertionError(f"{row.episode_id}: event sequence differs from the frozen row")
    if expected_observation_hash(m, row.update_ns) != row.expected_observation_hash:
        raise AssertionError(f"{row.episode_id}: rendered evidence differs from the frozen row")
    if list(m.target_box_at(row.event_ns).center()) != list(row.target_start_position):
        raise AssertionError(f"{row.episode_id}: target start position differs from the frozen row")


def load_empirical_windows(path: Path) -> tuple[float, ...]:
    """Real application-window durations in seconds: a JSON list, or {"window_s": [...]}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    xs = d["window_s"] if isinstance(d, dict) else d
    xs = tuple(float(x) for x in xs)
    if not xs or any(x <= 0 for x in xs):
        raise ValueError("empirical windows must be a non-empty list of positive seconds")
    return xs


__all__ = ["GenConfig", "EpisodeRow", "ManifestSet", "generate", "generate_row", "build_manifest", "proxy_scenario",
           "verify_row_against_manifest", "fixed_gates_ns", "first_target_frame_ns", "load_empirical_windows",
           "log_uniform", "TASK_FAMILIES", "ROW_FIELDS", "MANIFEST_SCHEMA", "S", "EVENT_NS", "EVENT_TARGET_APPEARS",
           "sha256_hex"]
