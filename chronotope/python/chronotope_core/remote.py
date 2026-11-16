"""K4b (2026-08-27): the agent as a separate C++ process. `SocketCore` has the method names and return dicts of
`chronotope_core.Core` (the ctypes wrapper), but every call is a request to `chronotope --serve <socket>` — one process that owns the
CUDA core and, with `--planner trt`, the TensorRT Edge-LLM engine; the model sees the frame at B on the device.
The referee (the gym) keeps its clocks; this file is the only Python between it and the agent.

Wire format: one line of flat JSON per request; "admit"/"track" append raw payload bytes; one JSON line per reply
("read_frame" is followed by the raw RGB). Field names of the replies are the C structs', so the dicts match Core's.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
from typing import Any

from . import ACTION, ACTION_NAME, BINDING, BINDING_ID, OUTCOME, PENDING_EVENT, RELEVANCE, ROOT

STALE, NO_DECISION = 1, 2


def _bin() -> str:
    env = os.environ.get("CHRONOTOPE_BIN")
    for c in ([env] if env else []) + [str(ROOT / "build" / "chronotope"), str(ROOT / "build-release" / "chronotope")]:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("chronotope binary not found: build chronotope/ (cmake -S chronotope -B chronotope/build && make -C chronotope/build) or set CHRONOTOPE_BIN")


class SocketCore:
    """Spawn `chronotope --serve` with a config equivalent to Core(**kwargs), or attach to `socket_path` if given."""

    def __init__(self, *, width: int, height: int, template_radius_px: int = 12, min_match_score: float = 0.8,
                 policy: str = "late_bind", restart_threshold_px: int | None = None, target_rgb: tuple[int, int, int] | None = None,
                 coordinate_offset_px: int = 0, movement_threshold_px: int = 6, validate_kind: bool = False, rgb_template: bool = False,
                 track_every_frame: bool = True, max_evidence_age_ns: int = 0, pending_radius_px: int = 0, pending_timeout_ns: int = 0,
                 detector_at_d: int = 0, ack_aware_pending: bool = False, receipt_timeout_ns: int = 0,
                 planner: str = "detector", engines_root: str | None = None, prompt_version: int = 5,
                 image_scale: int = 2, max_tokens: int = 64, temperature: float = 0.0, socket_path: str | None = None,
                 spawn_timeout_s: float = 300.0, model_id: str | None = None,
                 api_endpoint: str | None = None, reasoning_effort: str | None = None, **_ignored: Any) -> None:
        self.width, self.height = width, height
        self.planner = planner
        self._proc: subprocess.Popen | None = None
        self._path = socket_path or os.path.join(tempfile.mkdtemp(prefix="chronotope_sock_"), "agent.sock")
        if socket_path is None:
            args = [_bin(), "--serve", self._path, "--width", str(width), "--height", str(height), "--pixfmt", "RGB24",
                    "--planner", planner, "--policy", policy, "--radius", str(template_radius_px), "--min-score", str(min_match_score),
                    "--restart-threshold", str(restart_threshold_px if restart_threshold_px is not None else 6),
                    "--image-scale", str(image_scale), "--prompt-version", str(prompt_version), "--max-tokens", str(max_tokens),
                    "--temperature", str(temperature), "--detector-at-d", str(detector_at_d), "--max-evidence-age-ms", str(max_evidence_age_ns / 1e6),
                    "--pending-radius", str(pending_radius_px), "--pending-timeout-ms", str(pending_timeout_ns / 1e6), "--receipt-timeout-ms", str(receipt_timeout_ns / 1e6)]
            if ack_aware_pending:
                args += ["--ack-aware"]
            if target_rgb is not None:
                args += ["--target-rgb", ",".join(str(int(v)) for v in target_rgb)]
            if validate_kind:
                args += ["--validate-kind"]
            if planner == "trt":
                if not engines_root:
                    raise ValueError("planner trt needs engines_root (…/engines with llm/ and the visual engine)")
                args += ["--engine-dir", os.path.join(engines_root, "llm"), "--multimodal-engine-dir", engines_root]
                if os.environ.get("CHRONOTOPE_AGENT_S3_PROMPT"):
                    args += ["--agents3-prompt", os.environ["CHRONOTOPE_AGENT_S3_PROMPT"]]
                    if os.environ.get("CHRONOTOPE_AGENT_S3_AUDIT"):
                        args += ["--agents3-prompt-audit", os.environ["CHRONOTOPE_AGENT_S3_AUDIT"]]
            if planner == "api":
                providers = {'gpt-6.1-sol': ('openai', 'OPENAI_API_KEY'),
                             'claude-fable-5-1': ('anthropic', 'ANTHROPIC_API_KEY')}
                if model_id not in providers or not api_endpoint or not os.environ.get('CHRONOTOPE_PLANNER_AUDIT'):
                    raise ValueError('Hosted planner requires a supported model, endpoint and audit path')
                provider, key_env = providers[model_id]
                args += ['--model', model_id, '--base-url', api_endpoint, '--api-provider', provider,
                         '--api-key-env', key_env, '--api-effort', reasoning_effort or 'low',
                         '--api-audit', os.environ['CHRONOTOPE_PLANNER_AUDIT']]
            self._log = open(self._path + ".log", "wb")
            self._proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            t0 = time.time()
            while not os.path.exists(self._path):
                if self._proc.poll() is not None:
                    raise RuntimeError(f"chronotope --serve exited {self._proc.returncode}; see {self._path}.log")
                if time.time() - t0 > spawn_timeout_s:
                    self._proc.kill(); raise TimeoutError("chronotope --serve did not open its socket")
                time.sleep(0.02)
        self._s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        for _ in range(200):
            try:
                self._s.connect(self._path); break
            except (ConnectionRefusedError, FileNotFoundError):
                time.sleep(0.02)
        self._f = self._s.makefile("rb")
        self.hello = self._call({"op": "hello"})
        self.agent_process = f"chronotope --serve pid={self._proc.pid if self._proc else 'attached'} {self.hello.get('version')}"

    # ---- transport --------------------------------------------------------------------------------------------------
    def _send(self, req: dict, *payloads: bytes) -> None:
        self._s.sendall((json.dumps(req) + "\n").encode())
        for p in payloads:
            self._s.sendall(p)

    def _recv(self) -> dict:
        line = self._f.readline()
        if not line:
            raise RuntimeError("chronotope --serve closed the connection")
        return json.loads(line)

    def _call(self, req: dict, *payloads: bytes) -> dict:
        self._send(req, *payloads)
        r = self._recv()
        if not r.get("ok", False):
            raise RuntimeError(f"chronotope serve {req.get('op')}: {r.get('error')}")
        return r

    # ---- the Core interface -------------------------------------------------------------------------------------------
    def admit(self, payload: bytes, *, observation_id: int, source_event_id: int, source_time_ns: int, publication_time_ns: int) -> dict:
        d = self._call({"op": "admit", "observation_id": observation_id, "source_event_id": source_event_id, "source_time_ns": source_time_ns,
                        "publication_time_ns": publication_time_ns, "len": len(payload)}, payload)
        d["relevance_name"] = RELEVANCE[d["relevance"]]; d["pending_event_name"] = PENDING_EVENT[d["pending_event"]]
        return d

    def begin_decision(self, observation_id: int) -> dict:
        return self._call({"op": "begin_decision", "observation_id": observation_id})

    def decide_detector(self) -> dict:
        return self._call({"op": "decide_detector"})

    def decide_trt(self) -> dict:
        """The in-process model on the frame at B; blocks for the answer. Returns the answer with 'proposal' (or None)."""
        return self._call({"op": "decide_trt"})

    def decide_api(self) -> dict:
        """C++ stages the frozen frame, calls HTTPS, parses and proposes the action."""
        return self._call({"op": "decide_api"})

    def propose(self, kind: str, x: int = 0, y: int = 0, value: int = 0) -> dict:
        return self._call({"op": "propose", "kind": ACTION[kind], "x": int(x), "y": int(y), "value": int(value)})

    def abort_decision(self) -> None:
        self._call({"op": "abort_decision"})

    def restart(self) -> None:
        self._call({"op": "restart"})

    def decision_open(self) -> bool:
        return bool(self._call({"op": "decision_open"})["open"])

    def _commit(self, op: str) -> dict:
        d = self._call({"op": op})
        d["outcome_name"] = OUTCOME.get(d["outcome"], str(d["outcome"])); d["binding_name"] = BINDING.get(d["binding"], str(d["binding"]))
        d["kind_name"] = ACTION_NAME.get(d["kind"], str(d["kind"]))
        return d

    def resolve_and_commit(self) -> dict:
        return self._commit("resolve_and_commit")

    def resolve(self) -> dict:
        return self._commit("resolve")

    def commit_only(self) -> dict:
        return self._commit("commit_only")

    def test_recommit(self) -> dict:
        return self._commit("test_recommit")

    def read_frame(self) -> bytes:
        d = self._call({"op": "read_frame"})
        n = int(d["len"]); buf = bytearray()
        while len(buf) < n:
            chunk = self._f.read(n - len(buf))
            if not chunk:
                raise RuntimeError("short frame read")
            buf += chunk
        return bytes(buf)

    def override_binding(self, x: int, y: int) -> None:
        self._call({"op": "override_binding", "x": int(x), "y": int(y)})

    def bind_external(self, x: int, y: int, binding: str = "grounder") -> None:
        self._call({"op": "bind_external", "x": int(x), "y": int(y), "binding": BINDING_ID[binding]})

    def observe_receipt(self, decision_id: int, observation_publication_ns: int) -> None:
        self._call({"op": "observe_receipt", "decision_id": int(decision_id), "observation_publication_ns": int(observation_publication_ns)})

    def audit(self) -> dict:
        d = self._call({"op": "audit"})
        d["residency_ok"] = bool(d["residency_ok"]); d["window_open"] = bool(d["window_open"])
        return d

    def state_hash(self) -> tuple[int, int]:
        d = self._call({"op": "state_hash"}); return int(d["hash"]), int(d["version"])

    def ledger(self, n: int = 256) -> list[dict]:
        return list(self._call({"op": "ledger", "n": n})["rows"])

    def trace(self, n: int = 256) -> list[dict]:
        return list(self._call({"op": "trace", "n": n})["rows"])

    def track(self, frame_b: bytes, frame_d: bytes, x: int, y: int, radius: int = 12, rgb: bool = False) -> dict:
        assert len(frame_b) == len(frame_d)
        return self._call({"op": "track", "x": int(x), "y": int(y), "radius": int(radius), "rgb": int(rgb), "len": len(frame_b)}, frame_b, frame_d)

    def close(self) -> None:
        try:
            self._send({"op": "close"})
        except OSError:
            pass
        try:
            self._s.close()
        except OSError:
            pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
