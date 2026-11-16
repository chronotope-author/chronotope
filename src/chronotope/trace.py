"""Append-only JSONL trace. One record per line; `seq` is monotonic; nothing is ever rewritten.

Record types (experiments.md §13):
  trial_header, event, observation, observation_superseded, admission, decision_begin,
  action_proposed, commit, commit_failed, delivery, receipt, trial_end
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import canonical_json, hash_file


class TraceWriter:
    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"trace exists; traces are append-only and never overwritten: {path}")
        self.path = path
        self._f = path.open("a", encoding="utf-8")
        self._seq = 0

    def append(self, record: str, **fields: Any) -> None:
        self._seq += 1
        self._f.write(canonical_json({"seq": self._seq, "record": record, **fields}) + "\n")
        self._f.flush()

    def close(self) -> str:
        self._f.close()
        return hash_file(self.path)


class Trace:
    def __init__(self, records: list[dict[str, Any]], path: Path | None = None) -> None:
        self.records = records
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "Trace":
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for i, r in enumerate(recs, start=1):
            if r.get("seq") != i:
                raise ValueError(f"trace {path}: seq gap/reorder at line {i}")
        return cls(recs, path)

    def all(self, record: str) -> list[dict[str, Any]]:
        return [r for r in self.records if r["record"] == record]

    def first(self, record: str) -> dict[str, Any] | None:
        for r in self.records:
            if r["record"] == record:
                return r
        return None

    def committing_decision(self) -> dict[str, Any] | None:
        """The decision_begin record whose decision committed the action (or None)."""
        c = self.first("commit")
        if c is None:
            return None
        return next((d for d in self.all("decision_begin") if d["decision_id"] == c["decision_id"]), None)

    def header(self) -> dict[str, Any]:
        h = self.first("trial_header")
        if h is None:
            raise ValueError("trace has no trial_header")
        return h

    def file_hash(self) -> str:
        if self.path is None:
            raise ValueError("in-memory trace has no file hash")
        return hash_file(self.path)
