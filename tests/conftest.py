import copy
import json
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[1] / "specs" / "revealed_target_live.json"


def base_spec_dict(**over) -> dict:
    """Load the checked-in Live spec and apply dotted overrides, e.g. controller.compute_ns=5."""
    d = json.loads(SPEC_PATH.read_text())
    for k, v in over.items():
        parts = k.split("__")
        cur = d
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = v
    return d


@pytest.fixture
def spec_dict():
    return base_spec_dict()


@pytest.fixture
def ws(tmp_path):
    return tmp_path
