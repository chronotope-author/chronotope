"""Fixtures for the kGUA tests; re-exports chronotope's tests/conftest.py helpers (the test modules import
`from conftest import base_spec_dict`, and this directory's conftest shadows that name)."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("chronotope_tests_conftest", ROOT / "tests" / "conftest.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
base_spec_dict = _mod.base_spec_dict


@pytest.fixture
def ws(tmp_path):
    return tmp_path
