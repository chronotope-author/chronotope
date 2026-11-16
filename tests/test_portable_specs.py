"""Local configuration exports must not rewrite historical inputs."""

from pathlib import Path
import importlib.util
import json

import pytest


def load_exporter():
    path = Path(__file__).resolve().parents[1] / "studies" / "materialize_portable_specs.py"
    spec = importlib.util.spec_from_file_location("materialize_portable_specs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_preserves_original_json_and_records_new_hashes(tmp_path):
    exporter = load_exporter()
    source, output = tmp_path / "specs", tmp_path / "resolved"
    source.mkdir()
    original = {"base_url": "/home/USER/engines", "nested": ["/home/USER/chronotope/assets", 3]}
    file = source / "sample.json"
    file.write_text(json.dumps(original))
    old_bytes = file.read_bytes()
    report = exporter.export(source, output, home=Path("/tmp/machine"), repository=Path("/tmp/repo"))
    assert file.read_bytes() == old_bytes
    assert json.loads((output / "sample.json").read_text()) == {
        "base_url": "/tmp/machine/engines", "nested": ["/tmp/repo/assets", 3]}
    assert report[0]["original_sha256"] != report[0]["local_sha256"]
    assert report[0]["changed"]
    with pytest.raises(FileExistsError):
        exporter.export(source, output, home=Path("/tmp/machine"), repository=Path("/tmp/repo"))


def test_export_cannot_modify_source_tree(tmp_path):
    exporter = load_exporter()
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "sample.json").write_text('{}')
    with pytest.raises(ValueError):
        exporter.export(tmp_path / "specs", tmp_path / "specs" / "resolved",
                        home=tmp_path, repository=tmp_path)
