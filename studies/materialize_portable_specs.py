"""Write machine-local copies of historical JSON configurations.

Original specs and archived traces are read-only. The exported configurations
have new hashes and must not be used as if they were the original frozen inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remap_config(value, *, home: Path, repository: Path):
    """Return a copy with historical absolute-path placeholders resolved."""
    if isinstance(value, dict):
        return {key: remap_config(v, home=home, repository=repository) for key, v in value.items()}
    if isinstance(value, list):
        return [remap_config(v, home=home, repository=repository) for v in value]
    if isinstance(value, str):
        return (value.replace("/home/USER/chronotope", str(repository))
                     .replace("/home/USER", str(home)))
    return value


def export(source: Path, output: Path, *, home: Path, repository: Path) -> list[dict]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("output must be separate from the source tree")
    files = sorted(source.rglob("*.json"))
    if not files:
        raise ValueError(f"no JSON specs under {source}")
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for path in files:
        relative = path.relative_to(source)
        destination = output / relative
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        original = json.loads(path.read_text(encoding="utf-8"))
        resolved = remap_config(original, home=home, repository=repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report.append({"file": relative.as_posix(), "original_sha256": _hash(path),
                       "local_sha256": _hash(destination), "changed": original != resolved})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPOSITORY / "specs" / "d1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--home", type=Path, default=Path(os.environ.get("CHRONOTOPE_HOME", Path.home())))
    parser.add_argument("--repository", type=Path,
                        default=Path(os.environ.get("CHRONOTOPE_REPO_ROOT", REPOSITORY)))
    args = parser.parse_args()
    report = export(args.source, args.output, home=args.home.resolve(), repository=args.repository.resolve())
    print(json.dumps({"configs_written": len(report), "paths_changed": sum(r["changed"] for r in report),
                      "files": report}, indent=2))


if __name__ == "__main__":
    main()
