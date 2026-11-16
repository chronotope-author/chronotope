"""Load locked upstream source without importing desktop/Android entry points."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .common import SetupError

LOCK_PATH = Path(__file__).with_name('sources.lock.json')


def locked_file(root, project, relative):
    lock = json.loads(LOCK_PATH.read_text())[project]
    expected = lock['files'].get(relative)
    path = Path(root) / relative
    if expected is None or not path.is_file():
        raise SetupError(f'Missing pinned upstream source: {project}/{relative}')
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SetupError(f'Upstream source hash mismatch: {path}')
    return path


def load_module(root, project, relative):
    path = locked_file(root, project, relative)
    name = '_chronotope_upstream_' + hashlib.sha256(str(path).encode()).hexdigest()
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return sys.modules[name]


def load_methods(root, project, relative, class_name, methods, namespace, constants=()):
    """Compile unchanged method ASTs, avoiding the unreleased Android base class.

    This is a dependency boundary, not a rewrite of the candidate/selection
    algorithms. The complete original source is hash-checked before extraction.
    """
    path = locked_file(root, project, relative)
    tree = ast.parse(path.read_text(), filename=str(path))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if {n.name for n in body} != set(methods):
        raise SetupError('Missing upstream methods')
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in constants for t in n.targets)]
    cls = ast.ClassDef(name=class_name, bases=[], keywords=[], body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=assignments + [cls], type_ignores=[]))
    ns = dict(namespace)
    exec(compile(module, str(path), 'exec', flags=annotations.compiler_flag), ns)
    return ns[class_name]
