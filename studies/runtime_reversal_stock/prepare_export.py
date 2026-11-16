"""Freeze pristine NVIDIA sources and export the two recorded model checkpoints.

Preparation only reads source/model metadata. Export requires a paused evaluation
and exclusive GPU ownership even though export itself is forced onto the CPU.
No model-service calls or application trials are performed here.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
import tomllib

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM'))
COMMIT = 'bb291453bc10d9bd1142070c605e090d4e483ee2'
PYTHON = Path((_CHRONOTOPE_HOME + '/edgellm-venv/bin/python'))
LOCK = ROOT / 'runs/legacy_d1_gpu.lock'
EXPERIMENT = ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment'
MODELS = {
    'qwen08': ROOT / 'runs/agents3_policy_qwen08_20261028/experiment',
    'radixark27b': EXPERIMENT,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(repo, *arguments):
    return subprocess.check_output(['git', '-C', str(repo), *arguments])


def archive(repo, revision, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', revision],
                       stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as bundle:
            bundle.extractall(destination, filter='data')


def tree_files(repo, revision):
    entries = []
    for raw in git(repo, 'ls-tree', '-rz', revision).split(b'\0'):
        if not raw:
            continue
        metadata, path = raw.split(b'\t', 1)
        mode, kind, oid = metadata.decode().split()
        entries.append((mode, kind, oid, path.decode()))
    return entries


def freeze_source(out):
    source = out / 'source/TensorRT-Edge-LLM'
    assert not source.exists(), 'Refuse to overwrite existing upstream source'
    assert git(UPSTREAM, 'rev-parse', 'HEAD').decode().strip() == COMMIT
    archive(UPSTREAM, COMMIT, source)
    repositories = [(UPSTREAM, COMMIT, Path('.'))]
    submodules = []
    for mode, kind, oid, relative in tree_files(UPSTREAM, COMMIT):
        if kind == 'commit':
            original = UPSTREAM / relative
            assert git(original, 'rev-parse', 'HEAD').decode().strip() == oid
            archive(original, oid, source / relative)
            repositories.append((original, oid, Path(relative)))
            submodules.append({'path': relative, 'commit': oid})
    files = {}
    for repo, revision, prefix in repositories:
        for mode, kind, oid, relative in tree_files(repo, revision):
            if kind != 'blob':
                continue
            path = source / prefix / relative
            payload = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
            blob = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
            assert blob == oid, f'Archive differs from pinned upstream: {path}'
            files[str((prefix / relative))] = {
                'git_blob': oid, 'mode': mode, 'sha256': hashlib.sha256(payload).hexdigest(),
            }
    manifest = {'state': 'pristine_source_prepared', 'prepared_at': now(),
        'origin': 'https://github.com/NVIDIA/TensorRT-Edge-LLM', 'commit': COMMIT,
        'source': str(source), 'submodules': submodules, 'files': files,
        'tracked_files_verified_against_git_blobs': len(files),
        'local_source_patches': [], 'model_requests_executed': 0,
        'application_episodes': 0, 'qualification_trials': 0}
    write(out / 'upstream_source_manifest.json', manifest)
    return manifest


def environment(source):
    return dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='',
        LLM_SDK_DIR=str(source), PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def dependencies(source):
    required = tomllib.loads((source / 'pyproject.toml').read_text())['project']['dependencies']
    expected = dict(item.split('==', 1) for item in required)
    code = ('import importlib.metadata as m, importlib.util, json; '
            f'names={list(expected)!r}; '
            'print(json.dumps({"packages":{n:m.version(n) for n in names},'
            '"source":importlib.util.find_spec("tensorrt_edgellm").origin}))')
    result = subprocess.run([str(PYTHON), '-B', '-c', code], cwd=source,
        env=environment(source), check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout)
    assert observed['packages'] == expected, (observed['packages'], expected)
    assert Path(observed['source']).is_relative_to(source), observed['source']
    return {'python': str(PYTHON), 'required': expected, **observed}


def prepare(out):
    assert not (out / 'export_plan.json').exists(), 'Preparation already exists'
    out.mkdir(parents=True, exist_ok=True)
    pristine = freeze_source(out)
    source = Path(pristine['source'])
    packages = dependencies(source)
    models = {}
    for name, experiment in MODELS.items():
        execution = read(experiment / 'execution_manifest.json')
        checkpoint_hashes = execution['checkpoint_hashes']
        checkpoint_roots = {str(Path(path).parent) for path in checkpoint_hashes}
        assert len(checkpoint_roots) == 1
        checkpoint = Path(next(iter(checkpoint_roots)))
        config = read(checkpoint / 'config.json')
        destination = out / 'exports' / name
        models[name] = {'model': execution['model'], 'revision': execution['revision'],
            'source_precision': execution['precision'], 'source_experiment': str(experiment),
            'checkpoint': str(checkpoint), 'checkpoint_hashes': checkpoint_hashes,
            'checkpoint_weight_bytes': sum(Path(path).stat().st_size for path in checkpoint_hashes if path.endswith('.safetensors')),
            'model_type': config['model_type'], 'output': str(destination),
            'command': [str(PYTHON), '-B', '-m', 'tensorrt_edgellm.scripts.export', str(checkpoint), str(destination)],
            'dtype_argument': 'omitted: stock default float16 for visual/audio',
            'expected_runtime_precision': ('FP16 source and standard FP16 export' if name == 'qwen08' else
                'Source NVFP4 MLP/lm_head and FP8 projections retained; stock exporter converts BF16 dense/vision tensors to FP16'),
            'checkpoint_hash_verification': 'Deferred to locked export stage to avoid large reads during active evaluation'}
    plan = {'state': 'prepared', 'prepared_at': now(), 'source': str(source),
        'upstream_commit': COMMIT, 'source_manifest': str(out / 'upstream_source_manifest.json'),
        'source_manifest_sha256': sha(out / 'upstream_source_manifest.json'),
        'helper_sha256': sha(Path(__file__).resolve()), 'dependencies': packages, 'models': models,
        'environment_overrides': {key: value for key, value in environment(source).items()
            if key in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','LLM_SDK_DIR','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']},
        'model_requests_executed': 0, 'application_episodes': 0, 'qualification_trials': 0,
        'scope': 'Pristine dependency export preparation for recorded Disappearing Notification request profiling only',
        'precision_comparison_limit': 'Stock native export of Radix is not precision-identical to vLLM BF16 vision/dense sidecars; report observed exported dtypes separately.',
        'resource_notes': ['Run exports sequentially after evaluation pause and model-service shutdown.',
            'Historical modified Radix export ran about 2m38s; clean export timing is unmeasured.',
            'Exporter documentation recommends at least1.5x checkpoint bytes in CPU RAM; actual transient allocations may be higher.',
            'No output budget, task, policy or checkpoint substitutions are authorized by this helper.']}
    write(out / 'export_plan.json', plan)
    return plan


@contextmanager
def exclusive(lock_fd):
    if lock_fd is None:
        with LOCK.open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        return
    assert Path(f'/proc/self/fd/{lock_fd}').resolve() == LOCK.resolve()
    with LOCK.open('a+') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('Inherited descriptor did not already hold the GPU lock')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    yield


def verify_source(manifest):
    source = Path(manifest['source'])
    for relative, expected in manifest['files'].items():
        path = source / relative
        payload = os.readlink(path).encode() if expected['mode'] == '120000' else path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected['sha256'], path


def export_model(out, name, lock_fd):
    plan = read(out / 'export_plan.json')
    assert sha(Path(__file__).resolve()) == plan['helper_sha256'], 'Exporter helper changed after preparation'
    assert sha(plan['source_manifest']) == plan['source_manifest_sha256']
    selected = plan['models'][name]
    source = Path(plan['source'])
    destination = Path(selected['output'])
    result_path = out / ('export_' + name + '_result.json')
    assert not result_path.exists() and not destination.exists(), 'Refuse implicit retry/overwrite of export evidence'
    with exclusive(lock_fd):
        assert read(EXPERIMENT / 'status.json')['state'] in ('paused', 'complete')
        verify_source(read(plan['source_manifest']))
        assert dependencies(source) == plan['dependencies']
        for path, expected in selected['checkpoint_hashes'].items():
            assert sha(path) == expected, path
        result = {'state': 'exporting', 'started_at': now(), 'model': selected['model'],
            'revision': selected['revision'], 'command': selected['command'], 'cwd': str(source),
            'environment_overrides': plan['environment_overrides'], 'upstream_commit': COMMIT,
            'model_requests_executed': 0, 'application_episodes': 0, 'qualification_trials': 0}
        write(result_path, result)
        started = time.monotonic()
        try:
            with (out / ('export_' + name + '.log')).open('wb') as log:
                process = subprocess.run(selected['command'], cwd=source, env=environment(source),
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            result['returncode'] = process.returncode
            assert process.returncode == 0, f'Unmodified exporter exited {process.returncode}; preserve log for diagnosis'
            verify_source(read(plan['source_manifest']))
            assert (destination / 'llm/model.onnx').exists() and (destination / 'visual/model.onnx').exists()
            result.update(state='exported', output_hashes={str(p.relative_to(destination)): sha(p)
                for p in sorted(destination.rglob('*')) if p.is_file()})
        except BaseException as error:
            result.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            result.update(completed_at=now(), elapsed_seconds=time.monotonic()-started)
            write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--export', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / 'runs')
    if args.prepare:
        result = prepare(out)
    else:
        assert args.model is not None, '--export requires --model'
        result = export_model(out, args.model, args.lock_fd)
    print(json.dumps({key: result[key] for key in ['state','model_requests_executed','application_episodes','qualification_trials']}))


if __name__ == '__main__':
    main()
