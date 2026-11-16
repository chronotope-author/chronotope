"""Build the prompt-only legacy application against the existing pristine backend.

No model exports, engine builds, model requests, or evaluation episodes occur.
The old frozen experiment and its engines are never modified.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/nvidia_prompt_only_20261029'
OLD = ROOT / 'runs/agents3_policy_nvidia27b_20261029'
STOCK = ROOT / 'runs/runtime_reversal_stock_20261029'
SOURCE = STOCK / 'source/TensorRT-Edge-LLM'
BUILD = STOCK / 'build'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    status = json.loads((OLD / 'experiment/status.json').read_text())
    assert status['state'] in ('paused', 'superseded'), status
    assert status['automatic_resume'] is False and status['pid'] is None
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'runs/legacy_d1_gpu.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        app = OUT / 'build_source/chronotope'
        assert not app.exists(), 'Do not replace an existing source snapshot'
        assert not (ROOT / 'chronotope/src/agents3_policy.h').exists(), 'Worker port remains'
        assert (ROOT / 'chronotope/src/agents3_prompt.h').exists()
        for directory in ('src', 'tests'):
            shutil.copytree(ROOT / 'chronotope' / directory, app / directory)
        remote = ROOT / 'chronotope/python/chronotope_core/remote.py'
        remote_snapshot = app / 'python/chronotope_core/remote.py'
        remote_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote, remote_snapshot)
        # Reuse the already-reviewed application build shim. It only supplies an
        # explicit pristine backend source path; the upstream backend is untouched.
        shutil.copy2(OLD / 'build_source/chronotope/CMakeLists.txt', app / 'CMakeLists.txt')
        binaries = OUT / 'native_build'
        commands = [
            ['cmake', '-S', str(app), '-B', str(binaries),
             '-DCHRONOTOPE_EDGELLM_BUILD=' + str(BUILD),
             '-DCHRONOTOPE_EDGELLM_SOURCE=' + str(SOURCE),
             ('-DCHRONOTOPE_TRT_PREFIX=' + _CHRONOTOPE_HOME + '/trt-prefix'), '-DCMAKE_BUILD_TYPE=Release'],
            ['cmake', '--build', str(binaries), '--target', 'chronotope_main', 'chronotope', '-j6'],
        ]
        inputs = {str(p): sha(p) for p in sorted(app.rglob('*')) if p.is_file()}
        inputs[str(remote)] = sha(remote)
        result = {'state': 'building', 'started_at': datetime.now(timezone.utc).isoformat(),
                  'source_hashes': inputs, 'commands': commands,
                  'upstream_commit': 'bb291453bc10d9bd1142070c605e090d4e483ee2',
                  'source_patches': [], 'model_inference_calls': 0, 'qualification_trials': 0,
                  'scope': 'Restore legacy controller; opt-in static Agent S3 prompt and action parser; stock image API compatibility retained',
                  'build_helper_sha256': sha(Path(__file__))}
        write(OUT / 'native_build_result.json', result)
        env = dict(os.environ)
        env.update(EDGELLM_PLUGIN_PATH=str(BUILD / 'libNvInfer_edgellm_plugin.so'),
                   LD_LIBRARY_PATH=':'.join([str(BUILD), (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu'),
                                            '/usr/local/cuda/targets/sbsa-linux/lib']),
                   PYTHONDONTWRITEBYTECODE='1')
        try:
            for command, name in zip(commands, ('native_configure.log', 'native_build.log')):
                with (OUT / name).open('wb') as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            assert all(sha(path) == digest for path, digest in inputs.items())
            result.update(state='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                          binary_hashes={str(binaries / name): sha(binaries / name)
                                         for name in ('chronotope', 'libchronotope.so')})
        except BaseException as error:
            result.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            write(OUT / 'native_build_result.json', result)
        print(json.dumps({'state': result['state'], 'model_inference_calls': 0, 'out': str(OUT)}))


if __name__ == '__main__':
    main()
