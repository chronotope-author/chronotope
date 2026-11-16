"""Export the pinned NVIDIA checkpoint through pristine upstream code only."""

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/agents3_policy_nvidia27b_20261029'
STOCK = ROOT / 'runs/runtime_reversal_stock_20261029'
spec = importlib.util.spec_from_file_location('nvidia_export_utils', ROOT / 'studies/runtime_reversal_stock/prepare_export.py')
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

if __name__ == '__main__':
    result_path = OUT / 'export_result.json'
    assert not result_path.exists() and not (OUT / 'exports').exists()
    checkpoint = u.read(OUT / 'checkpoint_download.json')
    assert checkpoint['state'] == 'downloaded'
    assert checkpoint['revision'] == '482ca0f3832238542f8f5295dde86b5f22711d80'
    manifest = u.read(STOCK / 'upstream_source_manifest.json')
    source = Path(manifest['source'])
    command = [(_CHRONOTOPE_HOME + '/edgellm-venv/bin/python'), '-B', '-m',
               'tensorrt_edgellm.scripts.export', checkpoint['checkpoint'], str(OUT / 'exports')]
    env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES='',
               LLM_SDK_DIR=str(source), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1')
    with u.exclusive(None):
        assert u.read(ROOT / 'runs/agents3_policy_radixark27b_20261028/experiment/status.json')['state'] == 'superseded'
        u.verify_source(manifest)
        for path, expected in checkpoint['hashes'].items():
            assert u.sha(path) == expected, path
        result = {'state': 'exporting', 'started_at': u.now(), 'command': command,
                  'model': checkpoint['model'], 'revision': checkpoint['revision'],
                  'upstream_commit': manifest['commit'], 'source': str(source),
                  'source_manifest_sha256': u.sha(STOCK / 'upstream_source_manifest.json'),
                  'checkpoint_hashes': checkpoint['hashes'], 'source_patches': [],
                  'inference_calls': 0, 'qualification_trials': 0,
                  'environment_overrides': {k: env[k] for k in ['PYTHONPATH','CUDA_VISIBLE_DEVICES','LLM_SDK_DIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE']}}
        u.write(result_path, result)
        start = time.monotonic()
        try:
            with (OUT / 'export.log').open('wb') as log:
                subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            u.verify_source(manifest)
            result.update(state='exported', output_hashes={str(p.relative_to(OUT / 'exports')): u.sha(p)
                for p in sorted((OUT / 'exports').rglob('*')) if p.is_file()})
        except BaseException as error:
            result.update(state='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            result.update(completed_at=u.now(), elapsed_seconds=time.monotonic()-start)
            u.write(result_path, result)
        print(json.dumps({'state': result['state'], 'elapsed_seconds': result['elapsed_seconds']}))
