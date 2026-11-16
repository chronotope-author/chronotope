"""Download and hash the explicitly selected NVIDIA checkpoint; no inference."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/agents3_policy_nvidia27b_20261029'
DEST = ROOT / 'runs/model_cache/nvidia_qwen38_nvfp4_482ca0f'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'
os.environ['HF_HOME'] = str(ROOT / 'runs/model_cache/.hf_nvidia')
os.environ['HF_XET_CACHE'] = str(ROOT / 'runs/model_cache/.hf_nvidia/xet')
from huggingface_hub import snapshot_download

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(value):
    target = OUT / 'checkpoint_download.json'
    tmp = target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(target)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    record = {'state': 'downloading', 'model': 'nvidia/Qwen3.8-27B-NVFP4',
              'revision': REVISION, 'checkpoint': str(DEST),
              'started_at': datetime.now(timezone.utc).isoformat(),
              'inference_calls': 0, 'qualification_trials': 0}
    write(record)
    snapshot_download(record['model'], revision=REVISION, local_dir=DEST,
                      token=False, max_workers=4)
    hashes = {str(p): sha(p) for p in sorted(DEST.iterdir()) if p.is_file()}
    expected = [
        '7d0fd155118901373eb0fd13ed3aae68f95be747bc205be72ebc41739c25ee80',
        '98a7e9486baa860c792c9463a770cb9d017696bdd17606300a5b9334149f4c27',
        '0506ad35dc21469708e7813bd76c592cef08bd0317b4a1fe899745ebe2435271']
    for i, wanted in enumerate(expected, 1):
        assert hashes[str(DEST / f'model-{i:05d}-of-00003.safetensors')] == wanted
    record.update(state='downloaded', hashes=hashes,
                  completed_at=datetime.now(timezone.utc).isoformat())
    write(record)
    print(json.dumps({'state': 'downloaded', 'revision': REVISION, 'files': len(hashes)}))
