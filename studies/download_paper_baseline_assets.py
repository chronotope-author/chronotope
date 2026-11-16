"""Install pinned auxiliary assets; no model inference or evaluation episodes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'runs/paper_baseline_setup_20261108'
CACHE = ROOT / 'runs/model_cache/paper_baselines'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(repo, subdir, patterns, ignore=None):
    metadata = json.loads((SETUP / 'remote_models.json').read_text())[repo]
    target = CACHE / subdir
    snapshot_download(repo, revision=metadata['revision'], local_dir=target,
                      allow_patterns=patterns, ignore_patterns=ignore, max_workers=3)
    verified = {}
    for entry in metadata['files']:
        path = target / entry['name']
        if not path.is_file():
            continue
        if entry['lfs_sha256']:
            actual = digest(path)
            if actual != entry['lfs_sha256']:
                raise ValueError(f'Official LFS hash mismatch: {path}')
        else:
            data = path.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != entry['git_blob_id']:
                raise ValueError(f'Official Git blob hash mismatch: {path}')
            actual = hashlib.sha256(data).hexdigest()
        verified[str(path)] = actual
    if repo == 'UITron-hub/MobileDreamer':
        # The authors store tokenizer files beside merge_model, not inside it.
        for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                     'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja'):
            source = target / name
            if source.exists():
                dest = target / 'merge_model' / name
                shutil.copy2(source, dest)
                verified[str(dest)] = verified[str(source)]
    report = dict(repo=repo, revision=metadata['revision'], file_hashes=verified,
                  official_hashes_verified=True, model_inference_calls=0)
    (SETUP / (subdir + '_download.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Verified {repo}: {len(verified)} files', flush=True)


def clip():
    sha = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
    url = f'https://openaipublic.azureedge.net/clip/models/{sha}/ViT-B-16.pt'
    target = Path.home() / '.cache/clip/ViT-B-16.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix('.pt.downloading')
        urllib.request.urlretrieve(url, part)
        if digest(part) != sha:
            raise ValueError('Official CLIP hash mismatch')
        part.replace(target)
    if digest(target) != sha:
        raise ValueError('Existing CLIP checkpoint differs; not replacing it')
    (SETUP / 'clip_download.json').write_text(json.dumps(dict(url=url,
        revision=sha, file_hashes={str(target):sha}, official_hashes_verified=True), indent=2) + '\n')
    print('Verified CLIP ViT-B/16', flush=True)


if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, 'UITron-hub/MobileDreamer', 'MobileDreamer',
                    ['merge_model/*', 'tokenizer*', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt', 'chat_template.jinja']),
                   pool.submit(download, 'PaddlePaddle/PP-DocLayoutV2', 'PP-DocLayoutV2', ['*']),
                   pool.submit(download, 'PaddlePaddle/PaddleOCR-VL', 'PaddleOCR-VL', ['*'], ['PP-DocLayoutV2/*']),
                   pool.submit(clip)]
        errors = []
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
                print(errors[-1], flush=True)
        if errors:
            raise RuntimeError(errors)
