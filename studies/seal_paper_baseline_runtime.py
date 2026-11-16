"""Record verified auxiliary installation/launch evidence for an unlaunched plan."""
import argparse
import collections
import importlib.metadata
import json
from pathlib import Path
import sys
import urllib.request

from run_qwen08_five_arm import read, write, digest, require

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'runs/paper_baseline_setup_20261108'
CACHE = ROOT / 'runs/model_cache/paper_baselines'


def seal(out):
    require(read(out / 'status.json')['state'] == 'prepared', 'Unlaunched plan required')
    require(read(SETUP / 'auxiliary_loading.json')['state'] == 'passed', 'Model-loading checks failed')
    assets = {}
    specifications = (
        ('clip', 'clip', Path.home() / '.cache/clip', 'mixed FP16/FP32 JIT weights', 'FP32 CPU'),
        ('ocr_layout', 'OCRLegacyExports', CACHE / 'OCRLegacyExports/PP-DocLayoutV2',
         'FP32 weights; official graph also has boolean buffers', 'FP32 CPU; stock Paddle legacy IR'),
        ('ocr_vl', 'PaddleOCR-VL', CACHE / 'PaddleOCR-VL', 'BF16', 'FP32 CPU; stock Paddle conversion'),
        ('world_model', 'MobileDreamer', CACHE / 'MobileDreamer/merge_model',
         'FP32; all seven safetensors shard headers verified', 'FP16; stock vLLM conversion'))
    for name, report_name, root, source, effective in specifications:
        report = read(SETUP / (report_name + '_download.json'))
        files = {p: sha for p, sha in report['file_hashes'].items() if Path(p).is_relative_to(root)}
        require(files and report['official_hashes_verified'], 'Missing official asset verification')
        assets[name] = dict(root=str(root), revision=report['revision'], file_hashes=files,
            source_precision=source, effective_precision=effective,
            download_report_sha256=digest(SETUP / (report_name + '_download.json')))
    import torch
    clip = torch.jit.load(str(Path.home() / '.cache/clip/ViT-B-16.pt'), map_location='cpu')
    assets['clip']['source_parameter_dtypes'] = dict(collections.Counter(str(p.dtype) for p in clip.parameters()))
    assets['clip']['code_revision'] = 'd05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
    world = read(SETUP / 'world_launch.json')
    command = (Path('/proc') / str(world['pid']) / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    require(command == world['command'], 'World service process changed')
    log = (SETUP / 'world_service.log').read_text()
    require('dtype=torch.float16' in log and 'kv_cache_dtype=auto' in log and
            'Application startup complete' in log, 'Missing service startup evidence')
    with urllib.request.urlopen(world['base_url'] + '/models', timeout=10) as response:
        require([m['id'] for m in json.load(response)['data']] == [world['model']], 'Wrong world-model endpoint')
    world['launch_log'] = str(SETUP / 'world_service.log')
    versions = {}
    for package in ('torch', 'torchvision', 'clip', 'paddlepaddle', 'paddleocr', 'paddlex',
                    'numpy', 'pillow', 'openai', 'anthropic', 'transformers', 'vllm'):
        distribution = importlib.metadata.distribution(package)
        versions[package] = dict(version=distribution.version, root=str(distribution.locate_file('')))
    write(out / 'auxiliary_runtime.json', dict(schema='paper-baseline-runtime-1',
        python=str(Path(sys.executable).absolute()), plan_sha256=digest(out / 'plan.json'),
        assets=assets, world_service=world, packages=versions, model_loading=read(SETUP / 'auxiliary_loading.json'),
        required_environment={'PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK': 'True'},
        qualification_trials=0, model_inference_readiness_requests=0,
        numerical_notes=[
            'The two official layout exports have identical pdiparams SHA256; legacy IR avoids an ARM PIR loader crash.',
            'OCR VL source BF16 is converted to CPU FP32 by stock Paddle.',
            'CLIP encodes on CPU in FP32; MobileDreamer world inference uses GPU FP16 with FP16 KV.',
            'Model loading is setup; image encoding/OCR/prediction occur on the measured evaluation path.']))
    print('Sealed auxiliary runtime:', out / 'auxiliary_runtime.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    seal(parser.parse_args().out.resolve())
