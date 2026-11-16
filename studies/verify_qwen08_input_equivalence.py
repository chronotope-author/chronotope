"""CPU-only proof of the aligned Qwen0.8B request inputs; never runs a model.

Checks a stored evaluation screenshot against a CPU emulation of the frozen
native nearest-neighbor, normalization, patch ordering, and chat-template path.
This is source/CPU equivalence, not capture of the actual native GPU buffers.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys

# Set before importing libraries that can initialize accelerators/network clients.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents3_matched_request import SYSTEM_PROMPT, USER_TEXT, REQUEST_CONTRACT, contract_fingerprint


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def verify(reference_run, checkpoint, engine_root):
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from tokenizers import Tokenizer
    import transformers
    import tokenizers
    import torch

    sys.path.insert(0, str(reference_run / 'code/src'))
    from chronotope.png import encode_png
    from chronotope.controller import SYSTEM_PROMPTS
    assert SYSTEM_PROMPT == SYSTEM_PROMPTS[4]
    native_prompt_file = reference_run / 'code/chronotope/src/planner_vlm.cpp'
    native_line = next(line.strip().removesuffix(',') for line in native_prompt_file.read_text().splitlines()
                       if 'notification card' in line and '0-1000 grid' in line)
    assert json.loads(native_line) == SYSTEM_PROMPT
    trace_path = reference_run / 'trials/w16000000000/legacy_cpp/closing_window-8400000_live.jsonl'
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    begin = next(event for event in events if event['record'] == 'decision_begin')
    obs = next(event for event in events if event['record'] == 'observation'
               and event['observation_id'] == begin['observation_id'])
    invocation = next(event for event in events if event['record'] == 'decision_invocation'
                      and event['decision_id'] == begin['decision_id'])
    payload_file = reference_run / 'payloads' / (obs['payload_hash'] + '.bin')
    raw = payload_file.read_bytes()
    source = np.frombuffer(raw, dtype=np.uint8).reshape(240, 320, 3)
    native_rgb = source.repeat(2, axis=0).repeat(2, axis=1)
    image = Image.open(io.BytesIO(encode_png(raw, 320, 240, 2))).convert('RGB')
    host_rgb = np.array(image)
    assert image.size == (640, 480) and np.array_equal(host_rgb, native_rgb)

    processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer = processor.tokenizer
    export_tokenizer = Tokenizer.from_file(str(engine_root / 'llm/tokenizer.json'))
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
        {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': USER_TEXT}]},
    ]
    host_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
    template_file = engine_root / 'llm/processed_chat_template.json'
    template = json.loads(template_file.read_text())
    native_text = (template['roles']['system']['prefix'] + SYSTEM_PROMPT.strip()
                   + template['roles']['system']['suffix'] + template['roles']['user']['prefix']
                   + template['content_types']['image']['format'] + USER_TEXT.strip()
                   + template['roles']['user']['suffix'] + template['generation_prompt'])
    assert host_text == native_text
    assert tokenizer.encode(host_text, add_special_tokens=False) == export_tokenizer.encode(
        native_text, add_special_tokens=False).ids

    mm_kwargs = REQUEST_CONTRACT['mm_processor_kwargs']
    standardized = processor.image_processor._standardize_kwargs(**mm_kwargs)
    size = standardized['size']
    assert size.shortest_edge == 65536 and size.longest_edge == 1048576
    inputs = processor(text=[host_text], images=[image], return_tensors='pt', **mm_kwargs)
    assert inputs.image_grid_thw.tolist() == [[1, 30, 40]]
    image_tokens = int((inputs.input_ids == 248056).sum())
    assert image_tokens == 300
    expanded_native_ids = export_tokenizer.encode(
        native_text.replace('<|image_pad|>', '<|image_pad|>' * image_tokens), add_special_tokens=False).ids
    assert inputs.input_ids[0].tolist() == expanded_native_ids
    assert inputs.input_ids.shape[-1] == invocation['input_tokens'] == 449
    assert list(inputs.pixel_values.shape) == [1200, 1536]

    # Mirrors native normalizeImageKernel: FP32 arithmetic followed by FP16 storage.
    native_normalized = ((native_rgb.astype(np.float32) / np.float32(255) - np.float32(.5))
                         / np.float32(.5)).astype(np.float16)
    # Mirrors temporal duplication and transposeToPatchQwenKernel's 9D indexing.
    native_patches = np.stack([native_normalized, native_normalized]).transpose(0, 3, 1, 2)
    native_patches = native_patches.reshape(1, 2, 3, 15, 2, 16, 20, 2, 16)
    native_patches = native_patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(1200, 1536)
    host_patches = inputs.pixel_values.numpy().astype(np.float16)
    assert np.array_equal(native_patches, host_patches)
    max_diff = float(np.max(np.abs(native_patches.astype(np.float32) - host_patches.astype(np.float32))))
    assert max_diff == 0
    source_files = [Path(__file__).resolve(), Path(__file__).resolve().parent / 'agents3_matched_request.py',
        native_prompt_file, reference_run / 'code/chronotope/src/planner_trt.cpp',
        reference_run / 'code/chronotope/src/vision.cu', reference_run / 'code/src/chronotope/png.py',
        template_file, engine_root / 'llm/tokenizer.json', engine_root / 'visual/preprocessor_config.json',
        checkpoint / 'tokenizer.json', checkpoint / 'chat_template.jinja', checkpoint / 'preprocessor_config.json',
        Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/cpp/kernels/preprocessKernels/imageUtilKernels.cu')),
        Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/cpp/multimodal/qwenViTRunner.cpp')),
        Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/cpp/multimodal/qwen3vlViTRunner.cpp')),
        Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM/cpp/tokenizer/tokenizer.cpp')),
        Path(sys.modules[type(processor.image_processor).__module__].__file__), trace_path, payload_file]
    return {
        'created_at': datetime.now(timezone.utc).isoformat(), 'model': REQUEST_CONTRACT['model'],
        'contract_sha256': contract_fingerprint(), 'experimental_trials': 0, 'inference_calls': 0,
        'verification_scope': 'Stored first evaluation frame; native source emulated on CPU versus actual HF processor/tokenizer; no GPU-buffer capture or model inference',
        'limits': ['Does not prove equality of native GPU buffers or output logits',
                   'Backend kernels and numerical accumulation can still differ',
                   'Only the 640x480 static-image request path is checked'],
        'versions': {'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__, 'torch': torch.__version__},
        'rgb_equal': True, 'rgb_sha256': hashlib.sha256(host_rgb.tobytes()).hexdigest(),
        'source_observation_payload_hash': obs['payload_hash'], 'size_wh': [640, 480],
        'chat_text_equal': True, 'chat_text_sha256': hashlib.sha256(host_text.encode()).hexdigest(),
        'disabled_thinking_generation_prefix': template['generation_prompt'],
        'exported_and_serving_tokenizer_ids_equal': True, 'expanded_input_ids_equal': True,
        'input_token_count': 449, 'native_recorded_input_token_count': invocation['input_tokens'],
        'input_ids_sha256_int64_le': hashlib.sha256(inputs.input_ids.numpy().astype('<i8').tobytes()).hexdigest(),
        'image_tokens': image_tokens, 'image_grid_thw': [[1, 30, 40]], 'patch_shape': [1200, 1536],
        'mm_processor_kwargs': mm_kwargs, 'processor_kwargs_override_size_verified': True,
        'patch_fp16_equal': True, 'patch_max_abs_difference': max_diff,
        'patch_sha256_fp16_le': hashlib.sha256(host_patches.astype('<f2').tobytes()).hexdigest(),
        'source_file_sha256': {str(path): digest(path) for path in source_files},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=ROOT / 'runs/same_model_qwen08_20261028/experiment')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/model_cache/qwen35_0p8b_legacy_source')
    parser.add_argument('--engine-root', type=Path, default=Path((_CHRONOTOPE_HOME + '/tensorrt-edgellm-workspace/Qwen3.5-0.8B/engines')))
    args = parser.parse_args()
    report = verify(args.reference_run.resolve(), args.checkpoint.resolve(), args.engine_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ['rgb_equal', 'chat_text_equal', 'expanded_input_ids_equal',
        'input_token_count', 'patch_fp16_equal', 'patch_max_abs_difference', 'inference_calls']}))
