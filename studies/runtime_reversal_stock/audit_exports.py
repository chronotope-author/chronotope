"""Inspect completed pristine exports without loading weights or executing models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import onnx

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_metadata(path):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        assert length < 100_000_000
        header = json.loads(stream.read(length))
    return {key: {'dtype': value['dtype'], 'shape': value['shape']}
            for key, value in header.items() if key != '__metadata__'}


def graph_metadata(path):
    model = onnx.load_model(str(path), load_external_data=False)
    elements = Counter()
    tensors = Counter()
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        elements[dtype] += math.prod(initializer.dims)
        tensors[dtype] += 1
    return {'onnx_sha256': sha(path), 'initializer_elements_by_dtype': dict(elements),
        'initializer_count_by_dtype': dict(tensors), 'node_count': len(model.graph.node),
        'operators': dict(Counter(node.op_type for node in model.graph.node)),
        'inputs': {value.name: {'dtype': onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type),
            'shape': [dimension.dim_param or dimension.dim_value for dimension in value.type.tensor_type.shape.dim]}
            for value in model.graph.input},
        'outputs': {value.name: {'dtype': onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)}
            for value in model.graph.output}}


def audit(out, model):
    plan = read(out / 'export_plan.json')
    exported = read(out / f'export_{model}_result.json')
    assert exported['state'] == 'exported'
    folder = Path(plan['models'][model]['output'])
    result = {'state': 'auditing', 'model': plan['models'][model]['model'],
        'revision': plan['models'][model]['revision'], 'upstream_commit': plan['upstream_commit'],
        'source_checkpoint_precision': plan['models'][model]['source_precision'],
        'llm': graph_metadata(folder / 'llm/model.onnx'),
        'visual': graph_metadata(folder / 'visual/model.onnx'),
        'embedding': tensor_metadata(folder / 'llm/embedding.safetensors'),
        'inference_calls': 0, 'application_episodes': 0, 'qualification_trials': 0,
        'script_sha256': sha(Path(__file__).resolve())}
    assert result['visual']['initializer_elements_by_dtype'].get('BFLOAT16', 0) == 0
    assert result['visual']['initializer_elements_by_dtype'].get('FLOAT16', 0) > 0
    assert all(value['dtype'] == 'F16' for value in result['embedding'].values())
    if model == 'radixark27b':
        baseline = read(ROOT / 'runs/same_model_radixark_20261028/engine_build/onnx_precision_audit.json')
        for dtype in ['FLOAT4E2M1', 'FLOAT8E4M3FN']:
            assert result['llm']['initializer_elements_by_dtype'].get(dtype) == baseline['llm']['initializer_elements_by_dtype'][dtype]
        result['quantized_initializer_counts_match_prior_export'] = True
        result['runtime_precision'] = 'NVFP4 MLP/lm_head and FP8 attention projections; FP16 unquantized LLM tensors and FP16 vision'
    else:
        assert result['llm']['initializer_elements_by_dtype'].get('FLOAT4E2M1', 0) == 0
        assert result['llm']['initializer_elements_by_dtype'].get('FLOAT8E4M3FN', 0) == 0
        result['runtime_precision'] = 'FP16 LLM and vision'
    result['state'] = 'passed'
    path = out / f'export_{model}_precision_audit.json'
    assert not path.exists(), 'Refuse overwriting a prior precision audit'
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['state', 'model', 'runtime_precision']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['qwen08','radixark27b'], required=True)
    args = parser.parse_args()
    audit(args.out.resolve(), args.model)
