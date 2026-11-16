"""Compare historical and pristine ONNX exports on CPU without loading weights."""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/runtime_reversal_stock_20261029'
OLD = {
    'qwen08': Path((_CHRONOTOPE_HOME + '/tensorrt-edgellm-workspace/Qwen3.5-0.8B/onnx')),
    'radixark27b': ROOT / 'runs/same_model_radixark_20261028/engine_build/onnx_preserved',
}
LAUNCH = {
    'qwen08': ROOT / 'runs/agents3_policy_qwen08_20261028/engine_build_launch.json',
    'radixark27b': ROOT / 'runs/agents3_policy_radixark27b_20261028/engine_build_launch.json',
}


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def normalize(message):
    for field in message.DESCRIPTOR.fields:
        if field.name in ('doc_string', 'metadata_props'):
            message.ClearField(field.name)
            continue
        value = getattr(message, field.name)
        if field.name == 'external_data':
            for item in value:
                if item.key == 'location':
                    item.value = '<EXTERNAL_DATA_FILE>'
        elif field.message_type:
            if field.is_repeated:
                for item in value:
                    normalize(item)
            elif message.HasField(field.name):
                normalize(value)


def graph_summary(model):
    elements, counts = Counter(), Counter()
    for initializer in model.graph.initializer:
        dtype = onnx.TensorProto.DataType.Name(initializer.data_type)
        counts[dtype] += 1
        elements[dtype] += math.prod(initializer.dims)
    cast_to = Counter()
    for node in model.graph.node:
        if node.op_type == 'Cast':
            for attr in node.attribute:
                if attr.name == 'to':
                    cast_to[onnx.TensorProto.DataType.Name(attr.i)] += 1
    return {
        'node_count': len(model.graph.node),
        'operators': dict(sorted(Counter(node.domain + ':' + node.op_type for node in model.graph.node).items())),
        'initializer_elements_by_dtype': dict(elements),
        'initializer_count_by_dtype': dict(counts),
        'cast_target_counts': dict(cast_to),
        'inputs': {v.name: onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) for v in model.graph.input},
        'outputs': {v.name: onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type) for v in model.graph.output},
    }


def main():
    dest = OUT / 'export_comparison.json'
    assert not dest.exists(), 'Preserve earlier comparison evidence'
    result = {'normalization': ['Remove non-computational doc_string and metadata_props recursively',
                                'Normalize external_data.location; preserve offsets/lengths',
                                'Preserve all nodes, names, attributes, tensor metadata and inline tensor bytes'],
              'models': {}, 'model_requests_executed': 0, 'application_episodes': 0}
    for name, prior in OLD.items():
        launch = json.loads(LAUNCH[name].read_text())['inputs']
        exported = json.loads((OUT / f'export_{name}_result.json').read_text())['output_hashes']
        paired = {}
        for component in ('llm', 'visual'):
            old_dir, new_dir = prior / component, OUT / 'exports' / name / component
            old_graph, new_graph = old_dir / 'model.onnx', new_dir / 'model.onnx'
            old, new = [onnx.load_model(str(p), load_external_data=False) for p in (old_graph, new_graph)]
            old_summary, new_summary = graph_summary(old), graph_summary(new)
            files = {}
            for file in ('model.onnx', 'model.onnx.data', 'embedding.safetensors', 'config.json'):
                left, right = old_dir / file, new_dir / file
                if not left.exists() or not right.exists():
                    continue
                left_sha, right_sha = sha(left), sha(right)
                expected_old = launch.get(str(left))
                if expected_old:
                    assert left_sha == expected_old, str(left)
                assert right_sha == exported[f'{component}/{file}'], str(right)
                files[file] = {'old_sha256': left_sha, 'stock_sha256': right_sha,
                               'equal': left_sha == right_sha,
                               'old_sha_matches_frozen_engine_build_input': left_sha == expected_old if expected_old else None}
            old_initializers = {v.name: v for v in old.graph.initializer}
            new_initializers = {v.name: v for v in new.graph.initializer}
            dtype_changes = []
            for key in old_initializers.keys() & new_initializers.keys():
                a, b = old_initializers[key], new_initializers[key]
                if a.data_type != b.data_type:
                    dtype_changes.append({'name': key, 'old': onnx.TensorProto.DataType.Name(a.data_type),
                                          'stock': onnx.TensorProto.DataType.Name(b.data_type),
                                          'elements': math.prod(a.dims), 'shapes_match': list(a.dims) == list(b.dims)})
            normalize(old)
            normalize(new)
            a, b = old.SerializeToString(deterministic=True), new.SerializeToString(deterministic=True)
            paired[component] = {
                'old_path': str(old_dir), 'stock_path': str(new_dir),
                'old': old_summary, 'stock': new_summary,
                'normalized_graph_equal': a == b,
                'old_normalized_sha256': hashlib.sha256(a).hexdigest(),
                'stock_normalized_sha256': hashlib.sha256(b).hexdigest(),
                'file_hashes': files,
                'initializer_names_added': sorted(new_initializers.keys() - old_initializers.keys()),
                'initializer_names_removed': sorted(old_initializers.keys() - new_initializers.keys()),
                'initializer_dtype_changes': sorted(dtype_changes, key=lambda v: v['name']),
            }
        result['models'][name] = paired
        print(json.dumps({name: {k: {'normalized_graph_equal': v['normalized_graph_equal'],
                                    'external_weights_equal': v['file_hashes']['model.onnx.data']['equal'],
                                    'old_nodes': v['old']['node_count'], 'stock_nodes': v['stock']['node_count'],
                                    'dtype_changes': len(v['initializer_dtype_changes'])} for k, v in paired.items()}}), flush=True)
    dest.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
