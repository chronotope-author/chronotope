#!/usr/bin/env python3
"""CPU-only tests of the actual native penalty math and token-mask source.

The CUDA processor's host prompt-mask loop and one-token marking body are
compiled as CPU functions. Its shared scalar logit function is included
directly. Compare against the installed vLLM Torch reference on CPU; no model
weights are loaded and no CUDA operation, inference, or task trial is run.
"""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import ast
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from transformers import AutoConfig, GenerationConfig


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def function_body(source, name):
    start = source.index("{", source.index(name))
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start + 1 : end - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vllm-ops", type=Path, default=Path(
        (_CHRONOTOPE_HOME + '/qwen-serve/.venv/lib/python3.12/site-packages/vllm/_custom_ops.py')))
    args = parser.parse_args()
    cpp = args.runtime.resolve() / "cpp"
    kernel_path = cpp / "runtime/decoding/repetitionPenalty.cu"
    math_path = cpp / "runtime/decoding/repetitionPenaltyMath.h"
    kernel = kernel_path.read_text()
    start = kernel.index("        for (auto const token : context.rawBatchedInputIds.at(0))")
    end = kernel.index("        context.repetitionSeenTokens =", start)
    prompt_loop = kernel[start:end]
    marking = function_body(kernel, "__global__ void markRepetitionToken")
    assert "context.tokenIds[0].back()" in kernel
    assert "cudaMemcpyDeviceToHost" not in kernel
    wrapper = r'''
#include <cstdint>
#include <vector>
#include "runtime/decoding/repetitionPenaltyMath.h"
static void mark(int8_t* seen, int32_t token, int32_t vocabSize) {
''' + marking + r'''
}
extern "C" void calculate(float* logits, int32_t vocabSize,
    int32_t const* prompt, int32_t nPrompt, int32_t const* output, int32_t nOutput, float penalty) {
    struct Context {
        std::vector<std::vector<int32_t>> rawBatchedInputIds;
        std::vector<int8_t> repetitionPromptMask;
    } context;
    context.rawBatchedInputIds.push_back(std::vector<int32_t>(prompt, prompt+nPrompt));
    context.repetitionPromptMask.assign(vocabSize, 0);
''' + prompt_loop + r'''
    for (int32_t i=0; i<nOutput; ++i) mark(context.repetitionPromptMask.data(), output[i], vocabSize);
    for (int32_t i=0; i<vocabSize; ++i)
        logits[i] = trt_edgellm::rt::repetitionAdjustedLogit(logits[i], context.repetitionPromptMask[i] != 0, penalty);
}
'''
    # Execute only the installed library's standalone CPU reference function,
    # avoiding vLLM startup/import side effects and any model/GPU initialization.
    module = ast.parse(args.vllm_ops.read_text())
    reference_node = next(node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_repetition_penalties_torch")
    reference_module = ast.Module(body=[reference_node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(reference_module, str(args.vllm_ops), "exec"), namespace)
    reference = namespace[reference_node.name]
    cases = []
    with tempfile.TemporaryDirectory(prefix="chronotope-native-penalty-") as directory:
        directory = Path(directory)
        source, library = directory / "test.cpp", directory / "test.so"
        source.write_text(wrapper)
        subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I" + str(cpp),
            str(source), "-o", str(library)], check=True)
        function = ctypes.CDLL(str(library)).calculate
        float_pointer, int_pointer = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32)
        function.argtypes = [float_pointer, ctypes.c_int32, int_pointer, ctypes.c_int32,
            int_pointer, ctypes.c_int32, ctypes.c_float]
        function.restype = None
        rng = np.random.default_rng(20261028)
        worst_error = 0.0
        for penalty in (1.0, 1.05, 0.8, 2.0):
            for prompt, outputs, label in [
                ([], [], "empty"), ([0, 1, 1, 5, 7], [], "prompt duplicates"),
                ([], [1, 2, 2, 3], "generated duplicates"),
                ([1, 1, 3, 7], [1, 4, 4, 7], "prompt/output union"),
                ([31, 30, 0], [29, 30, 0], "special-token-shaped IDs")]:
                for count in range(len(outputs) + 1):
                    prompt_array = np.asarray(prompt, dtype=np.int32)
                    output_array = np.asarray(outputs[:count], dtype=np.int32)
                    logits = rng.normal(0, 10, 32).astype(np.float32)
                    logits[:6] = [0.0, -0.0, -8.0, 8.0, -0.001, 0.001]
                    actual = logits.copy()
                    function(actual.ctypes.data_as(float_pointer), 32,
                        prompt_array.ctypes.data_as(int_pointer), len(prompt_array),
                        output_array.ctypes.data_as(int_pointer), len(output_array), penalty)
                    prompt_mask, output_mask = torch.zeros((1, 32), dtype=torch.bool), torch.zeros((1, 32), dtype=torch.bool)
                    for index in prompt:
                        prompt_mask[0, index] = True
                    for index in outputs[:count]:
                        output_mask[0, index] = True
                    expected = torch.from_numpy(logits.copy()).view(1, -1)
                    reference(expected, prompt_mask, output_mask, torch.tensor([penalty], dtype=torch.float32))
                    # Torch CPU uses reciprocal-then-multiply for positive values;
                    # native CUDA uses division. Their final rounding may differ.
                    np.testing.assert_allclose(actual, expected.numpy()[0], rtol=2e-7, atol=0)
                    untouched = ~(prompt_mask | output_mask).numpy()[0]
                    np.testing.assert_array_equal(actual[untouched], logits[untouched])
                    worst_error = max(worst_error, float(np.max(np.abs(actual - expected.numpy()[0]))))
                    cases.append({"penalty": penalty, "case": label, "generated_prefix_length": count})
    model_tokenizer = json.loads((args.model / "tokenizer_config.json").read_text())
    engine_config = json.loads((args.engine / "llm/config.json").read_text())
    engine_tokenizer = json.loads((args.engine / "llm/tokenizer_config.json").read_text())
    eos = model_tokenizer["eos_token"]
    assert eos == engine_tokenizer["eos_token"]
    engine_vocab = json.loads((args.engine / "llm/tokenizer.json").read_text())
    native_eos_id = next(entry["id"] for entry in engine_vocab["added_tokens"] if entry["content"] == eos)
    # vLLM's neutral *sampling* defaults still retain special-token metadata:
    # InputProcessor.try_get_generation_config() loads the file, or falls back
    # to GenerationConfig.from_model_config, then adds tokenizer EOS separately.
    try:
        generation_config = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    except OSError:
        generation_config = GenerationConfig.from_model_config(
            AutoConfig.from_pretrained(args.model, local_files_only=True))
    generation_eos = generation_config.eos_token_id
    generation_ids = generation_eos if isinstance(generation_eos, list) else [generation_eos]
    host_eos_ids = sorted({native_eos_id, *[x for x in generation_ids if x is not None]})
    engine_eos = engine_config.get("eos_token_id", [])
    assert isinstance(engine_eos, list), "Native additional EOS configuration must be an array"
    native_eos_ids = sorted({native_eos_id, *engine_eos})
    assert native_eos_ids == host_eos_ids, f"EOS mismatch: native={native_eos_ids}, host={host_eos_ids}"
    runtime_source = (cpp / "runtime/llmInferenceRuntime.cpp").read_text()
    decoder_source = (cpp / "runtime/decoding/vanillaDecoder.cpp").read_text()
    assert "applyRepetitionPenalty(mPipelineIO->outputLogits, context, context.stream);" in runtime_source
    assert "applyRepetitionPenalty(mRuntime.base.pipelineIO.outputLogits, context, context.stream);" in decoder_source
    assert "static_cast<int64_t>(promptIds.size()) + request.maxGenerateLength > kvCacheCapacity" in runtime_source
    assert "EDGELLM_CONTEXT_BUDGET_EXCEEDED: generation budget would be clipped" in runtime_source
    config = engine_config["builder_config"]
    assert config["max_input_len"] == config["max_kv_cache_capacity"] == 16384
    report = {"passed": True, "evaluation_trials": 0, "inference_calls": 0, "cuda_operations": 0,
        "comparison": "actual native shared scalar math and extracted prompt/generated mask code versus installed vLLM Torch CPU reference",
        "fixture_count": len(cases), "max_absolute_error": worst_error,
        "tolerance": {"rtol": 2e-7, "atol": 0},
        "rounding_note": "CPU reference positive values use reciprocal multiply; native uses divide. Backend bit equality is not claimed.",
        "eos": {"token": eos, "tokenizer_id": native_eos_id, "native_stop_ids": native_eos_ids,
            "host_stop_ids": host_eos_ids, "host_generation_config": "vllm"},
        "strict_budget": "reject prompt+requested_output>16384 and any runtime attempt to shorten output cap",
        "penalty_applied_before": ["prefill selection", "each vanilla decode selection"],
        "source_hashes": {str(path): digest(path) for path in [kernel_path, math_path, args.vllm_ops]},
        "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "fixtures": len(cases), "native_eos_id": native_eos_id, "report": str(args.out)}))


if __name__ == "__main__":
    main()
