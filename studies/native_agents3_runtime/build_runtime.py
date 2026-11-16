#!/usr/bin/env python3
"""Build an isolated native runtime extension, without inference or upstream edits.

Reuse the installed archive's unchanged CUDA objects. Recompile every C++
translation unit whose compiler dependency file names either changed ABI
header, then replace those members and append the new CUDA processor.
"""

from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Expected one source anchor in {path}: {old!r}")
    path.write_text(text.replace(old, new))


def patch_source(root):
    cpp = root / "cpp"
    request = cpp / "runtime/llmRuntimeUtils.h"
    replace_once(request,
        "    ContextCacheCommitPolicy contextCacheCommitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};",
        "    ContextCacheCommitPolicy contextCacheCommitPolicy{ContextCacheCommitPolicy::kIncludingGeneratedTokens};\n"
        "\n    // Chronotope matched Agent S3 request: prompt+output repetition penalty.\n"
        "    float repetitionPenalty{1.0f};\n"
        "    bool strictGenerationBudget{false};")
    context = cpp / "runtime/state/decodingInferenceContext.h"
    replace_once(context, "    std::unique_ptr<LayerDebugger> layerDebugger;",
        "    std::unique_ptr<LayerDebugger> layerDebugger;\n"
        "\n    float repetitionPenalty{1.0f};\n"
        "    std::unique_ptr<rt::Tensor> repetitionSeenTokens;\n"
        "    std::vector<int8_t> repetitionPromptMask;")
    context_cpp = cpp / "runtime/state/decodingInferenceContext.cpp"
    replace_once(context_cpp, "    logitBiasGpuDirty = false;",
        "    logitBiasGpuDirty = false;\n"
        "    repetitionPenalty = 1.0f;\n"
        "    repetitionSeenTokens.reset();\n"
        "    repetitionPromptMask.clear();")
    runtime = cpp / "runtime/llmInferenceRuntime.cpp"
    replace_once(runtime, '#include "runtime/decoding/decoderRegistry.h"',
        '#include "runtime/decoding/decoderRegistry.h"\n'
        '#include "runtime/decoding/repetitionPenalty.h"')
    replace_once(runtime, "        mLastInputTokenIds = context.rawBatchedInputIds;\n", "")
    replace_once(runtime,
        "    // Guard: reject inputs longer than the engine's built max input length with a clear,",
        "    // Expose exact post-template/media-expansion input IDs for the paired request audit.\n"
        "    // These IDs already reside on the CPU; no hidden-state capture or GPU readback.\n"
        "    mLastInputTokenIds = context.rawBatchedInputIds;\n\n"
        "    // Guard: reject inputs longer than the engine's built max input length with a clear,")
    replace_once(runtime, "    context.topK = forceGreedySpecDecode ? 0 : request.topK;",
        "    context.topK = forceGreedySpecDecode ? 0 : request.topK;\n"
        "    context.repetitionPenalty = request.repetitionPenalty;")
    replace_once(runtime,
        "    applyLogitBias(mLogitBias, mPipelineIO->outputLogits, context, context.stream);",
        "    applyLogitBias(mLogitBias, mPipelineIO->outputLogits, context, context.stream);\n"
        "    applyRepetitionPenalty(mPipelineIO->outputLogits, context, context.stream);")
    replace_once(runtime,
        "bool LLMInferenceRuntime::validateRequestConfig(LLMGenerationRequest const& request)\n{",
        "bool LLMInferenceRuntime::validateRequestConfig(LLMGenerationRequest const& request)\n{\n"
        "    if (!std::isfinite(request.repetitionPenalty) || request.repetitionPenalty <= 0.0f)\n"
        "    {\n        LOG_ERROR(\"repetitionPenalty must be finite and positive\");\n        return false;\n    }\n"
        "    if (request.repetitionPenalty != 1.0f && (request.requests.size() != 1\n"
        "        || mDeployment.base.reducedVocabSize > 0\n"
        "        || mDeployment.specDecodeMode() != SpecDecodeMode::kNONE))\n"
        "    {\n        LOG_ERROR(\"repetitionPenalty requires batch=1, full vocabulary, and vanilla decoding\");\n"
        "        return false;\n    }")
    replace_once(runtime,
        "    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(",
        "    if (request.strictGenerationBudget)\n"
        "    {\n        for (auto const& promptIds : context.rawBatchedInputIds)\n        {\n"
        "            if (static_cast<int64_t>(promptIds.size()) + request.maxGenerateLength > kvCacheCapacity)\n"
        "            {\n                throw std::runtime_error(\"EDGELLM_CONTEXT_BUDGET_EXCEEDED: prompt plus requested output exceeds KV capacity\");\n"
        "            }\n        }\n    }\n"
        "    int32_t const clampedMaxGenerateLength = clampMaxGenerateLengthForKVCapacity(")
    replace_once(runtime,
        "    if (clampedMaxGenerateLength != context.maxGenerateLength)\n    {",
        "    if (clampedMaxGenerateLength != context.maxGenerateLength)\n    {\n"
        "        if (request.strictGenerationBudget)\n"
        "        {\n            throw std::runtime_error(\"EDGELLM_CONTEXT_BUDGET_EXCEEDED: generation budget would be clipped\");\n        }")
    decoder = cpp / "runtime/decoding/vanillaDecoder.cpp"
    replace_once(decoder, '#include "runtime/decoding/logitBias.h"',
        '#include "runtime/decoding/logitBias.h"\n#include "runtime/decoding/repetitionPenalty.h"')
    replace_once(decoder,
        "    applyLogitBias(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, context.stream);",
        "    applyLogitBias(mRuntime.logitBias, mRuntime.base.pipelineIO.outputLogits, context, context.stream);\n"
        "    applyRepetitionPenalty(mRuntime.base.pipelineIO.outputLogits, context, context.stream);")
    for name in ("repetitionPenalty.h", "repetitionPenaltyMath.h", "repetitionPenalty.cu"):
        shutil.copy2(HERE / name, cpp / "runtime/decoding" / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path((_CHRONOTOPE_HOME + '/TensorRT-Edge-LLM')))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    source, out = args.upstream.resolve(), args.out.resolve()
    workspace = HERE.parents[1]
    if not out.is_relative_to(workspace / "runs") or out == source:
        raise ValueError("Output must be a new runtime directory under this workspace's runs/")
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}; choose a new output directory")
    out.mkdir(parents=True)
    shutil.copytree(source / "cpp", out / "cpp", symlinks=True)
    for name in ("examples", "3rdParty"):
        (out / name).symlink_to(source / name, target_is_directory=True)
    (out / "build/cpp").mkdir(parents=True)
    for name in ("generated",):
        (out / "build" / name).symlink_to(source / "build" / name, target_is_directory=True)
    for path in (source / "build").glob("libNvInfer_edgellm_plugin.so*"):
        (out / "build" / path.name).symlink_to(path.resolve())
    patch_source(out)
    objroot = source / "build/cpp/CMakeFiles/edgellmCore.dir"
    flag_text = (objroot / "flags.make").read_text()
    flags = {}
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        raw = re.search(rf"^{key} = (.*)$", flag_text, re.MULTILINE).group(1)
        flags[key] = shlex.split(raw.replace(str(source / "cpp"), str(out / "cpp")))
    compiler_flags = flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
    dependencies = []
    for dep in sorted(objroot.rglob("*.o.d")):
        text = dep.read_text()
        if "/runtime/llmRuntimeUtils.h" in text or "/runtime/state/decodingInferenceContext.h" in text:
            relative = Path(str(dep.relative_to(objroot))[:-4])
            if relative.suffix != ".cpp":
                raise ValueError(f"Unexpected non-C++ ABI dependency: {relative}")
            dependencies.append(relative)
    if len(dependencies) < 10:
        raise ValueError("Incomplete installed compiler dependency files; cannot safely rebuild ABI users")
    objects = out / "build/rebuilt_objects"
    objects.mkdir()
    def compile_one(relative):
        target = objects / (relative.name + ".o")
        command = ["/usr/bin/aarch64-linux-gnu-g++", *compiler_flags, "-c", str(out / "cpp" / relative), "-o", str(target)]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (objects / (relative.name + ".log")).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"Compile failed: {relative}\n{result.stdout[-5000:]}")
        return target
    print(f"Rebuilding {len(dependencies)} dependent C++ objects in {out}", flush=True)
    rebuilt = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compile_one, path) for path in dependencies]
        for future in as_completed(futures):
            rebuilt.append(future.result())
            if len(rebuilt) % 5 == 0:
                print(f"Compiled {len(rebuilt)}/{len(dependencies)}", flush=True)
    kernel_obj = objects / "repetitionPenalty.cu.o"
    kernel_command = ["/usr/local/cuda/bin/nvcc", "-dc", "-O3", "-std=c++17", "-Xcompiler=-fPIC",
        "--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]",
        "-I" + str(out / "cpp"), "-I" + str(source / "3rdParty/nlohmannJson/include"),
        ('-I' + _CHRONOTOPE_HOME + '/trt-prefix/usr/include/aarch64-linux-gnu'),
        "-I/usr/local/cuda/targets/sbsa-linux/include",
        str(out / "cpp/runtime/decoding/repetitionPenalty.cu"), "-o", str(kernel_obj)]
    subprocess.run(kernel_command, check=True)
    original_archive = source / "build/cpp/libedgellmCore.a"
    archive = out / "build/cpp/libedgellmCore.a"
    members = subprocess.check_output(["ar", "t", str(original_archive)], text=True).splitlines()
    for obj in rebuilt:
        if members.count(obj.name) != 1:
            raise ValueError(f"Archive member is ambiguous/missing: {obj.name}")
    shutil.copy2(original_archive, archive)
    subprocess.run(["ar", "r", str(archive), *map(str, rebuilt), str(kernel_obj)], check=True)
    subprocess.run(["ranlib", str(archive)], check=True)
    changed = ["runtime/llmRuntimeUtils.h", "runtime/state/decodingInferenceContext.h",
        "runtime/state/decodingInferenceContext.cpp", "runtime/llmInferenceRuntime.cpp",
        "runtime/decoding/vanillaDecoder.cpp", "runtime/decoding/repetitionPenalty.h",
        "runtime/decoding/repetitionPenaltyMath.h", "runtime/decoding/repetitionPenalty.cu"]
    report = {"inference_calls": 0, "evaluation_trials": 0, "upstream": str(source), "runtime": str(out),
        "upstream_core_sha256": sha(original_archive), "patched_core_sha256": sha(archive),
        "build_script_sha256": sha(__file__), "recompiled_objects": list(map(str, dependencies)),
        "kernel_compile_command": kernel_command,
        "patched_sources": {name: sha(out / "cpp" / name) for name in changed},
        "limitations": ["repetitionPenalty!=1 requires batch1/full-vocabulary/non-speculative decoding"],
        "penalty_scope": "union of entire expanded prompt token IDs and all generated token IDs; sign-aware once per distinct token"}
    (out / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "built", "runtime": str(out), "archive": str(archive), "objects_recompiled": len(rebuilt)}), flush=True)


if __name__ == "__main__":
    main()
