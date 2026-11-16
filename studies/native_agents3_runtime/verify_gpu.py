#!/usr/bin/env python3
"""Synthetic native CUDA penalty verification; zero model calls/task trials."""
from __future__ import annotations

from os import environ as _CHRONOTOPE_ENV
from pathlib import Path as _CHRONOTOPE_PATH
_CHRONOTOPE_HOME = _CHRONOTOPE_ENV.get('CHRONOTOPE_HOME', str(_CHRONOTOPE_PATH.home()))

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from vllm import _custom_ops


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executable", type=Path, help="Use an already compiled test executable")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    source = Path(__file__).with_name("verify_gpu.cu").resolve()
    archive = runtime / "build/cpp/libedgellmCore.a"
    trtlib = (_CHRONOTOPE_HOME + '/trt-prefix/usr/lib/aarch64-linux-gnu')
    cudalib = "/usr/local/cuda/targets/sbsa-linux/lib"
    with tempfile.TemporaryDirectory(prefix="chronotope-native-gpu-unit-") as directory:
        directory = Path(directory)
        executable = args.executable.resolve() if args.executable else directory / "test"
        compile_command = ["/usr/local/cuda/bin/nvcc", "-std=c++17", "-O2", "-rdc=true",
            "--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]", "-Wno-deprecated-declarations",
            "-I" + str(runtime / "cpp"), "-I" + str(runtime / "3rdParty/nlohmannJson/include"),
            ('-I' + _CHRONOTOPE_HOME + '/trt-prefix/usr/include/aarch64-linux-gnu'), str(source), str(archive),
            str(runtime / "cpp/kernels/cuteDSLArtifact/aarch64/sm_121/libcutedsl_aarch64.a"),
            "-L" + trtlib, "-L" + cudalib, "-L" + cudalib + "/stubs", "-lnvonnxparser", "-lnvinfer",
            "-lcuda", "-ldl", "-lrt", "-lpthread", "-Xlinker=--unresolved-symbols=ignore-in-shared-libs",
            "-o", str(executable)]
        if not args.executable:
            subprocess.run(compile_command, check=True)
        output = directory / "actual.json"
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = trtlib + ":" + cudalib + ":" + environment.get("LD_LIBRARY_PATH", "")
        subprocess.run([str(executable), str(output)], env=environment, check=True)
        rows = json.loads(output.read_text())
        bit_exact = True
        max_absolute_error = 0.0
        for row in rows:
            original = np.asarray(row["input_logits"], dtype=np.float32)
            actual = np.asarray(row["actual_logits"], dtype=np.float32)
            logits = torch.tensor(original, device="cuda", dtype=torch.float32).view(1, -1)
            prompt_mask = torch.zeros_like(logits, dtype=torch.bool)
            output_mask = torch.zeros_like(logits, dtype=torch.bool)
            for token in row["prompt"]:
                prompt_mask[0, token] = True
            for token in row["generated"]:
                output_mask[0, token] = True
            penalty = torch.tensor([row["penalty"]], dtype=torch.float32, device="cuda")
            _custom_ops.apply_repetition_penalties_cuda(logits, prompt_mask, output_mask, penalty)
            expected = logits.cpu().numpy()[0]
            # Require exact float bits from the real installed vLLM CUDA op,
            # including negative zero. This does not assert model parity.
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32),
                err_msg=f"native/vLLM CUDA mismatch: request={row['request']} step={row['step']}")
            max_absolute_error = max(max_absolute_error, float(np.max(np.abs(actual - expected))))
        assert len(rows) == 18 and {row["request"] for row in rows} == {0, 1, 2}
        report = {"passed": True, "inference_calls": 0, "evaluation_trials": 0,
            "test_type": "synthetic CUDA processor unit test, no model or task",
            "cases": len(rows), "vllm_cuda_bit_exact": bit_exact, "max_absolute_error": max_absolute_error,
            "checks": ["prompt mask before first sampling", "incremental generated-token mask",
                "duplicate IDs penalized once", "positive, negative, positive-zero and negative-zero logits",
                "257-token boundary crosses CUDA blocks", "request-local mask reset", "penalty1 bypass",
                "actual applyRepetitionPenalty runtime entrypoint", "actual installed vLLM CUDA custom op"],
            "model_numerical_equality_claimed": False,
            "compile_command": compile_command,
            "source_hashes": {str(path): sha(path) for path in [source, Path(__file__),
                runtime / "cpp/runtime/decoding/repetitionPenalty.cu",
                runtime / "cpp/runtime/decoding/repetitionPenaltyMath.h", archive,
                Path(_custom_ops.__file__)]},
            "device": torch.cuda.get_device_name(0)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "cases": len(rows), "vllm_cuda_bit_exact": True, "report": str(args.out)}))


if __name__ == "__main__":
    main()
