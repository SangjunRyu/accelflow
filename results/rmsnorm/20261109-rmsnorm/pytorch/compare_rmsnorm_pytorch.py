"""Compare saved RMSNorm outputs with the official PyTorch CPU functional operator."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import struct
import sys

import torch
import torch.nn.functional as functional


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(actual, reference, atol=1e-4, rtol=1e-4):
    actual = actual.to(torch.float64)
    reference = reference.to(torch.float64)
    if actual.shape != reference.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} versus {reference.shape}")
    error = (actual - reference).abs()
    allowed = atol + rtol * reference.abs()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()):
        raise AssertionError("Non-finite output")
    if not bool((error <= allowed).all()):
        raise AssertionError(f"Output mismatch: max_abs_error={error.max().item()}")
    return error.max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    destination = root / "pytorch"
    destination.mkdir(exist_ok=True)
    manifest = load_json(root / "fixtures.json")
    measurements = load_json(root / "cuda-measurements.json")
    if not manifest["passed"]:
        raise AssertionError("Fixture manifest did not pass")

    npu = {}
    source_reports = ["fixtures.json", "cuda-measurements.json"]
    for depth in (1, 2):
        name = f"npu-depth{depth}.json"
        data = load_json(root / name)
        if not data["passed"]:
            raise AssertionError(f"NPU depth {depth} did not pass")
        npu[depth] = {case["key"]: case for case in data["cases"]}
        source_reports.append(name)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    with torch.inference_mode():
        for fixture in manifest["fixtures"]:
            if fixture["performance"]:
                continue
            raw = (root / fixture["binary"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != fixture["sha256"]:
                raise AssertionError("Fixture SHA-256 mismatch")
            magic, h, w, hp, wp, epsilon = struct.unpack_from("<8sIIIIf", raw)
            if magic != b"AFRM0001" or len(raw) != 28 + (hp * wp + wp) * 2:
                raise AssertionError("Invalid fixture")

            # Decode the original BF16 bit patterns, then widen exactly to FP32.
            values = torch.frombuffer(bytearray(raw[28:]), dtype=torch.bfloat16).float()
            x = values[:hp * wp].reshape(hp, wp)[:h, :w].contiguous()
            gamma = values[hp * wp:][:w].contiguous()
            reference32 = functional.rms_norm(x, (w,), weight=gamma, eps=epsilon)
            reference64 = functional.rms_norm(x.double(), (w,), weight=gamma.double(), eps=epsilon)
            if reference32.dtype != torch.float32 or reference64.dtype != torch.float64:
                raise AssertionError("Unexpected PyTorch output dtype")

            # Check the framework FP64 path against the earlier independent formula.
            python_reference = []
            for row in x.tolist():
                scale = 1 / math.sqrt(math.fsum(v * v for v in row) / w + epsilon)
                python_reference.append([v * scale * g for v, g in zip(row, gamma.tolist(), strict=True)])
            formula_error = compare(reference64, torch.tensor(python_reference, dtype=torch.float64),
                                    atol=1e-12, rtol=1e-12)
            reference_error = compare(reference32, reference64)

            saved = {}
            for depth in (1, 2):
                case = npu[depth][fixture["key"]]
                if not case["passed"]:
                    raise AssertionError("NPU fixture did not pass")
                saved[f"npu_depth{depth}"] = case["logical_output"]
            for variant in ("unfused", "fused"):
                runs = [m for m in measurements if m["key"] == fixture["key"] and m["variant"] == variant]
                if len(runs) != 1 or runs[0]["exit_code"] != 0:
                    raise AssertionError("Expected one saved correctness run per CUDA variant")
                report_name = runs[0]["report"]
                data = load_json(root / report_name)
                if not data["passed"]:
                    raise AssertionError("CUDA fixture did not pass")
                saved[f"cuda_{variant}"] = data["logical_output"]
                source_reports.append(report_name)

            errors = {}
            for backend, output in saved.items():
                actual = torch.tensor(output, dtype=torch.float64).reshape(h, w)
                errors[backend] = {
                    "vs_pytorch_fp32_max_abs_error": compare(actual, reference32),
                    "vs_pytorch_fp64_max_abs_error": compare(actual, reference64),
                }
            rows.append({"key": fixture["key"], "shape_hw": [h, w], "epsilon": epsilon,
                         "fixture_sha256": fixture["sha256"], "passed": True,
                         "pytorch_fp32_vs_fp64_max_abs_error": reference_error,
                         "pytorch_fp64_vs_python_max_abs_error": formula_error,
                         "backends": errors})
            print(f"PYTORCH_RMSNORM_PASS {fixture['key']}", flush=True)

    if len(rows) != 20:
        raise AssertionError("Expected the 20 shared NPU/CUDA fixtures")
    backends = list(rows[0]["backends"])
    summary = {
        "passed": True, "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
        "torch_cuda_version": torch.version.cuda, "device": "cpu", "threads": 1,
        "python": sys.version, "platform": platform.platform(), "command": [sys.executable, *sys.argv],
        "operator": "torch.nn.functional.rms_norm", "cases": len(rows),
        "saved_backend_outputs_compared": len(rows) * len(backends),
        "reference_dtypes": ["float32", "float64"], "input_policy": "original BF16 bits widened to FP32",
        "epsilon_policy": "explicit serialized FP32 epsilon; never framework default",
        "atol": 1e-4, "rtol": 1e-4,
        "max_abs_error_by_backend": {
            b: {key: max(row["backends"][b][key] for row in rows)
                for key in rows[0]["backends"][b]} for b in backends
        },
        "pytorch_fp32_vs_fp64_max_abs_error": max(row["pytorch_fp32_vs_fp64_max_abs_error"] for row in rows),
        "pytorch_fp64_vs_python_max_abs_error": max(row["pytorch_fp64_vs_python_max_abs_error"] for row in rows),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_report_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_reports},
        "scope": "saved finite shared correctness fixtures; no new kernel runs, GPU timings, backward, or BF16 output test",
        "rows": rows,
    }
    (destination / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (destination / "torch-config.txt").write_text(torch.__config__.show(), encoding="utf-8")
    (destination / "compare_rmsnorm_pytorch.py").write_bytes(Path(__file__).read_bytes())
    print(f"PYTORCH_RMSNORM_ALL_PASS cases={len(rows)} outputs={len(rows) * len(backends)}", flush=True)


if __name__ == "__main__":
    main()
