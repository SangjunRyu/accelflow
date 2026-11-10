"""Check fused outputs against the saved baseline, CUDA outputs and PyTorch references."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import torch
import torch.nn.functional as functional


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=root / "results/rmsnorm/20261109-rmsnorm")
    parser.add_argument("--depth", type=int, action="append", choices=[1, 2])
    args = parser.parse_args()
    depths = sorted(set(args.depth or [1, 2]))
    out, baseline = args.results.resolve(), args.baseline.resolve()
    summary = read_json(out / "summary.json") if depths == [1, 2] else {"passed": True}
    if not summary["passed"]:
        raise AssertionError("The fused simulator suite did not pass")
    manifest = read_json(out / "fixtures.json")
    reports = {}
    source_hashes = {}
    for name, path in [("fused_depth1", out / "fused-depth1.json"),
                       ("fused_depth2", out / "fused-depth2.json"),
                       ("unfused_depth1", baseline / "npu-depth1.json"),
                       ("unfused_depth2", baseline / "npu-depth2.json")]:
        if name.startswith("fused_depth") and int(name[-1]) not in depths:
            continue
        data = read_json(path)
        if not data["passed"]:
            raise AssertionError("Saved backend report did not pass")
        reports[name] = {case["key"]: case for case in data["cases"]}
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    cuda = read_json(baseline / "cuda-measurements.json")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows = []
    with torch.inference_mode():
        for fixture in manifest["fixtures"]:
            if fixture["performance"]:
                continue
            key = fixture["key"]
            raw = (out / fixture["binary"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != fixture["sha256"]:
                raise AssertionError("Fixture hash mismatch")
            magic, h, w, hp, wp, epsilon = struct.unpack_from("<8sIIIIf", raw)
            if magic != b"AFRM0001" or len(raw) != 28 + (hp * wp + wp) * 2:
                raise AssertionError("Invalid fixture layout")
            values = torch.frombuffer(bytearray(raw[28:]), dtype=torch.bfloat16).float()
            x = values[:hp * wp].reshape(hp, wp)[:h, :w].contiguous()
            gamma = values[hp * wp:][:w].contiguous()
            references = {
                "pytorch_fp32": functional.rms_norm(x, (w,), weight=gamma, eps=epsilon).double(),
                "pytorch_fp64": functional.rms_norm(x.double(), (w,), weight=gamma.double(), eps=epsilon),
            }
            outputs = {name: cases[key]["logical_output"] for name, cases in reports.items()}
            for variant in ["unfused", "fused"]:
                record = next(c for c in cuda if c["key"] == key and c["variant"] == variant)
                path = baseline / record["report"]
                data = read_json(path)
                if record["exit_code"] != 0 or not data["passed"]:
                    raise AssertionError("Saved CUDA run did not pass")
                outputs["cuda_" + variant] = data["logical_output"]
                source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            errors = {}
            for name, data in outputs.items():
                actual = torch.tensor(data, dtype=torch.float64).reshape(h, w)
                errors[name] = {}
                for reference_name, reference in references.items():
                    difference = (actual - reference).abs()
                    if not bool(torch.isfinite(actual).all() and
                                (difference <= 1e-4 + 1e-4 * reference.abs()).all()):
                        raise AssertionError(f"{key}: {name} versus {reference_name}")
                    errors[name][reference_name] = difference.max().item()
            original = reports["unfused_depth1"][key]
            for name in ["fused_depth" + str(depth) for depth in depths]:
                fused = reports[name][key]
                if any(fused[k] != original[k] for k in ["input_fingerprint", "gamma_fingerprint", "output_fingerprint"]):
                    raise AssertionError("Frozen input or bitwise output differs")
            rows.append(dict(key=key, passed=True, max_abs_errors=errors, bitwise_equal_to_unfused=True))
            print("FUSED_PYTORCH_PASS " + key, flush=True)
    if len(rows) != 20:
        raise AssertionError("Expected 20 shared fixtures")
    metadata = read_json(out / "metadata.json")
    for name, digest in metadata["source_sha256"].items():
        if hashlib.sha256((out / "sources" / name).read_bytes()).hexdigest() != digest:
            raise AssertionError("Captured source hash mismatch")
    result = dict(passed=True, torch_version=torch.__version__, device="cpu", threads=1,
                  command=[sys.executable, *sys.argv], cases=20, backend_outputs_compared=20*(4+len(depths)), depths=depths, complete_suite=(depths == [1, 2]),
                  atol=1e-4, rtol=1e-4, bitwise_equal_to_unfused=True,
                  fused_max_abs_error_vs_pytorch_fp32=max(r["max_abs_errors"][b]["pytorch_fp32"] for r in rows for b in ["fused_depth" + str(depth) for depth in depths]),
                  fused_max_abs_error_vs_pytorch_fp64=max(r["max_abs_errors"][b]["pytorch_fp64"] for r in rows for b in ["fused_depth" + str(depth) for depth in depths]),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_report_sha256=source_hashes, rows=rows,
                  scope="Finite frozen BF16 corpus, FP32 output, no GPU rerun or real NPU performance measurement")
    report_name = "pytorch-audit.json" if depths == [1, 2] else "pytorch-depth" + "-".join(map(str, depths)) + "-audit.json"
    (out / report_name).write_text(json.dumps(result, indent=2) + "\n")
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"FUSED_PYTORCH_PASS cases=20 outputs={20*(4+len(depths))} complete_suite={depths == [1, 2]}", flush=True)


if __name__ == "__main__":
    main()
