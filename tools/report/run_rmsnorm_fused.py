"""Build and validate L1-retained single-core RMSNorm fusion on virtual Wormhole."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=root / "results/rmsnorm/20261109-rmsnorm")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error("repeats must be 1..3")
    out = (args.output or root / "results/rmsnorm_fused" /
           datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    if (out / "metadata.json").exists():
        raise SystemExit("Choose a new output directory; previous experiment records are immutable")
    out.mkdir(parents=True, exist_ok=True)
    metal = Path.home() / "dev/tenstorrent/tt-metal-v0.79.0"
    sim = metal.parent / "sim/libttsim_wh.so"
    revision = subprocess.check_output(["git", "-C", str(metal), "rev-parse", "HEAD"], text=True).strip()
    digest = hashlib.sha256(sim.read_bytes()).hexdigest()
    if revision != "de546d3b146758714d900f11b218c8f9c805f410" or digest != "196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a":
        raise SystemExit("Expected pinned TT-Metal v0.79.0 and TTSim v1.11.1")
    (sim.parent / "soc_descriptor.yaml").write_bytes(
        (metal / "tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml").read_bytes())
    subprocess.run(["python3", str(root / "tools/report/rmsnorm_fixtures.py"), "--output", str(out)], check=True)
    manifest = read_json(out / "fixtures.json")
    old_manifest = read_json(args.baseline / "fixtures.json")
    expected = {f["key"]: f["sha256"] for f in old_manifest["fixtures"]}
    for fixture in manifest["fixtures"]:
        if fixture["sha256"] != expected[fixture["key"]]:
            raise AssertionError("Regenerated fixture differs from the frozen baseline")
    sources = list((root / "npu/rmsnorm").rglob("*.cpp")) + [
        root / "npu/rmsnorm/CMakeLists.txt", root / "npu/vector_add/common.hpp",
        root / "tools/report/build_rmsnorm.py", root / "tools/report/rmsnorm_fixtures.py", Path(__file__)]
    metadata = dict(passed=False, tt_metal_revision=revision, ttsim_sha256=digest,
                    baseline=str(args.baseline), runs=[], memory_before=Path("/proc/meminfo").read_text(),
                    source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for source in sources:
        dest = out / "sources" / source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)

    def save():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    with (out / "build.log").open("w") as log:
        subprocess.run(["python3", str(root / "tools/report/build_rmsnorm.py")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    executable = root / "build/rmsnorm/rmsnorm"
    metadata["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    env = dict(os.environ, TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal),
               TT_METAL_SIMULATOR=str(sim), TT_METAL_SLOW_DISPATCH_MODE="1")

    def execute(label, fixtures, depth, repeats, variant="fused", expected_failure=False):
        command = [str(executable), str(fixtures), str(out / (label + ".json")), str(depth), str(repeats), variant]
        started = time.perf_counter()
        with (out / (label + ".log")).open("w") as log:
            try:
                code = subprocess.run(command, cwd=metal, env=env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=1200).returncode
            except subprocess.TimeoutExpired:
                code = 124
        metadata["runs"].append(dict(label=label, command=command, exit_code=code,
                                     process_wall_seconds=time.perf_counter() - started,
                                     expected_failure=expected_failure))
        save()
        if expected_failure:
            if code != 1 or "supports W <= 4097" not in (out / (label + ".log")).read_text():
                raise AssertionError("Unexpected resource guard result")
            print("PASS resource guard", flush=True)
            return None
        if code:
            raise SystemExit("RMSNorm execution failed: " + str(out / (label + ".log")))
        data = read_json(out / (label + ".json"))
        assert data["passed"]
        print("PASS " + label + " cases=" + str(len(data["cases"])), flush=True)
        return data

    # Run both default arithmetic and the new fused path on the short pilot first.
    execute("unfused-pilot", out / "pilot-fixtures.txt", 1, 1, "unfused")
    execute("fused-pilot", out / "pilot-fixtures.txt", 1, 1)
    one = execute("fused-depth1", out / "npu-fixtures.txt", 1, args.repeats)
    two = execute("fused-depth2", out / "npu-fixtures.txt", 2, args.repeats)
    baseline = {c["key"]: c for c in read_json(args.baseline / "npu-depth1.json")["cases"]}
    comparisons = []
    for depth, result in [(1, one), (2, two)]:
        assert len(result["cases"]) == 20 and result["stages"] == 1
        for row in result["cases"]:
            previous = baseline[row["key"]]
            h, w = row["shape_hw"]
            ht, wt = (h + 31) // 32, (w + 31) // 32
            assert row["intermediate_dram_bytes"] == 0
            assert row["dram_payload_bytes_model"] == 3 * ht * wt * 4096
            assert row["cb_peak_payload_bytes_model"] == (2 * wt + 6 * depth + 1) * 4096
            assert row["padding_zero"] and row["input_fingerprint"] == previous["input_fingerprint"]
            assert row["gamma_fingerprint"] == previous["gamma_fingerprint"]
            if row["output_fingerprint"] != previous["output_fingerprint"]:
                raise AssertionError("Fused arithmetic differs from the saved 3-stage reference: " + row["key"])
            comparisons.append(dict(key=row["key"], depth=depth, bitwise_output_equal=True,
                                    unfused_dram_payload_bytes_model=previous["dram_payload_bytes_model"],
                                    fused_dram_payload_bytes_model=row["dram_payload_bytes_model"],
                                    intermediate_dram_bytes_removed=previous["intermediate_dram_bytes"],
                                    cb_payload_bytes=row["cb_peak_payload_bytes_model"]))
    too_wide = next(f for f in manifest["fixtures"] if f["shape_hw"][1] == 8193)
    guard = out / "guard-fixtures.txt"
    guard.write_text(str(out / too_wide["binary"]) + "\n")
    execute("width-resource-guard", guard, 2, 1, expected_failure=True)
    summary = dict(passed=True, configurations=40, operator_submissions=40 * args.repeats,
                   fused_pilot_submissions=2, unfused_regression_pilot_submissions=2,
                   bitwise_equal_to_3_stage_reference=True, width_resource_guard_passed=True,
                   intermediate_dram_bytes=0, max_abs_error=max(c["max_abs_error"] for c in one["cases"]),
                   max_cb_payload_bytes=max(c["cb_payload_bytes"] for c in comparisons),
                   comparisons=comparisons,
                   performance_policy="Source-derived DRAM/CB model. Process wall time includes simulator/JIT/validation; not real NPU performance.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata["passed"] = True
    save()
    print("NPU_RMSNORM_FUSED_ALL_PASS " + str(out), flush=True)


if __name__ == "__main__":
    main()
