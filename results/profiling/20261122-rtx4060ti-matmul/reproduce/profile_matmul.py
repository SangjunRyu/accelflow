import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

P = C.c_void_p
U = C.c_uint
I = C.c_int
Z = C.c_size_t
D = C.c_uint64
PP = C.POINTER(P)
CUDA = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.5")
DLL_DIR = os.add_dll_directory(str(CUDA / "bin/x64"))


def api(lib, name, *types):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = I

    def call(*args):
        status = fn(*args)
        if status:
            raise RuntimeError(f"{name}: {status}")
    return call


def compile_source(source_path, target):
    full_source = source_path.read_bytes()
    source = full_source.decode("utf-8")
    kernels = source[source.index("// 한 thread"):source.index("static void check(")]
    os.environ["PATH"] = str(CUDA / "bin/x64") + os.pathsep + os.environ["PATH"]
    builtins = C.WinDLL(str(CUDA / "bin/x64/nvrtc-builtins64_135.dll"))
    rtc = C.WinDLL(str(CUDA / "bin/x64/nvrtc64_130_0.dll"))
    create = api(rtc, "nvrtcCreateProgram", PP, C.c_char_p, C.c_char_p, I, P, P)
    lower = api(rtc, "nvrtcGetLoweredName", P, C.c_char_p, C.POINTER(C.c_char_p))
    compile_program = api(rtc, "nvrtcCompileProgram", P, I, C.POINTER(C.c_char_p))
    program = P()
    create(C.byref(program), kernels.encode(), b"matmul-kernels.cu", 0, None, None)
    expressions = {"naive": b"&naive", "tiled": b"&tiled<16, false>"}
    for expression in expressions.values():
        api(rtc, "nvrtcAddNameExpression", P, C.c_char_p)(program, expression)
    options = (C.c_char_p * 3)(b"--gpu-architecture=sm_89", b"--std=c++17", b"--generate-line-info")
    try:
        compile_program(program, len(options), options)
    except RuntimeError:
        size = Z()
        api(rtc, "nvrtcGetProgramLogSize", P, C.POINTER(Z))(program, C.byref(size))
        result = C.create_string_buffer(size.value)
        api(rtc, "nvrtcGetProgramLog", P, P)(program, result)
        print(result.value.decode(errors="replace"))
        raise
    names = {}
    for key, expression in expressions.items():
        name = C.c_char_p()
        lower(program, expression, C.byref(name))
        names[key] = name.value.decode()
    size = Z()
    api(rtc, "nvrtcGetCUBINSize", P, C.POINTER(Z))(program, C.byref(size))
    cubin = C.create_string_buffer(size.value)
    api(rtc, "nvrtcGetCUBIN", P, P)(program, cubin)
    target.write_bytes(cubin.raw)
    target.with_suffix(".json").write_text(json.dumps({
        "names": names, "options": [v.decode() for v in options],
        "source_sha256": hashlib.sha256(full_source).hexdigest(),
        "cubin_sha256": hashlib.sha256(cubin.raw).hexdigest(),
    }, indent=2))
    api(rtc, "nvrtcDestroyProgram", PP)(C.byref(program))


def run_workload(args):
    driver = C.WinDLL(r"C:\Windows\System32\nvcuda.dll")
    def bind(name, *types):
        return api(driver, name, *types)

    init = bind("cuInit", U)
    get_device = bind("cuDeviceGet", C.POINTER(I), I)
    get_device_name = bind("cuDeviceGetName", P, I, I)
    retain = bind("cuDevicePrimaryCtxRetain", PP, I)
    release = bind("cuDevicePrimaryCtxRelease_v2", I)
    current = bind("cuCtxSetCurrent", P)
    load = bind("cuModuleLoadData", PP, P)
    get_function = bind("cuModuleGetFunction", PP, P, C.c_char_p)
    unload = bind("cuModuleUnload", P)
    stream_create = bind("cuStreamCreate", PP, U)
    stream_sync = bind("cuStreamSynchronize", P)
    stream_destroy = bind("cuStreamDestroy_v2", P)
    gpu_alloc = bind("cuMemAlloc_v2", C.POINTER(D), Z)
    gpu_free = bind("cuMemFree_v2", D)
    host_alloc = bind("cuMemAllocHost_v2", PP, Z)
    host_free = bind("cuMemFreeHost", P)
    upload = bind("cuMemcpyHtoDAsync_v2", D, P, Z, P)
    download = bind("cuMemcpyDtoHAsync_v2", P, D, Z, P)
    launch_kernel = bind("cuLaunchKernel", P, U, U, U, U, U, U, U, P, PP, PP)
    event_create = bind("cuEventCreate", PP, U)
    event_record = bind("cuEventRecord", P, P)
    event_wait = bind("cuEventSynchronize", P)
    event_elapsed = bind("cuEventElapsedTime", C.POINTER(C.c_float), P, P)
    event_destroy = bind("cuEventDestroy_v2", P)

    init(0)
    device = I()
    get_device(C.byref(device), 0)
    name = C.create_string_buffer(128)
    get_device_name(name, len(name), device)
    ctx, module, function, stream, start, end = (P() for _ in range(6))
    retain(C.byref(ctx), device)
    current(ctx)
    cubin = C.create_string_buffer(args.cubin.read_bytes())
    load(C.byref(module), cubin)
    names = json.loads(args.cubin.with_suffix(".json").read_text())["names"]
    get_function(C.byref(function), module, names[args.variant].encode())
    stream_create(C.byref(stream), 1)
    event_create(C.byref(start), 0)
    event_create(C.byref(end), 0)

    m,k,n = args.shape
    prng = np.random.default_rng(20261122)
    a = prng.integers(-8, 9, size=(m,k)).astype(np.float32)
    b = prng.integers(-8, 9, size=(k,n)).astype(np.float32)
    c = np.empty((m,n), dtype=np.float32)
    sizes = [a.nbytes, b.nbytes, c.nbytes]
    gpu = [D() for _ in sizes]
    host = [P() for _ in sizes]
    for dev, pin, size in zip(gpu, host, sizes):
        gpu_alloc(C.byref(dev), size)
        host_alloc(C.byref(pin), size)
    C.memmove(host[0], a.ctypes.data, sizes[0])
    C.memmove(host[1], b.ctypes.data, sizes[1])

    params = [*gpu, I(m), I(k), I(n)]
    if args.variant == "tiled":
        params.extend([D(), D()])
    argv = (P * len(params))(*(C.cast(C.byref(v), P) for v in params))

    def launch():
        launch_kernel(function, (n + 15)//16, (m + 15)//16, 1, 16, 16, 1, 0, stream, argv, None)

    def write_inputs():
        upload(gpu[0], host[0], sizes[0], stream)
        upload(gpu[1], host[1], sizes[1], stream)

    def read_output():
        download(host[2], gpu[2], sizes[2], stream)

    write_inputs()
    for _ in range(args.warmup):
        launch()
    stream_sync(stream)

    event_samples, pipeline_samples = [], []
    elapsed = C.c_float()
    for _ in range(args.samples):
        event_record(start, stream)
        for _ in range(args.batch):
            launch()
        event_record(end, stream)
        event_wait(end)
        event_elapsed(C.byref(elapsed), start, end)
        event_samples.append(elapsed.value / args.batch)

    for _ in range(args.pipeline):
        t0 = time.perf_counter_ns()
        write_inputs()
        launch()
        read_output()
        stream_sync(stream)
        pipeline_samples.append((time.perf_counter_ns() - t0) / 1e6)

    read_output()
    stream_sync(stream)
    C.memmove(c.ctypes.data, host[2], sizes[2])
    reference = a.astype(np.float64) @ b.astype(np.float64)
    error = float(np.max(np.abs(c-reference)))
    for obj in (start,end):
        event_destroy(obj)
    for dev,pin in zip(gpu,host):
        gpu_free(dev)
        host_free(pin)
    stream_destroy(stream)
    unload(module)
    current(None)
    release(device)

    result = {
        "device": name.value.decode(), "variant": args.variant, "shape_mkn": args.shape,
        "input_seed": 20261122, "block": [16,16], "input_dtype": "FP32 integers [-8, 8]",
        "warmup": args.warmup, "batch": args.batch, "event_kernel_ms": event_samples,
        "host_pipeline_ms": pipeline_samples, "max_abs_error": error,
        "event_kernel_median_ms": statistics.median(event_samples),
        "host_pipeline_median_ms": statistics.median(pipeline_samples) if pipeline_samples else None,
        "output_sha256": hashlib.sha256(c.tobytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}))
    if error:
        raise RuntimeError("output mismatch")


p = argparse.ArgumentParser()
p.add_argument("--cubin", type=Path, required=True)
p.add_argument("--compile", type=Path)
p.add_argument("--output", type=Path)
p.add_argument("--variant", choices=("naive", "tiled"))
p.add_argument("--shape", type=int, nargs=3, default=[640,320,640])
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--samples", type=int, default=15)
p.add_argument("--batch", type=int, default=20)
p.add_argument("--pipeline", type=int, default=15)
opts = p.parse_args()
if opts.compile:
    compile_source(opts.compile,opts.cubin)
else:
    run_workload(opts)
