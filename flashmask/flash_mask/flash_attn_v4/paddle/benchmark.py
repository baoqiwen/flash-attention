# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

"""Useful functions for writing benchmark/test code (Paddle backend)."""

import time

import paddle


def _dtype_str(amp_dtype) -> str:
    """Convert paddle dtype to the string expected by paddle.amp.auto_cast."""
    if amp_dtype == paddle.bfloat16:
        return "bfloat16"
    return "float16"


class BenchmarkResult:
    """Minimal benchmark result that mirrors the interface of
    torch.utils.benchmark.Measurement so callers can inspect .mean/.median."""

    def __init__(self, times, stmt=""):
        self.times = times
        self.stmt = stmt
        self.mean = sum(times) / len(times)
        sorted_times = sorted(times)
        self.median = sorted_times[len(sorted_times) // 2]

    def __repr__(self):
        return (
            f"{self.stmt}\n"
            f"  mean={self.mean * 1e6:.2f} us  "
            f"median={self.median * 1e6:.2f} us  "
            f"({len(self.times)} runs)"
        )


def _timeit(fn, repeats: int) -> BenchmarkResult:
    """Run *fn* *repeats* times and return a BenchmarkResult."""
    times = []
    for _ in range(repeats):
        paddle.device.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        paddle.device.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return BenchmarkResult(times)


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=paddle.float16, **kwinputs
):
    """Benchmark the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper():
        with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
            fn(*inputs, **kwinputs)

    m = _timeit(amp_wrapper, repeats)
    if verbose:
        print(m)
    return None, m  # (timer placeholder, measurement) to match torch API shape


def benchmark_backward(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=paddle.float16,
    **kwinputs,
):
    """Benchmark the backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Backward pass")
    with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = paddle.randn(y.shape, dtype=y.dtype)
    else:
        if list(grad.shape) != list(y.shape):
            raise RuntimeError("Grad shape does not match output shape")

    def f():
        # Clear gradients to avoid accumulation overhead
        for x in inputs:
            if isinstance(x, paddle.Tensor):
                x.clear_gradient()
        y.backward(grad, retain_graph=True)

    m = _timeit(f, repeats)
    if verbose:
        print(m)
    return None, m


def benchmark_combined(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=paddle.float16,
    **kwinputs,
):
    """Benchmark the forward+backward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward + Backward pass")
    with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
        y = fn(*inputs, **kwinputs)
        if type(y) is tuple:
            y = y[0]
    if grad is None:
        grad = paddle.randn(y.shape, dtype=y.dtype)
    else:
        if list(grad.shape) != list(y.shape):
            raise RuntimeError("Grad shape does not match output shape")

    def f():
        for x in inputs:
            if isinstance(x, paddle.Tensor):
                x.clear_gradient()
        with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        out.backward(grad, retain_graph=True)

    m = _timeit(f, repeats)
    if verbose:
        print(m)
    return None, m


def benchmark_fwd_bwd(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=paddle.float16,
    **kwinputs,
):
    """Benchmark forward and backward passes separately."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def benchmark_all(
    fn,
    *inputs,
    grad=None,
    repeats=10,
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=paddle.float16,
    **kwinputs,
):
    """Benchmark forward, backward and combined passes."""
    return (
        benchmark_forward(
            fn,
            *inputs,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_backward(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
        benchmark_combined(
            fn,
            *inputs,
            grad=grad,
            repeats=repeats,
            desc=desc,
            verbose=verbose,
            amp=amp,
            amp_dtype=amp_dtype,
            **kwinputs,
        ),
    )


def pytorch_profiler(
    fn,
    *inputs,
    trace_filename=None,
    backward=False,
    amp=False,
    amp_dtype=paddle.float16,
    cpu=False,
    verbose=True,
    **kwinputs,
):
    """Profile a function using Paddle profiler to see GPU information."""
    if backward:
        with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
            g = paddle.randn(out.shape, dtype=out.dtype)

    # Warm-up
    for _ in range(30):
        if backward:
            for x in inputs:
                if isinstance(x, paddle.Tensor):
                    x.clear_gradient()
        with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)

    targets = [paddle.profiler.ProfilerTarget.GPU]
    if cpu:
        targets = [paddle.profiler.ProfilerTarget.CPU] + targets

    with paddle.profiler.Profiler(targets=targets, record_shapes=True) as prof:
        if backward:
            for x in inputs:
                if isinstance(x, paddle.Tensor):
                    x.clear_gradient()
        with paddle.amp.auto_cast(enable=amp, dtype=_dtype_str(amp_dtype)):
            out = fn(*inputs, **kwinputs)
            if type(out) is tuple:
                out = out[0]
        if backward:
            out.backward(g, retain_graph=True)

    if verbose:
        prof.summary(op_detail=True, thread_sep=False, time_unit="ms")
    if trace_filename is not None:
        prof.export(path=trace_filename, format="json")


def benchmark_memory(fn, *inputs, desc="", verbose=True, **kwinputs):
    paddle.device.cuda.empty_cache()
    paddle.device.cuda.synchronize()
    mem_before = paddle.device.cuda.memory_allocated()
    fn(*inputs, **kwinputs)
    paddle.device.cuda.synchronize()
    mem_after = paddle.device.cuda.max_memory_allocated()
    mem = (mem_after - mem_before) / ((2**20) * 1000)
    if verbose:
        print(f"{desc} max memory: {mem:.3f}GB")
    paddle.device.cuda.empty_cache()
    return mem
