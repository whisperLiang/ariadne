"""Benchmark helpers for prepared split runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from ariadne.api import prepare_split, prepare_split_replay
from ariadne.compiler.torch_compile import training_compile_options
from ariadne.pattern.split_spec import SplitSpec
from ariadne.runtime.segment_runtime import SplitRuntime


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    average_latency_s: float
    total_latency_s: float
    iterations: int
    batch_size: int
    split_id: str
    execution_mode: str
    cuda_peak_memory_bytes: int | None = None
    active_backend: str | None = None
    fallback_reason: str | None = None


def benchmark_runtime(
    model: torch.nn.Module,
    runtime: SplitRuntime,
    inputs: Sequence[Any],
    *,
    targets: Any | None = None,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    iterations: int = 20,
    warmup: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark direct, prefix+suffix, suffix-only, and optional training paths."""
    input_tuple = tuple(inputs)
    batch_size = _batch_size(input_tuple)
    boundary = runtime.run_prefix(*input_tuple)
    results = [
        _measure(
            "direct_pytorch_forward",
            lambda: model(*input_tuple),
            runtime=runtime,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=True,
        ),
        _measure(
            f"{runtime.mode}_prefix_suffix",
            lambda: runtime.run_suffix(runtime.run_prefix(*input_tuple)),
            runtime=runtime,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=True,
        ),
        _measure(
            f"{runtime.mode}_suffix_only",
            lambda: runtime.run_suffix(boundary),
            runtime=runtime,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=True,
        ),
    ]
    if targets is not None:
        results.extend(
            _benchmark_training_paths(
                runtime,
                input_tuple,
                targets,
                loss_fn=loss_fn,
                batch_size=batch_size,
                iterations=iterations,
                warmup=warmup,
            )
        )
    return results


def benchmark_split_modes(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    inputs: Sequence[Any],
    split: SplitSpec | str,
    targets: Any | None = None,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    modes: tuple[str, ...] = ("debug_interpreter", "generated_eager", "compiled"),
    compile_options: dict[str, Any] | None = None,
    iterations: int = 20,
    warmup: int = 5,
) -> list[BenchmarkResult]:
    """Prepare and benchmark several execution modes with identical inputs."""
    results: list[BenchmarkResult] = []
    for mode in modes:
        runtime = prepare_split(
            model,
            example_inputs=example_inputs,
            split=split,
            mode=mode,  # type: ignore[arg-type]
            compile_options=compile_options,
        )
        results.extend(
            benchmark_runtime(
                model,
                runtime,
                inputs,
                targets=targets,
                loss_fn=loss_fn,
                iterations=iterations,
                warmup=warmup,
            )
        )
    return results


def benchmark_replay_optimization(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    inputs: Sequence[Any],
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None = None,
    compile_options: dict[str, Any] | None = None,
    materialize_boundary: bool = True,
    iterations: int = 20,
    warmup: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark generated split replay against the optimized replay runtime."""
    input_tuple = tuple(inputs)
    batch_size = _batch_size(input_tuple)
    baseline = prepare_split(
        model,
        example_inputs=example_inputs,
        split=split,
        mode="generated_eager",
        objective=objective,
    )
    optimized = prepare_split_replay(
        model,
        example_inputs=example_inputs,
        split=split,
        mode="compiled",
        objective=objective,
        compile_options=compile_options,
        materialize_boundary=materialize_boundary,
    )
    optimized.warmup(*input_tuple)

    return [
        _measure(
            "direct_pytorch_forward",
            lambda: model(*input_tuple),
            runtime=baseline,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=True,
        ),
        _measure(
            "baseline_generated_prefix_suffix",
            lambda: baseline.run_suffix(baseline.run_prefix(*input_tuple)),
            runtime=baseline,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=True,
        ),
        _measure_generic(
            "optimized_replay",
            lambda: optimized.run_suffix(optimized.run_prefix(*input_tuple)),
            split_id=optimized.split_id,
            execution_mode=optimized.mode,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=False,
            active_backend=optimized.active_backend,
            fallback_reason=optimized.fallback_reason,
        ),
    ]


def benchmark_split_retain_optimization(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    inputs: Sequence[Any],
    split: SplitSpec | str,
    targets: Any | None = None,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    objective: Mapping[str, Any] | None = None,
    compile_options: dict[str, Any] | None = None,
    iterations: int = 20,
    warmup: int = 5,
) -> list[BenchmarkResult]:
    """Benchmark generated split retain against compiled split retain training."""
    input_tuple = tuple(inputs)
    batch_size = _batch_size(input_tuple)
    baseline = prepare_split(
        model,
        example_inputs=example_inputs,
        split=split,
        mode="generated_eager",
        objective=objective,
    )
    optimized = prepare_split(
        model,
        example_inputs=example_inputs,
        split=split,
        mode="compiled",
        objective=objective,
        compile_options=compile_options,
    )
    backend = str(training_compile_options(compile_options).get("backend", "inductor"))

    return [
        _measure_generic(
            "baseline_split_retain_roundtrip",
            lambda: _split_train_roundtrip(
                baseline,
                input_tuple,
                targets,
                loss_fn=loss_fn,
            ),
            split_id=baseline.split_id,
            execution_mode=baseline.mode,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=False,
        ),
        _measure_generic(
            "optimized_split_retain_roundtrip",
            lambda: _split_train_roundtrip(
                optimized,
                input_tuple,
                targets,
                loss_fn=loss_fn,
            ),
            split_id=optimized.split_id,
            execution_mode=optimized.mode,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=False,
            active_backend=backend,
        ),
    ]


def _benchmark_training_paths(
    runtime: SplitRuntime,
    inputs: tuple[Any, ...],
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> list[BenchmarkResult]:
    grad_inputs = _clone_inputs_for_grad(inputs)

    def suffix_train() -> None:
        runtime.trace_plan.root_module.zero_grad(set_to_none=True)
        boundary = runtime.run_training_prefix(*grad_inputs)
        runtime.train_suffix(boundary, targets, loss_fn=loss_fn)

    def prefix_backward() -> None:
        runtime.trace_plan.root_module.zero_grad(set_to_none=True)
        boundary = runtime.run_training_prefix(*grad_inputs)
        _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=loss_fn)
        runtime.backward_prefix(boundary, boundary_grads=boundary_grads)

    return [
        _measure(
            f"{runtime.mode}_suffix_training",
            suffix_train,
            runtime=runtime,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=False,
        ),
        _measure(
            f"{runtime.mode}_prefix_backward",
            prefix_backward,
            runtime=runtime,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            use_no_grad=False,
        ),
    ]


def _measure(
    name: str,
    fn: Callable[[], Any],
    *,
    runtime: SplitRuntime,
    batch_size: int,
    iterations: int,
    warmup: int,
    use_no_grad: bool,
) -> BenchmarkResult:
    return _measure_generic(
        name,
        fn,
        split_id=runtime.split_id,
        execution_mode=runtime.mode,
        batch_size=batch_size,
        iterations=iterations,
        warmup=warmup,
        use_no_grad=use_no_grad,
    )


def _measure_generic(
    name: str,
    fn: Callable[[], Any],
    *,
    split_id: str,
    execution_mode: str,
    batch_size: int,
    iterations: int,
    warmup: int,
    use_no_grad: bool,
    active_backend: str | None = None,
    fallback_reason: str | None = None,
) -> BenchmarkResult:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    for _ in range(warmup):
        _call(fn, use_no_grad=use_no_grad)
    _sync_cuda()
    _reset_cuda_memory()
    start = perf_counter()
    for _ in range(iterations):
        _call(fn, use_no_grad=use_no_grad)
    _sync_cuda()
    total = perf_counter() - start
    return BenchmarkResult(
        name=name,
        average_latency_s=total / iterations,
        total_latency_s=total,
        iterations=iterations,
        batch_size=batch_size,
        split_id=split_id,
        execution_mode=execution_mode,
        cuda_peak_memory_bytes=_cuda_peak_memory(),
        active_backend=active_backend,
        fallback_reason=fallback_reason,
    )


def _call(fn: Callable[[], Any], *, use_no_grad: bool) -> Any:
    if use_no_grad:
        with torch.no_grad():
            return fn()
    return fn()


def _batch_size(inputs: tuple[Any, ...]) -> int:
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Benchmark inputs must include a batched tensor.")


def _clone_inputs_for_grad(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
            if tensor.is_floating_point() or tensor.is_complex():
                tensor.requires_grad_(True)
            cloned.append(tensor)
        else:
            cloned.append(value)
    return tuple(cloned)


def _split_train_roundtrip(
    runtime: SplitRuntime,
    inputs: tuple[Any, ...],
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
) -> torch.Tensor:
    runtime.trace_plan.root_module.zero_grad(set_to_none=True)
    grad_inputs = _clone_inputs_for_grad(inputs)
    boundary = runtime.run_training_prefix(*grad_inputs)
    loss, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=loss_fn)
    runtime.backward_prefix(boundary, boundary_grads=boundary_grads)
    return loss


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory() -> int | None:
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())
