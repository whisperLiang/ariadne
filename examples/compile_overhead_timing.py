from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ariadne import SplitSpec, prepare_split, prepare_split_replay

TABLE_HEADER = (
    "| model | mode | batch | baseline_prepare_ms | compiled_prepare_ms | "
    "compile_warmup_ms | baseline_avg_ms | optimized_avg_ms | speedup | "
    "break_even_iters | backend | fallback_reason |"
)
TABLE_SEPARATOR = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: tuple[int, ...]
    builder: Callable[[], nn.Module]


@dataclass(frozen=True)
class OverheadRow:
    model: str
    mode: str
    batch: int
    baseline_prepare_ms: float
    compiled_prepare_ms: float
    compile_warmup_ms: float
    baseline_avg_ms: float
    optimized_avg_ms: float
    speedup: float
    break_even_iters: float | None
    backend: str
    fallback_reason: str


def _resnet50() -> ModelSpec:
    import timm

    return ModelSpec(
        name="resnet50",
        input_shape=(3, 96, 96),
        builder=lambda: timm.create_model("resnet50", pretrained=False).eval(),
    )


def _mobilenet() -> ModelSpec:
    from torchvision.models import mobilenet_v3_large

    return ModelSpec(
        name="mobilenet_v3_large",
        input_shape=(3, 96, 96),
        builder=lambda: mobilenet_v3_large(weights=None).eval(),
    )


def _nested_tensor_loss(value: Any, _targets: Any = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return value.float().square().mean()
        return torch.tensor(0.0, device=value.device)
    if isinstance(value, (tuple, list)):
        return sum(_nested_tensor_loss(item) for item in value)
    if isinstance(value, dict):
        return sum(_nested_tensor_loss(item) for item in value.values())
    return torch.tensor(0.0)


def _run_replay(
    model: nn.Module,
    spec: ModelSpec,
    *,
    batch: int,
    trace_batch: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> OverheadRow:
    trace_inputs = torch.randn(trace_batch, *spec.input_shape, device=device)
    inputs = torch.randn(batch, *spec.input_shape, device=device)
    split = SplitSpec(
        boundary="auto",
        dynamic_batch=_dynamic_batch_range(trace_batch, batch),
        trace_batch_mode="batch_gt1",
    )
    objective = {"minimize": "boundary_bytes"}

    start = perf_counter()
    baseline = prepare_split(
        model,
        example_inputs=(trace_inputs,),
        split=split,
        objective=objective,
        mode="generated_eager",
    )
    _sync_cuda()
    baseline_prepare_ms = _elapsed_ms(start)

    start = perf_counter()
    optimized = prepare_split_replay(
        model,
        example_inputs=(trace_inputs,),
        split=split,
        objective=objective,
        mode="compiled",
    )
    _sync_cuda()
    compiled_prepare_ms = _elapsed_ms(start)

    start = perf_counter()
    optimized.warmup(inputs)
    _sync_cuda()
    compile_warmup_ms = _elapsed_ms(start)

    baseline_avg_ms = _measure(
        lambda: baseline.run_suffix(baseline.run_prefix(inputs)),
        iterations=iterations,
        warmup=warmup,
    )
    optimized_avg_ms = _measure(
        lambda: optimized.run_suffix(optimized.run_prefix(inputs)),
        iterations=iterations,
        warmup=warmup,
    )
    return _row(
        spec.name,
        "replay",
        batch,
        baseline_prepare_ms,
        compiled_prepare_ms,
        compile_warmup_ms,
        baseline_avg_ms,
        optimized_avg_ms,
        backend=optimized.active_backend,
        fallback_reason=optimized.fallback_reason,
    )


def _run_retain(
    model: nn.Module,
    spec: ModelSpec,
    *,
    batch: int,
    trace_batch: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> OverheadRow:
    trace_inputs = torch.randn(trace_batch, *spec.input_shape, device=device)
    inputs = torch.randn(batch, *spec.input_shape, device=device)
    split = SplitSpec(
        boundary="auto",
        dynamic_batch=_dynamic_batch_range(trace_batch, batch),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    objective = {
        "minimize": "boundary_bytes",
        "constraints": {"trainable_suffix": True},
    }

    start = perf_counter()
    baseline = prepare_split(
        model,
        example_inputs=(trace_inputs,),
        split=split,
        objective=objective,
        mode="generated_eager",
    )
    _sync_cuda()
    baseline_prepare_ms = _elapsed_ms(start)

    start = perf_counter()
    optimized = prepare_split(
        model,
        example_inputs=(trace_inputs,),
        split=split,
        objective=objective,
        mode="compiled",
    )
    _sync_cuda()
    compiled_prepare_ms = _elapsed_ms(start)

    start = perf_counter()
    _split_train_roundtrip(optimized, inputs)
    _sync_cuda()
    compile_warmup_ms = _elapsed_ms(start)

    baseline_avg_ms = _measure(
        lambda: _split_train_roundtrip(baseline, inputs),
        iterations=iterations,
        warmup=warmup,
    )
    optimized_avg_ms = _measure(
        lambda: _split_train_roundtrip(optimized, inputs),
        iterations=iterations,
        warmup=warmup,
    )
    return _row(
        spec.name,
        "retain",
        batch,
        baseline_prepare_ms,
        compiled_prepare_ms,
        compile_warmup_ms,
        baseline_avg_ms,
        optimized_avg_ms,
        backend="inductor",
        fallback_reason=None,
    )


def _split_train_roundtrip(runtime: Any, inputs: torch.Tensor) -> torch.Tensor:
    runtime.trace_plan.root_module.zero_grad(set_to_none=True)
    grad_inputs = inputs.detach().clone().requires_grad_(True)
    boundary = runtime.run_training_prefix(grad_inputs)
    loss, boundary_grads = runtime.train_suffix(
        boundary,
        None,
        loss_fn=_nested_tensor_loss,
    )
    runtime.backward_prefix(boundary, boundary_grads=boundary_grads)
    return loss


def _measure(fn: Callable[[], Any], *, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync_cuda()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    _sync_cuda()
    return _elapsed_ms(start) / iterations


def _row(
    model: str,
    mode: str,
    batch: int,
    baseline_prepare_ms: float,
    compiled_prepare_ms: float,
    compile_warmup_ms: float,
    baseline_avg_ms: float,
    optimized_avg_ms: float,
    *,
    backend: str,
    fallback_reason: str | None,
) -> OverheadRow:
    saved_ms = baseline_avg_ms - optimized_avg_ms
    break_even_iters = None
    if saved_ms > 0:
        break_even_iters = (
            compiled_prepare_ms + compile_warmup_ms - baseline_prepare_ms
        ) / saved_ms
    return OverheadRow(
        model=model,
        mode=mode,
        batch=batch,
        baseline_prepare_ms=baseline_prepare_ms,
        compiled_prepare_ms=compiled_prepare_ms,
        compile_warmup_ms=compile_warmup_ms,
        baseline_avg_ms=baseline_avg_ms,
        optimized_avg_ms=optimized_avg_ms,
        speedup=baseline_avg_ms / optimized_avg_ms if optimized_avg_ms else 0.0,
        break_even_iters=break_even_iters,
        backend=backend,
        fallback_reason=fallback_reason or "",
    )


def _elapsed_ms(start: float) -> float:
    return (perf_counter() - start) * 1000.0


def _dynamic_batch_range(trace_batch: int, batch: int) -> tuple[int, int]:
    return (min(trace_batch, batch), max(trace_batch, batch))


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_torch_compiler() -> None:
    compiler = getattr(torch, "compiler", None)
    reset = getattr(compiler, "reset", None)
    if reset is not None:
        reset()


def _print_rows(rows: list[OverheadRow]) -> None:
    print(TABLE_HEADER)
    print(TABLE_SEPARATOR)
    for row in rows:
        break_even = "" if row.break_even_iters is None else f"{row.break_even_iters:.1f}"
        print(
            f"| {row.model} | {row.mode} | {row.batch} | "
            f"{row.baseline_prepare_ms:.3f} | {row.compiled_prepare_ms:.3f} | "
            f"{row.compile_warmup_ms:.3f} | {row.baseline_avg_ms:.3f} | "
            f"{row.optimized_avg_ms:.3f} | {row.speedup:.3f} | "
            f"{break_even} | {row.backend} | {row.fallback_reason} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure torch.compile overhead.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet"],
        choices=["resnet50", "mobilenet", "all"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["replay", "retain"],
        choices=["replay", "retain"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 32, 256])
    parser.add_argument("--trace-batch", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark but is not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    specs = {"resnet50": _resnet50, "mobilenet": _mobilenet}
    selected = list(specs) if "all" in args.models else args.models
    rows: list[OverheadRow] = []
    for model_name in selected:
        spec = specs[model_name]()
        for mode in args.modes:
            for batch in args.batches:
                _reset_torch_compiler()
                model = spec.builder().to(device)
                if mode == "replay":
                    rows.append(
                        _run_replay(
                            model,
                            spec,
                            batch=batch,
                            trace_batch=args.trace_batch,
                            iterations=args.iterations,
                            warmup=args.warmup,
                            device=device,
                        )
                    )
                    continue
                rows.append(
                    _run_retain(
                        model,
                        spec,
                        batch=batch,
                        trace_batch=args.trace_batch,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        device=device,
                    )
                )
    _print_rows(rows)


if __name__ == "__main__":
    main()
