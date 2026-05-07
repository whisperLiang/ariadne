from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ariadne import SplitSpec
from ariadne.benchmark import BenchmarkResult, benchmark_split_retain_optimization


@dataclass(frozen=True)
class ModelSpec:
    name: str
    input_shape: tuple[int, ...]
    builder: Callable[[], nn.Module]


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
        losses = [_nested_tensor_loss(item) for item in value]
        return sum(loss for loss in losses if loss.numel() > 0)
    if isinstance(value, dict):
        losses = [_nested_tensor_loss(item) for item in value.values()]
        return sum(loss for loss in losses if loss.numel() > 0)
    return torch.tensor(0.0)


def _compile_options(backend: str) -> dict[str, object] | None:
    if backend == "aot_eager":
        return {"backend": "aot_eager", "dynamic": True}
    return None


def _run_model(
    spec: ModelSpec,
    *,
    batches: list[int],
    trace_batch: int,
    iterations: int,
    warmup: int,
    backend: str,
    device: torch.device,
) -> list[tuple[str, int, list[BenchmarkResult]]]:
    model = spec.builder().to(device)
    trace_inputs = torch.randn(trace_batch, *spec.input_shape, device=device)
    split = SplitSpec(
        boundary="auto",
        dynamic_batch=(min(batches), max(batches)),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    objective = {
        "minimize": "boundary_bytes",
        "constraints": {"trainable_suffix": True},
    }
    rows: list[tuple[str, int, list[BenchmarkResult]]] = []
    for batch_size in batches:
        inputs = torch.randn(batch_size, *spec.input_shape, device=device)
        results = benchmark_split_retain_optimization(
            model,
            example_inputs=(trace_inputs,),
            inputs=(inputs,),
            split=split,
            loss_fn=_nested_tensor_loss,
            objective=objective,
            compile_options=_compile_options(backend),
            iterations=iterations,
            warmup=warmup,
        )
        rows.append((spec.name, batch_size, results))
    return rows


def _print_rows(rows: list[tuple[str, int, list[BenchmarkResult]]]) -> None:
    print("| model | batch | metric | avg_ms | speedup_vs_baseline | backend |")
    print("|---|---:|---|---:|---:|---|")
    for model_name, batch_size, results in rows:
        by_name = {result.name: result for result in results}
        baseline = by_name["baseline_split_retain_roundtrip"].average_latency_s
        for result in results:
            speedup = baseline / result.average_latency_s if result.average_latency_s else 0.0
            print(
                f"| {model_name} | {batch_size} | {result.name} | "
                f"{result.average_latency_s * 1000.0:.3f} | {speedup:.3f} | "
                f"{result.active_backend or result.execution_mode} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark compiled split retain training.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["resnet50", "mobilenet", "all"],
    )
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 32, 256])
    parser.add_argument("--trace-batch", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backend",
        choices=["torch_compile", "aot_eager"],
        default="torch_compile",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise SystemExit("CUDA is required for this benchmark but is not available.")
    if args.device == "cuda" and not cuda_available:
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and cuda_available) else "cpu"
    )
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    specs = {"resnet50": _resnet50, "mobilenet": _mobilenet}
    selected = list(specs) if "all" in args.models else args.models

    all_rows: list[tuple[str, int, list[BenchmarkResult]]] = []
    for model_name in selected:
        all_rows.extend(
            _run_model(
                specs[model_name](),
                batches=args.batches,
                trace_batch=args.trace_batch,
                iterations=args.iterations,
                warmup=args.warmup,
                backend=args.backend,
                device=device,
            )
        )
    _print_rows(all_rows)


if __name__ == "__main__":
    main()
