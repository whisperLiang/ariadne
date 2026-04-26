from __future__ import annotations

import argparse
import contextlib
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ariadne import SplitSpec, prepare_split
from examples.real_model_functional_test import RFDETRTensorWrapper, nested_tensor_loss


@dataclass(frozen=True)
class TimingRow:
    model: str
    metric: str
    average_ms: float
    total_ms: float
    iterations: int
    trace_batch: int
    benchmark_batch: int
    split_id: str
    trace_nodes: int


@dataclass(frozen=True)
class RealModelSpec:
    name: str
    input_shape: tuple[int, ...]
    boundary: str
    builder: Callable[[], nn.Module]
    trace_batch: int = 2
    benchmark_batch: int = 3


def benchmark_yolo(*, iterations: int, warmup: int) -> list[TimingRow]:
    from ultralytics import YOLO

    def build() -> nn.Module:
        weights_dir = Path(".ariadne_models")
        weights_dir.mkdir(exist_ok=True)
        with _pushd(weights_dir):
            yolo = YOLO("yolov8n.pt")
        return yolo.model.eval()

    return _benchmark_spec(
        RealModelSpec(
            name="YOLOv8n",
            input_shape=(3, 64, 64),
            boundary="after:model.2",
            builder=build,
        ),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_timm_resnet50(*, iterations: int, warmup: int) -> list[TimingRow]:
    import timm

    return _benchmark_spec(
        RealModelSpec(
            name="timm resnet50",
            input_shape=(3, 96, 96),
            boundary="after:layer3",
            builder=lambda: timm.create_model("resnet50", pretrained=False).eval(),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_torchvision_mobilenet_v3_large(*, iterations: int, warmup: int) -> list[TimingRow]:
    from torchvision.models import mobilenet_v3_large

    return _benchmark_spec(
        RealModelSpec(
            name="torchvision mobilenet_v3_large",
            input_shape=(3, 96, 96),
            boundary="after:features.10",
            builder=lambda: mobilenet_v3_large(weights=None).eval(),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_timm_swin_tiny(*, iterations: int, warmup: int) -> list[TimingRow]:
    import timm

    return _benchmark_spec(
        RealModelSpec(
            name="timm swin_tiny_patch4_window7_224",
            input_shape=(3, 224, 224),
            boundary="after:layers.1",
            builder=lambda: timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=False,
            ).eval(),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_deeplabv3(*, iterations: int, warmup: int) -> list[TimingRow]:
    from torchvision.models.segmentation import deeplabv3_resnet50

    return _benchmark_spec(
        RealModelSpec(
            name="torchvision deeplabv3_resnet50",
            input_shape=(3, 96, 96),
            boundary="after:backbone.layer3",
            builder=lambda: deeplabv3_resnet50(weights=None, weights_backbone=None).eval(),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_rfdetr(*, iterations: int, warmup: int) -> list[TimingRow]:
    return _benchmark_spec(
        RealModelSpec(
            name="RF-DETR Nano",
            input_shape=(3, 128, 128),
            boundary="after:model.transformer.decoder.layers.0.norm3",
            builder=lambda: RFDETRTensorWrapper().eval(),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def _benchmark_spec(
    spec: RealModelSpec,
    *,
    iterations: int,
    warmup: int,
) -> list[TimingRow]:
    model = spec.builder()
    trace_inputs = _make_inputs(spec.trace_batch, spec.input_shape)
    benchmark_inputs = _make_inputs(spec.benchmark_batch, spec.input_shape)

    prepare_start = perf_counter()
    runtime = prepare_split(
        model,
        example_inputs=(trace_inputs,),
        split=SplitSpec(
            boundary=spec.boundary,
            dynamic_batch=(spec.trace_batch, spec.benchmark_batch),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )
    prepare_total = perf_counter() - prepare_start
    boundary = runtime.run_prefix(benchmark_inputs)
    train_boundary = runtime.run_prefix(benchmark_inputs.detach().clone().requires_grad_(True))
    _, boundary_grads = runtime.train_suffix(
        train_boundary,
        None,
        loss_fn=lambda outputs, _targets: nested_tensor_loss(outputs),
    )
    runtime.trace_plan.root_module.zero_grad(set_to_none=True)
    rows = [
        TimingRow(
            model=spec.name,
            metric="prepare_split",
            average_ms=prepare_total * 1000.0,
            total_ms=prepare_total * 1000.0,
            iterations=1,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.trace_batch,
            split_id=runtime.split_id,
            trace_nodes=len(runtime.trace_plan.nodes),
        )
    ]
    trace_nodes = len(runtime.trace_plan.nodes)
    split_id = runtime.split_id
    rows.append(
        _measure_eval(
            spec.name,
            "direct_forward",
            lambda: model(benchmark_inputs),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_eval(
            spec.name,
            "run_prefix",
            lambda: runtime.run_prefix(benchmark_inputs),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_eval(
            spec.name,
            "run_suffix",
            lambda: runtime.run_suffix(boundary),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_eval(
            spec.name,
            "prefix_plus_suffix",
            lambda: runtime.run_suffix(runtime.run_prefix(benchmark_inputs)),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_train(
            spec.name,
            "direct_train",
            lambda: _run_direct_train_step(model, benchmark_inputs),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_train(
            spec.name,
            "train_suffix",
            lambda: runtime.train_suffix(
                train_boundary,
                None,
                loss_fn=lambda outputs, _targets: nested_tensor_loss(outputs),
            ),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_train(
            spec.name,
            "backward_prefix",
            lambda: _run_split_backward_step(runtime, benchmark_inputs, boundary_grads),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure_train(
            spec.name,
            "split_train_roundtrip",
            lambda: _run_split_train_step(runtime, benchmark_inputs),
            split_id,
            trace_nodes,
            trace_batch=spec.trace_batch,
            benchmark_batch=spec.benchmark_batch,
            iterations=iterations,
            warmup=warmup,
        )
    )
    return rows


def _run_direct_train_step(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    direct_inputs = inputs.detach().clone().requires_grad_(True)
    loss = nested_tensor_loss(model(direct_inputs))
    loss.backward()
    return loss.detach()


def _run_split_train_step(runtime: Any, inputs: torch.Tensor) -> torch.Tensor:
    runtime.trace_plan.root_module.zero_grad(set_to_none=True)
    split_inputs = inputs.detach().clone().requires_grad_(True)
    boundary = runtime.run_prefix(split_inputs)
    loss, boundary_grads = runtime.train_suffix(
        boundary,
        None,
        loss_fn=lambda outputs, _targets: nested_tensor_loss(outputs),
    )
    runtime.backward_prefix(split_inputs, boundary_grads=boundary_grads)
    return loss.detach()


def _run_split_backward_step(
    runtime: Any,
    inputs: torch.Tensor,
    boundary_grads: dict[str, torch.Tensor | None],
) -> None:
    runtime.trace_plan.root_module.zero_grad(set_to_none=True)
    split_inputs = inputs.detach().clone().requires_grad_(True)
    runtime.backward_prefix(split_inputs, boundary_grads=boundary_grads)


def _make_inputs(batch_size: int, input_shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(batch_size, *input_shape)


def _measure_eval(
    model_name: str,
    metric: str,
    fn: Callable[[], Any],
    split_id: str,
    trace_nodes: int,
    *,
    trace_batch: int,
    benchmark_batch: int,
    iterations: int,
    warmup: int,
) -> TimingRow:
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    _sync_cuda()
    start = perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            fn()
    _sync_cuda()
    total = perf_counter() - start
    return TimingRow(
        model=model_name,
        metric=metric,
        average_ms=total * 1000.0 / iterations,
        total_ms=total * 1000.0,
        iterations=iterations,
        trace_batch=trace_batch,
        benchmark_batch=benchmark_batch,
        split_id=split_id,
        trace_nodes=trace_nodes,
    )


def _measure_train(
    model_name: str,
    metric: str,
    fn: Callable[[], Any],
    split_id: str,
    trace_nodes: int,
    *,
    trace_batch: int,
    benchmark_batch: int,
    iterations: int,
    warmup: int,
) -> TimingRow:
    for _ in range(warmup):
        fn()
    _sync_cuda()
    start = perf_counter()
    for _ in range(iterations):
        fn()
    _sync_cuda()
    total = perf_counter() - start
    return TimingRow(
        model=model_name,
        metric=metric,
        average_ms=total * 1000.0 / iterations,
        total_ms=total * 1000.0,
        iterations=iterations,
        trace_batch=trace_batch,
        benchmark_batch=benchmark_batch,
        split_id=split_id,
        trace_nodes=trace_nodes,
    )


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def _pushd(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _print_rows(rows: list[TimingRow]) -> None:
    print(
        "| model | metric | avg_ms | total_ms | iters | trace_batch | "
        "bench_batch | split_id | trace_nodes |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in rows:
        print(
            f"| {row.model} | {row.metric} | {row.average_ms:.3f} | "
            f"{row.total_ms:.3f} | {row.iterations} | {row.trace_batch} | "
            f"{row.benchmark_batch} | {row.split_id} | {row.trace_nodes} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = [
        *benchmark_yolo(iterations=args.iterations, warmup=args.warmup),
        *benchmark_timm_resnet50(iterations=args.iterations, warmup=args.warmup),
        *benchmark_torchvision_mobilenet_v3_large(iterations=args.iterations, warmup=args.warmup),
        *benchmark_timm_swin_tiny(iterations=args.iterations, warmup=args.warmup),
        *benchmark_deeplabv3(iterations=args.iterations, warmup=args.warmup),
        *benchmark_rfdetr(iterations=args.iterations, warmup=args.warmup),
    ]
    _print_rows(rows)


if __name__ == "__main__":
    main()
