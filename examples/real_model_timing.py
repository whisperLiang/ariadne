from __future__ import annotations

import argparse
import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split


@dataclass(frozen=True)
class TimingRow:
    model: str
    metric: str
    average_ms: float
    total_ms: float
    iterations: int
    split_id: str
    trace_nodes: int


class RFDETRTensorWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from rfdetr import RFDETRNano

        self.model = RFDETRNano(pretrain_weights=None).model.model.eval()

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        from rfdetr.utilities.tensors import NestedTensor

        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        output = self.model(NestedTensor(x, mask))
        if not isinstance(output, dict):
            raise TypeError("Expected RF-DETR model to return a dict.")
        return output


def benchmark_yolo(*, iterations: int, warmup: int) -> list[TimingRow]:
    from ultralytics import YOLO

    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    model = yolo.model.eval()
    x = torch.randn(1, 3, 64, 64)
    return _benchmark_model(
        name="YOLOv8n",
        model=model,
        x=x,
        split=SplitSpec(boundary="after:model.2", dynamic_batch=(1, 1)),
        iterations=iterations,
        warmup=warmup,
    )


def benchmark_rfdetr(*, iterations: int, warmup: int) -> list[TimingRow]:
    model = RFDETRTensorWrapper().eval()
    x = torch.randn(1, 3, 128, 128)
    return _benchmark_model(
        name="RF-DETR Nano",
        model=model,
        x=x,
        split=SplitSpec(
            boundary="after:model.transformer.decoder.layers.0.norm3",
            dynamic_batch=(1, 1),
        ),
        iterations=iterations,
        warmup=warmup,
    )


def _benchmark_model(
    *,
    name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    split: SplitSpec,
    iterations: int,
    warmup: int,
) -> list[TimingRow]:
    prepare_start = perf_counter()
    runtime = prepare_split(model, example_inputs=(x,), split=split)
    prepare_total = perf_counter() - prepare_start
    boundary = runtime.run_prefix(x)

    rows = [
        TimingRow(
            model=name,
            metric="prepare_split",
            average_ms=prepare_total * 1000.0,
            total_ms=prepare_total * 1000.0,
            iterations=1,
            split_id=runtime.split_id,
            trace_nodes=len(runtime.trace_plan.nodes),
        )
    ]
    rows.append(
        _measure(
            name,
            "direct_forward",
            lambda: model(x),
            runtime.split_id,
            len(runtime.trace_plan.nodes),
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure(
            name,
            "run_prefix",
            lambda: runtime.run_prefix(x),
            runtime.split_id,
            len(runtime.trace_plan.nodes),
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure(
            name,
            "run_suffix",
            lambda: runtime.run_suffix(boundary),
            runtime.split_id,
            len(runtime.trace_plan.nodes),
            iterations=iterations,
            warmup=warmup,
        )
    )
    rows.append(
        _measure(
            name,
            "prefix_plus_suffix",
            lambda: runtime.run_suffix(runtime.run_prefix(x)),
            runtime.split_id,
            len(runtime.trace_plan.nodes),
            iterations=iterations,
            warmup=warmup,
        )
    )
    return rows


def _measure(
    model_name: str,
    metric: str,
    fn: Any,
    split_id: str,
    trace_nodes: int,
    *,
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
    print("| model | metric | avg_ms | total_ms | iters | split_id | trace_nodes |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for row in rows:
        print(
            f"| {row.model} | {row.metric} | {row.average_ms:.3f} | "
            f"{row.total_ms:.3f} | {row.iterations} | {row.split_id} | {row.trace_nodes} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = [
        *benchmark_yolo(iterations=args.iterations, warmup=args.warmup),
        *benchmark_rfdetr(iterations=args.iterations, warmup=args.warmup),
    ]
    _print_rows(rows)


if __name__ == "__main__":
    main()
