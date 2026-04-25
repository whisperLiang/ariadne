from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split


def assert_nested_close(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f"{path}: length mismatch {len(left)} != {len(right)}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            assert_nested_close(left_item, right_item, path=f"{path}.{index}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: key mismatch {left.keys()} != {right.keys()}")
        for key in left:
            assert_nested_close(left[key], right[key], path=f"{path}.{key}")
        return
    if left != right:
        raise AssertionError(f"{path}: value mismatch")


def run_yolo_smoke() -> None:
    from ultralytics import YOLO

    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    model = yolo.model.eval()
    x = torch.randn(1, 3, 64, 64)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(boundary="after:model.2", dynamic_batch=(1, 2)),
    )
    for batch_size in (1, 2):
        x_batch = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            split_output = runtime.run_suffix(runtime.run_prefix(x_batch))
            direct_output = model(x_batch)
        assert_nested_close(split_output, direct_output)
    print(f"YOLO smoke ok: split_id={runtime.split_id} nodes={len(runtime.trace_plan.nodes)}")


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


def run_rfdetr_smoke() -> None:
    model = RFDETRTensorWrapper().eval()
    x = torch.randn(1, 3, 128, 128)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:model.transformer.decoder.layers.0.norm3",
            dynamic_batch=(1, 2),
        ),
    )
    for batch_size in (1, 2):
        x_batch = torch.randn(batch_size, 3, 128, 128)
        with torch.no_grad():
            split_output = runtime.run_suffix(runtime.run_prefix(x_batch))
            direct_output = model(x_batch)
        assert_nested_close(split_output, direct_output)
    print(f"RF-DETR smoke ok: split_id={runtime.split_id} nodes={len(runtime.trace_plan.nodes)}")


@contextlib.contextmanager
def _pushd(path: Path) -> Any:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    torch.manual_seed(0)
    run_yolo_smoke()
    run_rfdetr_smoke()


if __name__ == "__main__":
    main()
