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


def nested_tensor_loss(value: Any) -> torch.Tensor:
    terms = list(_iter_loss_terms(value))
    if not terms:
        raise TypeError("Expected at least one floating tensor output for differentiable loss.")
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def assert_split_train_equivalent(
    model: nn.Module,
    runtime: Any,
    x: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    model.zero_grad(set_to_none=True)
    direct_inputs = x.detach().clone().requires_grad_(True)
    direct_loss = nested_tensor_loss(model(direct_inputs))
    direct_loss.backward()
    direct_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }

    model.zero_grad(set_to_none=True)
    split_inputs = x.detach().clone().requires_grad_(True)
    boundary = runtime.run_prefix(split_inputs)
    split_loss, boundary_grads = runtime.train_suffix(
        boundary,
        None,
        loss_fn=lambda outputs, _targets: nested_tensor_loss(outputs),
    )
    runtime.backward_prefix(split_inputs, boundary_grads=boundary_grads)

    torch.testing.assert_close(split_loss, direct_loss.detach(), rtol=rtol, atol=atol)
    for name, parameter in model.named_parameters():
        expected = direct_grads[name]
        actual = parameter.grad
        if expected is None and actual is None:
            continue
        if expected is None or actual is None:
            raise AssertionError(f"Gradient presence mismatch for parameter {name!r}.")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    model.zero_grad(set_to_none=True)


def run_yolo_smoke() -> None:
    from ultralytics import YOLO

    weights_dir = Path(".ariadne_models")
    weights_dir.mkdir(exist_ok=True)
    with _pushd(weights_dir):
        yolo = YOLO("yolov8n.pt")
    model = yolo.model.eval()
    x = torch.randn(2, 3, 64, 64)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:model.2",
            dynamic_batch=(2, 3),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )
    for batch_size in (2, 3):
        x_batch = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            split_output = runtime.run_suffix(runtime.run_prefix(x_batch))
            direct_output = model(x_batch)
        assert_nested_close(split_output, direct_output)
    assert_split_train_equivalent(model, runtime, torch.randn(3, 3, 64, 64))
    print(
        "YOLO batch_gt1 ok: "
        f"split_id={runtime.split_id} nodes={len(runtime.trace_plan.nodes)}"
    )


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
    x = torch.randn(2, 3, 128, 128)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:model.transformer.decoder.layers.0.norm3",
            dynamic_batch=(2, 3),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )
    for batch_size in (2, 3):
        x_batch = torch.randn(batch_size, 3, 128, 128)
        with torch.no_grad():
            split_output = runtime.run_suffix(runtime.run_prefix(x_batch))
            direct_output = model(x_batch)
        assert_nested_close(split_output, direct_output)
    assert_split_train_equivalent(model, runtime, torch.randn(3, 3, 128, 128))
    print(
        "RF-DETR batch_gt1 ok: "
        f"split_id={runtime.split_id} nodes={len(runtime.trace_plan.nodes)}"
    )


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


def _iter_loss_terms(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return [value.float().square().mean()]
        return []
    if isinstance(value, (tuple, list)):
        return [term for item in value for term in _iter_loss_terms(item)]
    if isinstance(value, dict):
        return [term for item in value.values() for term in _iter_loss_terms(item)]
    return []
