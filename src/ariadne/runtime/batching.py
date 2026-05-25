"""Shared batch-size and input-shape helpers for split runtimes."""

from __future__ import annotations

from typing import Any

import torch

from ariadne.trace.tensor_meta import ShapeExpr
from ariadne.trace.trace_plan import TracePlan


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def first_batch_size(values: tuple[Any, ...]) -> int | None:
    for value in values:
        tensor = first_tensor(value)
        if tensor is not None and tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def batch_size_from_inputs(inputs: tuple[Any, ...]) -> int:
    batch_size = first_batch_size(inputs)
    if batch_size is None:
        raise ValueError("Ariadne requires at least one batched tensor input.")
    return batch_size


def resize_batch(value: Any, traced_batch: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == traced_batch:
            return resize_tensor_batch(value, batch_size)
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(resize_batch(item, traced_batch, batch_size) for item in value)
    if isinstance(value, list):
        return [resize_batch(item, traced_batch, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: resize_batch(item, traced_batch, batch_size) for key, item in value.items()}
    return value


def resize_tensor_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if int(tensor.shape[0]) >= batch_size:
        resized = tensor[:batch_size].detach().clone()
    else:
        repeats = [1 for _ in tensor.shape]
        repeats[0] = (batch_size + int(tensor.shape[0]) - 1) // int(tensor.shape[0])
        resized = tensor.repeat(*repeats)[:batch_size].detach().clone()
    if tensor.requires_grad and (resized.is_floating_point() or resized.is_complex()):
        resized.requires_grad_(True)
    return resized


def validate_inputs(plan: TracePlan, inputs: tuple[Any, ...]) -> None:
    batch_size = batch_size_from_inputs(inputs)
    plan.shape_env.validate_batch(batch_size)
    for index, meta in enumerate(plan.input_metas):
        if meta is None or index >= len(inputs) or not isinstance(inputs[index], torch.Tensor):
            continue
        tensor = inputs[index]
        if tensor.ndim != len(meta.symbolic_shape):
            raise ValueError(
                f"Input {index} rank {tensor.ndim} does not match traced rank "
                f"{len(meta.symbolic_shape)}."
            )
        for dim_index, (actual, expected) in enumerate(
            zip(tensor.shape, meta.symbolic_shape, strict=True)
        ):
            if expected == plan.shape_env.batch_symbol:
                continue
            if isinstance(expected, ShapeExpr):
                expected_int = expected.materialize({plan.shape_env.batch_symbol: batch_size})
                if int(actual) != expected_int:
                    raise ValueError(
                        f"Input {index} dimension {dim_index} is {int(actual)}; "
                        f"expected {expected_int} from {expected}."
                    )
                continue
            if isinstance(expected, int) and int(actual) != expected:
                raise ValueError(
                    f"Input {index} dimension {dim_index} is {int(actual)}; "
                    f"expected {expected}."
                )
