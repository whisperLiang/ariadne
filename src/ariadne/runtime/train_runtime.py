"""Split training helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ariadne.runtime.boundary import BoundaryPayload

BoundaryGradients = dict[str, torch.Tensor | None]


def train_suffix(
    runtime: Any,
    boundary: BoundaryPayload,
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.Tensor, BoundaryGradients]:
    """Train or differentiate the suffix from detached boundary tensors."""
    runtime.validate_boundary(boundary)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    detached_tensors: dict[str, torch.Tensor] = {}
    grad_roots: dict[str, torch.Tensor] = {}
    for label in runtime.segments.boundary_order:
        tensor = boundary.tensors[label].detach()
        if tensor.is_floating_point() or tensor.is_complex():
            grad_root = tensor.requires_grad_(True)
            grad_roots[label] = grad_root
            tensor = grad_root.clone()
        detached_tensors[label] = tensor

    detached_boundary = BoundaryPayload(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors=detached_tensors,
        schema=boundary.schema,
        requires_grad={label: tensor.requires_grad for label, tensor in detached_tensors.items()},
        weight_version=boundary.weight_version,
        passthrough_inputs=boundary.passthrough_inputs,
    )
    outputs = runtime.run_suffix(detached_boundary)
    loss = _default_loss(outputs, targets) if loss_fn is None else loss_fn(outputs, targets)
    loss.backward()

    if optimizer is not None:
        optimizer.step()

    grads = {
        label: grad_roots[label].grad
        for label in runtime.segments.boundary_order
        if label in grad_roots
    }
    return loss.detach(), grads


def backward_prefix(
    runtime: Any,
    inputs: tuple[Any, ...],
    boundary_grads: BoundaryGradients,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Recompute prefix boundary tensors and backpropagate cloud-side gradients."""
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    runtime._validate_inputs(inputs)
    boundary_values = _as_tuple(runtime.prefix_segment(*inputs))
    tensors: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for label, tensor in zip(runtime.segments.boundary_order, boundary_values, strict=True):
        grad = boundary_grads.get(label)
        if isinstance(tensor, torch.Tensor) and grad is not None:
            tensors.append(tensor)
            grads.append(grad)
    if tensors:
        torch.autograd.backward(tensors, grads)
    if optimizer is not None:
        optimizer.step()


def _default_loss(outputs: Any, targets: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor) and isinstance(targets, torch.Tensor):
        if targets.dtype == torch.long and outputs.ndim >= 2:
            return F.cross_entropy(outputs, targets)
        return F.mse_loss(outputs, targets)
    raise TypeError("A loss_fn is required for non-tensor outputs or targets.")


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)
