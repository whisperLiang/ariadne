"""Split training helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ariadne.pattern.boundary_value import (
    detach_boundary_value,
    encode_boundary_value,
    materialize_boundary_value,
)
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

    grad_roots: dict[str, torch.Tensor] = {}
    detached_items: list[Any] = []
    detached_tensors = {}
    value_schema = _runtime_value_schema(runtime)
    for value, spec in zip(boundary.values, value_schema, strict=True):
        materialized = materialize_boundary_value(value, spec, boundary.tensors)
        detached_value, item_grad_roots = detach_boundary_value(materialized, spec)
        encoded_value, value_tensors = encode_boundary_value(detached_value, spec)
        detached_items.append(encoded_value)
        detached_tensors.update(value_tensors)
        grad_roots.update(item_grad_roots)

    detached_boundary = BoundaryPayload(
        split_id=boundary.split_id,
        semantic_split_id=boundary.semantic_split_id,
        graph_signature=boundary.graph_signature,
        contract_signature=boundary.contract_signature,
        batch_size=boundary.batch_size,
        tensors=detached_tensors,
        schema=boundary.schema,
        requires_grad={label: tensor.requires_grad for label, tensor in detached_tensors.items()},
        weight_version=boundary.weight_version,
        passthrough_inputs=boundary.passthrough_inputs,
        values=tuple(detached_items),
        value_schema=value_schema,
    )
    outputs = runtime.run_suffix(detached_boundary)
    loss = _default_loss(outputs, targets) if loss_fn is None else loss_fn(outputs, targets)
    loss.backward()

    if optimizer is not None:
        optimizer.step()

    grads = {label: grad_root.grad for label, grad_root in grad_roots.items()}
    return loss.detach(), grads


def backward_prefix_from_boundary(
    runtime: Any,
    boundary: BoundaryPayload,
    boundary_grads: BoundaryGradients,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Backpropagate boundary gradients through the original prefix graph."""
    runtime.validate_boundary(boundary)
    if not boundary.supports_prefix_backward:
        raise ValueError(
            "Boundary payload was not produced by run_training_prefix(). "
            "Call run_training_prefix() and pass that BoundaryPayload to backward_prefix()."
        )
    if boundary.prefix_backward_owner_id != runtime.prefix_backward_owner_id:
        raise ValueError(
            "Boundary payload was produced by a different SplitRuntime. "
            "Call backward_prefix() on the same runtime that produced the training boundary."
        )
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    tensors: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    for label, tensor in boundary.tensors.items():
        grad = boundary_grads.get(label)
        if grad is not None:
            tensors.append(tensor)
            grads.append(grad.to(tensor.device))
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


def _runtime_value_schema(runtime: Any) -> tuple[Any, ...]:
    return tuple(
        runtime.candidate.boundary_contract_value_schema[label]
        for label in runtime.segments.boundary_order
    )
