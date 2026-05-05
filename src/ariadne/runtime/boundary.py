"""Boundary payload schema and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.trace.tensor_meta import ShapeEnv


@dataclass(frozen=True)
class BoundaryPayload:
    split_id: str
    graph_signature: str
    batch_size: int
    tensors: dict[str, torch.Tensor]
    schema: dict[str, BoundaryTensorSpec]
    requires_grad: dict[str, bool]
    weight_version: int | None = None
    passthrough_inputs: dict[str, Any] = field(default_factory=dict)
    supports_prefix_backward: bool = False
    prefix_backward_owner_id: str | None = field(default=None, repr=False)


def validate_boundary_payload(
    payload: BoundaryPayload,
    *,
    split_id: str,
    graph_signature: str,
    schema: dict[str, BoundaryTensorSpec],
    shape_env: ShapeEnv,
) -> None:
    if payload.split_id != split_id:
        raise ValueError(f"Boundary split_id {payload.split_id!r} does not match {split_id!r}.")
    if payload.graph_signature != graph_signature:
        raise ValueError(
            f"Boundary graph_signature {payload.graph_signature!r} does not match "
            f"{graph_signature!r}."
        )
    shape_env.validate_batch(payload.batch_size)

    missing = [label for label in schema if label not in payload.tensors]
    if missing:
        raise ValueError(f"Boundary payload is missing labels: {', '.join(missing)}.")

    for label, tensor_spec in schema.items():
        tensor = payload.tensors[label]
        tensor_spec.validate_tensor(tensor, shape_env, payload.batch_size)
