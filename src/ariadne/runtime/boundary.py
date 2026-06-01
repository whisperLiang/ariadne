"""Boundary payload schema and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ariadne.pattern.boundary_value import (
    BoundaryValueSpec,
    materialize_boundary_value,
    register_boundary_serializer,
    validate_boundary_value,
)
from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.trace.tensor_meta import ShapeEnv

__all__ = [
    "BoundaryPayload",
    "register_boundary_serializer",
    "validate_boundary_payload",
]


@dataclass(frozen=True)
class BoundaryPayload:
    split_id: str
    graph_signature: str
    batch_size: int
    tensors: dict[str, torch.Tensor]
    schema: dict[str, BoundaryTensorSpec]
    requires_grad: dict[str, bool]
    semantic_split_id: str | None = None
    contract_signature: str | None = None
    weight_version: int | None = None
    passthrough_inputs: dict[str, Any] = field(default_factory=dict)
    supports_prefix_backward: bool = False
    prefix_backward_owner_id: str | None = field(default=None, repr=False)
    protocol_version: int = 2
    values: tuple[Any, ...] = ()
    value_schema: tuple[BoundaryValueSpec, ...] = ()


def validate_boundary_payload(
    payload: BoundaryPayload,
    *,
    split_id: str,
    graph_signature: str,
    schema: dict[str, BoundaryTensorSpec],
    shape_env: ShapeEnv,
    value_schema: tuple[BoundaryValueSpec, ...] | None = None,
    semantic_split_id: str | None = None,
    contract_signature: str | None = None,
) -> None:
    if payload.split_id != split_id:
        raise ValueError(f"Boundary split_id {payload.split_id!r} does not match {split_id!r}.")
    if semantic_split_id is not None and payload.semantic_split_id != semantic_split_id:
        raise ValueError(
            f"Boundary semantic_split_id {payload.semantic_split_id!r} does not match "
            f"{semantic_split_id!r}."
        )
    expected_contract = contract_signature
    if expected_contract is not None:
        if payload.contract_signature != expected_contract:
            raise ValueError(
                f"Boundary contract_signature {payload.contract_signature!r} does not match "
                f"{expected_contract!r}."
            )
    elif payload.graph_signature != graph_signature:
        raise ValueError(
            f"Boundary graph_signature {payload.graph_signature!r} does not match "
            f"{graph_signature!r}."
        )
    if payload.schema != schema:
        raise ValueError("BoundaryPayload schema does not match runtime schema.")
    shape_env.validate_batch(payload.batch_size)

    if payload.protocol_version != 2:
        raise ValueError("BoundaryPayload only supports protocol_version=2.")
    expected_value_schema = payload.value_schema if value_schema is None else value_schema
    if value_schema is not None and payload.value_schema != expected_value_schema:
        raise ValueError("BoundaryPayload v2 value_schema does not match runtime schema.")
    if len(payload.values) != len(expected_value_schema):
        raise ValueError(
            f"BoundaryPayload has {len(payload.values)} value(s); "
            f"expected {len(expected_value_schema)} schema item(s)."
        )
    for value, value_spec in zip(payload.values, expected_value_schema, strict=True):
        materialized = materialize_boundary_value(
            value,
            value_spec,
            payload.tensors,
        )
        validate_boundary_value(
            materialized,
            value_spec,
            shape_env=shape_env,
            batch_size=payload.batch_size,
        )
