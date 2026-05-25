"""Structured boundary value schemas for BoundaryPayload v2."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

import torch

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.trace.tensor_meta import ShapeEnv, ShapeExpr


@dataclass(frozen=True)
class BoundaryTensorValueSpec:
    label: str
    tensor_spec: BoundaryTensorSpec


@dataclass(frozen=True)
class BoundaryTensorRef:
    label: str


@dataclass(frozen=True)
class BoundarySequenceValueSpec:
    label: str
    container_type: str
    length_expr: int | str | ShapeExpr
    element_spec: BoundaryValueSpec


@dataclass(frozen=True)
class BoundaryTupleValueSpec:
    items: tuple[BoundaryValueSpec, ...]


@dataclass(frozen=True)
class BoundaryListValueSpec:
    items: tuple[BoundaryValueSpec, ...]


@dataclass(frozen=True)
class BoundaryDictValueSpec:
    items: tuple[tuple[Hashable, BoundaryValueSpec], ...]


@dataclass(frozen=True)
class BoundarySliceValueSpec:
    start: BoundaryValueSpec
    stop: BoundaryValueSpec
    step: BoundaryValueSpec


@dataclass(frozen=True)
class BoundaryLiteralValueSpec:
    value: Any


@dataclass(frozen=True)
class BoundaryCustomValueSpec:
    type_name: str
    schema: Any


BoundaryValueSpec = (
    BoundaryTensorValueSpec
    | BoundarySequenceValueSpec
    | BoundaryTupleValueSpec
    | BoundaryListValueSpec
    | BoundaryDictValueSpec
    | BoundarySliceValueSpec
    | BoundaryLiteralValueSpec
    | BoundaryCustomValueSpec
)


@dataclass(frozen=True)
class BoundarySerializer:
    type_: type[Any]
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    schema: Any


_SERIALIZERS: dict[type[Any], BoundarySerializer] = {}


def register_boundary_serializer(
    type_: type[Any],
    encode: Callable[[Any], Any],
    decode: Callable[[Any], Any],
    schema: Any,
) -> None:
    """Register a cross-process-safe serializer for a custom boundary value."""
    _SERIALIZERS[type_] = BoundarySerializer(type_, encode, decode, schema)


def flatten_boundary_tensors(
    value: Any,
    spec: BoundaryValueSpec,
) -> dict[str, torch.Tensor]:
    _encoded, tensors = encode_boundary_value(value, spec)
    return tensors


def encode_boundary_value(
    value: Any,
    spec: BoundaryValueSpec,
) -> tuple[Any, dict[str, torch.Tensor]]:
    tensors: dict[str, torch.Tensor] = {}
    encoded = _encode_value(value, spec, tensors)
    return encoded, tensors


def materialize_boundary_value(
    value: Any,
    spec: BoundaryValueSpec,
    tensors: dict[str, torch.Tensor],
) -> Any:
    return _materialize_value(value, spec, tensors)


def validate_boundary_value(
    value: Any,
    spec: BoundaryValueSpec,
    *,
    shape_env: ShapeEnv,
    batch_size: int,
) -> None:
    if isinstance(spec, BoundaryTensorValueSpec):
        if isinstance(value, BoundaryTensorRef):
            raise ValueError(
                f"Boundary value {spec.label!r} must be materialized before validation."
            )
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Boundary value {spec.label!r} expected Tensor, got {type(value).__name__}."
            )
        spec.tensor_spec.validate_tensor(value, shape_env, batch_size)
        return
    if isinstance(spec, BoundarySequenceValueSpec):
        if spec.container_type == "tuple" and not isinstance(value, tuple):
            raise ValueError(f"Boundary sequence {spec.label!r} expected tuple.")
        if spec.container_type == "list" and not isinstance(value, list):
            raise ValueError(f"Boundary sequence {spec.label!r} expected list.")
        expected_length = _materialize_length(spec.length_expr, shape_env, batch_size)
        if len(value) != expected_length:
            raise ValueError(
                f"Boundary sequence {spec.label!r} length {len(value)}; "
                f"expected {expected_length}."
            )
        for item in value:
            validate_boundary_value(
                item,
                spec.element_spec,
                shape_env=shape_env,
                batch_size=batch_size,
            )
        return
    if isinstance(spec, BoundaryTupleValueSpec):
        _validate_fixed_sequence(value, spec.items, tuple, shape_env, batch_size)
        return
    if isinstance(spec, BoundaryListValueSpec):
        _validate_fixed_sequence(value, spec.items, list, shape_env, batch_size)
        return
    if isinstance(spec, BoundaryDictValueSpec):
        if not isinstance(value, dict):
            raise ValueError(f"Boundary dict expected dict, got {type(value).__name__}.")
        expected_keys = tuple(key for key, _spec in spec.items)
        if tuple(value.keys()) != expected_keys:
            raise ValueError(
                f"Boundary dict keys {tuple(value.keys())!r}; expected {expected_keys!r}."
            )
        for key, item_spec in spec.items:
            validate_boundary_value(
                value[key],
                item_spec,
                shape_env=shape_env,
                batch_size=batch_size,
            )
        return
    if isinstance(spec, BoundarySliceValueSpec):
        if not isinstance(value, slice):
            raise ValueError(f"Boundary slice expected slice, got {type(value).__name__}.")
        validate_boundary_value(value.start, spec.start, shape_env=shape_env, batch_size=batch_size)
        validate_boundary_value(value.stop, spec.stop, shape_env=shape_env, batch_size=batch_size)
        validate_boundary_value(value.step, spec.step, shape_env=shape_env, batch_size=batch_size)
        return
    if isinstance(spec, BoundaryLiteralValueSpec):
        if value != spec.value:
            raise ValueError(f"Boundary literal {value!r}; expected {spec.value!r}.")
        return
    if isinstance(spec, BoundaryCustomValueSpec):
        serializer = _serializer_for_type_name(spec.type_name)
        serializer.encode(value)
        return
    raise TypeError(f"Unsupported boundary value spec {type(spec).__name__}.")


def detach_boundary_value(
    value: Any,
    spec: BoundaryValueSpec,
) -> tuple[Any, dict[str, torch.Tensor]]:
    grad_roots: dict[str, torch.Tensor] = {}
    detached = _detach_value(value, spec, grad_roots)
    return detached, grad_roots


def _encode_value(
    value: Any,
    spec: BoundaryValueSpec,
    tensors: dict[str, torch.Tensor],
) -> Any:
    if isinstance(spec, BoundaryTensorValueSpec):
        if isinstance(value, torch.Tensor):
            tensors[spec.label] = value
            return BoundaryTensorRef(spec.label)
        raise ValueError(
            f"Boundary value {spec.label!r} expected Tensor, got {type(value).__name__}."
        )
    if isinstance(spec, BoundarySequenceValueSpec):
        values = [
            _encode_value(
                item,
                _relabel_sequence_element(spec.element_spec, spec.label, index),
                tensors,
            )
            for index, item in enumerate(value)
        ]
        return tuple(values) if spec.container_type == "tuple" else values
    if isinstance(spec, (BoundaryTupleValueSpec, BoundaryListValueSpec)):
        values = [
            _encode_value(item, item_spec, tensors)
            for item, item_spec in zip(value, spec.items, strict=True)
        ]
        return tuple(values) if isinstance(spec, BoundaryTupleValueSpec) else values
    if isinstance(spec, BoundaryDictValueSpec):
        return {
            key: _encode_value(value[key], item_spec, tensors)
            for key, item_spec in spec.items
        }
    if isinstance(spec, BoundarySliceValueSpec):
        return slice(
            _encode_value(value.start, spec.start, tensors),
            _encode_value(value.stop, spec.stop, tensors),
            _encode_value(value.step, spec.step, tensors),
        )
    if isinstance(spec, BoundaryCustomValueSpec):
        serializer = _serializer_for_type_name(spec.type_name)
        return serializer.encode(value)
    return value


def _materialize_value(
    value: Any,
    spec: BoundaryValueSpec,
    tensors: dict[str, torch.Tensor],
) -> Any:
    if isinstance(spec, BoundaryTensorValueSpec):
        if not isinstance(value, BoundaryTensorRef):
            raise ValueError(f"Boundary tensor {spec.label!r} expected a tensor ref.")
        try:
            return tensors[value.label]
        except KeyError as error:
            raise ValueError(
                f"Boundary payload is missing tensor label {value.label!r}."
            ) from error
    if isinstance(spec, BoundarySequenceValueSpec):
        values = [
            _materialize_value(
                item,
                _relabel_sequence_element(spec.element_spec, spec.label, index),
                tensors,
            )
            for index, item in enumerate(value)
        ]
        return tuple(values) if spec.container_type == "tuple" else values
    if isinstance(spec, BoundaryTupleValueSpec):
        return tuple(
            _materialize_value(item, item_spec, tensors)
            for item, item_spec in zip(value, spec.items, strict=True)
        )
    if isinstance(spec, BoundaryListValueSpec):
        return [
            _materialize_value(item, item_spec, tensors)
            for item, item_spec in zip(value, spec.items, strict=True)
        ]
    if isinstance(spec, BoundaryDictValueSpec):
        return {
            key: _materialize_value(value[key], item_spec, tensors)
            for key, item_spec in spec.items
        }
    if isinstance(spec, BoundarySliceValueSpec):
        return slice(
            _materialize_value(value.start, spec.start, tensors),
            _materialize_value(value.stop, spec.stop, tensors),
            _materialize_value(value.step, spec.step, tensors),
        )
    if isinstance(spec, BoundaryCustomValueSpec):
        serializer = _serializer_for_type_name(spec.type_name)
        return serializer.decode(value)
    return value


def _detach_value(
    value: Any,
    spec: BoundaryValueSpec,
    grad_roots: dict[str, torch.Tensor],
) -> Any:
    if isinstance(spec, BoundaryTensorValueSpec):
        tensor = value.detach()
        if value.requires_grad and (tensor.is_floating_point() or tensor.is_complex()):
            grad_root = tensor.requires_grad_(True)
            grad_roots[spec.label] = grad_root
            return grad_root.clone()
        return tensor
    if isinstance(spec, BoundarySequenceValueSpec):
        values = [
            _detach_value(
                item,
                _relabel_sequence_element(spec.element_spec, spec.label, index),
                grad_roots,
            )
            for index, item in enumerate(value)
        ]
        return tuple(values) if spec.container_type == "tuple" else values
    if isinstance(spec, BoundaryTupleValueSpec):
        return tuple(
            _detach_value(item, item_spec, grad_roots)
            for item, item_spec in zip(value, spec.items, strict=True)
        )
    if isinstance(spec, BoundaryListValueSpec):
        return [
            _detach_value(item, item_spec, grad_roots)
            for item, item_spec in zip(value, spec.items, strict=True)
        ]
    if isinstance(spec, BoundaryDictValueSpec):
        return {
            key: _detach_value(value[key], item_spec, grad_roots)
            for key, item_spec in spec.items
        }
    if isinstance(spec, BoundarySliceValueSpec):
        return slice(
            _detach_value(value.start, spec.start, grad_roots),
            _detach_value(value.stop, spec.stop, grad_roots),
            _detach_value(value.step, spec.step, grad_roots),
        )
    return value


def _relabel_sequence_element(
    spec: BoundaryValueSpec,
    sequence_label: str,
    index: int,
) -> BoundaryValueSpec:
    if isinstance(spec, BoundaryTensorValueSpec):
        return BoundaryTensorValueSpec(
            label=f"{sequence_label}.{index}",
            tensor_spec=spec.tensor_spec,
        )
    return spec


def _validate_fixed_sequence(
    value: Any,
    specs: tuple[BoundaryValueSpec, ...],
    container_type: type[Any],
    shape_env: ShapeEnv,
    batch_size: int,
) -> None:
    if not isinstance(value, container_type):
        raise ValueError(
            f"Boundary value expected {container_type.__name__}, got {type(value).__name__}."
        )
    if len(value) != len(specs):
        raise ValueError(f"Boundary sequence length {len(value)}; expected {len(specs)}.")
    for item, item_spec in zip(value, specs, strict=True):
        validate_boundary_value(item, item_spec, shape_env=shape_env, batch_size=batch_size)


def _materialize_length(value: int | str | ShapeExpr, shape_env: ShapeEnv, batch_size: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value != shape_env.batch_symbol:
            raise ValueError(f"Unknown boundary length symbol {value!r}.")
        return batch_size
    return value.materialize({shape_env.batch_symbol: batch_size})


def _serializer_for_type_name(type_name: str) -> BoundarySerializer:
    for type_, serializer in _SERIALIZERS.items():
        if f"{type_.__module__}.{type_.__qualname__}" == type_name:
            return serializer
    raise ValueError(f"No boundary serializer registered for {type_name!r}.")
