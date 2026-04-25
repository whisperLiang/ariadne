"""Lightweight tensor and symbolic-shape metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch


@dataclass(frozen=True)
class ShapeExpr:
    """A tiny symbolic shape expression.

    ``expression`` names the driving symbol, usually ``B``. ``multiplier`` and
    ``offset`` cover affine batch-derived dimensions such as ``4 * B`` that
    appear when real models fold batch with token or window counts.
    """

    expression: str
    multiplier: int = 1
    offset: int = 0

    def materialize(self, symbols: dict[str, int]) -> int:
        if self.expression not in symbols:
            raise ValueError(f"Missing value for shape symbol {self.expression!r}.")
        return self.multiplier * symbols[self.expression] + self.offset

    def __str__(self) -> str:
        if self.multiplier == 1 and self.offset == 0:
            return self.expression
        if self.offset == 0:
            return f"{self.multiplier}*{self.expression}"
        sign = "+" if self.offset > 0 else "-"
        return f"{self.multiplier}*{self.expression}{sign}{abs(self.offset)}"


Dimension: TypeAlias = int | str | ShapeExpr


@dataclass(frozen=True)
class ShapeEnv:
    """Symbolic shape policy for one observed forward path."""

    batch_symbol: str = "B"
    traced_batch_size: int | None = None
    dynamic_batch: tuple[int, int] | None = None

    def canonicalize_shape(self, shape: tuple[int, ...]) -> tuple[Dimension, ...]:
        if self.traced_batch_size is None or not shape:
            return shape
        if shape[0] == self.traced_batch_size:
            return (self.batch_symbol, *shape[1:])
        return shape

    def validate_batch(self, batch_size: int) -> None:
        if self.dynamic_batch is None:
            return
        low, high = self.dynamic_batch
        if not low <= batch_size <= high:
            raise ValueError(
                f"Batch size {batch_size} is outside dynamic_batch range [{low}, {high}]."
            )


@dataclass(frozen=True)
class TensorMeta:
    """Metadata kept by default for tensors observed during tracing."""

    shape: tuple[int, ...]
    symbolic_shape: tuple[Dimension, ...]
    dtype: str
    device_type: str
    requires_grad: bool
    stride: tuple[int, ...]
    numel: int
    element_size: int

    @property
    def nbytes(self) -> int:
        return self.numel * self.element_size


@dataclass(frozen=True)
class ParamRef:
    """Reference to a model parameter used by a trace node."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool


@dataclass(frozen=True)
class BufferRef:
    """Reference to a model buffer used by a trace node."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class ParentRef:
    """Symbolic dependency on a previous trace node."""

    node: str


def tensor_meta_from_tensor(tensor: torch.Tensor, shape_env: ShapeEnv) -> TensorMeta:
    shape = tuple(int(dim) for dim in tensor.shape)
    return TensorMeta(
        shape=shape,
        symbolic_shape=shape_env.canonicalize_shape(shape),
        dtype=str(tensor.dtype),
        device_type=tensor.device.type,
        requires_grad=bool(tensor.requires_grad),
        stride=tuple(int(dim) for dim in tensor.stride()),
        numel=int(tensor.numel()),
        element_size=int(tensor.element_size()),
    )


def param_ref_from_parameter(name: str, parameter: torch.nn.Parameter) -> ParamRef:
    return ParamRef(
        name=name,
        shape=tuple(int(dim) for dim in parameter.shape),
        dtype=str(parameter.dtype),
        requires_grad=bool(parameter.requires_grad),
    )


def buffer_ref_from_tensor(name: str, tensor: torch.Tensor) -> BufferRef:
    return BufferRef(
        name=name,
        shape=tuple(int(dim) for dim in tensor.shape),
        dtype=str(tensor.dtype),
    )
