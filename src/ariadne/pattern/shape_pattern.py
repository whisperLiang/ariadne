"""Boundary shape schemas."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ariadne.trace.tensor_meta import Dimension, ShapeEnv, TensorMeta


@dataclass(frozen=True)
class BoundaryTensorSpec:
    """Expected tensor metadata for a boundary tensor."""

    label: str
    symbolic_shape: tuple[Dimension, ...]
    dtype: str
    requires_grad: bool
    device_type: str | None = None

    @classmethod
    def from_meta(cls, label: str, meta: TensorMeta) -> BoundaryTensorSpec:
        return cls(
            label=label,
            symbolic_shape=meta.symbolic_shape,
            dtype=meta.dtype,
            requires_grad=meta.requires_grad,
            device_type=None if meta.device_type == "meta" else meta.device_type,
        )

    def validate_tensor(self, tensor: torch.Tensor, shape_env: ShapeEnv, batch_size: int) -> None:
        if str(tensor.dtype) != self.dtype:
            raise ValueError(
                f"Boundary tensor {self.label!r} has dtype {tensor.dtype}; expected {self.dtype}."
            )
        if len(tensor.shape) != len(self.symbolic_shape):
            raise ValueError(
                f"Boundary tensor {self.label!r} rank {tensor.ndim} does not match schema "
                f"rank {len(self.symbolic_shape)}."
            )
        for index, (actual, expected) in enumerate(
            zip(tensor.shape, self.symbolic_shape, strict=True)
        ):
            actual_int = int(actual)
            if expected == shape_env.batch_symbol:
                if actual_int != batch_size:
                    raise ValueError(
                        f"Boundary tensor {self.label!r} dimension {index} has batch "
                        f"{actual_int}; expected payload batch {batch_size}."
                    )
            elif isinstance(expected, int) and actual_int != expected:
                raise ValueError(
                    f"Boundary tensor {self.label!r} dimension {index} is {actual_int}; "
                    f"expected {expected}."
                )
        if self.device_type is not None and tensor.device.type != self.device_type:
            raise ValueError(
                f"Boundary tensor {self.label!r} is on {tensor.device.type}; "
                f"expected {self.device_type}."
            )
