from __future__ import annotations

import pytest
import torch

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.pattern.split_spec import SplitSpec
from ariadne.pattern.validator import validate_split_spec
from ariadne.trace.tensor_meta import ShapeEnv


def test_split_spec_accepts_after_boundary() -> None:
    spec = SplitSpec(boundary="after:layer3", batch_symbol="B", dynamic_batch=(1, 64))

    validate_split_spec(spec)


def test_split_spec_rejects_unknown_boundary_pattern() -> None:
    with pytest.raises(ValueError, match="after"):
        validate_split_spec(SplitSpec(boundary="layer3"))


def test_boundary_tensor_spec_validates_batch_and_static_dims() -> None:
    spec = BoundaryTensorSpec(
        label="act",
        symbolic_shape=("B", 8),
        dtype=str(torch.float32),
        requires_grad=True,
    )
    shape_env = ShapeEnv(batch_symbol="B", traced_batch_size=4, dynamic_batch=(1, 8))

    spec.validate_tensor(torch.randn(2, 8), shape_env, batch_size=2)
    with pytest.raises(ValueError, match="dimension 1"):
        spec.validate_tensor(torch.randn(2, 9), shape_env, batch_size=2)
    with pytest.raises(ValueError, match="dtype"):
        spec.validate_tensor(torch.ones(2, 8, dtype=torch.float64), shape_env, batch_size=2)
