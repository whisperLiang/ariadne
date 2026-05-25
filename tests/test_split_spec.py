from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from ariadne import prepare_split, register_boundary_serializer
from ariadne.pattern.boundary_value import BoundaryLiteralValueSpec
from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.pattern.split_spec import SplitSpec
from ariadne.pattern.validator import validate_split_spec
from ariadne.trace.tensor_meta import ShapeEnv


class TinyNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin()


class TwoStepNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sin().cos()


def test_split_spec_accepts_after_boundary() -> None:
    spec = SplitSpec(boundary="after:layer3", batch_symbol="B", dynamic_batch=(1, 64))

    validate_split_spec(spec)


def test_register_boundary_serializer_is_public() -> None:
    class Token:
        def __init__(self, value: str) -> None:
            self.value = value

    register_boundary_serializer(
        Token,
        encode=lambda token: token.value,
        decode=Token,
        schema={"type": "token"},
    )


def test_tensor_only_boundary_uses_structured_payload() -> None:
    model = TwoStepNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(2, 4),),
        split=SplitSpec(
            boundary="after:node_0",
            dynamic_batch=(1, 4),
            trace_batch_mode="batch_gt1",
        ),
    )

    inputs = torch.randn(3, 4)
    boundary = runtime.run_prefix(inputs)
    assert boundary.protocol_version == 2
    assert boundary.values
    assert boundary.value_schema
    torch.testing.assert_close(runtime.run_suffix(boundary), model(inputs))


def test_runtime_rejects_payload_with_mismatched_v2_value_schema() -> None:
    model = TwoStepNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(2, 4),),
        split=SplitSpec(
            boundary="after:node_0",
            dynamic_batch=(1, 4),
            trace_batch_mode="batch_gt1",
        ),
    )

    boundary = runtime.run_prefix(torch.randn(3, 4))
    tampered = replace(
        boundary,
        value_schema=(BoundaryLiteralValueSpec(boundary.values[0]),),
    )
    with pytest.raises(ValueError, match="value_schema"):
        runtime.validate_boundary(tampered)


def test_split_spec_accepts_percent_boundary() -> None:
    validate_split_spec(
        SplitSpec(
            boundary="percent:50",
            dynamic_batch=(2, 16),
            trace_batch_mode="batch_gt1",
        )
    )
    validate_split_spec(
        SplitSpec(
            boundary="50%",
            dynamic_batch=(2, 16),
            trace_batch_mode="batch_gt1",
        )
    )


def test_split_spec_rejects_out_of_range_percent_boundary() -> None:
    with pytest.raises(ValueError, match="greater than 0 and less than 100"):
        validate_split_spec(
            SplitSpec(
                boundary="percent:100",
                dynamic_batch=(2, 16),
                trace_batch_mode="batch_gt1",
            )
        )


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


def test_split_spec_allows_batch_gt1_runtime_range_to_include_singleton() -> None:
    validate_split_spec(
        SplitSpec(boundary="after:layer3", dynamic_batch=(1, 8), trace_batch_mode="batch_gt1")
    )


def test_prepare_split_rejects_batch_mode_input_mismatch() -> None:
    model = TinyNet()
    with pytest.raises(ValueError, match="batch size 1"):
        prepare_split(
            model,
            example_inputs=(torch.randn(2, 4),),
            split=SplitSpec(
                boundary="after:input_0",
                dynamic_batch=(1, 8),
                trace_batch_mode="batch_1",
            ),
        )
    with pytest.raises(ValueError, match="greater than 1"):
        prepare_split(
            model,
            example_inputs=(torch.randn(1, 4),),
            split=SplitSpec(
                boundary="after:input_0",
                dynamic_batch=(2, 8),
                trace_batch_mode="batch_gt1",
            ),
        )
