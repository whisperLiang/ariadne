from __future__ import annotations

import pytest
import torch
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.validation.equivalence import assert_forward_equivalent


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def test_generated_eager_forward_matches_direct_model() -> None:
    model = TinyNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(boundary="after:act", trainable=True),
    )

    assert_forward_equivalent(model, runtime, (torch.randn(6, 5),))


def test_boundary_only_suffix_is_deterministic_for_same_payload() -> None:
    model = TinyNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(boundary="after:act", trainable=True),
    )

    boundary = runtime.run_prefix(torch.randn(4, 5))
    first = runtime.run_suffix(boundary)
    second = runtime.run_suffix(boundary)

    torch.testing.assert_close(first, second)


def test_boundary_validation_rejects_missing_label_and_wrong_static_dim() -> None:
    model = TinyNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(boundary="after:act", trainable=True),
    )
    boundary = runtime.run_prefix(torch.randn(4, 5))

    missing = dict(boundary.tensors)
    missing.pop("act")
    bad_missing = boundary.__class__(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors=missing,
        schema=boundary.schema,
        requires_grad=boundary.requires_grad,
        passthrough_inputs=boundary.passthrough_inputs,
    )
    with pytest.raises(ValueError, match="missing"):
        runtime.run_suffix(bad_missing)

    bad_tensors = dict(boundary.tensors)
    bad_tensors["act"] = torch.randn(4, 9)
    bad_shape = boundary.__class__(
        split_id=boundary.split_id,
        graph_signature=boundary.graph_signature,
        batch_size=boundary.batch_size,
        tensors=bad_tensors,
        schema=boundary.schema,
        requires_grad=boundary.requires_grad,
        passthrough_inputs=boundary.passthrough_inputs,
    )
    with pytest.raises(ValueError, match="dimension 1"):
        runtime.run_suffix(bad_shape)
