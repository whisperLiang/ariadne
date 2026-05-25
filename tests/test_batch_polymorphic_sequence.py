from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split, prepare_split_replay, validate_dynamic_batches
from ariadne.trace.interception import BatchDimArg
from ariadne.trace.tracer import trace_model
from ariadne.validation.equivalence import assert_forward_equivalent
from ariadne.validation.gradient import assert_gradient_equivalent


class SplitCatNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.relu()
        batch = x.shape[0]
        y = x.repeat(batch, 1)
        pieces = y.split(batch, dim=0)
        return self.out(torch.cat(pieces, dim=0))


class UnbindStackNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(x)
        pieces = torch.unbind(y, dim=0)
        return self.out(torch.stack(pieces, dim=0))


class DynamicSequenceOutputNet(nn.Module):
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.relu()
        batch = x.shape[0]
        y = x.repeat(batch, 1)
        return y.split(batch, dim=0)


class IndexedDynamicSequenceNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.relu()
        batch = x.shape[0]
        y = x.repeat(batch, 1)
        pieces = y.split(batch, dim=0)
        return pieces[0] + pieces[-1]


class AmbiguousFeatureSequenceNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.relu()
        batch = x.shape[0]
        y = x.repeat(batch, 1)
        pieces = y.split(batch, dim=0)
        return torch.cat(pieces, dim=0)


class PrefixTrainableSequenceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).relu()
        batch = x.shape[0]
        y = x.repeat(batch, 1)
        pieces = y.split(batch, dim=0)
        return self.out(torch.cat(pieces, dim=0))


def _split_boundary_for(model: nn.Module, example: torch.Tensor) -> str:
    plan = trace_model(
        model,
        example_inputs=(example,),
        dynamic_batch=(1, 20),
        trace_batch_mode="batch_gt1",
    )
    split_op = next(op for op in plan.runtime_artifact.ops if "split" in op.target)
    return f"after:{split_op.output_template.name}"


def test_split_size_is_symbolized_as_batch_dim_arg() -> None:
    plan = trace_model(
        SplitCatNet(),
        example_inputs=(torch.randn(2, 4),),
        dynamic_batch=(1, 20),
        trace_batch_mode="batch_gt1",
    )
    split_op = next(op for op in plan.runtime_artifact.ops if "split" in op.target)

    assert isinstance(split_op.args_template[1], BatchDimArg)
    assert split_op.args_template[1].symbol == "B"


def test_dynamic_sequence_graph_signature_omits_concrete_trace_batch() -> None:
    model = UnbindStackNet()
    plan_batch_2 = trace_model(
        model,
        example_inputs=(torch.randn(2, 4),),
        dynamic_batch=(1, 20),
        trace_batch_mode="batch_gt1",
    )
    plan_batch_3 = trace_model(
        model,
        example_inputs=(torch.randn(3, 4),),
        dynamic_batch=(1, 20),
        trace_batch_mode="batch_gt1",
    )

    unbind_2 = next(op for op in plan_batch_2.runtime_artifact.ops if "unbind" in op.target)
    unbind_3 = next(op for op in plan_batch_3.runtime_artifact.ops if "unbind" in op.target)
    assert len(unbind_2.output_template.element_names) != len(
        unbind_3.output_template.element_names
    )
    assert plan_batch_2.graph_signature == plan_batch_3.graph_signature


@pytest.mark.parametrize(
    ("model", "boundary", "target_shape"),
    [
        (SplitCatNet(), "after:node_0", lambda batch: (batch * batch, 2)),
        (UnbindStackNet(), "after:proj", lambda batch: (batch, 2)),
    ],
)
def test_dynamic_sequence_whole_consumption_replays_and_trains_across_batches(
    model: nn.Module,
    boundary: str,
    target_shape: object,
) -> None:
    torch.manual_seed(0)
    direct_model = copy.deepcopy(model)
    split_model = copy.deepcopy(direct_model)
    example = torch.randn(2, 4)
    runtime = prepare_split(
        split_model,
        example_inputs=(example,),
        split=SplitSpec(
            boundary=boundary,
            dynamic_batch=(1, 20),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    validate_dynamic_batches(runtime, (example.detach(),), (1, 4, 20))
    for batch_size in (1, 4, 20):
        inputs = (torch.randn(batch_size, 4),)
        assert_forward_equivalent(direct_model, runtime, inputs)

        x_direct = torch.randn(batch_size, 4, requires_grad=True)
        x_split = x_direct.detach().clone().requires_grad_(True)
        targets = torch.randn(target_shape(batch_size))
        assert_gradient_equivalent(
            direct_model,
            runtime,
            (x_direct,),
            (x_split,),
            targets,
            loss_fn=F.mse_loss,
        )


def test_dynamic_sequence_can_be_final_suffix_output() -> None:
    model = DynamicSequenceOutputNet()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(2, 4),),
        split=SplitSpec(
            boundary="after:node_0",
            dynamic_batch=(1, 20),
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
    )

    inputs = (torch.randn(20, 4),)
    actual = runtime.run_suffix(runtime.run_prefix(*inputs))
    expected = model(*inputs)

    assert isinstance(actual, type(expected))
    assert len(actual) == 20
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_item, expected_item)


def test_dynamic_sequence_elementwise_use_is_rejected() -> None:
    with pytest.raises(ValueError, match="element-wise"):
        prepare_split(
            IndexedDynamicSequenceNet(),
            example_inputs=(torch.randn(2, 4),),
            split=SplitSpec(
                boundary="after:node_0",
                dynamic_batch=(1, 20),
                trace_batch_mode="batch_gt1",
            ),
        )


def test_dynamic_sequence_element_shape_does_not_over_symbolize_static_dims() -> None:
    model = AmbiguousFeatureSequenceNet()
    example = torch.randn(2, 2)
    runtime = prepare_split(
        model,
        example_inputs=(example,),
        split=SplitSpec(
            boundary=_split_boundary_for(model, example),
            dynamic_batch=(1, 20),
            trace_batch_mode="batch_gt1",
        ),
    )

    assert_forward_equivalent(model, runtime, (torch.randn(3, 2),))


def test_dynamic_sequence_boundary_replays_and_trains() -> None:
    torch.manual_seed(0)
    direct_model = SplitCatNet()
    split_model = copy.deepcopy(direct_model)
    example = torch.randn(2, 4)
    runtime = prepare_split(
        split_model,
        example_inputs=(example,),
        split=SplitSpec(
            boundary="after:node_2",
            dynamic_batch=(1, 20),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    validate_dynamic_batches(runtime, (example.detach(),), (1, 4, 20))
    for batch_size in (1, 4, 20):
        x_direct = torch.randn(batch_size, 4, requires_grad=True)
        x_split = x_direct.detach().clone().requires_grad_(True)
        targets = torch.randn(batch_size * batch_size, 2)
        assert_gradient_equivalent(
            direct_model,
            runtime,
            (x_direct,),
            (x_split,),
            targets,
            loss_fn=F.mse_loss,
        )


def test_dynamic_sequence_boundary_replay_runtime_supports_structured_payload() -> None:
    model = SplitCatNet()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(2, 4),),
        split=SplitSpec(
            boundary="after:node_2",
            dynamic_batch=(1, 20),
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
        validation="strict",
    )

    inputs = torch.randn(4, 4)
    boundary = runtime.run_prefix(inputs)
    assert boundary.protocol_version == 2
    assert len(boundary.values[0]) == 4
    actual = runtime.run_suffix(boundary)
    expected = model(inputs)
    torch.testing.assert_close(actual, expected)


def test_dynamic_sequence_boundary_backpropagates_element_grads_to_prefix() -> None:
    torch.manual_seed(0)
    direct_model = PrefixTrainableSequenceNet()
    split_model = copy.deepcopy(direct_model)
    example = torch.randn(2, 2)
    runtime = prepare_split(
        split_model,
        example_inputs=(example,),
        split=SplitSpec(
            boundary=_split_boundary_for(split_model, example),
            dynamic_batch=(1, 20),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    batch_size = 3
    targets = torch.randn(batch_size * batch_size, 1)
    boundary = runtime.run_training_prefix(torch.randn(batch_size, 2, requires_grad=True))
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)
    boundary_prefix = f"{runtime.segments.boundary_order[0]}."
    assert any(label.startswith(boundary_prefix) for label in boundary_grads)

    x_direct = torch.randn(batch_size, 2, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    assert_gradient_equivalent(
        direct_model,
        runtime,
        (x_direct,),
        (x_split,),
        targets,
        loss_fn=F.mse_loss,
    )
