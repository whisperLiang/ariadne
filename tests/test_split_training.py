from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.planner.frontier import enumerate_frontier_splits
from ariadne.trace.tracer import trace_model
from ariadne.validation.gradient import assert_gradient_equivalent


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.Tanh()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


class DropoutNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.fc1(x)))


class DetachedBoundaryNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Linear(5, 8)
        self.suffix = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.prefix(x)
        detached = torch.ones_like(y).detach()
        return self.suffix(y + detached)


def test_split_training_matches_full_backward() -> None:
    torch.manual_seed(0)
    direct_model = TinyNet()
    split_model = copy.deepcopy(direct_model)
    x_direct = torch.randn(4, 5, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        split_model,
        example_inputs=(x_split,),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    assert_gradient_equivalent(
        direct_model,
        runtime,
        (x_direct,),
        (x_split,),
        targets,
        loss_fn=F.mse_loss,
    )


def test_compiled_split_training_matches_full_backward_with_aot_eager() -> None:
    torch.manual_seed(0)
    direct_model = TinyNet()
    split_model = copy.deepcopy(direct_model)
    x_direct = torch.randn(4, 5, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        split_model,
        example_inputs=(x_split,),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
        mode="compiled",
        compile_options={"backend": "aot_eager", "dynamic": True},
    )

    assert_gradient_equivalent(
        direct_model,
        runtime,
        (x_direct,),
        (x_split,),
        targets,
        loss_fn=F.mse_loss,
    )


def test_backward_prefix_accepts_prompt_style_positional_gradients() -> None:
    model = TinyNet()
    x = torch.randn(4, 5, requires_grad=True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    boundary = runtime.run_training_prefix(x)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)
    runtime.backward_prefix(boundary, boundary_grads)

    assert model.layer1.weight.grad is not None


def test_direct_boundary_backward_requires_training_prefix_payload() -> None:
    model = TinyNet()
    x = torch.randn(4, 5, requires_grad=True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    boundary = runtime.run_prefix(x)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)

    with pytest.raises(ValueError, match="run_training_prefix"):
        runtime.backward_prefix(boundary, boundary_grads=boundary_grads)


def test_backward_prefix_rejects_training_boundary_from_other_runtime() -> None:
    torch.manual_seed(0)
    model_a = TinyNet()
    model_b = copy.deepcopy(model_a)
    x_a = torch.randn(4, 5, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    targets = torch.randn(4, 3)
    split = SplitSpec(
        boundary="after:act",
        dynamic_batch=(2, 64),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    runtime_a = prepare_split(model_a, example_inputs=(x_a,), split=split)
    runtime_b = prepare_split(model_b, example_inputs=(x_b,), split=split)

    boundary = runtime_a.run_training_prefix(x_a)
    _, boundary_grads = runtime_b.train_suffix(boundary, targets, loss_fn=F.mse_loss)

    with pytest.raises(ValueError, match="different SplitRuntime"):
        runtime_b.backward_prefix(boundary, boundary_grads=boundary_grads)
    assert model_b.layer1.weight.grad is None


def test_backward_prefix_ignores_non_grad_boundary_tensors() -> None:
    model = DetachedBoundaryNet()
    x = torch.randn(4, 5, requires_grad=True)
    plan = trace_model(
        model,
        example_inputs=(x,),
        dynamic_batch=(2, 64),
        trace_batch_mode="batch_gt1",
    )
    candidate = next(
        candidate
        for candidate in enumerate_frontier_splits(plan)
        if candidate.trainable_suffix and len(candidate.boundary_nodes) > 1
    )
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary=candidate.split_id,
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    targets = torch.randn(4, 3)
    boundary = runtime.run_training_prefix(x)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)

    non_grad_labels = {
        label for label, tensor in boundary.tensors.items() if not tensor.requires_grad
    }
    assert non_grad_labels
    assert not non_grad_labels.intersection(boundary_grads)

    runtime.backward_prefix(boundary, boundary_grads=boundary_grads)

    assert model.prefix.weight.grad is not None
    assert model.suffix.weight.grad is not None


def test_training_prefix_backpropagates_through_rng_sensitive_prefix() -> None:
    torch.manual_seed(0)
    direct_model = DropoutNet().train()
    split_model = copy.deepcopy(direct_model).train()
    x_direct = torch.randn(4, 5, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        split_model,
        example_inputs=(x_split,),
        split=SplitSpec(
            boundary="after:drop",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    torch.manual_seed(123)
    direct_loss = F.mse_loss(direct_model(x_direct), targets)
    direct_loss.backward()

    torch.manual_seed(123)
    boundary = runtime.run_training_prefix(x_split)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)
    runtime.backward_prefix(boundary, boundary_grads=boundary_grads)

    direct_params = dict(direct_model.named_parameters())
    split_params = dict(split_model.named_parameters())
    for name, direct_param in direct_params.items():
        split_param = split_params[name]
        assert direct_param.grad is not None
        assert split_param.grad is not None
        torch.testing.assert_close(split_param.grad, direct_param.grad)


def test_backward_prefix_rejects_raw_inputs() -> None:
    model = DropoutNet().train()
    x = torch.randn(4, 5, requires_grad=True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(
            boundary="after:drop",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    boundary = runtime.run_training_prefix(x)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)

    with pytest.raises(TypeError, match="BoundaryPayload"):
        runtime.backward_prefix(x, boundary_grads=boundary_grads)
