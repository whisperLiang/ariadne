from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.validation.gradient import assert_gradient_equivalent


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.Tanh()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


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

    boundary = runtime.run_prefix(x)
    _, boundary_grads = runtime.train_suffix(boundary, targets, loss_fn=F.mse_loss)
    runtime.backward_prefix(x, boundary_grads)

    assert model.layer1.weight.grad is not None
