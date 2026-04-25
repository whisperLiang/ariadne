"""Gradient equivalence checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def assert_gradient_equivalent(
    direct_model: torch.nn.Module,
    split_runtime: Any,
    direct_inputs: tuple[Any, ...],
    split_inputs: tuple[Any, ...],
    targets: Any,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    for model in (direct_model, split_runtime.trace_plan.root_module):
        model.zero_grad(set_to_none=True)
    direct_loss = loss_fn(direct_model(*direct_inputs), targets)
    direct_loss.backward()

    boundary = split_runtime.run_prefix(*split_inputs)
    _, boundary_grads = split_runtime.train_suffix(boundary, targets, loss_fn=loss_fn)
    split_runtime.backward_prefix(*split_inputs, boundary_grads=boundary_grads)

    direct_params = dict(direct_model.named_parameters())
    split_params = dict(split_runtime.trace_plan.root_module.named_parameters())
    for name, direct_param in direct_params.items():
        split_param = split_params[name]
        if direct_param.grad is None or split_param.grad is None:
            raise AssertionError(f"Missing gradient for parameter {name!r}.")
        torch.testing.assert_close(direct_param.grad, split_param.grad, atol=atol, rtol=rtol)
