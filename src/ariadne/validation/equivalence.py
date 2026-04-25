"""Forward equivalence checks."""

from __future__ import annotations

from typing import Any

import torch


def assert_forward_equivalent(
    model: torch.nn.Module,
    runtime: Any,
    inputs: tuple[Any, ...],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    with torch.no_grad():
        direct = model(*inputs)
        split = runtime.run_suffix(runtime.run_prefix(*inputs))
    _assert_close(direct, split, rtol=rtol, atol=atol)


def _assert_close(left: Any, right: Any, *, rtol: float, atol: float) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        for left_item, right_item in zip(left, right, strict=True):
            _assert_close(left_item, right_item, rtol=rtol, atol=atol)
        return
    raise TypeError(f"Unsupported outputs for equivalence: {type(left)!r}, {type(right)!r}.")
