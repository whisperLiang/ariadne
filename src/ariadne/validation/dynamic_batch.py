"""Dynamic batch validation helpers."""

from __future__ import annotations

from typing import Any

import torch

from ariadne.validation.equivalence import assert_forward_equivalent


def assert_dynamic_batch_reuse(
    model: torch.nn.Module,
    runtime: Any,
    *,
    input_shape: tuple[int, ...],
    batches: tuple[int, ...],
) -> None:
    non_batch = input_shape[1:]
    for batch in batches:
        x = torch.randn((batch, *non_batch))
        assert_forward_equivalent(model, runtime, (x,))
