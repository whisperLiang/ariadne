"""Dynamic batch validation helpers."""

from __future__ import annotations

from typing import Any

import torch

from ariadne.runtime.batching import first_batch_size, resize_batch
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


def validate_dynamic_batches(
    runtime: Any,
    example_inputs: tuple[Any, ...],
    batch_sizes: tuple[int, ...],
) -> tuple[Any, ...]:
    """Run prefix/suffix replay for resized inputs at each requested batch size."""
    traced_batch = first_batch_size(example_inputs)
    if traced_batch is None:
        raise ValueError("Ariadne requires at least one batched tensor input.")
    outputs: list[Any] = []
    for batch_size in batch_sizes:
        inputs = resize_batch(example_inputs, traced_batch, batch_size)
        outputs.append(runtime.run_suffix(runtime.run_prefix(*inputs)))
    return tuple(outputs)
