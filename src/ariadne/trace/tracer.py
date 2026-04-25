"""Runtime-interception tracing entry point."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ariadne.trace.interception import trace_model_interception
from ariadne.trace.trace_plan import TracePlan


def trace_model(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = None,
    trace_batch_mode: str = "batch_1",
) -> TracePlan:
    """Trace a model by executing its real forward path under interception."""
    return trace_model_interception(
        model,
        example_inputs=tuple(example_inputs),
        batch_symbol=batch_symbol,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
    )
