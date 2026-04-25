"""Top-level preparation API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from ariadne.codegen.segment_builder import build_segments
from ariadne.compiler.torch_compile import maybe_compile_segments
from ariadne.pattern.split_spec import SplitSpec
from ariadne.pattern.validator import validate_split_spec
from ariadne.planner.selector import select_split
from ariadne.runtime.segment_runtime import SplitRuntime
from ariadne.trace.tracer import trace_model

ExecutionMode = Literal["debug_interpreter", "generated_eager", "compiled"]


def prepare_split(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    split: SplitSpec | str,
    mode: ExecutionMode = "generated_eager",
    objective: Mapping[str, Any] | None = None,
    compile_options: Mapping[str, Any] | None = None,
) -> SplitRuntime:
    """Trace, plan, generate, and package a split runtime.

    ``torch.compile`` is only applied after generated prefix/suffix segments exist.
    It is not used for graph capture.
    """
    if mode not in {"debug_interpreter", "generated_eager", "compiled"}:
        raise ValueError(
            "mode must be one of 'debug_interpreter', 'generated_eager', or 'compiled'"
        )

    spec = _normalize_split_spec(split)
    validate_split_spec(spec)

    plan = trace_model(
        model,
        example_inputs=tuple(example_inputs),
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
    )
    candidate = select_split(plan, split=spec if split != "auto" else "auto", objective=objective)
    segments = build_segments(plan, candidate)
    segments = maybe_compile_segments(segments, mode=mode, compile_options=compile_options)

    return SplitRuntime(
        trace_plan=plan,
        split_spec=spec,
        candidate=candidate,
        segments=segments,
        mode=mode,
    )


def _normalize_split_spec(split: SplitSpec | str) -> SplitSpec:
    if isinstance(split, SplitSpec):
        return split
    if split == "auto":
        return SplitSpec(boundary="auto")
    return SplitSpec(boundary=split)
