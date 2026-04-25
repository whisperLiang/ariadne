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
from ariadne.trace.trace_plan import TracePlan
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
    _validate_trace_batch_mode(spec, tuple(example_inputs))

    plan = trace_model(
        model,
        example_inputs=tuple(example_inputs),
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    runtime = _prepare_runtime_from_plan(
        plan,
        spec=spec,
        split=split,
        objective=objective,
        mode=mode,
        compile_options=compile_options,
    )
    variants = _prepare_batch_variants(
        model,
        example_inputs=tuple(example_inputs),
        spec=spec,
        split=split,
        objective=objective,
        mode=mode,
        compile_options=compile_options,
    )

    return SplitRuntime(
        trace_plan=runtime.trace_plan,
        split_spec=runtime.split_spec,
        candidate=runtime.candidate,
        segments=runtime.segments,
        mode=runtime.mode,
        variants=variants,
    )


def _normalize_split_spec(split: SplitSpec | str) -> SplitSpec:
    if isinstance(split, SplitSpec):
        return split
    if split == "auto":
        return SplitSpec(boundary="auto")
    return SplitSpec(boundary=split)


def _prepare_runtime_from_plan(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None,
    mode: ExecutionMode,
    compile_options: Mapping[str, Any] | None,
    batch_range: tuple[int, int] | None = None,
) -> SplitRuntime:
    candidate = select_split(plan, split=spec if split != "auto" else "auto", objective=objective)
    segments = build_segments(plan, candidate)
    segments = maybe_compile_segments(segments, mode=mode, compile_options=compile_options)
    return SplitRuntime(
        trace_plan=plan,
        split_spec=spec,
        candidate=candidate,
        segments=segments,
        mode=mode,
        batch_range=batch_range,
    )


def _prepare_batch_variants(
    model: torch.nn.Module,
    *,
    example_inputs: tuple[Any, ...],
    spec: SplitSpec,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None,
    mode: ExecutionMode,
    compile_options: Mapping[str, Any] | None,
) -> tuple[SplitRuntime, ...]:
    if spec.trace_batch_mode != "batch_1" or spec.dynamic_batch is None:
        return ()
    traced_batch = _first_batch_size(example_inputs)
    if traced_batch != 1:
        return ()
    low, high = spec.dynamic_batch
    if high < 2:
        return ()
    variant_batch = max(2, low)
    if variant_batch > high:
        return ()
    variant_inputs = _resize_batch(example_inputs, traced_batch, variant_batch)
    variant_plan = trace_model(
        model,
        example_inputs=variant_inputs,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    return (
        _prepare_runtime_from_plan(
            variant_plan,
            spec=spec,
            split=split,
            objective=objective,
            mode=mode,
            compile_options=compile_options,
            batch_range=(variant_batch, high),
        ),
    )


def _first_batch_size(values: tuple[Any, ...]) -> int | None:
    for value in values:
        tensor = _first_tensor(value)
        if tensor is not None and tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _resize_batch(value: Any, traced_batch: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == traced_batch:
            return _resize_tensor_batch(value, batch_size)
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_resize_batch(item, traced_batch, batch_size) for item in value)
    if isinstance(value, list):
        return [_resize_batch(item, traced_batch, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _resize_batch(item, traced_batch, batch_size) for key, item in value.items()}
    return value


def _resize_tensor_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if int(tensor.shape[0]) >= batch_size:
        resized = tensor[:batch_size].detach().clone()
    else:
        repeats = [1 for _ in tensor.shape]
        repeats[0] = (batch_size + int(tensor.shape[0]) - 1) // int(tensor.shape[0])
        resized = tensor.repeat(*repeats)[:batch_size].detach().clone()
    if tensor.requires_grad and (resized.is_floating_point() or resized.is_complex()):
        resized.requires_grad_(True)
    return resized


def _validate_trace_batch_mode(spec: SplitSpec, example_inputs: tuple[Any, ...]) -> None:
    traced_batch = _first_batch_size(example_inputs)
    if traced_batch is None:
        raise ValueError("Ariadne requires at least one tensor input with a batch dimension.")
    if spec.trace_batch_mode == "batch_1" and traced_batch != 1:
        raise ValueError("batch_1 mode requires example_inputs to use batch size 1.")
    if spec.trace_batch_mode == "batch_gt1" and traced_batch <= 1:
        raise ValueError("batch_gt1 mode requires example_inputs to use batch size greater than 1.")
