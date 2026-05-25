"""Top-level preparation API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from ariadne.codegen.segment_builder import build_replay_segments, build_segments
from ariadne.compiler.torch_compile import maybe_compile_segments
from ariadne.pattern.split_spec import SplitSpec, parse_boundary_percent
from ariadne.pattern.validator import validate_split_spec
from ariadne.planner.candidate_validation import validate_split_candidates
from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits
from ariadne.planner.selector import select_split
from ariadne.runtime.batching import first_batch_size, resize_batch
from ariadne.runtime.replay_runtime import (
    ReplayValidationMode,
    SplitReplayRuntime,
)
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
        example_inputs=tuple(example_inputs),
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


def prepare_split_replay(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    split: SplitSpec | str,
    mode: ExecutionMode = "compiled",
    objective: Mapping[str, Any] | None = None,
    compile_options: Mapping[str, Any] | None = None,
    validation: ReplayValidationMode = "fast",
    materialize_boundary: bool = True,
) -> SplitReplayRuntime:
    """Prepare an inference-only split replay runtime.

    This path skips split-retain/training-prefix construction and packages
    boundary values in tuple order for lower-overhead replay.
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
    runtime = _prepare_replay_runtime_from_plan(
        plan,
        spec=spec,
        split=split,
        example_inputs=tuple(example_inputs),
        objective=objective,
        mode=mode,
        compile_options=compile_options,
        validation=validation,
        materialize_boundary=materialize_boundary,
    )
    variants = _prepare_replay_batch_variants(
        model,
        example_inputs=tuple(example_inputs),
        spec=spec,
        split=split,
        objective=objective,
        mode=mode,
        compile_options=compile_options,
        validation=validation,
        materialize_boundary=materialize_boundary,
    )

    return SplitReplayRuntime(
        trace_plan=runtime.trace_plan,
        split_spec=runtime.split_spec,
        candidate=runtime.candidate,
        segments=runtime.segments,
        mode=runtime.mode,
        validation=runtime.validation,
        materialize_boundary=runtime.materialize_boundary,
        compile_options=dict(compile_options or {}),
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
    example_inputs: tuple[Any, ...],
    mode: ExecutionMode,
    compile_options: Mapping[str, Any] | None,
    batch_range: tuple[int, int] | None = None,
) -> SplitRuntime:
    candidates = _validated_candidates_for_split(
        plan,
        spec=spec,
        split=split,
        objective=objective,
        example_inputs=example_inputs,
        require_training=True,
    )
    candidate = select_split(
        plan,
        split=spec if split != "auto" else "auto",
        objective=objective,
        candidates=candidates,
    )
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


def _prepare_replay_runtime_from_plan(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None,
    example_inputs: tuple[Any, ...],
    mode: ExecutionMode,
    compile_options: Mapping[str, Any] | None,
    validation: ReplayValidationMode,
    materialize_boundary: bool,
    batch_range: tuple[int, int] | None = None,
) -> SplitReplayRuntime:
    candidates = _validated_candidates_for_split(
        plan,
        spec=spec,
        split=split,
        objective=objective,
        example_inputs=example_inputs,
        require_training=False,
    )
    candidate = select_split(
        plan,
        split=spec if split != "auto" else "auto",
        objective=objective,
        candidates=candidates,
    )
    segments = build_replay_segments(plan, candidate)
    return SplitReplayRuntime(
        trace_plan=plan,
        split_spec=spec,
        candidate=candidate,
        segments=segments,
        mode=mode,
        validation=validation,
        materialize_boundary=materialize_boundary,
        compile_options=dict(compile_options or {}),
        batch_range=batch_range,
    )


def _validated_candidates_for_split(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None,
    example_inputs: tuple[Any, ...],
    require_training: bool,
) -> tuple[SplitCandidate, ...]:
    candidates = enumerate_frontier_splits(plan)
    if _requires_validated_candidate_pool(split, spec):
        return validate_split_candidates(
            plan,
            spec=spec,
            example_inputs=example_inputs,
            candidates=candidates,
            require_training=require_training,
        )

    selected = select_split(
        plan,
        split=spec,
        objective=objective,
        candidates=candidates,
    )
    return validate_split_candidates(
        plan,
        spec=spec,
        example_inputs=example_inputs,
        candidates=(selected,),
        require_training=require_training,
    )


def _requires_validated_candidate_pool(split: SplitSpec | str, spec: SplitSpec) -> bool:
    if split == "auto" or spec.boundary == "auto":
        return True
    return parse_boundary_percent(spec.boundary) is not None


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
    traced_batch = first_batch_size(example_inputs)
    if traced_batch != 1:
        return ()
    low, high = spec.dynamic_batch
    if high < 2:
        return ()
    variant_batch = max(2, low)
    if variant_batch > high:
        return ()
    variant_inputs = resize_batch(example_inputs, traced_batch, variant_batch)
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
            example_inputs=variant_inputs,
            objective=objective,
            mode=mode,
            compile_options=compile_options,
            batch_range=(variant_batch, high),
        ),
    )


def _prepare_replay_batch_variants(
    model: torch.nn.Module,
    *,
    example_inputs: tuple[Any, ...],
    spec: SplitSpec,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None,
    mode: ExecutionMode,
    compile_options: Mapping[str, Any] | None,
    validation: ReplayValidationMode,
    materialize_boundary: bool,
) -> tuple[SplitReplayRuntime, ...]:
    if spec.trace_batch_mode != "batch_1" or spec.dynamic_batch is None:
        return ()
    traced_batch = first_batch_size(example_inputs)
    if traced_batch != 1:
        return ()
    low, high = spec.dynamic_batch
    if high < 2:
        return ()
    variant_batch = max(2, low)
    if variant_batch > high:
        return ()
    variant_inputs = resize_batch(example_inputs, traced_batch, variant_batch)
    variant_plan = trace_model(
        model,
        example_inputs=variant_inputs,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    return (
        _prepare_replay_runtime_from_plan(
            variant_plan,
            spec=spec,
            split=split,
            example_inputs=variant_inputs,
            objective=objective,
            mode=mode,
            compile_options=compile_options,
            validation=validation,
            materialize_boundary=materialize_boundary,
            batch_range=(variant_batch, high),
        ),
    )


def _validate_trace_batch_mode(spec: SplitSpec, example_inputs: tuple[Any, ...]) -> None:
    traced_batch = first_batch_size(example_inputs)
    if traced_batch is None:
        raise ValueError("Ariadne requires at least one tensor input with a batch dimension.")
    if spec.trace_batch_mode == "batch_1" and traced_batch != 1:
        raise ValueError("batch_1 mode requires example_inputs to use batch size 1.")
    if spec.trace_batch_mode == "batch_gt1" and traced_batch <= 1:
        raise ValueError("batch_gt1 mode requires example_inputs to use batch size greater than 1.")
