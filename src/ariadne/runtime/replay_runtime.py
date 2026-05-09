"""Inference-only split replay runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast
from uuid import uuid4

import torch

from ariadne.codegen.interception_segments import as_debug_interpreter
from ariadne.codegen.segment_builder import ReplaySegmentBundle
from ariadne.compiler.torch_compile import replay_compile_options
from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import SplitCandidate
from ariadne.runtime.boundary import BoundaryPayload, validate_boundary_payload
from ariadne.trace.tensor_meta import ShapeExpr
from ariadne.trace.trace_plan import TracePlan

ReplayExecutionMode = Literal["debug_interpreter", "generated_eager", "compiled"]
ReplayValidationMode = Literal["fast", "strict"]


@dataclass(frozen=True)
class ReplayBoundary:
    """Lightweight boundary for inference replay.

    Values stay in generated segment order so the hot suffix path can avoid
    dict construction and per-label metadata.
    """

    split_id: str
    graph_signature: str
    batch_size: int
    values: tuple[Any, ...]
    passthrough_values: tuple[Any, ...] = ()
    owner_id: str | None = field(default=None, repr=False)


@dataclass
class SplitReplayRuntime:
    """Prepared inference-only split runtime."""

    trace_plan: TracePlan
    split_spec: SplitSpec
    candidate: SplitCandidate
    segments: ReplaySegmentBundle
    mode: ReplayExecutionMode = "generated_eager"
    validation: ReplayValidationMode = "fast"
    materialize_boundary: bool = True
    compile_options: dict[str, Any] | None = None
    variants: tuple[SplitReplayRuntime, ...] = ()
    batch_range: tuple[int, int] | None = None
    fallback_reason: str | None = field(default=None, init=False)
    active_backend: str = field(default="eager", init=False)
    owner_id: str = field(default_factory=lambda: uuid4().hex, init=False, repr=False)
    _passthrough_indices: tuple[int, ...] = field(default=(), init=False, repr=False)
    eager_prefix_segment: torch.nn.Module = field(init=False, repr=False)
    eager_suffix_segment: torch.nn.Module = field(init=False, repr=False)
    prefix_segment: torch.nn.Module = field(init=False, repr=False)
    suffix_segment: torch.nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.validation not in {"fast", "strict"}:
            raise ValueError("validation must be either 'fast' or 'strict'.")
        if self.mode not in {"debug_interpreter", "generated_eager", "compiled"}:
            raise ValueError(
                "mode must be one of 'debug_interpreter', 'generated_eager', or 'compiled'."
            )

        self._passthrough_indices: tuple[int, ...] = tuple(
            self.trace_plan.input_node_names.index(label)
            for label in self.segments.passthrough_order
        )
        if self.mode == "debug_interpreter":
            self.eager_prefix_segment: torch.nn.Module = as_debug_interpreter(
                self.segments.prefix
            )
            self.eager_suffix_segment: torch.nn.Module = as_debug_interpreter(
                self.segments.suffix
            )
            self.active_backend = "debug_interpreter"
        else:
            self.eager_prefix_segment = self.segments.prefix
            self.eager_suffix_segment = self.segments.suffix
            self.active_backend = "eager"

        self._use_eager_segments()
        if self.mode == "compiled":
            self._compile_segments(replay_compile_options(self.compile_options))

    @property
    def split_id(self) -> str:
        return self.candidate.split_id

    @property
    def graph_signature(self) -> str:
        return self.trace_plan.graph_signature

    @property
    def boundary_order(self) -> tuple[str, ...]:
        return self.segments.boundary_order

    @property
    def passthrough_order(self) -> tuple[str, ...]:
        return self.segments.passthrough_order

    def warmup(self, *inputs: Any) -> None:
        """Trigger compilation outside the measured split replay path.

        For compiled runtimes, the first attempt uses ``fullgraph=True``. If
        that fails during warmup, the runtime retries with ``fullgraph=False``;
        if that also fails, it falls back to eager split replay and records
        ``fallback_reason``.
        """
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            variant.warmup(*inputs)
            return

        self._validate_inputs_for_mode(inputs, batch_size=batch_size)
        if self.mode != "compiled":
            self._warmup_once(*inputs)
            return

        try:
            self._warmup_once(*inputs)
            return
        except Exception as error:  # pragma: no cover - exercised on backend-specific failures.
            self.fallback_reason = f"fullgraph compile failed: {_short_error(error)}"

        try:
            self._compile_segments(replay_compile_options(self.compile_options, fullgraph=False))
            self._warmup_once(*inputs)
            return
        except Exception as error:  # pragma: no cover - exercised on backend-specific failures.
            previous = self.fallback_reason
            self.fallback_reason = (
                f"{previous}; non-fullgraph compile failed: {_short_error(error)}"
            )
            self._use_eager_segments()
            self._warmup_once(*inputs)

    def run_prefix(self, *inputs: Any) -> ReplayBoundary:
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            return variant.run_prefix(*inputs)

        self._validate_inputs_for_mode(inputs, batch_size=batch_size)
        with torch.inference_mode():
            values = _as_tuple(self.prefix_segment(*inputs))
            if self._should_materialize_boundary():
                values = _materialize_boundary_values(values)
        return ReplayBoundary(
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=batch_size,
            values=values,
            passthrough_values=self._passthrough_values(inputs),
            owner_id=self.owner_id,
        )

    def run_suffix(self, boundary: ReplayBoundary) -> Any:
        if not isinstance(boundary, ReplayBoundary):
            raise TypeError("run_suffix requires a ReplayBoundary from run_prefix().")
        variant = self._variant_for_boundary(boundary)
        if variant is not None:
            return variant.run_suffix(boundary)

        self.validate_boundary(boundary)
        with torch.inference_mode():
            return self.suffix_segment(
                *self._suffix_inputs(boundary),
                batch_size=boundary.batch_size,
            )

    def replay(self, *inputs: Any) -> Any:
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            return variant.replay(*inputs)

        self._validate_inputs_for_mode(inputs, batch_size=batch_size)
        return self.run_suffix(self.run_prefix(*inputs))

    def validate_boundary(self, boundary: ReplayBoundary) -> None:
        if self.validation == "fast":
            if boundary.owner_id != self.owner_id:
                raise ValueError("ReplayBoundary was produced by a different SplitReplayRuntime.")
            self.trace_plan.shape_env.validate_batch(boundary.batch_size)
            if len(boundary.values) != len(self.segments.boundary_order):
                raise ValueError(
                    f"ReplayBoundary has {len(boundary.values)} values; "
                    f"expected {len(self.segments.boundary_order)}."
                )
            if len(boundary.passthrough_values) != len(self.segments.passthrough_order):
                raise ValueError(
                    f"ReplayBoundary has {len(boundary.passthrough_values)} passthrough "
                    f"values; expected {len(self.segments.passthrough_order)}."
                )
            return

        validate_boundary_payload(
            self._to_boundary_payload(boundary),
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            schema=self.candidate.boundary_schema,
            shape_env=self.trace_plan.shape_env,
        )

    def with_validation(self, validation: ReplayValidationMode) -> SplitReplayRuntime:
        updated = replace(self, validation=validation)
        return updated

    def _compile_segments(self, options: dict[str, Any]) -> None:
        self.prefix_segment = cast(
            torch.nn.Module,
            torch.compile(self.eager_prefix_segment, **options),
        )
        self.suffix_segment = cast(
            torch.nn.Module,
            torch.compile(self.eager_suffix_segment, **options),
        )
        self.active_backend = str(options.get("backend", "inductor"))

    def _use_eager_segments(self) -> None:
        self.prefix_segment = self.eager_prefix_segment
        self.suffix_segment = self.eager_suffix_segment
        if self.mode == "debug_interpreter":
            self.active_backend = "debug_interpreter"
        else:
            self.active_backend = "eager"

    def _warmup_once(self, *inputs: Any) -> None:
        boundary = self.run_prefix(*inputs)
        self.run_suffix(boundary)
        _sync_cuda()

    def _suffix_inputs(self, boundary: ReplayBoundary) -> tuple[Any, ...]:
        return (*boundary.values, *boundary.passthrough_values)

    def _should_materialize_boundary(self) -> bool:
        return (
            self.materialize_boundary
            and self.mode == "compiled"
        )

    def _passthrough_values(self, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(inputs[index] for index in self._passthrough_indices)

    def _to_boundary_payload(self, boundary: ReplayBoundary) -> BoundaryPayload:
        if boundary.split_id != self.split_id:
            raise ValueError(
                f"Boundary split_id {boundary.split_id!r} does not match {self.split_id!r}."
            )
        if boundary.graph_signature != self.graph_signature:
            raise ValueError(
                f"Boundary graph_signature {boundary.graph_signature!r} does not match "
                f"{self.graph_signature!r}."
            )
        tensors = {
            label: value
            for label, value in zip(self.segments.boundary_order, boundary.values, strict=True)
            if isinstance(value, torch.Tensor)
        }
        passthrough_inputs = dict(
            zip(self.segments.passthrough_order, boundary.passthrough_values, strict=True)
        )
        return BoundaryPayload(
            split_id=boundary.split_id,
            graph_signature=boundary.graph_signature,
            batch_size=boundary.batch_size,
            tensors=tensors,
            schema=self.candidate.boundary_schema,
            requires_grad={label: tensor.requires_grad for label, tensor in tensors.items()},
            passthrough_inputs=passthrough_inputs,
        )

    def _validate_inputs_for_mode(self, inputs: tuple[Any, ...], *, batch_size: int) -> None:
        if self.validation == "strict":
            self._validate_inputs(inputs)
        else:
            self.trace_plan.shape_env.validate_batch(batch_size)

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> None:
        batch_size = self._batch_size_from_inputs(inputs)
        self.trace_plan.shape_env.validate_batch(batch_size)
        for index, meta in enumerate(self.trace_plan.input_metas):
            if meta is None or index >= len(inputs) or not isinstance(inputs[index], torch.Tensor):
                continue
            tensor = inputs[index]
            if tensor.ndim != len(meta.symbolic_shape):
                raise ValueError(
                    f"Input {index} rank {tensor.ndim} does not match traced rank "
                    f"{len(meta.symbolic_shape)}."
                )
            for dim_index, (actual, expected) in enumerate(
                zip(tensor.shape, meta.symbolic_shape, strict=True)
            ):
                if expected == self.trace_plan.shape_env.batch_symbol:
                    continue
                if isinstance(expected, ShapeExpr):
                    expected_int = expected.materialize(
                        {self.trace_plan.shape_env.batch_symbol: batch_size}
                    )
                    if int(actual) != expected_int:
                        raise ValueError(
                            f"Input {index} dimension {dim_index} is {int(actual)}; "
                            f"expected {expected_int} from {expected}."
                        )
                    continue
                if isinstance(expected, int) and int(actual) != expected:
                    raise ValueError(
                        f"Input {index} dimension {dim_index} is {int(actual)}; "
                        f"expected {expected}."
                    )

    def _batch_size_from_inputs(self, inputs: tuple[Any, ...]) -> int:
        for value in inputs:
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        raise ValueError("Ariadne requires at least one batched tensor input.")

    def _variant_for_batch(self, batch_size: int) -> SplitReplayRuntime | None:
        for variant in self.variants:
            if variant._matches_batch(batch_size):
                return variant
        return None

    def _variant_for_boundary(self, boundary: ReplayBoundary) -> SplitReplayRuntime | None:
        for variant in self.variants:
            if boundary.graph_signature != variant.graph_signature:
                continue
            if boundary.split_id != variant.split_id:
                continue
            if boundary.owner_id is not None:
                if boundary.owner_id == variant.owner_id:
                    return variant
                continue
            if variant._matches_batch(boundary.batch_size):
                return variant
        return None

    def _matches_batch(self, batch_size: int) -> bool:
        if self.batch_range is None:
            return False
        low, high = self.batch_range
        return low <= batch_size <= high


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _materialize_boundary_values(values: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_materialize_boundary_value(value) for value in values)


def _materialize_boundary_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_materialize_boundary_value(item) for item in value)
    if isinstance(value, list):
        return [_materialize_boundary_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _materialize_boundary_value(item) for key, item in value.items()}
    return value


def _short_error(error: Exception) -> str:
    message = str(error).splitlines()[0] if str(error) else error.__class__.__name__
    return f"{error.__class__.__name__}: {message}"
