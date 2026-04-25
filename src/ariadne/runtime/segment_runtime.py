"""Runtime execution for generated split segments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from ariadne.codegen.interception_segments import as_debug_interpreter
from ariadne.codegen.segment_builder import SegmentBundle
from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import SplitCandidate
from ariadne.runtime.boundary import BoundaryPayload, validate_boundary_payload
from ariadne.runtime.train_runtime import backward_prefix, train_suffix
from ariadne.trace.tensor_meta import ShapeExpr
from ariadne.trace.trace_plan import TracePlan

BoundaryGradients = dict[str, torch.Tensor | None]


@dataclass
class SplitRuntime:
    """Prepared split runtime."""

    trace_plan: TracePlan
    split_spec: SplitSpec
    candidate: SplitCandidate
    segments: SegmentBundle
    mode: Literal["debug_interpreter", "generated_eager", "compiled"] = "generated_eager"
    variants: tuple[SplitRuntime, ...] = ()
    batch_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.mode == "debug_interpreter":
            self.prefix_segment: torch.nn.Module = as_debug_interpreter(self.segments.prefix)
            self.suffix_segment: torch.nn.Module = as_debug_interpreter(self.segments.suffix)
        else:
            self.prefix_segment = self.segments.prefix
            self.suffix_segment = self.segments.suffix

    @property
    def split_id(self) -> str:
        return self.candidate.split_id

    @property
    def graph_signature(self) -> str:
        return self.trace_plan.graph_signature

    def run_prefix(self, *inputs: Any) -> BoundaryPayload:
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            return variant.run_prefix(*inputs)

        self._validate_inputs(inputs)
        boundary_values = _as_tuple(self.prefix_segment(*inputs))
        tensors = {
            label: value
            for label, value in zip(self.segments.boundary_order, boundary_values, strict=True)
            if isinstance(value, torch.Tensor)
        }
        passthrough_inputs = {
            label: inputs[self.trace_plan.input_node_names.index(label)]
            for label in self.segments.passthrough_order
        }
        return BoundaryPayload(
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            batch_size=batch_size,
            tensors=tensors,
            schema=self.candidate.boundary_schema,
            requires_grad={label: tensor.requires_grad for label, tensor in tensors.items()},
            passthrough_inputs=passthrough_inputs,
        )

    def run_suffix(self, boundary: BoundaryPayload) -> Any:
        variant = self._variant_for_boundary(boundary)
        if variant is not None:
            return variant.run_suffix(boundary)

        self.validate_boundary(boundary)
        suffix_inputs = self._suffix_inputs(boundary)
        return self.suffix_segment(*suffix_inputs)

    def train_suffix(
        self,
        boundary: BoundaryPayload,
        targets: Any,
        *,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple[torch.Tensor, BoundaryGradients]:
        variant = self._variant_for_boundary(boundary)
        if variant is not None:
            return variant.train_suffix(boundary, targets, loss_fn=loss_fn, optimizer=optimizer)

        return train_suffix(
            self,
            boundary,
            targets,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

    def backward_prefix(
        self,
        *inputs: Any,
        boundary_grads: BoundaryGradients | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        raw_inputs, grads = _normalize_backward_prefix_args(inputs, boundary_grads)
        batch_size = self._batch_size_from_inputs(raw_inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            variant.backward_prefix(*raw_inputs, boundary_grads=grads, optimizer=optimizer)
            return

        backward_prefix(self, raw_inputs, grads, optimizer=optimizer)

    def validate_boundary(self, boundary: BoundaryPayload) -> None:
        validate_boundary_payload(
            boundary,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            schema=self.candidate.boundary_schema,
            shape_env=self.trace_plan.shape_env,
        )

    def _suffix_inputs(self, boundary: BoundaryPayload) -> tuple[Any, ...]:
        boundary_values = tuple(boundary.tensors[label] for label in self.segments.boundary_order)
        passthrough_values = tuple(
            boundary.passthrough_inputs[label] for label in self.segments.passthrough_order
        )
        return (*boundary_values, *passthrough_values)

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

    def _variant_for_batch(self, batch_size: int) -> SplitRuntime | None:
        for variant in self.variants:
            if variant._matches_batch(batch_size):
                return variant
        return None

    def _variant_for_boundary(self, boundary: BoundaryPayload) -> SplitRuntime | None:
        for variant in self.variants:
            if (
                boundary.graph_signature == variant.graph_signature
                and boundary.split_id == variant.split_id
            ):
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


def _normalize_backward_prefix_args(
    inputs: tuple[Any, ...],
    boundary_grads: BoundaryGradients | None,
) -> tuple[tuple[Any, ...], BoundaryGradients]:
    if boundary_grads is not None:
        return inputs, boundary_grads
    if not inputs:
        raise TypeError("backward_prefix requires raw inputs and boundary gradients.")
    *raw_inputs, maybe_grads = inputs
    if not isinstance(maybe_grads, dict):
        raise TypeError(
            "Pass boundary gradients as the final positional argument or as boundary_grads=..."
        )
    return tuple(raw_inputs), maybe_grads
