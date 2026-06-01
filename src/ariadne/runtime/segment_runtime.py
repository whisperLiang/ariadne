"""Runtime execution for generated split segments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import torch

from ariadne.codegen.interception_segments import as_debug_interpreter
from ariadne.codegen.segment_builder import SegmentBundle
from ariadne.pattern.boundary_value import encode_boundary_value, materialize_boundary_value
from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import SplitCandidate
from ariadne.runtime.batching import batch_size_from_inputs, validate_inputs
from ariadne.runtime.boundary import BoundaryPayload, validate_boundary_payload
from ariadne.runtime.train_runtime import (
    backward_prefix_from_boundary,
    train_suffix,
)
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
    prefix_backward_owner_id: str = field(
        default_factory=lambda: uuid4().hex,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.mode == "debug_interpreter":
            self.prefix_segment: torch.nn.Module = as_debug_interpreter(self.segments.prefix)
            self.training_prefix_segment: torch.nn.Module = as_debug_interpreter(
                self.segments.training_prefix
            )
            self.suffix_segment: torch.nn.Module = as_debug_interpreter(self.segments.suffix)
        else:
            self.prefix_segment = self.segments.prefix
            self.training_prefix_segment = self.segments.training_prefix
            self.suffix_segment = self.segments.suffix

    @property
    def split_id(self) -> str:
        return self.candidate.split_id

    @property
    def semantic_split_id(self) -> str:
        return self.candidate.semantic_split_id

    @property
    def graph_signature(self) -> str:
        return self.trace_plan.graph_signature

    @property
    def contract_signature(self) -> str:
        return self.candidate.contract_signature

    def visualize(
        self,
        *,
        outpath: str = "ariadne_runtime_split_graph",
        fileformat: str = "svg",
        view: str = "split",
        save_only: bool = True,
        return_dot: bool = False,
        **kwargs: Any,
    ) -> str | None:
        from ariadne.visualization.graphviz_renderer import render_split_graph, render_trace_graph

        if view == "trace":
            return render_trace_graph(
                self.trace_plan,
                outpath=outpath,
                fileformat=fileformat,
                save_only=save_only,
                return_dot=return_dot,
                **kwargs,
            )
        if view == "split":
            return render_split_graph(
                self.trace_plan,
                self.candidate,
                outpath=outpath,
                fileformat=fileformat,
                save_only=save_only,
                return_dot=return_dot,
                **kwargs,
            )
        raise ValueError("view must be either 'trace' or 'split'.")

    def run_prefix(self, *inputs: Any) -> BoundaryPayload:
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            return variant.run_prefix(*inputs)

        self._validate_inputs(inputs)
        boundary_values = _as_tuple(self.prefix_segment(*inputs))
        return self._make_boundary_payload(
            inputs,
            boundary_values,
            batch_size=batch_size,
            supports_prefix_backward=False,
        )

    def run_training_prefix(self, *inputs: Any) -> BoundaryPayload:
        batch_size = self._batch_size_from_inputs(inputs)
        variant = self._variant_for_batch(batch_size)
        if variant is not None:
            return variant.run_training_prefix(*inputs)

        self._validate_inputs(inputs)
        boundary_values = _as_tuple(self.training_prefix_segment(*inputs))
        return self._make_boundary_payload(
            inputs,
            boundary_values,
            batch_size=batch_size,
            supports_prefix_backward=True,
        )

    def _make_boundary_payload(
        self,
        inputs: tuple[Any, ...],
        boundary_values: tuple[Any, ...],
        *,
        batch_size: int,
        supports_prefix_backward: bool,
    ) -> BoundaryPayload:
        value_schema = self._boundary_value_schema()
        tensors: dict[str, torch.Tensor] = {}
        payload_values: list[Any] = []
        for value, spec in zip(boundary_values, value_schema, strict=True):
            encoded, value_tensors = encode_boundary_value(value, spec)
            payload_values.append(encoded)
            tensors.update(value_tensors)
        passthrough_inputs = {
            label: inputs[self.trace_plan.input_node_names.index(label)]
            for label in self.segments.passthrough_order
        }
        return BoundaryPayload(
            split_id=self.split_id,
            semantic_split_id=self.semantic_split_id,
            graph_signature=self.graph_signature,
            contract_signature=self.contract_signature,
            batch_size=batch_size,
            tensors=tensors,
            schema=self.candidate.boundary_contract_schema,
            requires_grad={label: tensor.requires_grad for label, tensor in tensors.items()},
            passthrough_inputs=passthrough_inputs,
            supports_prefix_backward=supports_prefix_backward,
            prefix_backward_owner_id=(
                self.prefix_backward_owner_id if supports_prefix_backward else None
            ),
            values=tuple(payload_values),
            value_schema=value_schema,
        )

    def run_suffix(self, boundary: BoundaryPayload) -> Any:
        variant = self._variant_for_boundary(boundary)
        if variant is not None:
            return variant.run_suffix(boundary)

        self.validate_boundary(boundary)
        suffix_inputs = self._suffix_inputs(boundary)
        return self.suffix_segment(*suffix_inputs, batch_size=boundary.batch_size)

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
        boundary: BoundaryPayload,
        boundary_grads: BoundaryGradients | None = None,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if not isinstance(boundary, BoundaryPayload):
            raise TypeError(
                "backward_prefix requires a BoundaryPayload from run_training_prefix()."
            )
        if boundary_grads is None:
            raise TypeError("backward_prefix requires boundary_grads.")

        variant = self._variant_for_boundary(boundary)
        if variant is not None:
            variant.backward_prefix(boundary, boundary_grads=boundary_grads, optimizer=optimizer)
            return

        backward_prefix_from_boundary(self, boundary, boundary_grads, optimizer=optimizer)

    def validate_boundary(self, boundary: BoundaryPayload) -> None:
        validate_boundary_payload(
            boundary,
            split_id=self.split_id,
            graph_signature=self.graph_signature,
            semantic_split_id=self.semantic_split_id,
            contract_signature=self.contract_signature,
            schema=self.candidate.boundary_contract_schema,
            shape_env=self.trace_plan.shape_env,
            value_schema=self._boundary_value_schema(),
        )

    def _suffix_inputs(self, boundary: BoundaryPayload) -> tuple[Any, ...]:
        value_schema = self._boundary_value_schema()
        device = _runtime_device(self.trace_plan.root_module)
        boundary_values = tuple(
            _move_tree(
                materialize_boundary_value(value, spec, boundary.tensors),
                device=device,
            )
            for value, spec in zip(boundary.values, value_schema, strict=True)
        )
        passthrough_values = tuple(
            _move_tree(boundary.passthrough_inputs[label], device=device)
            for label in self.segments.passthrough_order
        )
        return (*boundary_values, *passthrough_values)

    def _boundary_value_schema(self) -> tuple[Any, ...]:
        return tuple(
            self.candidate.boundary_contract_value_schema[label]
            for label in self.segments.boundary_order
        )

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> None:
        validate_inputs(self.trace_plan, inputs)

    def _batch_size_from_inputs(self, inputs: tuple[Any, ...]) -> int:
        return batch_size_from_inputs(inputs)

    def _variant_for_batch(self, batch_size: int) -> SplitRuntime | None:
        for variant in self.variants:
            if variant._matches_batch(batch_size):
                return variant
        return None

    def _variant_for_boundary(self, boundary: BoundaryPayload) -> SplitRuntime | None:
        for variant in self.variants:
            if (
                boundary.contract_signature == variant.contract_signature
                and boundary.split_id == variant.split_id
                and boundary.semantic_split_id == variant.semantic_split_id
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


def _runtime_device(module: torch.nn.Module) -> torch.device | None:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return None


def _move_tree(value: Any, *, device: torch.device | None) -> Any:
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device=device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tree(item, device=device) for key, item in value.items()}
    return value
