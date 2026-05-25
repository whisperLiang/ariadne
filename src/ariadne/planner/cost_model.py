"""Small cost model for split candidates."""

from __future__ import annotations

from dataclasses import dataclass

from ariadne.pattern.boundary_value import (
    BoundaryDictValueSpec,
    BoundaryListValueSpec,
    BoundarySequenceValueSpec,
    BoundarySliceValueSpec,
    BoundaryTupleValueSpec,
    BoundaryValueSpec,
)
from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.trace.trace_plan import TraceNode


@dataclass(frozen=True)
class SplitCost:
    boundary_bytes: int
    prefix_node_count: int
    suffix_node_count: int
    prefix_memory_bytes: int | None = None
    suffix_memory_bytes: int | None = None
    prefix_flops: float | None = None
    suffix_flops: float | None = None


def estimate_boundary_bytes(
    schema: dict[str, BoundaryTensorSpec],
    nodes: tuple[TraceNode, ...],
) -> int:
    node_by_name = {node.name: node for node in nodes}
    total = 0
    for label in schema:
        meta = node_by_name[label].tensor_meta
        if meta is not None:
            total += meta.nbytes
    return total


def estimate_split_cost(
    *,
    schema: dict[str, BoundaryTensorSpec],
    value_schema: dict[str, BoundaryValueSpec] | None = None,
    nodes: tuple[TraceNode, ...],
    prefix_nodes: tuple[str, ...],
    suffix_nodes: tuple[str, ...],
) -> SplitCost:
    return SplitCost(
        boundary_bytes=estimate_boundary_bytes(schema, nodes)
        + _estimate_value_schema_overhead(value_schema or {}),
        prefix_node_count=len(prefix_nodes),
        suffix_node_count=len(suffix_nodes),
    )


def _estimate_value_schema_overhead(schema: dict[str, BoundaryValueSpec]) -> int:
    return sum(_estimate_value_spec_overhead(spec) for spec in schema.values())


def _estimate_value_spec_overhead(spec: BoundaryValueSpec) -> int:
    if isinstance(spec, BoundarySequenceValueSpec):
        return 64 + _estimate_value_spec_overhead(spec.element_spec)
    if isinstance(spec, (BoundaryTupleValueSpec, BoundaryListValueSpec)):
        return 32 + sum(_estimate_value_spec_overhead(item) for item in spec.items)
    if isinstance(spec, BoundaryDictValueSpec):
        return 48 + sum(_estimate_value_spec_overhead(item) for _key, item in spec.items)
    if isinstance(spec, BoundarySliceValueSpec):
        return (
            24
            + _estimate_value_spec_overhead(spec.start)
            + _estimate_value_spec_overhead(spec.stop)
            + _estimate_value_spec_overhead(spec.step)
        )
    return 0
