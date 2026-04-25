"""Frontier-based split enumeration."""

from __future__ import annotations

from dataclasses import dataclass

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.planner.cost_model import SplitCost, estimate_split_cost
from ariadne.trace.trace_plan import TraceNode, TracePlan


@dataclass(frozen=True)
class SplitCandidate:
    split_id: str
    boundary_after: str
    boundary_nodes: tuple[str, ...]
    prefix_nodes: tuple[str, ...]
    suffix_nodes: tuple[str, ...]
    boundary_schema: dict[str, BoundaryTensorSpec]
    passthrough_inputs: tuple[str, ...]
    cost: SplitCost
    trainable_suffix: bool


def enumerate_frontier_splits(plan: TracePlan) -> tuple[SplitCandidate, ...]:
    """Enumerate valid one-frontier splits over a TracePlan."""
    compute_nodes = [node for node in plan.nodes if node.is_compute]
    candidates: list[SplitCandidate] = []
    for split_node in compute_nodes:
        candidate = _candidate_after(plan, split_node)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _candidate_after(plan: TracePlan, split_node: TraceNode) -> SplitCandidate | None:
    split_index = plan.index_of(split_node.name)
    prefix_set = {
        node.name
        for node in plan.nodes[: split_index + 1]
        if not node.is_output and not node.is_attr and not node.is_placeholder
    }
    suffix_set = {
        node.name
        for node in plan.nodes[split_index + 1 :]
        if not node.is_output and not node.is_attr and not node.is_placeholder
    }
    if not suffix_set:
        return None

    suffix_and_output = [
        node for node in plan.nodes[split_index + 1 :] if node.is_compute or node.is_output
    ]
    boundary_nodes: list[str] = []
    passthrough_inputs: list[str] = []
    hidden_prefix_deps: list[str] = []
    placeholders = set(plan.input_node_names)

    for node in suffix_and_output:
        for parent in node.parents:
            if parent in placeholders:
                _append_unique(passthrough_inputs, parent)
            elif parent in prefix_set:
                parent_node = plan.get_node(parent)
                if parent_node.tensor_meta is None:
                    hidden_prefix_deps.append(parent)
                else:
                    _append_unique(boundary_nodes, parent)

    if hidden_prefix_deps or not boundary_nodes:
        return None

    schema = {
        label: BoundaryTensorSpec.from_meta(label, plan.get_node(label).tensor_meta)  # type: ignore[arg-type]
        for label in boundary_nodes
    }
    prefix_nodes = tuple(node.name for node in plan.nodes if node.name in prefix_set)
    suffix_nodes = tuple(node.name for node in plan.nodes if node.name in suffix_set)
    trainable_suffix = any(plan.get_node(name).param_refs for name in suffix_nodes)
    cost = estimate_split_cost(
        schema=schema,
        nodes=plan.nodes,
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
    )
    split_id = f"after:{_friendly_boundary_label(split_node)}"
    return SplitCandidate(
        split_id=split_id,
        boundary_after=_friendly_boundary_label(split_node),
        boundary_nodes=tuple(boundary_nodes),
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
        boundary_schema=schema,
        passthrough_inputs=tuple(passthrough_inputs),
        cost=cost,
        trainable_suffix=trainable_suffix,
    )


def _friendly_boundary_label(node: TraceNode) -> str:
    return node.module_path or node.name


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
