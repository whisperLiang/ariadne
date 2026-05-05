"""Frontier-based split enumeration."""

from __future__ import annotations

from dataclasses import dataclass

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.planner.cost_model import SplitCost, estimate_split_cost
from ariadne.trace.trace_plan import TraceNode, TracePlan

_PLANNING_AUXILIARY_TARGETS = frozenset({"detach.default", "empty.memory_format"})
_NON_SPLIT_AUXILIARY_TARGETS = frozenset({"t.default"})
_NodeGroupKey = tuple[str, tuple[str, ...], str | None]


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
    auxiliary_nodes = _planning_auxiliary_node_names(plan)
    compute_nodes = [
        node
        for node in plan.nodes
        if (
            node.is_compute
            and node.name not in auxiliary_nodes
            and not _is_non_split_auxiliary_node(node)
        )
    ]
    candidates: list[SplitCandidate] = []
    for split_node in compute_nodes:
        candidate = _candidate_after(plan, split_node, auxiliary_nodes)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _candidate_after(
    plan: TracePlan,
    split_node: TraceNode,
    auxiliary_nodes: frozenset[str],
) -> SplitCandidate | None:
    split_index = plan.index_of(split_node.name)
    prefix_set = {
        node.name
        for node in plan.nodes[: split_index + 1]
        if (
            node.name not in auxiliary_nodes
            and not node.is_output
            and not node.is_attr
            and not node.is_placeholder
        )
    }
    suffix_set = {
        node.name
        for node in plan.nodes[split_index + 1 :]
        if (
            node.name not in auxiliary_nodes
            and not node.is_output
            and not node.is_attr
            and not node.is_placeholder
        )
    }
    if not suffix_set:
        return None

    suffix_and_output = [
        node
        for node in plan.nodes[split_index + 1 :]
        if node.name not in auxiliary_nodes and (node.is_compute or node.is_output)
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


def _planning_auxiliary_node_names(plan: TracePlan) -> frozenset[str]:
    consumer_counts: dict[str, int] = {}
    for node in plan.nodes:
        for parent in node.parents:
            consumer_counts[parent] = consumer_counts.get(parent, 0) + 1
    output_names = {node.name for node in plan.nodes if node.is_output}
    node_groups = _node_groups(plan)
    return frozenset(
        node.name
        for node in plan.nodes
        if consumer_counts.get(node.name, 0) == 0
        and node.name not in output_names
        and (
            node.target in _PLANNING_AUXILIARY_TARGETS
            or _is_unused_multi_output_node(node, node_groups, consumer_counts, output_names)
        )
    )


def _node_groups(plan: TracePlan) -> dict[_NodeGroupKey, tuple[str, ...]]:
    grouped: dict[_NodeGroupKey, list[str]] = {}
    for node in plan.nodes:
        if not node.is_compute:
            continue
        key = (node.target, node.parents, node.module_path)
        grouped.setdefault(key, []).append(node.name)
    return {key: tuple(names) for key, names in grouped.items()}


def _is_unused_multi_output_node(
    node: TraceNode,
    node_groups: dict[tuple[str, tuple[str, ...], str | None], tuple[str, ...]],
    consumer_counts: dict[str, int],
    output_names: set[str],
) -> bool:
    siblings = node_groups.get((node.target, node.parents, node.module_path), ())
    return len(siblings) > 1 and any(
        sibling != node.name and (consumer_counts.get(sibling, 0) > 0 or sibling in output_names)
        for sibling in siblings
    )


def _is_non_split_auxiliary_node(node: TraceNode) -> bool:
    return (
        node.target in _NON_SPLIT_AUXILIARY_TARGETS
        and not node.parents
        and bool(node.param_refs)
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
