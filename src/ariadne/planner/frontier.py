"""Frontier-based split enumeration."""

from __future__ import annotations

from dataclasses import dataclass

from ariadne.pattern.boundary_value import (
    BoundarySequenceValueSpec,
    BoundaryTensorValueSpec,
    BoundaryValueSpec,
)
from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.planner.cost_model import SplitCost, estimate_split_cost
from ariadne.trace.interception import (
    BatchDimArg,
    InterceptionTraceArtifact,
    SequenceOutputTemplate,
)
from ariadne.trace.tensor_meta import ShapeExpr, TensorMeta
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
    boundary_value_schema: dict[str, BoundaryValueSpec]
    passthrough_inputs: tuple[str, ...]
    cost: SplitCost
    trainable_suffix: bool
    rejection_reason: str | None = None


def enumerate_frontier_splits(plan: TracePlan) -> tuple[SplitCandidate, ...]:
    """Enumerate valid one-frontier splits over captured operation groups."""
    auxiliary_nodes = _planning_auxiliary_node_names(plan)
    candidates: list[SplitCandidate] = []
    for group in _split_frontier_groups(plan, auxiliary_nodes):
        candidate = _candidate_after_group(plan, group, auxiliary_nodes)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates)


def _candidate_after_group(
    plan: TracePlan,
    split_group: tuple[TraceNode, ...],
    auxiliary_nodes: frozenset[str],
) -> SplitCandidate | None:
    split_op_index = split_group[-1].op_index
    split_index = plan.index_of(split_group[-1].name)

    prefix_set = {
        node.name
        for node in plan.nodes
        if (
            _is_prefix_cut(
                plan,
                node,
                split_op_index=split_op_index,
                split_index=split_index,
            )
            and node.name not in auxiliary_nodes
            and not node.is_output
            and not node.is_attr
            and not node.is_placeholder
        )
    }
    suffix_set = {
        node.name
        for node in plan.nodes
        if (
            _is_suffix_cut(
                plan,
                node,
                split_op_index=split_op_index,
                split_index=split_index,
            )
            and node.name not in auxiliary_nodes
            and not node.is_output
            and not node.is_attr
            and not node.is_placeholder
        )
    }
    if not suffix_set:
        return None

    suffix_and_output = [
        node
        for node in plan.nodes
        if (
            node.name not in auxiliary_nodes
            and (
                node.is_output
                or (
                    node.is_compute
                    and _is_suffix_cut(
                        plan,
                        node,
                        split_op_index=split_op_index,
                        split_index=split_index,
                    )
                )
            )
        )
    ]
    boundary_nodes: list[str] = []
    boundary_value_schema: dict[str, BoundaryValueSpec] = {}
    passthrough_inputs: list[str] = []
    hidden_prefix_deps: list[str] = []
    placeholders = set(plan.input_node_names)

    for node in suffix_and_output:
        for parent in node.parents:
            if parent in placeholders:
                _append_unique(passthrough_inputs, parent)
            elif parent in prefix_set:
                value_spec = _boundary_value_spec_for_node(plan, parent)
                if value_spec is None:
                    hidden_prefix_deps.append(parent)
                else:
                    _append_unique(boundary_nodes, parent)
                    boundary_value_schema[parent] = value_spec
            elif parent not in suffix_set:
                hidden_prefix_deps.append(parent)

    rejection_reason = None
    if hidden_prefix_deps:
        rejection_reason = _hidden_dependency_rejection_reason(plan, hidden_prefix_deps)
    elif not boundary_nodes:
        return None

    schema = {
        label: BoundaryTensorSpec.from_meta(label, plan.get_node(label).tensor_meta)  # type: ignore[arg-type]
        for label in boundary_nodes
        if plan.get_node(label).tensor_meta is not None
    }
    prefix_nodes = tuple(node.name for node in plan.nodes if node.name in prefix_set)
    suffix_nodes = tuple(node.name for node in plan.nodes if node.name in suffix_set)
    trainable_suffix = any(plan.get_node(name).param_refs for name in suffix_nodes)
    cost = estimate_split_cost(
        schema=schema,
        value_schema=boundary_value_schema,
        nodes=plan.nodes,
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
    )
    split_node = split_group[-1]
    split_id = f"after:{_friendly_boundary_label(split_node)}"
    return SplitCandidate(
        split_id=split_id,
        boundary_after=_friendly_boundary_label(split_node),
        boundary_nodes=tuple(boundary_nodes),
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
        boundary_schema=schema,
        boundary_value_schema=boundary_value_schema,
        passthrough_inputs=tuple(passthrough_inputs),
        cost=cost,
        trainable_suffix=trainable_suffix,
        rejection_reason=rejection_reason,
    )


def _split_frontier_groups(
    plan: TracePlan,
    auxiliary_nodes: frozenset[str],
) -> tuple[tuple[TraceNode, ...], ...]:
    groups: dict[int, list[TraceNode]] = {}
    fallback_groups: list[tuple[TraceNode, ...]] = []
    for node in plan.nodes:
        if (
            not node.is_compute
            or node.name in auxiliary_nodes
            or _is_non_split_auxiliary_node(node)
        ):
            continue
        if node.op_index is None:
            fallback_groups.append((node,))
            continue
        groups.setdefault(node.op_index, []).append(node)
    ordered_groups = [
        tuple(nodes)
        for _op_index, nodes in sorted(groups.items(), key=lambda item: item[0])
        if nodes
    ]
    return (*ordered_groups, *fallback_groups)


def _is_prefix_cut(
    plan: TracePlan,
    node: TraceNode,
    *,
    split_op_index: int | None,
    split_index: int,
) -> bool:
    if split_op_index is None:
        return plan.index_of(node.name) <= split_index
    return node.op_index is not None and node.op_index <= split_op_index


def _is_suffix_cut(
    plan: TracePlan,
    node: TraceNode,
    *,
    split_op_index: int | None,
    split_index: int,
) -> bool:
    if split_op_index is None:
        return plan.index_of(node.name) > split_index
    return node.op_index is not None and node.op_index > split_op_index


def _friendly_boundary_label(node: TraceNode) -> str:
    return node.module_path or node.name


def _boundary_value_spec_for_node(
    plan: TracePlan,
    node_name: str,
) -> BoundaryValueSpec | None:
    node = plan.get_node(node_name)
    if node.tensor_meta is not None:
        tensor_spec = BoundaryTensorSpec.from_meta(node_name, node.tensor_meta)
        return BoundaryTensorValueSpec(node_name, tensor_spec)

    sequence_template = _sequence_output_template_for_node(plan, node_name)
    if sequence_template is None or not sequence_template.element_names:
        return None
    first_element = plan.get_node(sequence_template.element_names[0])
    if first_element.tensor_meta is None:
        return None
    element_tensor_spec = _sequence_element_tensor_spec(
        plan,
        f"{node_name}.*",
        first_element.tensor_meta,
    )
    return BoundarySequenceValueSpec(
        label=node_name,
        container_type=sequence_template.container_type,
        length_expr=_boundary_length_expr(sequence_template.length_expr),
        element_spec=BoundaryTensorValueSpec(f"{node_name}.*", element_tensor_spec),
    )


def _sequence_output_template_for_node(
    plan: TracePlan,
    node_name: str,
) -> SequenceOutputTemplate | None:
    artifact = plan.runtime_artifact
    if not isinstance(artifact, InterceptionTraceArtifact):
        return None
    for op in artifact.ops:
        if (
            isinstance(op.output_template, SequenceOutputTemplate)
            and op.output_template.name == node_name
        ):
            return op.output_template
    return None


def _boundary_length_expr(value: int | BatchDimArg | ShapeExpr) -> int | str | ShapeExpr:
    if isinstance(value, BatchDimArg):
        return value.symbol
    return value


def _sequence_element_tensor_spec(
    plan: TracePlan,
    label: str,
    meta: TensorMeta,
) -> BoundaryTensorSpec:
    return BoundaryTensorSpec.from_meta(label, meta)


def _hidden_dependency_rejection_reason(
    plan: TracePlan,
    hidden_prefix_deps: list[str],
) -> str:
    labels: list[str] = []
    for name in hidden_prefix_deps:
        try:
            node = plan.get_node(name)
        except KeyError:
            labels.append(name)
            continue
        labels.append(node.module_path or node.name)
    return (
        "split crosses a dependency that cannot be represented in the structured "
        f"boundary payload ({', '.join(labels)})"
    )


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
