"""Frontier-based split enumeration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

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
_LAYOUT_ONLY_CLONE_CONSUMERS = frozenset(
    {
        "_unsafe_view.default",
        "clone.default",
        "flatten.using_ints",
        "reshape.default",
        "view.default",
    }
)
_NodeGroupKey = tuple[str, tuple[str, ...], str | None]


@dataclass(frozen=True)
class SplitCandidate:
    split_id: str
    semantic_split_id: str
    contract_signature: str
    boundary_after: str
    boundary_nodes: tuple[str, ...]
    boundary_keys: tuple[str, ...]
    prefix_nodes: tuple[str, ...]
    suffix_nodes: tuple[str, ...]
    boundary_schema: dict[str, BoundaryTensorSpec]
    boundary_value_schema: dict[str, BoundaryValueSpec]
    boundary_contract_schema: dict[str, BoundaryTensorSpec]
    boundary_contract_value_schema: dict[str, BoundaryValueSpec]
    boundary_node_to_key: dict[str, str]
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
    return _with_semantic_contracts(plan, tuple(candidates))


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
        semantic_split_id=split_id,
        contract_signature="",
        boundary_after=_friendly_boundary_label(split_node),
        boundary_nodes=tuple(boundary_nodes),
        boundary_keys=(),
        prefix_nodes=prefix_nodes,
        suffix_nodes=suffix_nodes,
        boundary_schema=schema,
        boundary_value_schema=boundary_value_schema,
        boundary_contract_schema={},
        boundary_contract_value_schema={},
        boundary_node_to_key={},
        passthrough_inputs=tuple(passthrough_inputs),
        cost=cost,
        trainable_suffix=trainable_suffix,
        rejection_reason=rejection_reason,
    )


def _with_semantic_contracts(
    plan: TracePlan,
    candidates: tuple[SplitCandidate, ...],
) -> tuple[SplitCandidate, ...]:
    occurrence_counts: dict[str, int] = {}
    contracted: list[SplitCandidate] = []
    for candidate in candidates:
        occurrence = occurrence_counts.get(candidate.boundary_after, 0)
        occurrence_counts[candidate.boundary_after] = occurrence + 1
        semantic_split_id = f"after:{candidate.boundary_after}#{occurrence}"
        boundary_keys = tuple(
            _semantic_boundary_key(
                plan,
                node_name,
                position=index,
                semantic_split_id=semantic_split_id,
            )
            for index, node_name in enumerate(candidate.boundary_nodes)
        )
        node_to_key = dict(zip(candidate.boundary_nodes, boundary_keys, strict=True))
        contract_schema = {
            node_to_key[label]: _device_agnostic_tensor_spec(node_to_key[label], spec)
            for label, spec in candidate.boundary_schema.items()
        }
        contract_value_schema = {
            node_to_key[label]: _relabel_value_spec(spec, node_to_key[label])
            for label, spec in candidate.boundary_value_schema.items()
        }
        contract_signature = _contract_signature(
            semantic_split_id,
            boundary_keys,
            contract_value_schema,
        )
        contracted.append(
            replace(
                candidate,
                semantic_split_id=semantic_split_id,
                contract_signature=contract_signature,
                boundary_keys=boundary_keys,
                boundary_contract_schema=contract_schema,
                boundary_contract_value_schema=contract_value_schema,
                boundary_node_to_key=node_to_key,
            )
        )
    return tuple(contracted)


def _semantic_boundary_key(
    plan: TracePlan,
    node_name: str,
    *,
    position: int,
    semantic_split_id: str,
) -> str:
    node = plan.get_node(node_name)
    meta = node.tensor_meta
    shape = meta.symbolic_shape if meta is not None else _sequence_shape_signature(plan, node_name)
    dtype = meta.dtype if meta is not None else _sequence_dtype_signature(plan, node_name)
    parts = (
        semantic_split_id,
        f"b{position}",
        _canonical_target(node.target),
        node.module_path or "",
        _stringify_shape(shape),
        dtype or "",
    )
    digest = sha256("|".join(parts).encode()).hexdigest()[:12]
    return f"{parts[0]}:{parts[1]}:{digest}"


def _canonical_target(target: str) -> str:
    if target in {"native_batch_norm.default", "cudnn_batch_norm.default"}:
        return "batch_norm.backend"
    if target.startswith("_scaled_dot_product_") and target.endswith(".default"):
        return "scaled_dot_product_attention.backend"
    return target


def _sequence_shape_signature(plan: TracePlan, node_name: str) -> tuple[Any, ...]:
    template = _sequence_output_template_for_node(plan, node_name)
    if template is None or not template.element_names:
        return ()
    meta = plan.get_node(template.element_names[0]).tensor_meta
    return meta.symbolic_shape if meta is not None else ()


def _sequence_dtype_signature(plan: TracePlan, node_name: str) -> str | None:
    template = _sequence_output_template_for_node(plan, node_name)
    if template is None or not template.element_names:
        return None
    meta = plan.get_node(template.element_names[0]).tensor_meta
    return meta.dtype if meta is not None else None


def _stringify_shape(shape: tuple[Any, ...]) -> str:
    return ",".join(str(dimension) for dimension in shape)


def _device_agnostic_tensor_spec(label: str, spec: BoundaryTensorSpec) -> BoundaryTensorSpec:
    return replace(spec, label=label, device_type=None)


def _relabel_value_spec(spec: BoundaryValueSpec, label: str) -> BoundaryValueSpec:
    if isinstance(spec, BoundaryTensorValueSpec):
        return BoundaryTensorValueSpec(
            label=label,
            tensor_spec=_device_agnostic_tensor_spec(label, spec.tensor_spec),
        )
    if isinstance(spec, BoundarySequenceValueSpec):
        element_label = f"{label}.*"
        return BoundarySequenceValueSpec(
            label=label,
            container_type=spec.container_type,
            length_expr=spec.length_expr,
            element_spec=_relabel_value_spec(spec.element_spec, element_label),
        )
    return spec


def _contract_signature(
    semantic_split_id: str,
    boundary_keys: tuple[str, ...],
    value_schema: dict[str, BoundaryValueSpec],
) -> str:
    digest = sha256()
    digest.update(semantic_split_id.encode())
    for key in boundary_keys:
        digest.update(key.encode())
        digest.update(repr(_schema_fingerprint(value_schema[key])).encode())
    return digest.hexdigest()[:16]


def _schema_fingerprint(spec: BoundaryValueSpec) -> Any:
    if isinstance(spec, BoundaryTensorValueSpec):
        tensor = spec.tensor_spec
        return (
            "tensor",
            tensor.symbolic_shape,
            tensor.dtype,
            tensor.requires_grad,
        )
    if isinstance(spec, BoundarySequenceValueSpec):
        return (
            "sequence",
            spec.container_type,
            spec.length_expr,
            _schema_fingerprint(spec.element_spec),
        )
    return repr(spec)


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
            or _is_non_split_auxiliary_node(plan, node)
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


def _is_non_split_auxiliary_node(plan: TracePlan, node: TraceNode) -> bool:
    if node.target in _NON_SPLIT_AUXILIARY_TARGETS and not node.parents and bool(node.param_refs):
        return True
    return _is_layout_only_clone_node(plan, node)


def _is_layout_only_clone_node(plan: TracePlan, node: TraceNode) -> bool:
    if node.target != "clone.default" or len(node.parents) != 1 or node.tensor_meta is None:
        return False
    try:
        parent = plan.get_node(node.parents[0])
    except KeyError:
        return False
    if parent.tensor_meta is None:
        return False
    if (
        node.tensor_meta.symbolic_shape != parent.tensor_meta.symbolic_shape
        or node.tensor_meta.dtype != parent.tensor_meta.dtype
        or node.tensor_meta.requires_grad != parent.tensor_meta.requires_grad
    ):
        return False

    consumer_targets: list[str] = []
    for consumer in plan.nodes:
        if node.name not in consumer.parents:
            continue
        if not consumer.is_compute:
            return False
        consumer_targets.append(consumer.target)
    return bool(consumer_targets) and all(
        target in _LAYOUT_ONLY_CLONE_CONSUMERS for target in consumer_targets
    )


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)
