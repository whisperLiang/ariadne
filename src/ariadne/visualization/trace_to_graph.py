"""Convert Ariadne TracePlan objects into render-only graph models."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Literal

from ariadne.trace.trace_plan import TraceNode, TracePlan
from ariadne.visualization.graph_model import (
    RenderBuffer,
    RenderEdge,
    RenderGraph,
    RenderNode,
    RenderParam,
)

_VISUAL_AUXILIARY_TARGETS = frozenset({"detach.default", "empty.memory_format"})
_VISUAL_PARAMETER_PREP_TARGETS = frozenset({"t.default"})
_NodeGroupKey = tuple[str, tuple[str, ...], str | None]
ViewDetail = Literal["module", "operation"]
DEFAULT_MODULE_DEPTH = 4


def build_visual_graph(
    plan: TracePlan,
    *,
    view_detail: ViewDetail = "module",
    max_module_depth: int | None = DEFAULT_MODULE_DEPTH,
) -> RenderGraph:
    """Build the default user-facing graph for visualization exports."""
    graph = prune_visual_auxiliary_nodes(build_render_graph(plan))
    if view_detail == "module":
        return collapse_module_nodes(graph, max_module_depth=max_module_depth)
    if view_detail == "operation":
        return graph
    raise ValueError("view_detail must be either 'module' or 'operation'.")


def build_render_graph(plan: TracePlan) -> RenderGraph:
    """Build a visualization graph from existing trace metadata."""
    nodes = tuple(_render_node(index, node, plan) for index, node in enumerate(plan.nodes))
    edges = tuple(
        RenderEdge(source=parent, target=node.name)
        for node in plan.nodes
        for parent in node.parents
    )
    output_node_names = tuple(node.name for node in plan.nodes if node.is_output)
    metadata: dict[str, Any] = {
        "graph_signature": plan.graph_signature,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "input_node_names": plan.input_node_names,
        "output_node_names": output_node_names,
        "batch_symbol": plan.shape_env.batch_symbol,
        "dynamic_batch": plan.shape_env.dynamic_batch,
        "trace_batch_mode": plan.shape_env.trace_batch_mode,
        "traced_batch_size": plan.shape_env.traced_batch_size,
        "module_types": {
            name: module.__class__.__name__
            for name, module in plan.root_module.named_modules()
            if name
        },
        "module_child_counts": _module_child_counts(plan),
    }
    return RenderGraph(
        graph_signature=plan.graph_signature,
        nodes=nodes,
        edges=edges,
        input_node_names=plan.input_node_names,
        output_node_names=output_node_names,
        metadata=metadata,
    )


def prune_visual_auxiliary_nodes(graph: RenderGraph) -> RenderGraph:
    """Hide graph leaves that come from tracing/runtime bookkeeping."""
    consumer_counts: dict[str, int] = {}
    non_output_consumer_counts: dict[str, int] = {}
    for edge in graph.edges:
        consumer_counts[edge.source] = consumer_counts.get(edge.source, 0) + 1
        if edge.target not in graph.output_node_names:
            non_output_consumer_counts[edge.source] = (
                non_output_consumer_counts.get(edge.source, 0) + 1
            )
    output_names = set(graph.output_node_names)
    node_groups = _node_groups(graph)
    hidden_names = frozenset(
        node.name
        for node in graph.nodes
        if _is_parameter_prep_node(node, non_output_consumer_counts)
        or (
            consumer_counts.get(node.name, 0) == 0
            and node.name not in output_names
            and (
                node.target in _VISUAL_AUXILIARY_TARGETS
                or _is_unused_multi_output_node(node, node_groups, consumer_counts, output_names)
            )
        )
    )
    if not hidden_names:
        return graph

    metadata = dict(graph.metadata)
    metadata["hidden_auxiliary_node_names"] = tuple(
        node.name for node in graph.nodes if node.name in hidden_names
    )
    nodes = tuple(node for node in graph.nodes if node.name not in hidden_names)
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.source not in hidden_names and edge.target not in hidden_names
    )
    metadata["node_count"] = len(nodes)
    metadata["edge_count"] = len(edges)
    return RenderGraph(
        graph_signature=graph.graph_signature,
        nodes=nodes,
        edges=edges,
        input_node_names=graph.input_node_names,
        output_node_names=graph.output_node_names,
        metadata=metadata,
    )


def collapse_module_nodes(
    graph: RenderGraph,
    *,
    max_module_depth: int | None = None,
) -> RenderGraph:
    """Collapse traced operations into module-level nodes when possible.

    The trace still originates from ATen interception, but this view follows the
    model's module hierarchy so rendered graphs read like the original model.
    """
    grouped: dict[str, list[RenderNode]] = {}
    group_module_paths: dict[str, str | None] = {}
    group_module_types: dict[str, str | None] = {}
    ordered_group_names: list[str] = []
    node_to_group: dict[str, str] = {}
    group_occurrence_counts: dict[str, int] = {}
    last_group_for_base_name: dict[str, str] = {}
    previous_base_name: str | None = None

    for node in graph.nodes:
        prepared_node, base_group_name, group_module_path = _render_group(
            graph,
            node,
            max_module_depth=max_module_depth,
        )
        group_name = _occurrence_group_name(
            base_group_name,
            node=prepared_node,
            previous_base_name=previous_base_name,
            group_occurrence_counts=group_occurrence_counts,
            last_group_for_base_name=last_group_for_base_name,
        )
        previous_base_name = base_group_name

        if group_name not in grouped:
            grouped[group_name] = []
            group_module_paths[group_name] = group_module_path
            group_module_types[group_name] = _graph_module_type(graph, group_module_path)
            ordered_group_names.append(group_name)
        grouped[group_name].append(prepared_node)
        for source_name in prepared_node.source_names:
            node_to_group[source_name] = group_name

    nodes = tuple(
        _collapse_group(
            grouped[group_name],
            group_name,
            group_module_paths[group_name],
            group_module_types[group_name],
        )
        for group_name in ordered_group_names
    )

    edge_keys: dict[tuple[str, str], RenderEdge] = {}
    for edge in graph.edges:
        source = node_to_group.get(edge.source, edge.source)
        target = node_to_group.get(edge.target, edge.target)
        if source == target:
            continue
        key = (source, target)
        if key not in edge_keys:
            edge_keys[key] = RenderEdge(source=source, target=target, kind=edge.kind)

    output_names = set(graph.output_node_names)
    output_node_names = tuple(
        node.name for node in nodes if any(source in output_names for source in node.source_names)
    )
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "view_detail": "module",
            "max_module_depth": max_module_depth,
            "node_count": len(nodes),
            "edge_count": len(edge_keys),
        }
    )
    return RenderGraph(
        graph_signature=graph.graph_signature,
        nodes=nodes,
        edges=tuple(edge_keys.values()),
        input_node_names=graph.input_node_names,
        output_node_names=output_node_names,
        metadata=metadata,
    )


def _node_groups(graph: RenderGraph) -> dict[_NodeGroupKey, tuple[str, ...]]:
    grouped: dict[_NodeGroupKey, list[str]] = {}
    for node in graph.nodes:
        if not node.is_compute:
            continue
        key = (node.target, node.parents, node.module_path)
        grouped.setdefault(key, []).append(node.name)
    return {key: tuple(names) for key, names in grouped.items()}


def _is_unused_multi_output_node(
    node: RenderNode,
    node_groups: dict[_NodeGroupKey, tuple[str, ...]],
    consumer_counts: dict[str, int],
    output_names: set[str],
) -> bool:
    siblings = node_groups.get((node.target, node.parents, node.module_path), ())
    return len(siblings) > 1 and any(
        sibling != node.name and (consumer_counts.get(sibling, 0) > 0 or sibling in output_names)
        for sibling in siblings
    )


def _is_parameter_prep_node(
    node: RenderNode,
    non_output_consumer_counts: dict[str, int],
) -> bool:
    return (
        node.target in _VISUAL_PARAMETER_PREP_TARGETS
        and not node.parents
        and node.param_count > 0
        and non_output_consumer_counts.get(node.name, 0) > 0
    )


def _render_node(index: int, node: TraceNode, plan: TracePlan) -> RenderNode:
    tensor_meta = node.tensor_meta
    param_refs, buffer_refs = _param_buffer_refs(plan, node)
    return RenderNode(
        name=node.name,
        source_names=(node.name,),
        index=index,
        op=node.op,
        target=node.target,
        parents=node.parents,
        module_path=node.module_path,
        module_type=_module_type(plan, node.module_path),
        symbolic_shape=_symbolic_shape(tensor_meta.symbolic_shape if tensor_meta else None),
        dtype=tensor_meta.dtype if tensor_meta else None,
        nbytes=tensor_meta.nbytes if tensor_meta else None,
        param_count=len(param_refs),
        buffer_count=len(buffer_refs),
        param_refs=param_refs,
        buffer_refs=buffer_refs,
        is_placeholder=node.is_placeholder,
        is_output=node.is_output,
        is_attr=node.is_attr,
        is_compute=node.is_compute,
        rng_sensitive=node.rng_sensitive,
        has_alias_metadata=node.alias_metadata is not None,
        has_mutation_metadata=node.mutation_metadata is not None,
    )


def _render_group(
    graph: RenderGraph,
    node: RenderNode,
    *,
    max_module_depth: int | None,
) -> tuple[RenderNode, str, str | None]:
    if _is_expanded_container_op(graph, node, max_module_depth=max_module_depth):
        return replace(node, module_type=None), node.name, node.module_path

    module_path = _module_group_path(node.module_path, max_module_depth=max_module_depth)
    if module_path is not None and node.is_compute:
        return node, _module_node_name(module_path), module_path
    return node, node.name, node.module_path


def _occurrence_group_name(
    base_group_name: str,
    *,
    node: RenderNode,
    previous_base_name: str | None,
    group_occurrence_counts: dict[str, int],
    last_group_for_base_name: dict[str, str],
) -> str:
    if base_group_name == node.name:
        return base_group_name
    if previous_base_name == base_group_name:
        return last_group_for_base_name[base_group_name]

    occurrence_count = group_occurrence_counts.get(base_group_name, 0) + 1
    group_occurrence_counts[base_group_name] = occurrence_count
    if occurrence_count == 1:
        group_name = base_group_name
    else:
        group_name = f"{base_group_name}__call_{occurrence_count}"
    last_group_for_base_name[base_group_name] = group_name
    return group_name


def _is_expanded_container_op(
    graph: RenderGraph,
    node: RenderNode,
    *,
    max_module_depth: int | None,
) -> bool:
    if not node.is_compute or node.module_path is None:
        return False
    if not _module_has_children(graph, node.module_path):
        return False
    if max_module_depth is None:
        return True
    return len(node.module_path.split(".")) < max_module_depth


def _module_group_path(module_path: str | None, *, max_module_depth: int | None) -> str | None:
    if module_path is None:
        return None
    if max_module_depth is None:
        return module_path
    if max_module_depth <= 0:
        return None
    return ".".join(module_path.split(".")[:max_module_depth])


def _module_node_name(module_path: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", module_path).strip("_")
    if not sanitized:
        sanitized = "root"
    return f"module__{sanitized}"


def _collapse_group(
    nodes: list[RenderNode],
    group_name: str,
    module_path: str | None,
    module_type: str | None,
) -> RenderNode:
    if len(nodes) == 1:
        node = nodes[0]
        if node.name == group_name:
            return node
        return replace(
            node,
            name=group_name,
            module_path=module_path,
            module_type=module_type or node.module_type,
        )

    last_node = nodes[-1]
    source_names = tuple(source for node in nodes for source in node.source_names)
    source_name_set = set(source_names)
    param_refs = _dedupe_params(nodes)
    buffer_refs = _dedupe_buffers(nodes)
    return RenderNode(
        name=group_name,
        source_names=source_names,
        index=nodes[0].index,
        op="module",
        target=last_node.target,
        parents=tuple(
            dict.fromkeys(
                parent for node in nodes for parent in node.parents if parent not in source_name_set
            )
        ),
        module_path=module_path,
        module_type=module_type or _collapsed_module_type(nodes, module_path),
        symbolic_shape=last_node.symbolic_shape,
        dtype=last_node.dtype,
        nbytes=last_node.nbytes,
        param_count=len(param_refs),
        buffer_count=len(buffer_refs),
        param_refs=param_refs,
        buffer_refs=buffer_refs,
        is_placeholder=any(node.is_placeholder for node in nodes),
        is_output=any(node.is_output for node in nodes),
        is_attr=any(node.is_attr for node in nodes),
        is_compute=any(node.is_compute for node in nodes),
        rng_sensitive=any(node.rng_sensitive for node in nodes),
        has_alias_metadata=any(node.has_alias_metadata for node in nodes),
        has_mutation_metadata=any(node.has_mutation_metadata for node in nodes),
    )


def _collapsed_module_type(nodes: list[RenderNode], module_path: str | None) -> str | None:
    if module_path is None:
        return nodes[0].module_type
    if all(node.module_path == module_path for node in nodes):
        return nodes[0].module_type
    return None


def _graph_module_type(graph: RenderGraph, module_path: str | None) -> str | None:
    if module_path is None:
        return None
    module_types = graph.metadata.get("module_types", {})
    if isinstance(module_types, dict):
        value = module_types.get(module_path)
        if isinstance(value, str):
            return value
    return None


def _module_has_children(graph: RenderGraph, module_path: str) -> bool:
    child_counts = graph.metadata.get("module_child_counts", {})
    if isinstance(child_counts, dict):
        return int(child_counts.get(module_path, 0)) > 0
    return False


def _module_child_counts(plan: TracePlan) -> dict[str, int]:
    return {
        name: sum(1 for _child_name, _child in module.named_children())
        for name, module in plan.root_module.named_modules()
        if name
    }


def _module_type(plan: TracePlan, module_path: str | None) -> str | None:
    if module_path is None:
        return None
    try:
        module = plan.root_module.get_submodule(module_path)
    except AttributeError:
        return None
    return module.__class__.__name__


def _param_buffer_refs(
    plan: TracePlan,
    node: TraceNode,
) -> tuple[tuple[RenderParam, ...], tuple[RenderBuffer, ...]]:
    if node.module_path is None:
        return (
            tuple(
                RenderParam(name=ref.name, shape=ref.shape, trainable=ref.requires_grad)
                for ref in node.param_refs
            ),
            tuple(RenderBuffer(name=ref.name, shape=ref.shape) for ref in node.buffer_refs),
        )
    try:
        module = plan.root_module.get_submodule(node.module_path)
    except AttributeError:
        return (
            tuple(
                RenderParam(name=ref.name, shape=ref.shape, trainable=ref.requires_grad)
                for ref in node.param_refs
            ),
            tuple(RenderBuffer(name=ref.name, shape=ref.shape) for ref in node.buffer_refs),
        )
    param_refs = tuple(
        RenderParam(
            name=f"{node.module_path}.{name}",
            shape=tuple(int(dim) for dim in parameter.shape),
            trainable=bool(parameter.requires_grad),
        )
        for name, parameter in module.named_parameters(recurse=False)
    )
    buffer_refs = tuple(
        RenderBuffer(
            name=f"{node.module_path}.{name}",
            shape=tuple(int(dim) for dim in buffer.shape),
        )
        for name, buffer in module.named_buffers(recurse=False)
    )
    return param_refs, buffer_refs


def _dedupe_params(nodes: list[RenderNode]) -> tuple[RenderParam, ...]:
    refs: dict[str, RenderParam] = {}
    for node in nodes:
        for ref in node.param_refs:
            refs.setdefault(ref.name, ref)
    return tuple(refs.values())


def _dedupe_buffers(nodes: list[RenderNode]) -> tuple[RenderBuffer, ...]:
    refs: dict[str, RenderBuffer] = {}
    for node in nodes:
        for ref in node.buffer_refs:
            refs.setdefault(ref.name, ref)
    return tuple(refs.values())


def _symbolic_shape(shape: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    if shape is None:
        return None
    return tuple(shape)
