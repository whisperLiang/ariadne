"""Convert Ariadne TracePlan objects into render-only graph models."""

from __future__ import annotations

from typing import Any

from ariadne.trace.trace_plan import TraceNode, TracePlan
from ariadne.visualization.graph_model import RenderEdge, RenderGraph, RenderNode

_VISUAL_AUXILIARY_TARGETS = frozenset({"detach.default", "empty.memory_format"})
_VISUAL_PARAMETER_PREP_TARGETS = frozenset({"t.default"})
_NodeGroupKey = tuple[str, tuple[str, ...], str | None]


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
    return RenderNode(
        name=node.name,
        index=index,
        op=node.op,
        target=node.target,
        parents=node.parents,
        module_path=node.module_path,
        module_type=_module_type(plan, node.module_path),
        symbolic_shape=_symbolic_shape(tensor_meta.symbolic_shape if tensor_meta else None),
        dtype=tensor_meta.dtype if tensor_meta else None,
        nbytes=tensor_meta.nbytes if tensor_meta else None,
        param_count=len(node.param_refs),
        buffer_count=len(node.buffer_refs),
        is_placeholder=node.is_placeholder,
        is_output=node.is_output,
        is_attr=node.is_attr,
        is_compute=node.is_compute,
        rng_sensitive=node.rng_sensitive,
        has_alias_metadata=node.alias_metadata is not None,
        has_mutation_metadata=node.mutation_metadata is not None,
    )


def _module_type(plan: TracePlan, module_path: str | None) -> str | None:
    if module_path is None:
        return None
    try:
        module = plan.root_module.get_submodule(module_path)
    except AttributeError:
        return None
    return module.__class__.__name__


def _symbolic_shape(shape: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    if shape is None:
        return None
    return tuple(shape)
