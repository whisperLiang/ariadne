"""Graphviz rendering for Ariadne trace visualizations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ariadne.planner.frontier import SplitCandidate
from ariadne.trace.trace_plan import TracePlan
from ariadne.visualization.graph_model import RenderGraph, RenderNode
from ariadne.visualization.split_overlay import (
    SplitOverlay,
    build_split_overlay,
    classify_node_for_split,
)
from ariadne.visualization.styles import cluster_name, edge_attrs, node_attrs, node_label
from ariadne.visualization.trace_to_graph import (
    build_render_graph,
    prune_visual_auxiliary_nodes,
)


def render_trace_graph(
    plan: TracePlan,
    *,
    outpath: str = "ariadne_trace_graph",
    fileformat: str = "svg",
    save_only: bool = True,
    direction: str = "LR",
    max_module_depth: int | None = 3,
    show_tensor_meta: bool = True,
    show_param_refs: bool = True,
    return_dot: bool = False,
) -> str | None:
    graphviz = _import_graphviz()
    graph = prune_visual_auxiliary_nodes(build_render_graph(plan))
    dot = _build_dot(
        graphviz,
        graph,
        overlay=None,
        direction=direction,
        max_module_depth=max_module_depth,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=show_param_refs,
        show_cost=False,
    )
    if return_dot:
        return str(dot.source)
    dot.render(outpath, format=fileformat, cleanup=True)
    if not save_only:
        dot.view()
    return None


def render_split_graph(
    plan: TracePlan,
    candidate: SplitCandidate,
    *,
    outpath: str = "ariadne_split_graph",
    fileformat: str = "svg",
    save_only: bool = True,
    direction: str = "LR",
    max_module_depth: int | None = 3,
    show_tensor_meta: bool = True,
    show_cost: bool = True,
    return_dot: bool = False,
) -> str | None:
    graphviz = _import_graphviz()
    graph = prune_visual_auxiliary_nodes(build_render_graph(plan))
    overlay = build_split_overlay(candidate)
    dot = _build_dot(
        graphviz,
        graph,
        overlay=overlay,
        direction=direction,
        max_module_depth=max_module_depth,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=True,
        show_cost=show_cost,
    )
    if return_dot:
        return str(dot.source)
    dot.render(outpath, format=fileformat, cleanup=True)
    if not save_only:
        dot.view()
    return None


def _import_graphviz() -> Any:
    try:
        import graphviz
    except ImportError as error:
        raise ImportError(
            "Install Ariadne with visualization extras: "
            "pip install ariadne-split[visualization]"
        ) from error
    return graphviz


def _build_dot(
    graphviz: Any,
    graph: RenderGraph,
    *,
    overlay: SplitOverlay | None,
    direction: str,
    max_module_depth: int | None,
    show_tensor_meta: bool,
    show_param_refs: bool,
    show_cost: bool,
) -> Any:
    dot = graphviz.Digraph(name="AriadneTrace")
    dot.attr(
        "graph",
        rankdir=direction,
        labelloc="t",
        labeljust="l",
        label=_graph_label(graph, overlay=overlay, show_cost=show_cost),
    )
    dot.attr("node", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    clustered, root_nodes = _nodes_by_cluster(graph.nodes, max_module_depth=max_module_depth)
    for cluster_path in sorted(clustered):
        with dot.subgraph(name=cluster_name(cluster_path)) as cluster:
            cluster.attr(label=cluster_path, color="#D1D5DB", fontname="Helvetica")
            _add_nodes(
                cluster,
                clustered[cluster_path],
                overlay=overlay,
                show_tensor_meta=show_tensor_meta,
                show_param_refs=show_param_refs,
            )
    _add_nodes(
        dot,
        root_nodes,
        overlay=overlay,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=show_param_refs,
    )
    for edge in graph.edges:
        split_role = (
            classify_node_for_split(edge.target, overlay) if overlay is not None else None
        )
        dot.edge(edge.source, edge.target, **edge_attrs(edge, split_role=split_role))
    return dot


def _add_nodes(
    dot: Any,
    nodes: Iterable[RenderNode],
    *,
    overlay: SplitOverlay | None,
    show_tensor_meta: bool,
    show_param_refs: bool,
) -> None:
    for node in nodes:
        split_role = (
            classify_node_for_split(node.name, overlay) if overlay is not None else None
        )
        attrs = node_attrs(node, split_role=split_role)
        attrs["label"] = node_label(
            node,
            show_tensor_meta=show_tensor_meta,
            show_param_refs=show_param_refs,
        )
        dot.node(node.name, **attrs)


def _nodes_by_cluster(
    nodes: tuple[RenderNode, ...],
    *,
    max_module_depth: int | None,
) -> tuple[dict[str, list[RenderNode]], list[RenderNode]]:
    clustered: dict[str, list[RenderNode]] = defaultdict(list)
    root_nodes: list[RenderNode] = []
    for node in nodes:
        cluster_path = _cluster_path(node.module_path, max_module_depth=max_module_depth)
        if cluster_path is None:
            root_nodes.append(node)
        else:
            clustered[cluster_path].append(node)
    return dict(clustered), root_nodes


def _cluster_path(module_path: str | None, *, max_module_depth: int | None) -> str | None:
    if module_path is None:
        return None
    if max_module_depth is None:
        return module_path
    if max_module_depth <= 0:
        return None
    return ".".join(module_path.split(".")[:max_module_depth])


def _graph_label(
    graph: RenderGraph,
    *,
    overlay: SplitOverlay | None,
    show_cost: bool,
) -> str:
    parts = [
        f"Ariadne TracePlan {graph.graph_signature}",
        f"nodes={len(graph.nodes)} edges={len(graph.edges)}",
    ]
    if overlay is not None:
        parts.extend([f"split_id={overlay.split_id}", f"boundary_after={overlay.boundary_after}"])
        if show_cost:
            parts.extend(
                [
                    f"boundary_bytes={overlay.boundary_bytes}",
                    f"prefix_nodes={overlay.prefix_node_count}",
                    f"suffix_nodes={overlay.suffix_node_count}",
                    f"trainable_suffix={overlay.trainable_suffix}",
                ]
            )
    return "\n".join(parts)
