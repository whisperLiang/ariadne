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
)
from ariadne.visualization.styles import cluster_name, edge_attrs, node_attrs, node_label
from ariadne.visualization.trace_to_graph import (
    DEFAULT_MODULE_DEPTH,
    ViewDetail,
    build_visual_graph,
)


def render_trace_graph(
    plan: TracePlan,
    *,
    outpath: str = "ariadne_trace_graph",
    fileformat: str = "svg",
    save_only: bool = True,
    direction: str = "BT",
    max_module_depth: int | None = DEFAULT_MODULE_DEPTH,
    view_detail: ViewDetail = "module",
    show_tensor_meta: bool = True,
    show_param_refs: bool = True,
    show_operation_targets: bool = False,
    show_debug_markers: bool = False,
    show_node_indices: bool = False,
    return_dot: bool = False,
) -> str | None:
    graphviz = _import_graphviz()
    graph = build_visual_graph(
        plan,
        view_detail=view_detail,
        max_module_depth=max_module_depth,
    )
    dot = _build_dot(
        graphviz,
        graph,
        overlay=None,
        direction=direction,
        max_module_depth=max_module_depth,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=show_param_refs,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
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
    direction: str = "BT",
    max_module_depth: int | None = DEFAULT_MODULE_DEPTH,
    view_detail: ViewDetail = "module",
    show_tensor_meta: bool = True,
    show_operation_targets: bool = False,
    show_debug_markers: bool = False,
    show_node_indices: bool = False,
    show_cost: bool = True,
    return_dot: bool = False,
) -> str | None:
    graphviz = _import_graphviz()
    graph = build_visual_graph(
        plan,
        view_detail=view_detail,
        max_module_depth=max_module_depth,
    )
    overlay = build_split_overlay(candidate)
    dot = _build_dot(
        graphviz,
        graph,
        overlay=overlay,
        direction=direction,
        max_module_depth=max_module_depth,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=True,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
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
            "Install Ariadne with visualization extras: pip install ariadne-split[visualization]"
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
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
    show_cost: bool,
) -> Any:
    dot = graphviz.Digraph(name="AriadneTrace")
    dot.attr(
        "graph",
        rankdir=direction,
        labelloc="t",
        labeljust="l",
        label=_graph_label(graph, overlay=overlay, show_cost=show_cost),
        ordering="out",
        compound="true",
        newrank="true",
        bgcolor="white",
        nodesep="0.28",
        ranksep="0.55",
        splines="ortho",
    )
    dot.attr("node", fontname="Helvetica", ordering="out")
    dot.attr("edge", fontname="Helvetica")

    clustered, root_nodes = _nodes_by_cluster(
        graph.nodes,
        max_module_depth=max_module_depth,
        cluster_by_parent=graph.metadata.get("view_detail") == "module",
    )
    cluster_tree = _cluster_tree(clustered)
    for cluster_path in cluster_tree.get(None, []):
        _add_cluster(
            dot,
            graph,
            cluster_path,
            cluster_tree,
            clustered,
            overlay=overlay,
            show_tensor_meta=show_tensor_meta,
            show_param_refs=show_param_refs,
            show_operation_targets=show_operation_targets,
            show_debug_markers=show_debug_markers,
            show_node_indices=show_node_indices,
        )
    _add_nodes(
        dot,
        root_nodes,
        overlay=overlay,
        show_tensor_meta=show_tensor_meta,
        show_param_refs=show_param_refs,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
    )
    nodes_by_name = {node.name: node for node in graph.nodes}
    for edge in graph.edges:
        split_role = _split_role_for_node(nodes_by_name[edge.target], overlay)
        dot.edge(edge.source, edge.target, **edge_attrs(edge, split_role=split_role))
    return dot


def _add_cluster(
    dot: Any,
    graph: RenderGraph,
    cluster_path: str,
    cluster_tree: dict[str | None, list[str]],
    clustered: dict[str, list[RenderNode]],
    *,
    overlay: SplitOverlay | None,
    show_tensor_meta: bool,
    show_param_refs: bool,
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
) -> None:
    max_depth = _max_cluster_depth(cluster_tree)
    with dot.subgraph(name=cluster_name(cluster_path)) as cluster:
        cluster.attr(**_cluster_attrs(graph, cluster_path, max_depth=max_depth))
        for child_path in cluster_tree.get(cluster_path, []):
            _add_cluster(
                cluster,
                graph,
                child_path,
                cluster_tree,
                clustered,
                overlay=overlay,
                show_tensor_meta=show_tensor_meta,
                show_param_refs=show_param_refs,
                show_operation_targets=show_operation_targets,
                show_debug_markers=show_debug_markers,
                show_node_indices=show_node_indices,
            )
        _add_nodes(
            cluster,
            clustered.get(cluster_path, []),
            overlay=overlay,
            show_tensor_meta=show_tensor_meta,
            show_param_refs=show_param_refs,
            show_operation_targets=show_operation_targets,
            show_debug_markers=show_debug_markers,
            show_node_indices=show_node_indices,
        )


def _add_nodes(
    dot: Any,
    nodes: Iterable[RenderNode],
    *,
    overlay: SplitOverlay | None,
    show_tensor_meta: bool,
    show_param_refs: bool,
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
) -> None:
    for node in nodes:
        split_role = _split_role_for_node(node, overlay)
        attrs = node_attrs(node, split_role=split_role)
        attrs["label"] = node_label(
            node,
            show_tensor_meta=show_tensor_meta,
            show_param_refs=show_param_refs,
            show_operation_targets=show_operation_targets,
            show_debug_markers=show_debug_markers,
            show_node_indices=show_node_indices,
        )
        dot.node(node.name, **attrs)


def _nodes_by_cluster(
    nodes: tuple[RenderNode, ...],
    *,
    max_module_depth: int | None,
    cluster_by_parent: bool,
) -> tuple[dict[str, list[RenderNode]], list[RenderNode]]:
    clustered: dict[str, list[RenderNode]] = defaultdict(list)
    root_nodes: list[RenderNode] = []
    for node in nodes:
        cluster_path = _cluster_path(
            node,
            max_module_depth=max_module_depth,
            cluster_by_parent=cluster_by_parent,
        )
        if cluster_path is None:
            root_nodes.append(node)
        else:
            clustered[cluster_path].append(node)
    return dict(clustered), root_nodes


def _cluster_tree(clustered: dict[str, list[RenderNode]]) -> dict[str | None, list[str]]:
    all_paths: set[str] = set()
    for cluster_path in clustered:
        parts = cluster_path.split(".")
        for depth in range(1, len(parts) + 1):
            all_paths.add(".".join(parts[:depth]))

    tree: dict[str | None, list[str]] = defaultdict(list)
    for cluster_path in all_paths:
        parent = cluster_path.rsplit(".", 1)[0] if "." in cluster_path else None
        tree[parent].append(cluster_path)

    return {
        parent: sorted(children, key=lambda path: (_cluster_min_index(path, clustered), path))
        for parent, children in tree.items()
    }


def _cluster_min_index(cluster_path: str, clustered: dict[str, list[RenderNode]]) -> int:
    prefix = f"{cluster_path}."
    indices = [
        node.index
        for path, nodes in clustered.items()
        if path == cluster_path or path.startswith(prefix)
        for node in nodes
    ]
    return min(indices) if indices else 0


def _max_cluster_depth(cluster_tree: dict[str | None, list[str]]) -> int:
    paths = [path for path in cluster_tree if path is not None]
    paths.extend(child for children in cluster_tree.values() for child in children)
    if not paths:
        return 0
    return max(len(path.split(".")) - 1 for path in paths)


def _cluster_attrs(
    graph: RenderGraph,
    cluster_path: str,
    *,
    max_depth: int,
) -> dict[str, str]:
    return {
        "label": _cluster_label(graph, cluster_path),
        "color": "#CBD5E1",
        "fillcolor": "#FAFAFA",
        "fontcolor": "#475569",
        "fontname": "Helvetica",
        "fontsize": "10",
        "labeljust": "l",
        "labelloc": "b",
        "margin": "8",
        "penwidth": _format_penwidth(_cluster_penwidth(cluster_path, max_depth)),
        "style": "rounded,filled",
    }


def _cluster_label(graph: RenderGraph, cluster_path: str) -> str:
    module_types = graph.metadata.get("module_types", {})
    module_type = module_types.get(cluster_path) if isinstance(module_types, dict) else None
    if isinstance(module_type, str):
        return f"{cluster_path}\n({module_type})"
    return cluster_path


def _cluster_penwidth(cluster_path: str, max_depth: int) -> float:
    depth = len(cluster_path.split(".")) - 1
    if max_depth <= 0:
        return 2.4
    depth_fraction = (max_depth - depth) / max_depth
    return 1.4 + max(0.0, min(1.0, depth_fraction)) * 1.0


def _format_penwidth(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _cluster_path(
    node: RenderNode,
    *,
    max_module_depth: int | None,
    cluster_by_parent: bool,
) -> str | None:
    module_path = node.module_path
    if module_path is None:
        return None
    parts = module_path.split(".")
    if cluster_by_parent and _node_represents_module(node):
        if len(parts) <= 1:
            return None
        parts = parts[:-1]
    if max_module_depth is not None:
        if max_module_depth <= 0:
            return None
        parts = parts[:max_module_depth]
    return ".".join(parts)


def _node_represents_module(node: RenderNode) -> bool:
    return node.name.startswith("module__")


def _split_role_for_node(node: RenderNode, overlay: SplitOverlay | None) -> str | None:
    if overlay is None:
        return None
    source_names = set(node.source_names)
    if source_names & overlay.boundary_nodes:
        return "boundary"
    has_prefix = bool(source_names & overlay.prefix_nodes)
    has_suffix = bool(source_names & overlay.suffix_nodes)
    if has_prefix and has_suffix:
        return "boundary"
    if has_prefix:
        return "prefix"
    if has_suffix:
        return "suffix"
    if source_names & overlay.passthrough_inputs:
        return "passthrough"
    return "neutral"


def _graph_label(
    graph: RenderGraph,
    *,
    overlay: SplitOverlay | None,
    show_cost: bool,
) -> str:
    parts = [
        f"Ariadne TracePlan {graph.graph_signature}",
        f"view={graph.metadata.get('view_detail', 'operation')}",
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
