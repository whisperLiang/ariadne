"""DOT and table exports for Ariadne visualization."""

from __future__ import annotations

from typing import Any

from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits
from ariadne.trace.trace_plan import TracePlan
from ariadne.visualization.graph_model import RenderGraph, RenderNode
from ariadne.visualization.split_overlay import SplitOverlay, build_split_overlay
from ariadne.visualization.styles import cluster_name, edge_attrs, node_attrs, node_label
from ariadne.visualization.trace_to_graph import (
    DEFAULT_MODULE_DEPTH,
    ViewDetail,
    build_visual_graph,
)


def export_trace_dot(
    plan: TracePlan,
    *,
    view_detail: ViewDetail = "module",
    max_module_depth: int | None = DEFAULT_MODULE_DEPTH,
    show_operation_targets: bool = False,
    show_debug_markers: bool = False,
    show_node_indices: bool = False,
    direction: str = "BT",
) -> str:
    return _render_dot(
        build_visual_graph(
            plan,
            view_detail=view_detail,
            max_module_depth=max_module_depth,
        ),
        overlay=None,
        max_module_depth=max_module_depth,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
        direction=direction,
    )


def export_split_dot(
    plan: TracePlan,
    candidate: SplitCandidate,
    *,
    view_detail: ViewDetail = "module",
    max_module_depth: int | None = DEFAULT_MODULE_DEPTH,
    show_operation_targets: bool = False,
    show_debug_markers: bool = False,
    show_node_indices: bool = False,
    direction: str = "BT",
) -> str:
    return _render_dot(
        build_visual_graph(
            plan,
            view_detail=view_detail,
            max_module_depth=max_module_depth,
        ),
        overlay=build_split_overlay(candidate),
        max_module_depth=max_module_depth,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
        direction=direction,
    )


def export_split_candidates_table(plan: TracePlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in enumerate_frontier_splits(plan):
        rows.append(
            {
                "split_id": candidate.split_id,
                "boundary_after": candidate.boundary_after,
                "boundary_nodes": candidate.boundary_nodes,
                "boundary_bytes": candidate.cost.boundary_bytes,
                "prefix_node_count": candidate.cost.prefix_node_count,
                "suffix_node_count": candidate.cost.suffix_node_count,
                "trainable_suffix": candidate.trainable_suffix,
                "passthrough_inputs": candidate.passthrough_inputs,
            }
        )
    return rows


def _render_dot(
    graph: RenderGraph,
    overlay: SplitOverlay | None,
    *,
    max_module_depth: int | None,
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
    direction: str,
) -> str:
    label = _graph_label(graph, overlay)
    lines = [
        "digraph AriadneTrace {",
        f'  graph [rankdir="{_escape_attr(direction)}", labelloc="t", labeljust="l", '
        f'label="{_escape_attr(label)}", ordering="out", compound="true", '
        'newrank="true", bgcolor="white", nodesep="0.28", ranksep="0.55", '
        'splines="ortho"];',
        '  node [fontname="Helvetica", ordering="out"];',
        '  edge [fontname="Helvetica"];',
    ]

    clustered, root_nodes = _nodes_by_cluster(graph, max_module_depth=max_module_depth)
    cluster_tree = _cluster_tree(clustered)
    for cluster_path in _ordered_cluster_paths(clustered):
        lines.extend(
            _cluster_to_dot_lines(
                graph,
                cluster_path,
                cluster_tree,
                clustered,
                overlay,
                show_operation_targets,
                show_debug_markers,
                show_node_indices,
                indent=2,
            )
        )
    for node in root_nodes:
        node_dot = _node_to_dot(
            node,
            overlay,
            show_operation_targets,
            show_debug_markers,
            show_node_indices,
        )
        lines.append(f"  {node_dot}")

    nodes_by_name = {node.name: node for node in graph.nodes}
    for edge in graph.edges:
        split_role = _split_role_for_node(nodes_by_name[edge.target], overlay)
        lines.append(
            f'  "{_escape_id(edge.source)}" -> "{_escape_id(edge.target)}" '
            f"[{_attrs_to_dot(edge_attrs(edge, split_role=split_role))}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _cluster_to_dot_lines(
    graph: RenderGraph,
    cluster_path: str,
    cluster_tree: dict[str | None, list[str]],
    clustered: dict[str, list[RenderNode]],
    overlay: SplitOverlay | None,
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
    *,
    indent: int,
) -> list[str]:
    prefix = " " * indent
    child_prefix = " " * (indent + 2)
    max_depth = _max_cluster_depth(cluster_tree)
    lines = [f'{prefix}subgraph "{cluster_name(cluster_path)}" {{']
    for key, value in _cluster_attrs(graph, cluster_path, max_depth=max_depth).items():
        lines.append(f'{child_prefix}{key}="{_escape_attr(value)}";')
    for child_path in cluster_tree.get(cluster_path, []):
        lines.extend(
            _cluster_to_dot_lines(
                graph,
                child_path,
                cluster_tree,
                clustered,
                overlay,
                show_operation_targets,
                show_debug_markers,
                show_node_indices,
                indent=indent + 2,
            )
        )
    for node in clustered.get(cluster_path, []):
        node_dot = _node_to_dot(
            node,
            overlay,
            show_operation_targets,
            show_debug_markers,
            show_node_indices,
        )
        lines.append(f"{child_prefix}{node_dot}")
    lines.append(f"{prefix}}}")
    return lines


def _node_to_dot(
    node: RenderNode,
    overlay: SplitOverlay | None,
    show_operation_targets: bool,
    show_debug_markers: bool,
    show_node_indices: bool,
) -> str:
    split_role = _split_role_for_node(node, overlay)
    attrs = node_attrs(node, split_role=split_role)
    attrs["label"] = node_label(
        node,
        show_tensor_meta=True,
        show_param_refs=True,
        show_operation_targets=show_operation_targets,
        show_debug_markers=show_debug_markers,
        show_node_indices=show_node_indices,
    )
    return f'"{_escape_id(node.name)}" [{_attrs_to_dot(attrs)}];'


def _nodes_by_cluster(
    graph: RenderGraph,
    *,
    max_module_depth: int | None,
) -> tuple[dict[str, list[RenderNode]], list[RenderNode]]:
    clustered: dict[str, list[RenderNode]] = {}
    root_nodes: list[RenderNode] = []
    cluster_by_parent = graph.metadata.get("view_detail") == "module"
    for node in graph.nodes:
        cluster_path = _cluster_path(
            node,
            max_module_depth=max_module_depth,
            cluster_by_parent=cluster_by_parent,
        )
        if cluster_path is None:
            root_nodes.append(node)
        else:
            clustered.setdefault(cluster_path, []).append(node)
    return clustered, root_nodes


def _ordered_cluster_paths(clustered: dict[str, list[RenderNode]]) -> list[str]:
    return _cluster_tree(clustered).get(None, [])


def _cluster_tree(clustered: dict[str, list[RenderNode]]) -> dict[str | None, list[str]]:
    all_paths: set[str] = set()
    for cluster_path in clustered:
        parts = cluster_path.split(".")
        for depth in range(1, len(parts) + 1):
            all_paths.add(".".join(parts[:depth]))

    tree: dict[str | None, list[str]] = {}
    for cluster_path in all_paths:
        parent = cluster_path.rsplit(".", 1)[0] if "." in cluster_path else None
        tree.setdefault(parent, []).append(cluster_path)

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


def _graph_label(graph: RenderGraph, overlay: SplitOverlay | None) -> str:
    parts = [
        f"Ariadne TracePlan {graph.graph_signature}",
        f"view={graph.metadata.get('view_detail', 'operation')}",
        f"nodes={len(graph.nodes)} edges={len(graph.edges)}",
    ]
    if overlay is not None:
        parts.extend(
            [
                f"split_id={overlay.split_id}",
                f"boundary_after={overlay.boundary_after}",
                f"boundary_bytes={overlay.boundary_bytes}",
                f"prefix_nodes={overlay.prefix_node_count}",
                f"suffix_nodes={overlay.suffix_node_count}",
                f"trainable_suffix={overlay.trainable_suffix}",
            ]
        )
    return "\n".join(parts)


def _attrs_to_dot(attrs: dict[str, str]) -> str:
    return ", ".join(f'{key}="{_escape_attr(value)}"' for key, value in sorted(attrs.items()))


def _escape_id(value: str) -> str:
    return _escape_attr(value)


def _escape_attr(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
