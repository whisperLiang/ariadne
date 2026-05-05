"""DOT and table exports for Ariadne visualization."""

from __future__ import annotations

from typing import Any

from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits
from ariadne.trace.trace_plan import TracePlan
from ariadne.visualization.graph_model import RenderGraph
from ariadne.visualization.split_overlay import (
    SplitOverlay,
    build_split_overlay,
    classify_node_for_split,
)
from ariadne.visualization.styles import edge_attrs, node_attrs, node_label
from ariadne.visualization.trace_to_graph import (
    build_render_graph,
    prune_visual_auxiliary_nodes,
)


def export_trace_dot(plan: TracePlan) -> str:
    return _render_dot(prune_visual_auxiliary_nodes(build_render_graph(plan)), overlay=None)


def export_split_dot(plan: TracePlan, candidate: SplitCandidate) -> str:
    return _render_dot(
        prune_visual_auxiliary_nodes(build_render_graph(plan)),
        overlay=build_split_overlay(candidate),
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


def _render_dot(graph: RenderGraph, overlay: SplitOverlay | None) -> str:
    label = _graph_label(graph, overlay)
    lines = [
        "digraph AriadneTrace {",
        '  graph [rankdir="LR", labelloc="t", labeljust="l", label='
        f'"{_escape_attr(label)}"];',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    for node in graph.nodes:
        split_role = (
            classify_node_for_split(node.name, overlay) if overlay is not None else None
        )
        attrs = node_attrs(node, split_role=split_role)
        attrs["label"] = node_label(
            node,
            show_tensor_meta=True,
            show_param_refs=True,
        )
        lines.append(f'  "{_escape_id(node.name)}" [{_attrs_to_dot(attrs)}];')
    for edge in graph.edges:
        split_role = (
            classify_node_for_split(edge.target, overlay) if overlay is not None else None
        )
        lines.append(
            f'  "{_escape_id(edge.source)}" -> "{_escape_id(edge.target)}" '
            f"[{_attrs_to_dot(edge_attrs(edge, split_role=split_role))}];"
        )
    lines.append("}")
    return "\n".join(lines)


def _graph_label(graph: RenderGraph, overlay: SplitOverlay | None) -> str:
    parts = [
        f"Ariadne TracePlan {graph.graph_signature}",
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
    return ", ".join(
        f'{key}="{_escape_attr(value)}"' for key, value in sorted(attrs.items())
    )


def _escape_id(value: str) -> str:
    return _escape_attr(value)


def _escape_attr(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
