"""Deterministic visualization styling helpers."""

from __future__ import annotations

import re

from ariadne.visualization.graph_model import RenderEdge, RenderNode


def node_label(node: RenderNode, show_tensor_meta: bool, show_param_refs: bool) -> str:
    parts = [f"{_display_name(node)}  #{node.index}", _display_kind(node)]
    if node.is_compute:
        parts.append(f"aten: {node.target}")
    markers: list[str] = []
    if node.rng_sensitive:
        markers.append("rng")
    if node.has_mutation_metadata:
        markers.append("mutation")
    if node.has_alias_metadata:
        markers.append("alias")
    if markers:
        parts.append(f"[!] {', '.join(markers)}")
    if node.module_path is not None:
        parts.append(f"trace node: {node.name}")
    if show_tensor_meta and node.symbolic_shape is not None:
        shape = ", ".join(str(dim) for dim in node.symbolic_shape)
        meta = f"shape: ({shape})"
        if node.dtype is not None:
            meta = f"{meta}\ndtype: {node.dtype}"
        if node.nbytes is not None:
            meta = f"{meta}\nnbytes: {node.nbytes}"
        parts.append(meta)
    if show_param_refs and (node.param_count or node.buffer_count):
        parts.append(f"params: {node.param_count}\nbuffers: {node.buffer_count}")
    return "\n".join(parts)


def node_attrs(node: RenderNode, split_role: str | None = None) -> dict[str, str]:
    attrs = {
        "shape": "box",
        "style": "filled",
        "fontname": "Helvetica",
        "fontsize": "10",
        "color": "#9CA3AF",
        "fillcolor": "#FFFFFF",
        "ariadne_node_type": _node_type(node),
    }
    if node.is_placeholder:
        attrs.update({"style": "rounded,filled", "fillcolor": "#E0F2FE", "color": "#0284C7"})
    elif node.is_output:
        attrs.update(
            {
                "style": "rounded,filled,bold",
                "fillcolor": "#FCE7F3",
                "color": "#BE185D",
                "penwidth": "2",
            }
        )
    elif node.is_attr:
        attrs.update({"fillcolor": "#F3F4F6", "color": "#6B7280"})
    else:
        attrs.update({"fillcolor": "#FFFFFF", "color": "#4B5563"})

    if split_role is not None:
        attrs["ariadne_split_role"] = split_role
        _apply_split_role(attrs, split_role)
    return attrs


def edge_attrs(edge: RenderEdge, split_role: str | None = None) -> dict[str, str]:
    attrs = {
        "fontname": "Helvetica",
        "fontsize": "9",
        "color": "#6B7280",
        "arrowsize": "0.7",
        "ariadne_edge_kind": edge.kind,
    }
    if edge.label is not None:
        attrs["label"] = edge.label
    if split_role is not None:
        attrs["ariadne_split_role"] = split_role
        if split_role == "boundary":
            attrs.update({"color": "#B45309", "penwidth": "2"})
        elif split_role == "suffix":
            attrs["color"] = "#047857"
        elif split_role == "prefix":
            attrs["color"] = "#1D4ED8"
        elif split_role == "passthrough":
            attrs.update({"style": "dashed", "color": "#7C3AED"})
    return attrs


def cluster_name(module_path: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", module_path).strip("_")
    if not sanitized:
        sanitized = "root"
    if sanitized[0].isdigit():
        sanitized = f"module_{sanitized}"
    return f"cluster_{sanitized}"


def _display_name(node: RenderNode) -> str:
    return node.module_path or node.name


def _display_kind(node: RenderNode) -> str:
    if node.is_placeholder:
        return "input"
    if node.is_output:
        return "output"
    if node.is_attr:
        return "attribute"
    if node.module_type is not None:
        return node.module_type
    return _friendly_target(node.target)


def _friendly_target(target: str) -> str:
    return target.removesuffix(".default")


def _node_type(node: RenderNode) -> str:
    if node.is_placeholder:
        return "placeholder"
    if node.is_output:
        return "output"
    if node.is_attr:
        return "get_attr"
    return "compute"


def _apply_split_role(attrs: dict[str, str], split_role: str) -> None:
    if split_role == "boundary":
        attrs.update({"fillcolor": "#FEF3C7", "color": "#B45309", "peripheries": "2"})
    elif split_role == "prefix":
        attrs.update({"fillcolor": "#DBEAFE", "color": "#1D4ED8"})
    elif split_role == "suffix":
        attrs.update({"fillcolor": "#D1FAE5", "color": "#047857"})
    elif split_role == "passthrough":
        attrs.update({"fillcolor": "#EDE9FE", "color": "#7C3AED", "style": "dashed,filled"})
