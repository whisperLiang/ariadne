"""Deterministic visualization styling helpers."""

from __future__ import annotations

import re

from ariadne.visualization.graph_model import RenderEdge, RenderNode


def node_label(
    node: RenderNode,
    show_tensor_meta: bool,
    show_param_refs: bool,
    show_operation_targets: bool = False,
    show_debug_markers: bool = False,
    show_node_indices: bool = False,
) -> str:
    parts = [_title_line(node, show_node_indices=show_node_indices)]
    if node.is_compute and show_operation_targets:
        parts.append(f"op: {_friendly_target(node.target)}")
    if show_debug_markers:
        markers: list[str] = []
        if node.rng_sensitive:
            markers.append("rng")
        if node.has_mutation_metadata:
            markers.append("mutation")
        if node.has_alias_metadata:
            markers.append("alias")
        if markers:
            parts.append(f"[!] {', '.join(markers)}")
    if show_operation_targets and node.module_path is not None:
        parts.append(_trace_node_label(node))
    if show_tensor_meta and node.symbolic_shape is not None:
        parts.append(_tensor_meta_line(node))
    if show_param_refs:
        param_line = _param_line(node)
        if param_line:
            parts.append(param_line)
    return "\n".join(parts)


def node_attrs(node: RenderNode, split_role: str | None = None) -> dict[str, str]:
    attrs = {
        "shape": _node_shape(node),
        "style": "filled",
        "fontname": "Helvetica",
        "fontsize": "10",
        "margin": "0.08,0.04",
        "width": "2.15",
        "height": "0.62",
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
    if node.module_path is not None:
        return node.module_path
    if node.is_compute:
        return _friendly_target(node.target)
    return node.name


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


def _title_line(node: RenderNode, *, show_node_indices: bool) -> str:
    display_name = _display_name(node)
    display_kind = _display_kind(node)
    if node.is_compute and display_name == display_kind:
        title = display_kind
    else:
        title = f"{display_name}: {display_kind}"
    if show_node_indices:
        return f"{title}  #{node.index}"
    return title


def _friendly_target(target: str) -> str:
    friendly = target.removeprefix("aten.")
    friendly = friendly.removesuffix(".default")
    friendly = friendly.split(".", 1)[0]
    return friendly.removesuffix("_")


def _node_type(node: RenderNode) -> str:
    if node.is_placeholder:
        return "placeholder"
    if node.is_output:
        return "output"
    if node.is_attr:
        return "get_attr"
    if node.op == "module":
        return "module"
    return "compute"


def _node_shape(node: RenderNode) -> str:
    if node.op == "module":
        return "box3d"
    if node.is_compute and node.module_path is None:
        return "ellipse"
    return "box"


def _trace_node_label(node: RenderNode) -> str:
    if len(node.source_names) == 1:
        return f"trace node: {node.source_names[0]}"
    return f"trace nodes: {len(node.source_names)}"


def _param_line(node: RenderNode) -> str:
    if not node.param_refs:
        return ""
    if len(node.param_refs) > 3:
        return _param_summary_line(node)
    parts = []
    for ref in node.param_refs:
        wrapper = ("(", ")") if ref.trainable else ("[", "]")
        name = _short_ref_name(node, ref.name)
        shape = _format_shape(ref.shape)
        parts.append(f"{name}{wrapper[0]}{shape}{wrapper[1]}")
    return "params: " + ", ".join(parts)


def _param_summary_line(node: RenderNode) -> str:
    total = sum(_numel(ref.shape) for ref in node.param_refs)
    trainable = sum(_numel(ref.shape) for ref in node.param_refs if ref.trainable)
    if total == trainable:
        return f"params: {_format_count(total)}"
    if trainable == 0:
        return f"params: {_format_count(total)} frozen"
    return f"params: {_format_count(total)} ({_format_count(trainable)} trainable)"


def _tensor_meta_line(node: RenderNode) -> str:
    shape = ", ".join(str(dim) for dim in node.symbolic_shape or ())
    if node.nbytes is None:
        return f"({shape})"
    return f"({shape}) | {_format_mb(node.nbytes)}"


def _short_ref_name(node: RenderNode, name: str) -> str:
    if node.module_path is not None:
        prefix = f"{node.module_path}."
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name.rsplit(".", 1)[-1]


def _format_mb(nbytes: int) -> str:
    mb = nbytes / (1024 * 1024)
    if 0 < mb < 0.001:
        return "<0.001 MB"
    return f"{mb:.3f} MB"


def _format_shape(shape: tuple[object, ...]) -> str:
    if len(shape) > 1:
        return "x".join(str(dim) for dim in shape)
    if len(shape) == 1:
        return f"x{shape[0]}"
    return "x1"


def _numel(shape: tuple[object, ...]) -> int:
    total = 1
    for dim in shape:
        if not isinstance(dim, int):
            return 0
        total *= dim
    return total


def _format_count(count: int) -> str:
    if count >= 1_000_000:
        value = count / 1_000_000
        return f"{value:.2f}M"
    if count >= 1_000:
        value = count / 1_000
        return f"{value:.1f}K"
    return str(count)


def _apply_split_role(attrs: dict[str, str], split_role: str) -> None:
    if split_role == "boundary":
        attrs.update({"fillcolor": "#FEF3C7", "color": "#B45309", "peripheries": "2"})
    elif split_role == "prefix":
        attrs.update({"fillcolor": "#DBEAFE", "color": "#1D4ED8"})
    elif split_role == "suffix":
        attrs.update({"fillcolor": "#D1FAE5", "color": "#047857"})
    elif split_role == "passthrough":
        attrs.update({"fillcolor": "#EDE9FE", "color": "#7C3AED", "style": "dashed,filled"})
