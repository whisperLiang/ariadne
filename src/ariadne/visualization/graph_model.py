"""Render-only graph data structures for Ariadne visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RenderNode:
    name: str
    index: int
    op: str
    target: str
    parents: tuple[str, ...]
    module_path: str | None
    module_type: str | None
    symbolic_shape: tuple[Any, ...] | None
    dtype: str | None
    nbytes: int | None
    param_count: int
    buffer_count: int
    is_placeholder: bool
    is_output: bool
    is_attr: bool
    is_compute: bool
    rng_sensitive: bool
    has_alias_metadata: bool
    has_mutation_metadata: bool


@dataclass(frozen=True)
class RenderEdge:
    source: str
    target: str
    label: str | None = None
    kind: str = "data"


@dataclass(frozen=True)
class RenderGraph:
    graph_signature: str
    nodes: tuple[RenderNode, ...]
    edges: tuple[RenderEdge, ...]
    input_node_names: tuple[str, ...]
    output_node_names: tuple[str, ...]
    metadata: dict[str, Any]
