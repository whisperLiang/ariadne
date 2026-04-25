"""Symbolic trace plan data structures."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch

from ariadne.trace.tensor_meta import BufferRef, ParamRef, ShapeEnv, TensorMeta


@dataclass(frozen=True)
class TraceNode:
    """One node in Ariadne's lightweight symbolic trace IR."""

    name: str
    op: str
    target: str
    args_template: Any
    kwargs_template: Any
    parents: tuple[str, ...]
    tensor_meta: TensorMeta | None = None
    param_refs: tuple[ParamRef, ...] = ()
    buffer_refs: tuple[BufferRef, ...] = ()
    module_path: str | None = None
    alias_metadata: dict[str, Any] | None = None
    mutation_metadata: dict[str, Any] | None = None
    rng_sensitive: bool = False

    @property
    def is_placeholder(self) -> bool:
        return self.op == "placeholder"

    @property
    def is_output(self) -> bool:
        return self.op == "output"

    @property
    def is_attr(self) -> bool:
        return self.op == "get_attr"

    @property
    def is_compute(self) -> bool:
        return not (self.is_placeholder or self.is_output or self.is_attr)


@dataclass(frozen=True)
class TracePlan:
    """One observed forward execution path with symbolic batch metadata."""

    graph_signature: str
    nodes: tuple[TraceNode, ...]
    input_metas: tuple[TensorMeta | None, ...]
    output_metas: tuple[TensorMeta, ...]
    shape_env: ShapeEnv
    input_node_names: tuple[str, ...]
    output_template: Any
    fx_graph_module: torch.nn.Module
    runtime_artifact: Any | None = None

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes)

    def get_node(self, name: str) -> TraceNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"Trace node {name!r} does not exist.")

    def index_of(self, name: str) -> int:
        for index, node in enumerate(self.nodes):
            if node.name == name:
                return index
        raise KeyError(f"Trace node {name!r} does not exist.")


def compute_graph_signature(nodes: tuple[TraceNode, ...], shape_env: ShapeEnv) -> str:
    """Create a stable signature that intentionally excludes concrete batch size."""
    digest = sha256()
    digest.update(shape_env.batch_symbol.encode())
    digest.update(repr(shape_env.dynamic_batch).encode())
    for node in nodes:
        if node.is_output:
            continue
        digest.update(node.name.encode())
        digest.update(node.op.encode())
        digest.update(node.target.encode())
        digest.update(repr(node.parents).encode())
        if node.tensor_meta is not None:
            digest.update(repr(node.tensor_meta.symbolic_shape).encode())
            digest.update(node.tensor_meta.dtype.encode())
    return digest.hexdigest()[:16]
