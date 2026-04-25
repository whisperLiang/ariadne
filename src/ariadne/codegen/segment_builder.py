"""Generate prefix and suffix FX GraphModules from a split candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.fx import Graph, GraphModule, Node

from ariadne.codegen.interception_segments import (
    build_interception_prefix,
    build_interception_suffix,
)
from ariadne.planner.frontier import SplitCandidate
from ariadne.trace.interception import InterceptionTraceArtifact
from ariadne.trace.trace_plan import TracePlan


@dataclass(frozen=True)
class SegmentBundle:
    prefix: torch.nn.Module
    suffix: torch.nn.Module
    boundary_order: tuple[str, ...]
    passthrough_order: tuple[str, ...]


def build_segments(plan: TracePlan, candidate: SplitCandidate) -> SegmentBundle:
    """Build generated eager prefix/suffix callables as FX GraphModules."""
    if isinstance(plan.runtime_artifact, InterceptionTraceArtifact):
        return SegmentBundle(
            prefix=build_interception_prefix(
                root=plan.fx_graph_module,
                artifact=plan.runtime_artifact,
                op_names=candidate.prefix_nodes,
                raw_input_names=plan.input_node_names,
                boundary_order=candidate.boundary_nodes,
            ),
            suffix=build_interception_suffix(
                root=plan.fx_graph_module,
                artifact=plan.runtime_artifact,
                op_names=candidate.suffix_nodes,
                boundary_order=candidate.boundary_nodes,
                passthrough_order=candidate.passthrough_inputs,
            ),
            boundary_order=candidate.boundary_nodes,
            passthrough_order=candidate.passthrough_inputs,
        )
    graph_module = cast(GraphModule, plan.fx_graph_module)
    return SegmentBundle(
        prefix=_build_prefix(graph_module, candidate),
        suffix=_build_suffix(graph_module, candidate),
        boundary_order=candidate.boundary_nodes,
        passthrough_order=candidate.passthrough_inputs,
    )


def _build_prefix(graph_module: GraphModule, candidate: SplitCandidate) -> GraphModule:
    graph = Graph()
    env: dict[Node, Node] = {}
    original_nodes = _nodes_by_name(graph_module)
    required = _dependency_closure(
        [original_nodes[name] for name in candidate.boundary_nodes],
        stop_names=set(),
    )

    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            env[node] = graph.placeholder(node.name)
        elif node.name in required:
            env[node] = graph.node_copy(node, lambda original: env[original])

    outputs = tuple(env[original_nodes[name]] for name in candidate.boundary_nodes)
    graph.output(outputs)
    graph.lint()
    return GraphModule(graph_module, graph, class_name="PrefixSegment")


def _build_suffix(graph_module: GraphModule, candidate: SplitCandidate) -> GraphModule:
    graph = Graph()
    env: dict[Node, Node] = {}
    original_nodes = _nodes_by_name(graph_module)
    boundary_names = set(candidate.boundary_nodes)
    passthrough_names = set(candidate.passthrough_inputs)
    output_node = _output_node(graph_module)

    for label in candidate.boundary_nodes:
        env[original_nodes[label]] = graph.placeholder(label)
    for label in candidate.passthrough_inputs:
        env[original_nodes[label]] = graph.placeholder(label)

    required = _dependency_closure(
        output_node.all_input_nodes,
        stop_names=boundary_names | passthrough_names,
    )
    prefix_only = set(candidate.prefix_nodes) - boundary_names
    invalid = sorted(required & prefix_only)
    if invalid:
        raise ValueError(
            "Suffix depends on hidden prefix-only nodes that are not in the boundary: "
            + ", ".join(invalid)
        )

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.name in boundary_names or node.name in passthrough_names:
            continue
        if node.name in required:
            env[node] = graph.node_copy(node, lambda original: env[original])

    graph.output(_map_fx_arg(output_node.args[0], env))
    graph.lint()
    return GraphModule(graph_module, graph, class_name="SuffixSegment")


def _nodes_by_name(graph_module: GraphModule) -> dict[str, Node]:
    return {node.name: node for node in graph_module.graph.nodes}


def _output_node(graph_module: GraphModule) -> Node:
    for node in graph_module.graph.nodes:
        if node.op == "output":
            return cast(Node, node)
    raise ValueError("FX graph has no output node.")


def _dependency_closure(start_nodes: Any, stop_names: set[str]) -> set[str]:
    pending = [start_nodes] if isinstance(start_nodes, Node) else list(start_nodes)
    required: set[str] = set()
    while pending:
        node = pending.pop()
        if node.name in stop_names or node.name in required:
            continue
        required.add(node.name)
        pending.extend(node.all_input_nodes)
    return required


def _map_fx_arg(value: Any, env: dict[Node, Node]) -> Any:
    if isinstance(value, Node):
        return env[value]
    if isinstance(value, tuple):
        return tuple(_map_fx_arg(item, env) for item in value)
    if isinstance(value, list):
        return [_map_fx_arg(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _map_fx_arg(item, env) for key, item in value.items()}
    if isinstance(value, slice):
        return slice(
            _map_fx_arg(value.start, env),
            _map_fx_arg(value.stop, env),
            _map_fx_arg(value.step, env),
        )
    return value
