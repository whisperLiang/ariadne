"""FX-backed lightweight tracing.

The first milestone uses ``torch.fx`` to capture common module, method, and
function graphs as a maintainable substrate for Ariadne's TracePlan. It keeps
the default metadata lean and leaves richer runtime interception as a later
trace backend, without changing the planner/runtime boundary.
"""

from __future__ import annotations

import builtins
import operator
from collections.abc import Sequence
from typing import Any

import torch
from torch.fx import GraphModule, Node, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.shape_prop import TensorMetadata as FxTensorMetadata

from ariadne.trace.interception import trace_model_interception
from ariadne.trace.tensor_meta import (
    BufferRef,
    ParamRef,
    ShapeEnv,
    ShapeExpr,
    TensorMeta,
    buffer_ref_from_tensor,
    param_ref_from_parameter,
    tensor_meta_from_tensor,
)
from ariadne.trace.trace_plan import TraceNode, TracePlan, compute_graph_signature

_SHAPE_METHODS = {"view", "reshape", "flatten", "unflatten", "expand", "repeat"}
_SHAPE_FUNCTION_NAMES = {
    "view",
    "reshape",
    "flatten",
    "unflatten",
    "expand",
    "repeat",
    "zeros",
    "ones",
    "empty",
    "randn",
    "arange",
}
_RNG_FUNCTION_NAMES = {"rand", "randn", "randint", "normal", "bernoulli", "dropout"}


def trace_model(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = None,
) -> TracePlan:
    """Trace a model with runtime interception."""
    return trace_model_interception(
        model,
        example_inputs=tuple(example_inputs),
        batch_symbol=batch_symbol,
        dynamic_batch=dynamic_batch,
    )


def trace_model_fx(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = None,
) -> TracePlan:
    """Trace a model and build Ariadne's lightweight symbolic TracePlan."""
    inputs = tuple(example_inputs)
    traced_batch_size = _infer_traced_batch_size(inputs)
    shape_env = ShapeEnv(
        batch_symbol=batch_symbol,
        traced_batch_size=traced_batch_size,
        dynamic_batch=dynamic_batch,
    )
    graph_module = symbolic_trace(model)
    _rewrite_batch_size_constants(graph_module, inputs, traced_batch_size)
    ShapeProp(graph_module).propagate(*inputs)

    nodes = tuple(
        _trace_node_from_fx(node, graph_module, shape_env) for node in graph_module.graph.nodes
    )
    input_metas = tuple(_meta_from_input(value, shape_env) for value in inputs)
    output_metas = tuple(
        meta
        for node in nodes
        if node.op == "output"
        for meta in _tensor_metas_from_template(node.tensor_meta)
    )
    plan = TracePlan(
        graph_signature=compute_graph_signature(nodes, shape_env),
        nodes=nodes,
        input_metas=input_metas,
        output_metas=output_metas,
        shape_env=shape_env,
        input_node_names=tuple(node.name for node in nodes if node.op == "placeholder"),
        output_template=_output_template(graph_module),
        fx_graph_module=graph_module,
        runtime_artifact=None,
    )
    return plan


def _infer_traced_batch_size(inputs: tuple[Any, ...]) -> int | None:
    for value in inputs:
        tensor = _first_tensor(value)
        if tensor is not None and tensor.ndim > 0:
            return int(tensor.shape[0])
    return None


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _rewrite_batch_size_constants(
    graph_module: GraphModule,
    inputs: tuple[Any, ...],
    traced_batch_size: int | None,
) -> None:
    if traced_batch_size is None:
        return
    graph = graph_module.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    if not placeholders:
        return
    fallback_source = placeholders[0]
    changed = False

    for node in list(graph.nodes):
        if not _is_shape_sensitive(node):
            continue
        source = _first_input_node(node) or fallback_source
        with graph.inserting_before(node):
            shape_node = graph.call_function(builtins.getattr, args=(source, "shape"))
            batch_node = graph.call_function(operator.getitem, args=(shape_node, 0))
        new_args = _replace_batch_constants(node.args, traced_batch_size, batch_node)
        new_kwargs = _replace_batch_constants(node.kwargs, traced_batch_size, batch_node)
        if new_args != node.args or new_kwargs != node.kwargs:
            node.args = new_args
            node.kwargs = new_kwargs
            changed = True
        else:
            graph.erase_node(batch_node)
            graph.erase_node(shape_node)

    if changed:
        graph.lint()
        graph_module.recompile()


def _is_shape_sensitive(node: Node) -> bool:
    if node.op == "call_method" and str(node.target) in _SHAPE_METHODS:
        return True
    return node.op == "call_function" and _target_name(node.target) in _SHAPE_FUNCTION_NAMES


def _first_input_node(node: Node) -> Node | None:
    for value in node.args:
        if isinstance(value, Node):
            return value
    return None


def _replace_batch_constants(value: Any, traced_batch_size: int, batch_node: Node) -> Any:
    if isinstance(value, int) and value == traced_batch_size:
        return batch_node
    if isinstance(value, tuple):
        return tuple(
            _replace_batch_constants(item, traced_batch_size, batch_node) for item in value
        )
    if isinstance(value, list):
        return [_replace_batch_constants(item, traced_batch_size, batch_node) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_batch_constants(item, traced_batch_size, batch_node)
            for key, item in value.items()
        }
    return value


def _trace_node_from_fx(node: Node, graph_module: GraphModule, shape_env: ShapeEnv) -> TraceNode:
    tensor_meta = _tensor_meta_from_fx(node.meta.get("tensor_meta"), shape_env)
    return TraceNode(
        name=node.name,
        op=node.op,
        target=str(node.target),
        args_template=_template(node.args, shape_env),
        kwargs_template=_template(node.kwargs, shape_env),
        parents=tuple(parent.name for parent in node.all_input_nodes),
        tensor_meta=tensor_meta,
        param_refs=_param_refs_for_node(node, graph_module),
        buffer_refs=_buffer_refs_for_node(node, graph_module),
        module_path=str(node.target) if node.op == "call_module" else None,
        alias_metadata=_alias_metadata(node),
        mutation_metadata=_mutation_metadata(node),
        rng_sensitive=_is_rng_sensitive(node),
    )


def _tensor_meta_from_fx(value: Any, shape_env: ShapeEnv) -> TensorMeta | None:
    if isinstance(value, FxTensorMetadata):
        shape = tuple(int(dim) for dim in value.shape)
        numel = 1
        for dim in shape:
            numel *= dim
        dtype = str(value.dtype)
        element_size = torch.empty((), dtype=value.dtype).element_size()
        return TensorMeta(
            shape=shape,
            symbolic_shape=shape_env.canonicalize_shape(shape),
            dtype=dtype,
            device_type="meta",
            requires_grad=bool(value.requires_grad),
            stride=tuple(int(dim) for dim in value.stride),
            numel=int(numel),
            element_size=int(element_size),
        )
    return None


def _meta_from_input(value: Any, shape_env: ShapeEnv) -> TensorMeta | None:
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    return tensor_meta_from_tensor(tensor, shape_env)


def _tensor_metas_from_template(value: TensorMeta | None) -> tuple[TensorMeta, ...]:
    if value is None:
        return ()
    return (value,)


def _output_template(graph_module: GraphModule) -> Any:
    for node in graph_module.graph.nodes:
        if node.op == "output":
            return _template(node.args, ShapeEnv())
    return None


def _template(value: Any, shape_env: ShapeEnv) -> Any:
    if isinstance(value, Node):
        return {"node": value.name}
    if (
        isinstance(value, int)
        and shape_env.traced_batch_size is not None
        and value == shape_env.traced_batch_size
    ):
        return ShapeExpr(shape_env.batch_symbol)
    if isinstance(value, tuple):
        return tuple(_template(item, shape_env) for item in value)
    if isinstance(value, list):
        return [_template(item, shape_env) for item in value]
    if isinstance(value, dict):
        return {key: _template(item, shape_env) for key, item in value.items()}
    if isinstance(value, (str, float, bool, type(None))):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    return repr(value)


def _param_refs_for_node(node: Node, graph_module: GraphModule) -> tuple[ParamRef, ...]:
    refs: list[ParamRef] = []
    named_parameters = dict(graph_module.named_parameters())
    if node.op == "call_module":
        module = graph_module.get_submodule(str(node.target))
        for local_name, parameter in module.named_parameters(recurse=True):
            full_name = f"{node.target}.{local_name}" if local_name else str(node.target)
            refs.append(param_ref_from_parameter(full_name, parameter))
    if node.op == "get_attr" and str(node.target) in named_parameters:
        refs.append(param_ref_from_parameter(str(node.target), named_parameters[str(node.target)]))
    return tuple(refs)


def _buffer_refs_for_node(node: Node, graph_module: GraphModule) -> tuple[BufferRef, ...]:
    refs: list[BufferRef] = []
    named_buffers = dict(graph_module.named_buffers())
    if node.op == "call_module":
        module = graph_module.get_submodule(str(node.target))
        for local_name, buffer in module.named_buffers(recurse=True):
            full_name = f"{node.target}.{local_name}" if local_name else str(node.target)
            refs.append(buffer_ref_from_tensor(full_name, buffer))
    if node.op == "get_attr" and str(node.target) in named_buffers:
        refs.append(buffer_ref_from_tensor(str(node.target), named_buffers[str(node.target)]))
    return tuple(refs)


def _alias_metadata(node: Node) -> dict[str, Any] | None:
    if node.op == "call_method" and str(node.target) in {"view", "reshape", "flatten"}:
        return {"view_like": True}
    return None


def _mutation_metadata(node: Node) -> dict[str, Any] | None:
    target = str(node.target)
    if target.endswith("_") or target.startswith("aten.") and "_" in target.split(".")[-1]:
        return {"possibly_mutating": True}
    return None


def _is_rng_sensitive(node: Node) -> bool:
    return _target_name(node.target) in _RNG_FUNCTION_NAMES


def _target_name(target: Any) -> str:
    return getattr(target, "__name__", str(target))
