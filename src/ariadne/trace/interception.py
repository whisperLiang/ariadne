"""TorchLens-style runtime interception tracing.

This backend executes the real PyTorch forward pass and records the aten
operations that actually run. It is intentionally lightweight: it stores graph
structure, callable references, templates, tensor metadata, and module context,
but not full activation tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ariadne.trace.tensor_meta import (
    BufferRef,
    ParamRef,
    ShapeEnv,
    TensorMeta,
    buffer_ref_from_tensor,
    param_ref_from_parameter,
    tensor_meta_from_tensor,
)
from ariadne.trace.trace_plan import TraceNode, TracePlan, compute_graph_signature


@dataclass(frozen=True)
class NodeArg:
    name: str


@dataclass(frozen=True)
class ParamArg:
    name: str


@dataclass(frozen=True)
class BufferArg:
    name: str


@dataclass(frozen=True)
class TensorAttrArg:
    path: str


@dataclass(frozen=True)
class ConstantTensorArg:
    value: torch.Tensor


@dataclass(frozen=True)
class BatchDimArg:
    symbol: str


@dataclass(frozen=True)
class CapturedOp:
    name: str
    target: str
    callable_ref: Any
    args_template: Any
    kwargs_template: Any
    output_template: Any
    output_names: tuple[str, ...]
    parents: tuple[str, ...]
    tensor_metas: dict[str, TensorMeta]
    param_refs: tuple[ParamRef, ...]
    buffer_refs: tuple[BufferRef, ...]
    module_path: str | None
    rng_sensitive: bool
    mutating: bool


@dataclass(frozen=True)
class InterceptionTraceArtifact:
    ops: tuple[CapturedOp, ...]
    output_template: Any
    input_node_names: tuple[str, ...]
    parameter_names_by_id: dict[int, str]
    buffer_names_by_id: dict[int, str]
    tensor_attr_names_by_id: dict[int, str]


class RuntimeTraceError(RuntimeError):
    """Raised when runtime interception cannot represent a value safely."""


class _Recorder(TorchDispatchMode):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        shape_env: ShapeEnv,
        inputs: tuple[Any, ...],
    ) -> None:
        super().__init__()
        self.model = model
        self.shape_env = shape_env
        self.tensor_to_node: dict[int, str] = {}
        self.tensor_meta_by_node: dict[str, TensorMeta] = {}
        self.ops: list[CapturedOp] = []
        self.module_stack: list[str] = []
        self.parameter_names_by_id = {
            id(parameter): name for name, parameter in model.named_parameters()
        }
        self.buffer_names_by_id = {id(buffer): name for name, buffer in model.named_buffers()}
        self.tensor_attr_names_by_id = _tensor_attr_names_by_id(
            model,
            excluded_ids=set(self.parameter_names_by_id) | set(self.buffer_names_by_id),
        )
        self.parameter_by_id = {id(parameter): parameter for parameter in model.parameters()}
        self.buffer_by_id = {id(buffer): buffer for buffer in model.buffers()}
        self.input_node_names = self._register_inputs(inputs)

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        target = _target_name(func)
        args_template = self._canonicalize_shape_args(self._template(args), target)
        kwargs_template = self._canonicalize_shape_args(self._template(kwargs), target)
        parents = tuple(dict.fromkeys(_node_refs(args_template) + _node_refs(kwargs_template)))
        result = func(*args, **kwargs)
        output_template, output_names, tensor_metas = self._register_outputs(result)

        if output_names:
            op_name = output_names[0]
            self.ops.append(
                CapturedOp(
                    name=op_name,
                    target=target,
                    callable_ref=func,
                    args_template=args_template,
                    kwargs_template=kwargs_template,
                    output_template=output_template,
                    output_names=tuple(output_names),
                    parents=parents,
                    tensor_metas=tensor_metas,
                    param_refs=self._param_refs(args_template, kwargs_template),
                    buffer_refs=self._buffer_refs(args_template, kwargs_template),
                    module_path=self.module_stack[-1] if self.module_stack else None,
                    rng_sensitive=_is_rng_sensitive(target),
                    mutating=_is_mutating(target),
                )
            )
        return result

    def _register_inputs(self, inputs: tuple[Any, ...]) -> tuple[str, ...]:
        names: list[str] = []

        def visit(value: Any, path: str) -> None:
            if isinstance(value, torch.Tensor):
                node_name = path
                self.tensor_to_node[id(value)] = node_name
                self.tensor_meta_by_node[node_name] = tensor_meta_from_tensor(value, self.shape_env)
                names.append(node_name)
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    visit(item, f"{path}_{index}")
            elif isinstance(value, dict):
                for key, item in value.items():
                    visit(item, f"{path}_{key}")

        for index, value in enumerate(inputs):
            visit(value, f"input_{index}")
        return tuple(names)

    def _register_outputs(self, result: Any) -> tuple[Any, list[str], dict[str, TensorMeta]]:
        output_names: list[str] = []
        tensor_metas: dict[str, TensorMeta] = {}

        def visit(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                node_name = f"node_{len(self.ops)}"
                if output_names:
                    node_name = f"{node_name}_{len(output_names)}"
                self.tensor_to_node[id(value)] = node_name
                meta = tensor_meta_from_tensor(value, self.shape_env)
                self.tensor_meta_by_node[node_name] = meta
                output_names.append(node_name)
                tensor_metas[node_name] = meta
                return NodeArg(node_name)
            if isinstance(value, tuple):
                return tuple(visit(item) for item in value)
            if isinstance(value, list):
                return [visit(item) for item in value]
            if isinstance(value, dict):
                return {key: visit(item) for key, item in value.items()}
            return value

        return visit(result), output_names, tensor_metas

    def _template(self, value: Any) -> Any:
        if isinstance(value, torch.nn.Parameter) and id(value) in self.parameter_names_by_id:
            return ParamArg(self.parameter_names_by_id[id(value)])
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if tensor_id in self.tensor_to_node:
                return NodeArg(self.tensor_to_node[tensor_id])
            if tensor_id in self.parameter_names_by_id:
                return ParamArg(self.parameter_names_by_id[tensor_id])
            if tensor_id in self.buffer_names_by_id:
                return BufferArg(self.buffer_names_by_id[tensor_id])
            if tensor_id in self.tensor_attr_names_by_id:
                return TensorAttrArg(self.tensor_attr_names_by_id[tensor_id])
            if value.numel() <= 16:
                return ConstantTensorArg(value.detach().clone())
            raise RuntimeTraceError(
                "Encountered an untracked non-parameter tensor argument with more than "
                "16 elements. Register it as an input, parameter, or buffer."
            )
        if isinstance(value, tuple):
            return tuple(self._template(item) for item in value)
        if isinstance(value, list):
            return [self._template(item) for item in value]
        if isinstance(value, dict):
            return {key: self._template(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self._template(value.start),
                self._template(value.stop),
                self._template(value.step),
            )
        return value

    def _canonicalize_shape_args(self, value: Any, target: str) -> Any:
        if not _is_shape_sensitive(target) or self.shape_env.traced_batch_size is None:
            return value
        if isinstance(value, int) and value == self.shape_env.traced_batch_size:
            return BatchDimArg(self.shape_env.batch_symbol)
        if isinstance(value, tuple):
            return tuple(self._canonicalize_shape_args(item, target) for item in value)
        if isinstance(value, list):
            return [self._canonicalize_shape_args(item, target) for item in value]
        if isinstance(value, dict):
            return {key: self._canonicalize_shape_args(item, target) for key, item in value.items()}
        return value

    def _param_refs(self, *templates: Any) -> tuple[ParamRef, ...]:
        names = tuple(dict.fromkeys(_param_refs(*templates)))
        return tuple(
            param_ref_from_parameter(
                name,
                self.parameter_by_id[_id_for_name(self.parameter_names_by_id, name)],
            )
            for name in names
        )

    def _buffer_refs(self, *templates: Any) -> tuple[BufferRef, ...]:
        names = tuple(dict.fromkeys(_buffer_refs(*templates)))
        return tuple(
            buffer_ref_from_tensor(
                name,
                self.buffer_by_id[_id_for_name(self.buffer_names_by_id, name)],
            )
            for name in names
        )


def trace_model_interception(
    model: torch.nn.Module,
    *,
    example_inputs: Sequence[Any],
    batch_symbol: str = "B",
    dynamic_batch: tuple[int, int] | None = None,
) -> TracePlan:
    inputs = tuple(example_inputs)
    traced_batch_size = _infer_traced_batch_size(inputs)
    shape_env = ShapeEnv(
        batch_symbol=batch_symbol,
        traced_batch_size=traced_batch_size,
        dynamic_batch=dynamic_batch,
    )
    recorder = _Recorder(model=model, shape_env=shape_env, inputs=inputs)
    hooks = _install_module_stack_hooks(model, recorder)
    try:
        with recorder:
            output = model(*inputs)
    finally:
        for handle in hooks:
            handle.remove()

    output_template = recorder._template(output)
    nodes = _build_trace_nodes(recorder, output_template)
    artifact = InterceptionTraceArtifact(
        ops=tuple(recorder.ops),
        output_template=output_template,
        input_node_names=recorder.input_node_names,
        parameter_names_by_id=recorder.parameter_names_by_id,
        buffer_names_by_id=recorder.buffer_names_by_id,
        tensor_attr_names_by_id=recorder.tensor_attr_names_by_id,
    )
    output_metas = tuple(
        recorder.tensor_meta_by_node[name]
        for name in _node_refs(output_template)
        if name in recorder.tensor_meta_by_node
    )
    return TracePlan(
        graph_signature=_interception_signature(nodes, shape_env),
        nodes=nodes,
        input_metas=tuple(recorder.tensor_meta_by_node[name] for name in recorder.input_node_names),
        output_metas=output_metas,
        shape_env=shape_env,
        input_node_names=recorder.input_node_names,
        output_template=output_template,
        fx_graph_module=model,
        runtime_artifact=artifact,
    )


def _build_trace_nodes(recorder: _Recorder, output_template: Any) -> tuple[TraceNode, ...]:
    nodes: list[TraceNode] = []
    for name in recorder.input_node_names:
        nodes.append(
            TraceNode(
                name=name,
                op="placeholder",
                target=name,
                args_template=(),
                kwargs_template={},
                parents=(),
                tensor_meta=recorder.tensor_meta_by_node[name],
            )
        )
    for op in recorder.ops:
        for output_name in op.output_names:
            nodes.append(
                TraceNode(
                    name=output_name,
                    op="call_function",
                    target=op.target,
                    args_template=op.args_template,
                    kwargs_template=op.kwargs_template,
                    parents=op.parents,
                    tensor_meta=op.tensor_metas[output_name],
                    param_refs=op.param_refs,
                    buffer_refs=op.buffer_refs,
                    module_path=op.module_path,
                    mutation_metadata={"possibly_mutating": True} if op.mutating else None,
                    rng_sensitive=op.rng_sensitive,
                )
            )
    nodes.append(
        TraceNode(
            name="output",
            op="output",
            target="output",
            args_template=output_template,
            kwargs_template={},
            parents=tuple(_node_refs(output_template)),
        )
    )
    return tuple(nodes)


def _install_module_stack_hooks(
    model: torch.nn.Module,
    recorder: _Recorder,
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    names_by_id = {id(module): name for name, module in model.named_modules() if name}

    def pre_hook(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        name = names_by_id.get(id(module))
        if name is not None:
            recorder.module_stack.append(name)

    def post_hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
        name = names_by_id.get(id(module))
        if name is not None and recorder.module_stack:
            recorder.module_stack.pop()

    for module in model.modules():
        if id(module) in names_by_id:
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
    return handles


def resolve_template(
    value: Any,
    *,
    env: dict[str, Any],
    parameters: dict[str, torch.nn.Parameter],
    buffers: dict[str, torch.Tensor],
    root: torch.nn.Module | None = None,
) -> Any:
    if isinstance(value, NodeArg):
        return env[value.name]
    if isinstance(value, ParamArg):
        return parameters[value.name]
    if isinstance(value, BufferArg):
        return buffers[value.name]
    if isinstance(value, TensorAttrArg):
        if root is None:
            raise ValueError("Resolving TensorAttrArg requires the root module.")
        return _resolve_tensor_attr(root, value.path)
    if isinstance(value, ConstantTensorArg):
        return value.value
    if isinstance(value, BatchDimArg):
        return env[value.symbol]
    if isinstance(value, tuple):
        return tuple(
            resolve_template(item, env=env, parameters=parameters, buffers=buffers, root=root)
            for item in value
        )
    if isinstance(value, list):
        return [
            resolve_template(item, env=env, parameters=parameters, buffers=buffers, root=root)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: resolve_template(item, env=env, parameters=parameters, buffers=buffers, root=root)
            for key, item in value.items()
        }
    if isinstance(value, slice):
        start = resolve_template(
            value.start,
            env=env,
            parameters=parameters,
            buffers=buffers,
            root=root,
        )
        stop = resolve_template(
            value.stop,
            env=env,
            parameters=parameters,
            buffers=buffers,
            root=root,
        )
        step = resolve_template(
            value.step,
            env=env,
            parameters=parameters,
            buffers=buffers,
            root=root,
        )
        return slice(start, stop, step)
    return value


def assign_template(value: Any, template: Any, env: dict[str, Any]) -> None:
    if isinstance(template, NodeArg):
        env[template.name] = value
    elif isinstance(template, (tuple, list)):
        for item, item_template in zip(value, template, strict=True):
            assign_template(item, item_template, env)
    elif isinstance(template, dict):
        for key, item_template in template.items():
            assign_template(value[key], item_template, env)


def materialize_template(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, NodeArg):
        return env[value.name]
    if isinstance(value, tuple):
        return tuple(materialize_template(item, env) for item in value)
    if isinstance(value, list):
        return [materialize_template(item, env) for item in value]
    if isinstance(value, dict):
        return {key: materialize_template(item, env) for key, item in value.items()}
    return value


def _node_refs(value: Any) -> list[str]:
    if isinstance(value, NodeArg):
        return [value.name]
    if isinstance(value, tuple):
        return [name for item in value for name in _node_refs(item)]
    if isinstance(value, list):
        return [name for item in value for name in _node_refs(item)]
    if isinstance(value, dict):
        return [name for item in value.values() for name in _node_refs(item)]
    if isinstance(value, slice):
        return _node_refs(value.start) + _node_refs(value.stop) + _node_refs(value.step)
    return []


def _param_refs(*values: Any) -> list[str]:
    refs: list[str] = []
    for value in values:
        if isinstance(value, ParamArg):
            refs.append(value.name)
        elif isinstance(value, (tuple, list)):
            refs.extend(name for item in value for name in _param_refs(item))
        elif isinstance(value, dict):
            refs.extend(name for item in value.values() for name in _param_refs(item))
        elif isinstance(value, slice):
            refs.extend(_param_refs(value.start, value.stop, value.step))
    return refs


def _buffer_refs(*values: Any) -> list[str]:
    refs: list[str] = []
    for value in values:
        if isinstance(value, BufferArg):
            refs.append(value.name)
        elif isinstance(value, (tuple, list)):
            refs.extend(name for item in value for name in _buffer_refs(item))
        elif isinstance(value, dict):
            refs.extend(name for item in value.values() for name in _buffer_refs(item))
        elif isinstance(value, slice):
            refs.extend(_buffer_refs(value.start, value.stop, value.step))
    return refs


def _tensor_attr_names_by_id(
    model: torch.nn.Module,
    *,
    excluded_ids: set[int],
) -> dict[int, str]:
    names: dict[int, str] = {}
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for attr_name, value in vars(module).items():
            if attr_name.startswith("_"):
                continue
            if isinstance(value, torch.Tensor) and id(value) not in excluded_ids:
                names[id(value)] = f"{prefix}{attr_name}"
    return names


def _resolve_tensor_attr(root: torch.nn.Module, path: str) -> torch.Tensor:
    if "." not in path:
        value = getattr(root, path)
    else:
        module_path, attr_name = path.rsplit(".", 1)
        value = getattr(root.get_submodule(module_path), attr_name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Module attribute {path!r} is no longer a tensor.")
    return value


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


def _target_name(target: Any) -> str:
    name = getattr(target, "__name__", str(target))
    return str(name)


def _is_shape_sensitive(target: str) -> bool:
    return any(
        token in target
        for token in (
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
        )
    )


def _is_rng_sensitive(target: str) -> bool:
    return any(token in target for token in ("rand", "normal", "bernoulli", "dropout"))


def _is_mutating(target: str) -> bool:
    return "_" in target.split(".")[0] or target.endswith("_")


def _id_for_name(names_by_id: dict[int, str], name: str) -> int:
    for object_id, candidate in names_by_id.items():
        if candidate == name:
            return object_id
    raise KeyError(name)


def _interception_signature(nodes: tuple[TraceNode, ...], shape_env: ShapeEnv) -> str:
    base = compute_graph_signature(nodes, shape_env)
    digest = sha256(base.encode())
    digest.update(b"interception")
    return digest.hexdigest()[:16]
