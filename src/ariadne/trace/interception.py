"""TorchLens-style runtime interception tracing.

This backend executes the real PyTorch forward pass and records the aten
operations that actually run. It is intentionally lightweight: it stores graph
structure, callable references, templates, tensor metadata, and module context,
but not full activation tensors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any
from weakref import ReferenceType, ref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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
from ariadne.trace.trace_plan import TraceNode, TracePlan


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
class SequenceArg:
    name: str
    container_type: str


@dataclass(frozen=True)
class SequenceElementArg:
    sequence_name: str
    index: int


@dataclass(frozen=True)
class SequenceOutputTemplate:
    name: str
    container_type: str
    length_expr: int | BatchDimArg | ShapeExpr
    element_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapturedOp:
    op_index: int
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
    dynamic_output_structure: bool = False
    dynamic_structure_reason: str | None = None


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
        self.tensor_to_node: dict[int, tuple[ReferenceType[torch.Tensor], str]] = {}
        self.tensor_meta_by_node: dict[str, TensorMeta] = {}
        self.sequence_containers: dict[int, tuple[Any, SequenceOutputTemplate]] = {}
        self.sequence_elements: dict[
            int,
            tuple[ReferenceType[torch.Tensor], SequenceElementArg],
        ] = {}
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
        self.meta_shape_env = ShapeEnv(
            batch_symbol=shape_env.batch_symbol,
            traced_batch_size=None,
            dynamic_batch=shape_env.dynamic_batch,
            trace_batch_mode=shape_env.trace_batch_mode,
        )
        self.output: Any = None
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
        args_template = self._template(args, allow_rebuilt_sequence=False)
        kwargs_template = self._template(kwargs)
        parents = tuple(dict.fromkeys(_node_refs(args_template) + _node_refs(kwargs_template)))
        result = func(*args, **kwargs)
        output_template, output_names, tensor_metas = self._register_outputs(
            result,
            target=target,
        )

        if output_names:
            op_name = output_names[0]
            op_index = len(self.ops)
            self.ops.append(
                CapturedOp(
                    op_index=op_index,
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
                self._remember_tensor_node(value, node_name)
                self.tensor_meta_by_node[node_name] = tensor_meta_from_tensor(
                    value,
                    self.meta_shape_env,
                )
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

    def _register_outputs(
        self,
        result: Any,
        *,
        target: str,
    ) -> tuple[Any, list[str], dict[str, TensorMeta]]:
        output_names: list[str] = []
        tensor_metas: dict[str, TensorMeta] = {}

        if _is_sequence_producing_target(target) and _is_tensor_sequence(result):
            sequence_name = f"node_{len(self.ops)}"
            element_names: list[str] = []
            for index, item in enumerate(result):
                element_name = f"{sequence_name}_{index}"
                element_names.append(element_name)
                self._remember_tensor_node(item, element_name)
                meta = tensor_meta_from_tensor(item, self.meta_shape_env)
                self.tensor_meta_by_node[element_name] = meta
                tensor_metas[element_name] = meta
            template = SequenceOutputTemplate(
                name=sequence_name,
                container_type=_container_type_name(result),
                length_expr=len(result),
                element_names=tuple(element_names),
            )
            self._remember_sequence(result, template)
            return template, [sequence_name, *element_names], tensor_metas

        def visit(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                node_name = f"node_{len(self.ops)}"
                if output_names:
                    node_name = f"{node_name}_{len(output_names)}"
                self._remember_tensor_node(value, node_name)
                meta = tensor_meta_from_tensor(value, self.meta_shape_env)
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

    def _template(self, value: Any, *, allow_rebuilt_sequence: bool = True) -> Any:
        if isinstance(value, torch.nn.Parameter) and id(value) in self.parameter_names_by_id:
            return ParamArg(self.parameter_names_by_id[id(value)])
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            sequence_element = self._sequence_element_for_tensor(value)
            if sequence_element is not None:
                return sequence_element
            node_name = self._node_for_tensor(value)
            if node_name is not None:
                return NodeArg(node_name)
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
            sequence_arg = self._sequence_arg_for_container(value)
            if sequence_arg is not None:
                return sequence_arg
            templated_tuple = tuple(self._template(item) for item in value)
            if allow_rebuilt_sequence:
                return (
                    self._sequence_arg_for_rebuilt_container(templated_tuple, "tuple")
                    or templated_tuple
                )
            return templated_tuple
        if isinstance(value, list):
            sequence_arg = self._sequence_arg_for_container(value)
            if sequence_arg is not None:
                return sequence_arg
            templated_list = [self._template(item) for item in value]
            if allow_rebuilt_sequence:
                return (
                    self._sequence_arg_for_rebuilt_container(templated_list, "list")
                    or templated_list
                )
            return templated_list
        if isinstance(value, dict):
            return {key: self._template(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(
                self._template(value.start),
                self._template(value.stop),
                self._template(value.step),
            )
        return value

    def _remember_sequence(self, value: Any, template: SequenceOutputTemplate) -> None:
        self.sequence_containers[id(value)] = (value, template)
        for index, item in enumerate(value):
            if isinstance(item, torch.Tensor):
                element = SequenceElementArg(template.name, index)
                self.sequence_elements[id(item)] = (ref(item), element)

    def _sequence_arg_for_container(self, value: Any) -> SequenceArg | None:
        entry = self.sequence_containers.get(id(value))
        if entry is None:
            return None
        container, template = entry
        if container is not value:
            return None
        return SequenceArg(template.name, _container_type_name(value))

    def _sequence_arg_for_rebuilt_container(
        self,
        value: tuple[Any, ...] | list[Any],
        container_type: str,
    ) -> SequenceArg | None:
        if not value or not all(isinstance(item, SequenceElementArg) for item in value):
            return None
        elements = [item for item in value if isinstance(item, SequenceElementArg)]
        sequence_name = elements[0].sequence_name
        if any(element.sequence_name != sequence_name for element in elements):
            return None
        if [element.index for element in elements] != list(range(len(elements))):
            return None
        sequence_template = next(
            (
                template
                for _container, template in self.sequence_containers.values()
                if template.name == sequence_name
            ),
            None,
        )
        if sequence_template is None or sequence_template.length_expr != len(elements):
            return None
        return SequenceArg(sequence_name, container_type)

    def _remember_tensor_node(self, tensor: torch.Tensor, node_name: str) -> None:
        self.tensor_to_node[id(tensor)] = (ref(tensor), node_name)

    def _sequence_element_for_tensor(self, tensor: torch.Tensor) -> SequenceElementArg | None:
        entry = self.sequence_elements.get(id(tensor))
        if entry is None:
            return None
        tensor_ref, element = entry
        if tensor_ref() is tensor:
            return element
        del self.sequence_elements[id(tensor)]
        return None

    def _node_for_tensor(self, tensor: torch.Tensor) -> str | None:
        tensor_id = id(tensor)
        entry = self.tensor_to_node.get(tensor_id)
        if entry is None:
            return None
        tensor_ref, node_name = entry
        if tensor_ref() is tensor:
            return node_name
        del self.tensor_to_node[tensor_id]
        return None

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
    trace_batch_mode: str = "batch_1",
) -> TracePlan:
    inputs = tuple(example_inputs)
    traced_batch_size = _infer_traced_batch_size(inputs)
    shape_env = ShapeEnv(
        batch_symbol=batch_symbol,
        traced_batch_size=traced_batch_size,
        dynamic_batch=dynamic_batch,
        trace_batch_mode=trace_batch_mode,
    )
    buffer_snapshot = _snapshot_buffers(model)
    rng_snapshot = _snapshot_rng()
    try:
        recorder = _record_forward(model=model, shape_env=shape_env, inputs=inputs)
        probe_batch = _choose_probe_batch(shape_env)
        probe_recorder = None
        if probe_batch is not None and traced_batch_size is not None:
            probe_inputs = _make_probe_inputs(inputs, traced_batch_size, probe_batch)
            probe_shape_env = ShapeEnv(
                batch_symbol=batch_symbol,
                traced_batch_size=probe_batch,
                dynamic_batch=dynamic_batch,
                trace_batch_mode=trace_batch_mode,
            )
            probe_recorder = _record_forward(
                model=model,
                shape_env=probe_shape_env,
                inputs=probe_inputs,
            )
    finally:
        _restore_buffers(model, buffer_snapshot)
        _restore_rng(rng_snapshot)

    probe_ops_by_index = _align_probe_ops(
        recorder.ops,
        probe_recorder.ops if probe_recorder else None,
    )
    probe_meta_by_node = _probe_meta_by_node(
        recorder,
        probe_recorder=probe_recorder,
        probe_ops_by_index=probe_ops_by_index,
    )
    _canonicalize_tensor_metas(
        recorder,
        probe_meta_by_node=probe_meta_by_node,
        shape_env=shape_env,
        probe_batch=probe_batch,
    )
    ops, sequence_templates, dynamic_sequences = _canonicalize_ops(
        recorder,
        probe_ops_by_index=probe_ops_by_index,
        shape_env=shape_env,
        probe_batch=probe_batch,
    )
    output_template = _canonicalize_sequence_refs(
        recorder._template(recorder.output),
        sequence_templates=sequence_templates,
        dynamic_sequences=dynamic_sequences,
    )
    nodes = _build_trace_nodes(recorder, ops, output_template)
    artifact = InterceptionTraceArtifact(
        ops=ops,
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
        graph_signature=_interception_signature(nodes, shape_env, ops=ops),
        nodes=nodes,
        input_metas=tuple(recorder.tensor_meta_by_node[name] for name in recorder.input_node_names),
        output_metas=output_metas,
        shape_env=shape_env,
        input_node_names=recorder.input_node_names,
        output_template=output_template,
        root_module=model,
        runtime_artifact=artifact,
    )


def _record_forward(
    *,
    model: torch.nn.Module,
    shape_env: ShapeEnv,
    inputs: tuple[Any, ...],
) -> _Recorder:
    recorder = _Recorder(model=model, shape_env=shape_env, inputs=inputs)
    hooks = _install_module_stack_hooks(model, recorder)
    try:
        with recorder:
            recorder.output = model(*inputs)
    finally:
        for handle in hooks:
            handle.remove()
    return recorder


def _canonicalize_tensor_metas(
    recorder: _Recorder,
    *,
    probe_meta_by_node: dict[str, TensorMeta],
    shape_env: ShapeEnv,
    probe_batch: int | None,
) -> None:
    for name, meta in tuple(recorder.tensor_meta_by_node.items()):
        symbolic_shape = _symbolic_shape_for_node(
            name,
            meta,
            recorder=recorder,
            probe_meta_by_node=probe_meta_by_node,
            shape_env=shape_env,
            probe_batch=probe_batch,
        )
        recorder.tensor_meta_by_node[name] = replace(meta, symbolic_shape=symbolic_shape)


def _symbolic_shape_for_node(
    name: str,
    meta: TensorMeta,
    *,
    recorder: _Recorder,
    probe_meta_by_node: dict[str, TensorMeta],
    shape_env: ShapeEnv,
    probe_batch: int | None,
) -> tuple[Any, ...]:
    traced_batch = shape_env.traced_batch_size
    if traced_batch is None or not meta.shape:
        return meta.shape
    if name in recorder.input_node_names:
        if meta.shape[0] == traced_batch:
            return (shape_env.batch_symbol, *meta.shape[1:])
        return meta.shape
    if probe_batch is not None:
        probe_meta = probe_meta_by_node.get(name)
        if probe_meta is not None and len(probe_meta.shape) == len(meta.shape):
            return tuple(
                _symbolic_dimension(
                    value,
                    probe_value,
                    traced_batch=traced_batch,
                    probe_batch=probe_batch,
                    symbol=shape_env.batch_symbol,
                )
                for value, probe_value in zip(meta.shape, probe_meta.shape, strict=True)
            )
        return meta.shape
    if meta.shape[0] == traced_batch and traced_batch != 1:
        return (shape_env.batch_symbol, *meta.shape[1:])
    return meta.shape


def _canonicalize_ops(
    recorder: _Recorder,
    *,
    probe_ops_by_index: tuple[CapturedOp | None, ...],
    shape_env: ShapeEnv,
    probe_batch: int | None,
) -> tuple[
    tuple[CapturedOp, ...],
    dict[str, SequenceOutputTemplate],
    dict[str, SequenceOutputTemplate],
]:
    ops: list[CapturedOp] = []
    sequence_templates: dict[str, SequenceOutputTemplate] = {}
    dynamic_sequences: dict[str, SequenceOutputTemplate] = {}
    for index, op in enumerate(recorder.ops):
        args_template = op.args_template
        kwargs_template = op.kwargs_template
        dynamic_output_structure = False
        dynamic_structure_reason = None
        output_template = op.output_template
        probe_op = probe_ops_by_index[index]
        if probe_op is not None and probe_batch is not None:
            arg_exprs = _batch_exprs(
                op.args_template,
                probe_op.args_template,
                traced_batch=shape_env.traced_batch_size,
                probe_batch=probe_batch,
                symbol=shape_env.batch_symbol,
            )
            kwarg_exprs = _batch_exprs(
                op.kwargs_template,
                probe_op.kwargs_template,
                traced_batch=shape_env.traced_batch_size,
                probe_batch=probe_batch,
                symbol=shape_env.batch_symbol,
            )
            args_template = _replace_batch_exprs(op.args_template, arg_exprs)
            kwargs_template = _replace_batch_exprs(op.kwargs_template, kwarg_exprs)
        args_template = _canonicalize_sequence_refs(
            args_template,
            sequence_templates=sequence_templates,
            dynamic_sequences=dynamic_sequences,
        )
        kwargs_template = _canonicalize_sequence_refs(
            kwargs_template,
            sequence_templates=sequence_templates,
            dynamic_sequences=dynamic_sequences,
        )

        if isinstance(op.output_template, SequenceOutputTemplate):
            output_template, dynamic_structure_reason = _canonicalize_sequence_output(
                op.output_template,
                probe_op.output_template if probe_op is not None else None,
                traced_batch=shape_env.traced_batch_size,
                probe_batch=probe_batch,
                symbol=shape_env.batch_symbol,
            )
            sequence_templates[op.output_template.name] = op.output_template
            if dynamic_structure_reason is None and isinstance(
                output_template,
                SequenceOutputTemplate,
            ):
                dynamic_sequences[output_template.name] = output_template
        elif probe_op is not None and not _compatible_container_structure(
            op.output_template,
            probe_op.output_template,
        ):
            dynamic_output_structure = True
            dynamic_structure_reason = (
                "batch-dependent Python container structure is not representable "
                "as a supported dynamic sequence"
            )
        if dynamic_structure_reason is not None:
            dynamic_output_structure = True
        ops.append(
            replace(
                op,
                args_template=args_template,
                kwargs_template=kwargs_template,
                parents=tuple(
                    dict.fromkeys(_node_refs(args_template) + _node_refs(kwargs_template))
                ),
                output_template=output_template,
                tensor_metas={
                    name: recorder.tensor_meta_by_node[name] for name in op.output_names
                    if name in recorder.tensor_meta_by_node
                },
                dynamic_output_structure=dynamic_output_structure,
                dynamic_structure_reason=dynamic_structure_reason,
            )
        )
    return tuple(ops), sequence_templates, dynamic_sequences


def _canonicalize_sequence_refs(
    value: Any,
    *,
    sequence_templates: dict[str, SequenceOutputTemplate],
    dynamic_sequences: dict[str, SequenceOutputTemplate],
) -> Any:
    if isinstance(value, SequenceArg):
        if value.name in dynamic_sequences:
            return value
        template = sequence_templates.get(value.name)
        return _fixed_sequence_template(template, value.container_type) if template else value
    if isinstance(value, SequenceElementArg):
        if value.sequence_name in dynamic_sequences:
            return value
        template = sequence_templates.get(value.sequence_name)
        if template is None or value.index >= len(template.element_names):
            return value
        return NodeArg(template.element_names[value.index])
    if isinstance(value, tuple):
        return tuple(
            _canonicalize_sequence_refs(
                item,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            )
            for item in value
        )
    if isinstance(value, list):
        return [
            _canonicalize_sequence_refs(
                item,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _canonicalize_sequence_refs(
                item,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            )
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return slice(
            _canonicalize_sequence_refs(
                value.start,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            ),
            _canonicalize_sequence_refs(
                value.stop,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            ),
            _canonicalize_sequence_refs(
                value.step,
                sequence_templates=sequence_templates,
                dynamic_sequences=dynamic_sequences,
            ),
        )
    return value


def _canonicalize_sequence_output(
    template: SequenceOutputTemplate,
    probe_template: Any,
    *,
    traced_batch: int | None,
    probe_batch: int | None,
    symbol: str,
) -> tuple[Any, str | None]:
    if not isinstance(probe_template, SequenceOutputTemplate) or probe_batch is None:
        return _fixed_sequence_template(template), None
    if template.container_type != probe_template.container_type:
        return template, "dynamic sequence container type changed between trace and probe"
    trace_length = _sequence_length_value(template.length_expr)
    probe_length = _sequence_length_value(probe_template.length_expr)
    if trace_length == probe_length:
        return _fixed_sequence_template(template), None
    if traced_batch is None:
        return template, "dynamic sequence length changed without a known batch symbol"
    expression = _template_expression_for_values(
        trace_length,
        probe_length,
        traced_batch=traced_batch,
        probe_batch=probe_batch,
        symbol=symbol,
    )
    if expression is None:
        return (
            template,
            "dynamic sequence length changed but did not fit an affine batch expression",
        )
    return replace(template, length_expr=expression), None


def _fixed_sequence_template(
    template: SequenceOutputTemplate,
    container_type: str | None = None,
) -> tuple[NodeArg, ...] | list[NodeArg]:
    values = [NodeArg(name) for name in template.element_names]
    if (container_type or template.container_type) == "tuple":
        return tuple(values)
    return values


def _sequence_length_value(value: int | BatchDimArg | ShapeExpr) -> int:
    if isinstance(value, int):
        return value
    raise TypeError("Sequence length must be concrete before canonicalization.")


def _build_trace_nodes(
    recorder: _Recorder,
    ops: tuple[CapturedOp, ...],
    output_template: Any,
) -> tuple[TraceNode, ...]:
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
    for op in ops:
        alias_metadata = _op_alias_metadata(op)
        for output_index, output_name in enumerate(op.output_names):
            nodes.append(
                TraceNode(
                    name=output_name,
                    op="call_function",
                    target=op.target,
                    args_template=op.args_template,
                    kwargs_template=op.kwargs_template,
                    parents=op.parents,
                    tensor_meta=op.tensor_metas.get(output_name),
                    param_refs=op.param_refs,
                    buffer_refs=op.buffer_refs,
                    module_path=op.module_path,
                    op_index=op.op_index,
                    output_index=output_index,
                    alias_metadata=alias_metadata,
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


def _op_alias_metadata(op: CapturedOp) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    if op.dynamic_output_structure:
        metadata["dynamic_output_structure"] = True
        metadata["dynamic_structure_reason"] = op.dynamic_structure_reason
    if _contains_sequence_element_arg(
        op.args_template,
    ) or _contains_sequence_element_arg(op.kwargs_template):
        metadata["unsupported_dynamic_sequence_use"] = True
        metadata["dynamic_structure_reason"] = (
            "dynamic sequence is indexed element-wise after trace"
        )
    return metadata or None


def _contains_sequence_element_arg(value: Any) -> bool:
    if isinstance(value, SequenceElementArg):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_sequence_element_arg(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_sequence_element_arg(item) for item in value.values())
    if isinstance(value, slice):
        return any(
            _contains_sequence_element_arg(item)
            for item in (value.start, value.stop, value.step)
        )
    return False


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


def _align_probe_ops(
    ops: list[CapturedOp],
    probe_ops: list[CapturedOp] | None,
) -> tuple[CapturedOp | None, ...]:
    if probe_ops is None:
        return tuple(None for _ in ops)
    if len(ops) == len(probe_ops) and all(
        _op_alignment_key(op) == _op_alignment_key(probe_op)
        for op, probe_op in zip(ops, probe_ops, strict=True)
    ):
        return tuple(probe_ops)

    aligned: list[CapturedOp | None] = [None for _ in ops]
    matcher = SequenceMatcher(
        None,
        [_op_alignment_key(op) for op in ops],
        [_op_alignment_key(op) for op in probe_ops],
        autojunk=False,
    )
    for match in matcher.get_matching_blocks():
        for offset in range(match.size):
            aligned[match.a + offset] = probe_ops[match.b + offset]
    return tuple(aligned)


def _probe_meta_by_node(
    recorder: _Recorder,
    *,
    probe_recorder: _Recorder | None,
    probe_ops_by_index: tuple[CapturedOp | None, ...],
) -> dict[str, TensorMeta]:
    if probe_recorder is None:
        return {}
    metas: dict[str, TensorMeta] = {}
    for name in recorder.input_node_names:
        probe_meta = probe_recorder.tensor_meta_by_node.get(name)
        if probe_meta is not None:
            metas[name] = probe_meta
    for op, probe_op in zip(recorder.ops, probe_ops_by_index, strict=True):
        if probe_op is None:
            continue
        if isinstance(op.output_template, SequenceOutputTemplate) and isinstance(
            probe_op.output_template,
            SequenceOutputTemplate,
        ):
            for output_name, probe_output_name in zip(
                op.output_template.element_names,
                probe_op.output_template.element_names,
                strict=False,
            ):
                probe_meta = probe_op.tensor_metas.get(probe_output_name)
                if probe_meta is not None:
                    metas[output_name] = probe_meta
        if len(op.output_names) != len(probe_op.output_names) and not (
            isinstance(op.output_template, SequenceOutputTemplate)
            and isinstance(probe_op.output_template, SequenceOutputTemplate)
        ):
            continue
        for output_name, probe_output_name in zip(
            op.output_names,
            probe_op.output_names,
            strict=False,
        ):
            probe_meta = probe_op.tensor_metas.get(probe_output_name)
            if probe_meta is not None:
                metas[output_name] = probe_meta
    return metas


def _op_alignment_key(op: CapturedOp) -> tuple[str, str | None]:
    return op.target, op.module_path


def _batch_exprs(
    value: Any,
    probe_value: Any,
    *,
    traced_batch: int | None,
    probe_batch: int,
    symbol: str,
    path: tuple[Any, ...] = (),
) -> dict[tuple[Any, ...], BatchDimArg | ShapeExpr]:
    if traced_batch is None:
        return {}
    if type(value) is int and type(probe_value) is int:
        expression = _template_expression_for_values(
            value,
            probe_value,
            traced_batch=traced_batch,
            probe_batch=probe_batch,
            symbol=symbol,
        )
        return {} if expression is None else {path: expression}
    expressions: dict[tuple[Any, ...], BatchDimArg | ShapeExpr] = {}
    if (
        isinstance(value, (tuple, list))
        and isinstance(probe_value, type(value))
        and len(value) == len(probe_value)
    ):
        for index, (item, probe_item) in enumerate(zip(value, probe_value, strict=True)):
            expressions.update(
                _batch_exprs(
                    item,
                    probe_item,
                    traced_batch=traced_batch,
                    probe_batch=probe_batch,
                    symbol=symbol,
                    path=(*path, index),
                )
            )
    elif isinstance(value, dict) and isinstance(probe_value, dict):
        for key in value.keys() & probe_value.keys():
            expressions.update(
                _batch_exprs(
                    value[key],
                    probe_value[key],
                    traced_batch=traced_batch,
                    probe_batch=probe_batch,
                    symbol=symbol,
                    path=(*path, key),
                )
            )
    elif isinstance(value, slice) and isinstance(probe_value, slice):
        for attr in ("start", "stop", "step"):
            expressions.update(
                _batch_exprs(
                    getattr(value, attr),
                    getattr(probe_value, attr),
                    traced_batch=traced_batch,
                    probe_batch=probe_batch,
                    symbol=symbol,
                    path=(*path, attr),
                )
            )
    return expressions


def _replace_batch_exprs(
    value: Any,
    expressions: dict[tuple[Any, ...], BatchDimArg | ShapeExpr],
    *,
    path: tuple[Any, ...] = (),
) -> Any:
    if path in expressions and type(value) is int:
        return expressions[path]
    if isinstance(value, tuple):
        return tuple(
            _replace_batch_exprs(item, expressions, path=(*path, index))
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _replace_batch_exprs(item, expressions, path=(*path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_batch_exprs(item, expressions, path=(*path, key))
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return slice(
            _replace_batch_exprs(value.start, expressions, path=(*path, "start")),
            _replace_batch_exprs(value.stop, expressions, path=(*path, "stop")),
            _replace_batch_exprs(value.step, expressions, path=(*path, "step")),
        )
    return value


def _symbolic_dimension(
    value: int,
    probe_value: int,
    *,
    traced_batch: int | None,
    probe_batch: int,
    symbol: str,
) -> int | str | ShapeExpr:
    if value == probe_value:
        return value
    expression = _shape_expression_for_values(
        value,
        probe_value,
        traced_batch=traced_batch,
        probe_batch=probe_batch,
        symbol=symbol,
    )
    if expression is None:
        return value
    if expression.multiplier == 1 and expression.offset == 0:
        return symbol
    return expression


def _template_expression_for_values(
    value: int,
    probe_value: int,
    *,
    traced_batch: int,
    probe_batch: int,
    symbol: str,
) -> BatchDimArg | ShapeExpr | None:
    expression = _shape_expression_for_values(
        value,
        probe_value,
        traced_batch=traced_batch,
        probe_batch=probe_batch,
        symbol=symbol,
    )
    if expression is None:
        return None
    if expression.multiplier == 1 and expression.offset == 0:
        return BatchDimArg(symbol)
    return expression


def _shape_expression_for_values(
    value: int,
    probe_value: int,
    *,
    traced_batch: int | None,
    probe_batch: int,
    symbol: str,
) -> ShapeExpr | None:
    if traced_batch is None or probe_batch == traced_batch:
        return None
    delta_batch = probe_batch - traced_batch
    delta_value = probe_value - value
    if delta_value == 0 or delta_value % delta_batch != 0:
        return None
    multiplier = delta_value // delta_batch
    offset = value - multiplier * traced_batch
    if multiplier == 0 or multiplier * probe_batch + offset != probe_value:
        return None
    return ShapeExpr(symbol, multiplier=multiplier, offset=offset)


def _replace_matching_batch_constants(value: Any, traced_batch: int, symbol: str) -> Any:
    if type(value) is int and value == traced_batch:
        return BatchDimArg(symbol)
    if isinstance(value, tuple):
        return tuple(
            _replace_matching_batch_constants(item, traced_batch, symbol) for item in value
        )
    if isinstance(value, list):
        return [_replace_matching_batch_constants(item, traced_batch, symbol) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_matching_batch_constants(item, traced_batch, symbol)
            for key, item in value.items()
        }
    if isinstance(value, slice):
        return slice(
            _replace_matching_batch_constants(value.start, traced_batch, symbol),
            _replace_matching_batch_constants(value.stop, traced_batch, symbol),
            _replace_matching_batch_constants(value.step, traced_batch, symbol),
        )
    return value


def _compatible_container_structure(left: Any, right: Any) -> bool:
    if isinstance(left, NodeArg) and isinstance(right, NodeArg):
        return True
    if isinstance(left, SequenceOutputTemplate) and isinstance(right, SequenceOutputTemplate):
        return left.container_type == right.container_type
    if isinstance(left, SequenceArg) and isinstance(right, SequenceArg):
        return left.container_type == right.container_type
    if isinstance(left, SequenceElementArg) and isinstance(right, SequenceElementArg):
        return left.sequence_name == right.sequence_name and left.index == right.index
    if isinstance(left, (ParamArg, BufferArg, TensorAttrArg, ConstantTensorArg)):
        return type(left) is type(right)
    if isinstance(left, tuple):
        return (
            isinstance(right, tuple)
            and len(left) == len(right)
            and all(
                _compatible_container_structure(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(
                _compatible_container_structure(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_compatible_container_structure(left[key], right[key]) for key in left)
        )
    return type(left) is type(right)


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
    if isinstance(value, SequenceArg):
        sequence = env[value.name]
        return _coerce_sequence_container(sequence, value.container_type)
    if isinstance(value, SequenceElementArg):
        return env[value.sequence_name][value.index]
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
    if isinstance(value, ShapeExpr):
        symbol_value = env.get(value.expression)
        if not isinstance(symbol_value, int):
            raise ValueError(f"Missing shape symbol {value.expression!r} in runtime env.")
        return value.materialize({value.expression: symbol_value})
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
    elif isinstance(template, SequenceOutputTemplate):
        if not isinstance(value, (tuple, list)):
            raise RuntimeTraceError(
                "Expected a Python sequence for dynamic sequence output "
                f"{template.name!r}, got {type(value).__name__}."
            )
        expected_length = _materialize_length_expr(template.length_expr, env)
        if len(value) != expected_length:
            raise RuntimeTraceError(
                "Dynamic sequence length changed in a non-symbolic way "
                f"for {template.name!r}: expected {expected_length}, got {len(value)}."
            )
        env[template.name] = _coerce_sequence_container(value, template.container_type)
    elif isinstance(template, (tuple, list)):
        if not isinstance(value, type(template)):
            raise RuntimeTraceError(
                "Captured Python container type changed at replay time "
                f"({type(template).__name__} traced, {type(value).__name__} runtime)."
            )
        if len(value) != len(template):
            raise RuntimeTraceError(
                "Captured Python container length changed at replay time "
                f"({len(template)} traced element(s), {len(value)} runtime element(s)). "
                "Trace with a stable container structure or choose a different split boundary."
            )
        for item, item_template in zip(value, template, strict=True):
            assign_template(item, item_template, env)
    elif isinstance(template, dict):
        for key, item_template in template.items():
            assign_template(value[key], item_template, env)


def materialize_template(value: Any, env: dict[str, Any]) -> Any:
    if isinstance(value, NodeArg):
        return env[value.name]
    if isinstance(value, SequenceArg):
        return _coerce_sequence_container(env[value.name], value.container_type)
    if isinstance(value, SequenceElementArg):
        return env[value.sequence_name][value.index]
    if isinstance(value, SequenceOutputTemplate):
        return _coerce_sequence_container(env[value.name], value.container_type)
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
    if isinstance(value, SequenceArg):
        return [value.name]
    if isinstance(value, SequenceElementArg):
        return [value.sequence_name]
    if isinstance(value, SequenceOutputTemplate):
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
        elif isinstance(value, (SequenceArg, SequenceElementArg, SequenceOutputTemplate)):
            continue
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
        elif isinstance(value, (SequenceArg, SequenceElementArg, SequenceOutputTemplate)):
            continue
        elif isinstance(value, (tuple, list)):
            refs.extend(name for item in value for name in _buffer_refs(item))
        elif isinstance(value, dict):
            refs.extend(name for item in value.values() for name in _buffer_refs(item))
        elif isinstance(value, slice):
            refs.extend(_buffer_refs(value.start, value.stop, value.step))
    return refs


def _materialize_length_expr(value: int | BatchDimArg | ShapeExpr, env: dict[str, Any]) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, BatchDimArg):
        symbol_value = env.get(value.symbol)
        if not isinstance(symbol_value, int):
            raise RuntimeTraceError(f"Missing batch symbol {value.symbol!r} for sequence length.")
        return symbol_value
    symbol_value = env.get(value.expression)
    if not isinstance(symbol_value, int):
        raise RuntimeTraceError(f"Missing shape symbol {value.expression!r} for sequence length.")
    return value.materialize({value.expression: symbol_value})


def _coerce_sequence_container(value: Any, container_type: str) -> Any:
    if container_type == "tuple":
        return tuple(value)
    if container_type == "list":
        return list(value)
    raise RuntimeTraceError(f"Unsupported dynamic sequence container type {container_type!r}.")


def _choose_probe_batch(shape_env: ShapeEnv) -> int | None:
    traced_batch = shape_env.traced_batch_size
    if traced_batch is None or shape_env.dynamic_batch is None:
        return None
    low, high = shape_env.dynamic_batch
    if shape_env.trace_batch_mode == "batch_1":
        if traced_batch == 1 and low <= 2 <= high:
            return 2
        if low <= high and low != traced_batch:
            return low
        if high != traced_batch:
            return high
        return None
    if shape_env.trace_batch_mode == "batch_gt1":
        next_batch = traced_batch + 1
        if low <= next_batch <= high:
            return next_batch
        if low <= high and low != traced_batch and low > 1:
            return low
        if high != traced_batch and high > 1:
            return high
        return None
    if low <= high and low != traced_batch:
        return low
    if high != traced_batch:
        return high
    return None


def _make_probe_inputs(value: Any, traced_batch: int, probe_batch: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and int(value.shape[0]) == traced_batch:
            return _resize_tensor_batch(value, probe_batch)
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_make_probe_inputs(item, traced_batch, probe_batch) for item in value)
    if isinstance(value, list):
        return [_make_probe_inputs(item, traced_batch, probe_batch) for item in value]
    if isinstance(value, dict):
        return {
            key: _make_probe_inputs(item, traced_batch, probe_batch)
            for key, item in value.items()
        }
    return value


def _resize_tensor_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if int(tensor.shape[0]) >= batch_size:
        resized = tensor[:batch_size].detach().clone()
    else:
        repeats = [1 for _ in tensor.shape]
        repeats[0] = (batch_size + int(tensor.shape[0]) - 1) // int(tensor.shape[0])
        resized = tensor.repeat(*repeats)[:batch_size].detach().clone()
    if tensor.requires_grad and (resized.is_floating_point() or resized.is_complex()):
        resized.requires_grad_(True)
    return resized


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _restore_buffers(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    buffers = dict(model.named_buffers())
    for name, saved in snapshot.items():
        if name in buffers and buffers[name].shape == saved.shape:
            buffers[name].detach().copy_(saved)


def _snapshot_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = snapshot
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


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


def _is_sequence_producing_target(target: str) -> bool:
    return any(token in target for token in ("split", "chunk", "unbind"))


def _is_tensor_sequence(value: Any) -> bool:
    return isinstance(value, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in value
    )


def _container_type_name(value: Any) -> str:
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    raise TypeError(f"Unsupported sequence container type {type(value).__name__}.")


def _is_rng_sensitive(target: str) -> bool:
    return any(token in target for token in ("rand", "normal", "bernoulli", "dropout"))


def _is_mutating(target: str) -> bool:
    return "_" in target.split(".")[0] or target.endswith("_")


def _id_for_name(names_by_id: dict[int, str], name: str) -> int:
    for object_id, candidate in names_by_id.items():
        if candidate == name:
            return object_id
    raise KeyError(name)


def _interception_signature(
    nodes: tuple[TraceNode, ...],
    shape_env: ShapeEnv,
    *,
    ops: tuple[CapturedOp, ...],
) -> str:
    dynamic_sequence_templates = _dynamic_sequence_templates(ops)
    dynamic_sequence_elements = {
        element_name
        for template in dynamic_sequence_templates.values()
        for element_name in template.element_names
    }
    nodes_by_name = {node.name: node for node in nodes}

    digest = sha256()
    digest.update(shape_env.batch_symbol.encode())
    digest.update(repr(shape_env.dynamic_batch).encode())
    digest.update(shape_env.trace_batch_mode.encode())
    for node in nodes:
        if node.is_output or node.name in dynamic_sequence_elements:
            continue
        digest.update(node.name.encode())
        digest.update(node.op.encode())
        digest.update(node.target.encode())
        digest.update(repr(node.parents).encode())
        if node.tensor_meta is not None:
            digest.update(repr(node.tensor_meta.symbolic_shape).encode())
            digest.update(node.tensor_meta.dtype.encode())
        sequence_template = dynamic_sequence_templates.get(node.name)
        if sequence_template is not None:
            element_meta = _dynamic_sequence_element_meta(sequence_template, nodes_by_name)
            digest.update(repr(_template_schema(sequence_template.length_expr)).encode())
            if element_meta is not None:
                digest.update(repr(element_meta.symbolic_shape).encode())
                digest.update(element_meta.dtype.encode())

    digest.update(b"interception")
    for node in nodes:
        if node.name in dynamic_sequence_elements:
            continue
        digest.update(repr(_template_schema(node.args_template)).encode())
        digest.update(repr(_template_schema(node.kwargs_template)).encode())
    for op in ops:
        digest.update(repr(_template_schema(op.output_template)).encode())
    return digest.hexdigest()[:16]


def _dynamic_sequence_templates(ops: tuple[CapturedOp, ...]) -> dict[str, SequenceOutputTemplate]:
    return {
        op.output_template.name: op.output_template
        for op in ops
        if (
            isinstance(op.output_template, SequenceOutputTemplate)
            and not isinstance(op.output_template.length_expr, int)
        )
    }


def _dynamic_sequence_element_meta(
    template: SequenceOutputTemplate,
    nodes_by_name: dict[str, TraceNode],
) -> TensorMeta | None:
    for element_name in template.element_names:
        node = nodes_by_name.get(element_name)
        if node is not None and node.tensor_meta is not None:
            return node.tensor_meta
    return None


def _template_schema(value: Any) -> Any:
    if isinstance(value, NodeArg):
        return ("node", value.name)
    if isinstance(value, SequenceArg):
        return ("sequence", value.name, value.container_type)
    if isinstance(value, SequenceElementArg):
        return ("sequence_element", value.sequence_name, value.index)
    if isinstance(value, SequenceOutputTemplate):
        element_schema = value.element_names if isinstance(value.length_expr, int) else ("*",)
        return (
            "sequence_output",
            value.name,
            value.container_type,
            _template_schema(value.length_expr),
            element_schema,
        )
    if isinstance(value, BatchDimArg):
        return ("batch", value.symbol)
    if isinstance(value, ShapeExpr):
        return ("shape_expr", value.expression, value.multiplier, value.offset)
    if isinstance(value, ParamArg):
        return ("param", value.name)
    if isinstance(value, BufferArg):
        return ("buffer", value.name)
    if isinstance(value, TensorAttrArg):
        return ("tensor_attr", value.path)
    if isinstance(value, ConstantTensorArg):
        return ("constant_tensor", tuple(value.value.shape), str(value.value.dtype))
    if isinstance(value, tuple):
        return ("tuple", tuple(_template_schema(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_template_schema(item) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple((key, _template_schema(item)) for key, item in value.items()))
    if isinstance(value, slice):
        return (
            "slice",
            _template_schema(value.start),
            _template_schema(value.stop),
            _template_schema(value.step),
        )
    return value
