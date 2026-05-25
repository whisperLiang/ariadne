"""Prepare-time validation for safe split candidates."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from ariadne.codegen.segment_builder import build_segments
from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import SplitCandidate
from ariadne.runtime.batching import first_batch_size, resize_batch
from ariadne.runtime.segment_runtime import SplitRuntime
from ariadne.trace.trace_plan import TracePlan


def validate_split_candidates(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    example_inputs: tuple[Any, ...],
    candidates: Sequence[SplitCandidate],
    require_training: bool,
) -> tuple[SplitCandidate, ...]:
    """Return candidates annotated with rejection reasons for unsafe frontiers."""
    validation_inputs = _validation_input_batches(spec, example_inputs)
    validated: list[SplitCandidate] = []
    for candidate in candidates:
        reason = candidate.rejection_reason or _static_rejection_reason(
            plan,
            spec=spec,
            candidate=candidate,
        )
        if reason is None:
            reason = _runtime_rejection_reason(
                plan,
                spec=spec,
                candidate=candidate,
                validation_inputs=validation_inputs,
                require_training=require_training,
            )
        validated.append(replace(candidate, rejection_reason=reason))
    return tuple(validated)


def _static_rejection_reason(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    candidate: SplitCandidate,
) -> str | None:
    for node_name in (*candidate.prefix_nodes, *candidate.suffix_nodes):
        node = plan.get_node(node_name)
        metadata = node.alias_metadata or {}
        if metadata.get("unsupported_dynamic_sequence_use"):
            label = node.module_path or node.name
            return (
                f"operation {label!r} uses a dynamic sequence element-wise after trace; "
                "batch-polymorphic replay only supports whole-sequence consumption"
            )
        if metadata.get("dynamic_output_structure"):
            label = node.module_path or node.name
            reason = metadata.get("dynamic_structure_reason")
            if isinstance(reason, str) and reason:
                return f"operation {label!r} is not batch-polymorphic: {reason}"
            return (
                f"operation {label!r} produced a batch-dependent Python container "
                "structure during trace/probe validation; static generated segments "
                "cannot safely replay this boundary for arbitrary batch sizes"
            )
    return None


def _runtime_rejection_reason(
    plan: TracePlan,
    *,
    spec: SplitSpec,
    candidate: SplitCandidate,
    validation_inputs: tuple[tuple[Any, ...], ...],
    require_training: bool,
) -> str | None:
    try:
        runtime = SplitRuntime(
            trace_plan=plan,
            split_spec=spec,
            candidate=candidate,
            segments=build_segments(plan, candidate),
            mode="generated_eager",
        )
    except Exception as error:
        return f"failed to build generated split segments: {_short_error(error)}"

    for inputs in validation_inputs:
        reason = _forward_rejection_reason(plan.root_module, runtime, inputs)
        if reason is not None:
            return reason
        if require_training:
            reason = _training_rejection_reason(plan.root_module, runtime, inputs)
            if reason is not None:
                return reason
    return None


def _validation_input_batches(
    spec: SplitSpec,
    example_inputs: tuple[Any, ...],
) -> tuple[tuple[Any, ...], ...]:
    traced_batch = first_batch_size(example_inputs)
    if traced_batch is None:
        return (example_inputs,)
    batches: list[tuple[Any, ...]] = [example_inputs]
    if spec.trace_batch_mode != "batch_gt1" or spec.dynamic_batch is None:
        return tuple(batches)
    low, high = spec.dynamic_batch
    probe_batch = _choose_probe_batch(traced_batch, spec.dynamic_batch)
    if probe_batch is not None:
        batches.append(resize_batch(example_inputs, traced_batch, probe_batch))
    if low == 1 and traced_batch != 1 and probe_batch != 1:
        batches.append(resize_batch(example_inputs, traced_batch, 1))
    if high != traced_batch and high != probe_batch and high != 1:
        batches.append(resize_batch(example_inputs, traced_batch, high))
    return tuple(batches)


def _choose_probe_batch(
    traced_batch: int,
    dynamic_batch: tuple[int, int],
) -> int | None:
    low, high = dynamic_batch
    next_batch = traced_batch + 1
    if low <= next_batch <= high:
        return next_batch
    if low <= high and low != traced_batch:
        return low
    if high != traced_batch:
        return high
    return None


def _forward_rejection_reason(
    model: torch.nn.Module,
    runtime: SplitRuntime,
    inputs: tuple[Any, ...],
) -> str | None:
    buffer_snapshot = _snapshot_buffers(model)
    rng_snapshot = _snapshot_rng()
    try:
        with torch.no_grad():
            expected = model(*_clone_tree(inputs))
        _restore_buffers(model, buffer_snapshot)
        _restore_rng(rng_snapshot)
        with torch.no_grad():
            actual = runtime.run_suffix(runtime.run_prefix(*_clone_tree(inputs)))
        _assert_nested_close(actual, expected)
    except Exception as error:
        return f"forward replay validation failed: {_short_error(error)}"
    finally:
        _restore_buffers(model, buffer_snapshot)
        _restore_rng(rng_snapshot)
    return None


def _training_rejection_reason(
    model: torch.nn.Module,
    runtime: SplitRuntime,
    inputs: tuple[Any, ...],
) -> str | None:
    buffer_snapshot = _snapshot_buffers(model)
    rng_snapshot = _snapshot_rng()
    try:
        model.zero_grad(set_to_none=True)
        direct_inputs = _clone_tree(inputs, requires_grad=True)
        direct_loss = _synthetic_loss(model(*direct_inputs))
        direct_loss.backward()
        expected_grads = _parameter_grads(model)

        _restore_buffers(model, buffer_snapshot)
        _restore_rng(rng_snapshot)
        model.zero_grad(set_to_none=True)
        split_inputs = _clone_tree(inputs, requires_grad=True)
        boundary = runtime.run_training_prefix(*split_inputs)
        split_loss, boundary_grads = runtime.train_suffix(
            boundary,
            None,
            loss_fn=lambda outputs, _targets: _synthetic_loss(outputs),
        )
        runtime.backward_prefix(boundary, boundary_grads=boundary_grads)
        actual_grads = _parameter_grads(model)

        torch.testing.assert_close(split_loss, direct_loss.detach(), rtol=1e-4, atol=1e-5)
        _assert_gradients_close(actual_grads, expected_grads)
    except Exception as error:
        return f"split training validation failed: {_short_error(error)}"
    finally:
        model.zero_grad(set_to_none=True)
        _restore_buffers(model, buffer_snapshot)
        _restore_rng(rng_snapshot)
    return None


def _clone_tree(value: Any, *, requires_grad: bool = False) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if requires_grad and (cloned.is_floating_point() or cloned.is_complex()):
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, tuple):
        return tuple(_clone_tree(item, requires_grad=requires_grad) for item in value)
    if isinstance(value, list):
        return [_clone_tree(item, requires_grad=requires_grad) for item in value]
    if isinstance(value, dict):
        return {key: _clone_tree(item, requires_grad=requires_grad) for key, item in value.items()}
    return value


def _synthetic_loss(value: Any) -> torch.Tensor:
    terms = _loss_terms(value)
    if not terms:
        raise TypeError("synthetic validation loss requires at least one floating tensor output")
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def _loss_terms(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return [value.float().square().mean()]
        return []
    if isinstance(value, (tuple, list)):
        return [term for item in value for term in _loss_terms(item)]
    if isinstance(value, dict):
        return [term for item in value.values() for term in _loss_terms(item)]
    return []


def _assert_nested_close(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise AssertionError(f"tuple length mismatch {len(left)} != {len(right)}")
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"list length mismatch {len(left)} != {len(right)}")
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"dict key mismatch {left.keys()} != {right.keys()}")
        for key in left:
            _assert_nested_close(left[key], right[key])
        return
    if left != right:
        raise AssertionError(f"value mismatch {left!r} != {right!r}")


def _parameter_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_gradients_close(
    actual: dict[str, torch.Tensor | None],
    expected: dict[str, torch.Tensor | None],
) -> None:
    for name, expected_grad in expected.items():
        actual_grad = actual[name]
        if expected_grad is None and actual_grad is None:
            continue
        if expected_grad is None or actual_grad is None:
            raise AssertionError(f"gradient presence mismatch for parameter {name!r}")
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _restore_buffers(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    buffers = dict(model.named_buffers())
    for name, saved in snapshot.items():
        buffer = buffers.get(name)
        if buffer is not None and tuple(buffer.shape) == tuple(saved.shape):
            buffer.detach().copy_(saved)


def _snapshot_rng() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def _restore_rng(snapshot: tuple[torch.Tensor, list[torch.Tensor] | None]) -> None:
    cpu_state, cuda_state = snapshot
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _short_error(error: BaseException) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__
