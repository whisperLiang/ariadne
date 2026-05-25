from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.planner.candidate_validation import validate_split_candidates
from ariadne.planner.frontier import enumerate_frontier_splits
from ariadne.trace.tracer import trace_model
from ariadne.validation.gradient import assert_gradient_equivalent


class DictListNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> dict[str, object]:
        hidden = self.act(self.stem(x))
        logits = self.head(hidden)
        return {"logits": logits, "aux": [hidden.mean(dim=1, keepdim=True)]}


class FoldedBatchNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 6)
        self.act = nn.ReLU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = self.act(self.proj(x))
        x = x.reshape(-1, 3, 6).sum(dim=1)
        return self.out(x)


class MaxTupleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _indices = torch.max(x, dim=1)
        return self.out(values)


class BatchDependentContainerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = torch.unbind(x, dim=0)
        projected = [self.proj(piece) for piece in pieces]
        return self.out(torch.stack(projected, dim=0))


@pytest.mark.parametrize(
    ("model", "example", "targets"),
    [
        (
            DictListNet(),
            torch.randn(2, 5),
            {"logits": torch.randn(5, 3), "aux": [torch.randn(5, 1)]},
        ),
        (FoldedBatchNet(), torch.randn(2, 3, 4), torch.randn(5, 2)),
        (MaxTupleNet(), torch.randn(2, 6, 4), torch.randn(5, 2)),
    ],
)
def test_validated_safe_candidates_support_cross_batch_forward_and_training(
    model: nn.Module,
    example: torch.Tensor,
    targets: object,
) -> None:
    torch.manual_seed(0)
    spec = SplitSpec(
        boundary="auto",
        dynamic_batch=(1, 5),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    plan = trace_model(
        model,
        example_inputs=(example,),
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    candidates = validate_split_candidates(
        plan,
        spec=spec,
        example_inputs=(example,),
        candidates=enumerate_frontier_splits(plan),
        require_training=True,
    )
    safe_candidates = [
        candidate
        for candidate in candidates
        if candidate.rejection_reason is None and candidate.trainable_suffix
    ]

    assert safe_candidates
    assert all(candidate.rejection_reason is None for candidate in candidates)

    for candidate in safe_candidates:
        direct_model = copy.deepcopy(model)
        split_model = copy.deepcopy(direct_model)
        runtime = prepare_split(
            split_model,
            example_inputs=(example.detach().clone().requires_grad_(True),),
            split=SplitSpec(
                boundary=candidate.split_id,
                dynamic_batch=(1, 5),
                trainable=True,
                trace_batch_mode="batch_gt1",
            ),
        )
        for batch_size in (1, 5):
            direct_inputs = (torch.randn(batch_size, *example.shape[1:], requires_grad=True),)
            split_inputs = (direct_inputs[0].detach().clone().requires_grad_(True),)
            with torch.no_grad():
                _assert_nested_close(
                    runtime.run_suffix(runtime.run_prefix(*split_inputs)),
                    direct_model(*split_inputs),
                )
            assert_gradient_equivalent(
                direct_model,
                runtime,
                direct_inputs,
                split_inputs,
                _resize_targets(targets, batch_size),
                loss_fn=_loss_for(targets),
            )


def test_batch_dependent_python_container_splits_are_rejected_before_replay() -> None:
    model = BatchDependentContainerNet()
    spec = SplitSpec(
        boundary="auto",
        dynamic_batch=(2, 4),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    example = torch.randn(2, 4)
    plan = trace_model(
        model,
        example_inputs=(example,),
        dynamic_batch=spec.dynamic_batch,
        trace_batch_mode=spec.trace_batch_mode,
    )
    candidates = validate_split_candidates(
        plan,
        spec=spec,
        example_inputs=(example,),
        candidates=enumerate_frontier_splits(plan),
        require_training=True,
    )
    rejected = [
        candidate
        for candidate in candidates
        if candidate.rejection_reason
        and (
            "dynamic sequence boundaries are not supported" in candidate.rejection_reason
            or "element-wise" in candidate.rejection_reason
        )
    ]

    assert rejected
    with pytest.raises(ValueError, match="dynamic sequence|element-wise"):
        prepare_split(
            model,
            example_inputs=(example,),
            split=SplitSpec(
                boundary=rejected[-1].split_id,
                dynamic_batch=(2, 4),
                trainable=True,
                trace_batch_mode="batch_gt1",
            ),
        )


def _loss_for(targets: object):
    if isinstance(targets, dict):
        return lambda outputs, _targets: _nested_loss(outputs)
    return F.mse_loss


def _resize_targets(value: object, batch_size: int) -> object:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0:
            return value[:batch_size].detach().clone()
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_resize_targets(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_resize_targets(item, batch_size) for item in value]
    if isinstance(value, dict):
        return {key: _resize_targets(item, batch_size) for key, item in value.items()}
    return value


def _assert_nested_close(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-4, atol=1e-5)
        return
    if isinstance(left, tuple) and isinstance(right, tuple):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key])
        return
    assert left == right


def _nested_loss(value: object) -> torch.Tensor:
    terms = _loss_terms(value)
    assert terms
    loss = terms[0]
    for term in terms[1:]:
        loss = loss + term
    return loss


def _loss_terms(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return [value.float().square().mean()]
        return []
    if isinstance(value, (tuple, list)):
        return [term for item in value for term in _loss_terms(item)]
    if isinstance(value, dict):
        return [term for item in value.values() for term in _loss_terms(item)]
    return []
