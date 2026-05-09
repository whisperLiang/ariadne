from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ariadne import ReplayBoundary, SplitSpec, prepare_split_replay


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


class BatchStructuralVariantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6)
        self.act = nn.GELU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:  # noqa: SIM108 - explicit singleton branch for tracing.
            x = x.view(1, -1)
        else:
            x = x.clone().reshape(x.shape[0], -1)
        return self.out(self.act(self.proj(x)))


class FoldedBatchBoundaryNet(nn.Module):
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


def test_replay_runtime_matches_direct_forward() -> None:
    torch.manual_seed(0)
    model = TinyNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
    )
    inputs = torch.randn(6, 5)

    with torch.no_grad():
        expected = model(inputs)
    boundary = runtime.run_prefix(inputs)
    torch.testing.assert_close(runtime.run_suffix(boundary), expected)
    torch.testing.assert_close(runtime.replay(inputs), expected)


def test_replay_runtime_uses_structural_variant_for_batch_one_traces() -> None:
    torch.manual_seed(0)
    model = BatchStructuralVariantNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(1, 3, 4),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(1, 4),
            trace_batch_mode="batch_1",
        ),
        mode="generated_eager",
    )

    assert len(runtime.variants) == 1
    for batch_size in (1, 2, 4):
        inputs = torch.randn(batch_size, 3, 4)
        with torch.no_grad():
            expected = model(inputs)
        torch.testing.assert_close(runtime.replay(inputs), expected)


def test_replay_batch_one_boundary_stays_on_parent_runtime() -> None:
    torch.manual_seed(0)
    model = TinyNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(1, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(1, 4),
            trace_batch_mode="batch_1",
        ),
        mode="generated_eager",
    )
    assert len(runtime.variants) == 1
    for batch_size in (1, 3):
        inputs = torch.randn(batch_size, 5)
        with torch.no_grad():
            expected = model(inputs)
        boundary = runtime.run_prefix(inputs)

        torch.testing.assert_close(runtime.run_suffix(boundary), expected)
        torch.testing.assert_close(runtime.replay(inputs), expected)


def test_replay_runtime_supports_folded_batch_boundary() -> None:
    torch.manual_seed(0)
    model = FoldedBatchBoundaryNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(2, 3, 4),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 5),
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
        validation="strict",
    )
    inputs = torch.randn(5, 3, 4)

    with torch.no_grad():
        expected = model(inputs)
    torch.testing.assert_close(runtime.replay(inputs), expected)


def test_replay_strict_validation_rejects_wrong_boundary_shape() -> None:
    model = TinyNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 64),
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
        validation="strict",
    )

    boundary = runtime.run_prefix(torch.randn(4, 5))
    bad_boundary = replace(boundary, values=(torch.randn(4, 9),))

    try:
        runtime.run_suffix(bad_boundary)
    except ValueError as error:
        assert "dimension 1" in str(error)
    else:  # pragma: no cover
        raise AssertionError("strict validation accepted a wrong boundary shape")


def test_replay_compiled_smoke_with_aot_eager_backend() -> None:
    torch.manual_seed(0)
    model = TinyNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(2, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 8),
            trace_batch_mode="batch_gt1",
        ),
        mode="compiled",
        compile_options={"backend": "aot_eager", "dynamic": True},
    )
    inputs = torch.randn(3, 5)

    runtime.warmup(inputs)
    with torch.no_grad():
        expected = model(inputs)
    boundary = runtime.run_prefix(inputs)
    assert isinstance(boundary, ReplayBoundary)
    assert len(boundary.values) == len(runtime.boundary_order)
    torch.testing.assert_close(runtime.run_suffix(boundary), expected)
    torch.testing.assert_close(runtime.replay(inputs), expected)
    assert runtime.fallback_reason is None


def test_replay_compiled_boundary_is_materialized_by_default() -> None:
    model = TinyNet().eval()
    runtime = prepare_split_replay(
        model,
        example_inputs=(torch.randn(2, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 8),
            trace_batch_mode="batch_gt1",
        ),
        mode="compiled",
        compile_options={"backend": "aot_eager", "dynamic": True},
    )
    inputs = torch.randn(3, 5)

    raw_values = runtime.prefix_segment(inputs)
    boundary = runtime.run_prefix(inputs)

    assert isinstance(raw_values, tuple)
    assert isinstance(raw_values[0], torch.Tensor)
    assert boundary.values[0] is not raw_values[0]
    torch.testing.assert_close(boundary.values[0], raw_values[0])
