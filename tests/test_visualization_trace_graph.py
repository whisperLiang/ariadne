from __future__ import annotations

import pytest
import torch
from torch import nn

from ariadne.trace.tracer import trace_model
from ariadne.visualization.export import export_trace_dot


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


class PoolNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.pool(self.conv1(x)))


class NormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.out = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.bn(self.conv(x)))


class MaxValueNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _indices = torch.max(x, dim=1)
        return values * 2


def test_export_trace_dot_contains_trace_metadata_and_edges() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    dot = export_trace_dot(plan)

    assert plan.graph_signature in dot
    assert "input_0" in dot
    assert "output" in dot
    assert '"input_0" -> "node_' in dot
    assert "layer1" in dot
    assert "Linear" in dot
    assert dot.count("Linear") == 2
    assert "call_function" not in dot
    assert "detach.default" not in dot
    assert "t.default" not in dot


def test_render_trace_graph_return_dot_contains_trace_metadata() -> None:
    pytest.importorskip("graphviz")
    from ariadne.visualization import render_trace_graph

    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    dot = render_trace_graph(plan, return_dot=True)

    assert dot is not None
    assert plan.graph_signature in dot
    assert "input_0" in dot
    assert "output" in dot


def test_export_trace_dot_collapses_unused_maxpool_indices_output() -> None:
    plan = trace_model(
        PoolNet(),
        example_inputs=(torch.randn(2, 3, 8, 8),),
        dynamic_batch=(2, 4),
        trace_batch_mode="batch_gt1",
    )

    dot = export_trace_dot(plan)

    assert dot.count("MaxPool2d") == 1
    assert dot.count("max_pool2d_with_indices.default") == 1


def test_export_trace_dot_collapses_generic_unused_multi_output() -> None:
    plan = trace_model(
        MaxValueNet(),
        example_inputs=(torch.randn(3, 5),),
        dynamic_batch=(2, 8),
        trace_batch_mode="batch_gt1",
    )

    dot = export_trace_dot(plan)

    assert '"node_0"' in dot
    assert "node_0_1" not in dot


def test_export_trace_dot_collapses_unused_batchnorm_auxiliary_outputs() -> None:
    plan = trace_model(
        NormNet().eval(),
        example_inputs=(torch.randn(2, 3, 8, 8),),
        dynamic_batch=(2, 4),
        trace_batch_mode="batch_gt1",
    )

    dot = export_trace_dot(plan)

    assert dot.count("BatchNorm2d") == 1
    assert dot.count("native_batch_norm.default") == 1
    assert "empty.memory_format" not in dot
