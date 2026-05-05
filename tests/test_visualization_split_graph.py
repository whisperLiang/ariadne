from __future__ import annotations

import pytest
import torch
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.visualization.export import export_split_dot


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def _runtime_for_split() -> object:
    return prepare_split(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 16),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )


def test_export_split_dot_contains_split_overlay_metadata() -> None:
    runtime = _runtime_for_split()

    dot = export_split_dot(runtime.trace_plan, runtime.candidate)

    assert runtime.candidate.split_id in dot
    assert f"boundary_after={runtime.candidate.boundary_after}" in dot
    assert f"boundary_bytes={runtime.candidate.cost.boundary_bytes}" in dot
    assert 'ariadne_split_role="prefix"' in dot
    assert 'ariadne_split_role="suffix"' in dot
    assert 'ariadne_split_role="boundary"' in dot
    assert "detach.default" not in dot


def test_render_split_graph_return_dot_contains_split_overlay_metadata() -> None:
    pytest.importorskip("graphviz")
    from ariadne.visualization import render_split_graph

    runtime = _runtime_for_split()

    dot = render_split_graph(runtime.trace_plan, runtime.candidate, return_dot=True)

    assert dot is not None
    assert runtime.candidate.split_id in dot
    assert runtime.candidate.boundary_after in dot
    assert "boundary_bytes" in dot
    assert "ariadne_split_role=prefix" in dot
    assert "ariadne_split_role=suffix" in dot
    assert "ariadne_split_role=boundary" in dot
