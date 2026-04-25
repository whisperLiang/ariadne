from __future__ import annotations

import torch
from torch import nn

from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import enumerate_frontier_splits
from ariadne.planner.selector import select_split
from ariadne.trace.tracer import trace_model


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def test_frontier_planner_selects_named_module_boundary() -> None:
    plan = trace_model(TinyNet(), example_inputs=(torch.randn(4, 5),))

    candidate = select_split(plan, split=SplitSpec(boundary="after:act", trainable=True))

    assert candidate.boundary_nodes == ("act",)
    assert candidate.trainable_suffix
    assert candidate.boundary_schema["act"].symbolic_shape == ("B", 8)


def test_auto_split_returns_lowest_boundary_candidate() -> None:
    plan = trace_model(TinyNet(), example_inputs=(torch.randn(4, 5),))

    candidate = select_split(
        plan,
        split="auto",
        objective={"minimize": "boundary_bytes", "constraints": {"trainable_suffix": True}},
    )

    assert candidate in enumerate_frontier_splits(plan)
    assert candidate.trainable_suffix
