from __future__ import annotations

import torch
from torch import nn

from ariadne.trace.tracer import trace_model


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def test_trace_plan_captures_nodes_and_symbolic_batch() -> None:
    model = TinyNet()
    plan = trace_model(model, example_inputs=(torch.randn(4, 5),), dynamic_batch=(1, 16))

    assert plan.shape_env.traced_batch_size == 4
    assert plan.input_metas[0] is not None
    assert plan.input_metas[0].symbolic_shape == ("B", 5)
    assert any(node.module_path == "layer1" for node in plan.nodes)
    assert any(node.param_refs for node in plan.nodes if node.module_path == "layer2")
    assert plan.graph_signature
