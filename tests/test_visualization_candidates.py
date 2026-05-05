from __future__ import annotations

import torch
from torch import nn

from ariadne.trace.tracer import trace_model
from ariadne.visualization.export import export_split_candidates_table


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 8)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.act(self.layer1(x)))


def test_export_split_candidates_table_lists_required_fields() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    rows = export_split_candidates_table(plan)

    assert rows
    assert {
        "split_id",
        "boundary_after",
        "boundary_nodes",
        "boundary_bytes",
        "prefix_node_count",
        "suffix_node_count",
        "trainable_suffix",
        "passthrough_inputs",
    } <= rows[0].keys()
