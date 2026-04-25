from __future__ import annotations

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.trace.interception import InterceptionTraceArtifact


class BranchFunctionalNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(x.sum().item() > 0):  # noqa: SIM108 - keeps the branch explicit for tracing.
            x = self.proj(x)
            x = torch.sin(x) + torch.cos(x)
        else:
            x = self.proj(x)
            x = torch.relu(x)
        return self.out(x)


def test_runtime_interception_records_real_python_branch_and_functional_ops() -> None:
    model = BranchFunctionalNet()
    x = torch.ones(3, 4)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(boundary="after:proj", dynamic_batch=(1, 8), trainable=True),
    )

    assert isinstance(runtime.trace_plan.runtime_artifact, InterceptionTraceArtifact)
    assert any("sin" in node.target for node in runtime.trace_plan.nodes)

    with torch.no_grad():
        direct = model(x)
        split = runtime.run_suffix(runtime.run_prefix(x))
    torch.testing.assert_close(split, direct)
