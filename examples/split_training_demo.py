from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


def main() -> None:
    torch.manual_seed(0)
    model = DemoNet()
    x = torch.randn(4, 5, requires_grad=True)
    targets = torch.randn(4, 3)
    runtime = prepare_split(
        model,
        example_inputs=(x,),
        split=SplitSpec(boundary="after:layer2", dynamic_batch=(1, 64), trainable=True),
    )

    suffix_optimizer = torch.optim.SGD(model.layer3.parameters(), lr=0.01)
    prefix_optimizer = torch.optim.SGD(
        list(model.layer1.parameters()) + list(model.layer2.parameters()),
        lr=0.01,
    )

    boundary = runtime.run_prefix(x)
    loss, boundary_grads = runtime.train_suffix(
        boundary,
        targets,
        loss_fn=F.mse_loss,
        optimizer=suffix_optimizer,
    )
    runtime.backward_prefix(x, boundary_grads=boundary_grads, optimizer=prefix_optimizer)
    print(f"loss={float(loss):.6f} boundary_labels={list(boundary_grads)}")


if __name__ == "__main__":
    main()
