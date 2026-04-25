from __future__ import annotations

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split


class DemoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(5, 16)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


def main() -> None:
    torch.manual_seed(0)
    model = DemoNet().eval()
    example = torch.randn(4, 5)
    runtime = prepare_split(
        model,
        example_inputs=(example,),
        split=SplitSpec(
            boundary="after:layer2",
            dynamic_batch=(2, 64),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
        mode="generated_eager",
    )

    x_batch = torch.randn(8, 5)
    boundary = runtime.run_prefix(x_batch)
    output = runtime.run_suffix(boundary)
    direct = model(x_batch)
    torch.testing.assert_close(output, direct)
    print(
        f"split_id={boundary.split_id} "
        f"batch={boundary.batch_size} output_shape={tuple(output.shape)}"
    )


if __name__ == "__main__":
    main()
