from __future__ import annotations

import torch
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.runtime.cache import RuntimeCacheKey
from ariadne.validation.equivalence import assert_forward_equivalent


class ReshapeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6)
        self.act = nn.ReLU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.out(self.act(self.proj(x)))


def test_dynamic_batch_reuses_trace_for_multiple_batch_sizes() -> None:
    model = ReshapeNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(4, 3, 4),),
        split=SplitSpec(boundary="after:act", dynamic_batch=(1, 16), trainable=True),
    )

    for batch_size in (1, 2, 8, 16):
        inputs = (torch.randn(batch_size, 3, 4),)
        assert_forward_equivalent(model, runtime, inputs)


def test_runtime_cache_key_omits_concrete_batch_size() -> None:
    key_1 = RuntimeCacheKey.from_inputs(
        graph_signature="abc",
        split_id="after:act",
        mode="generated_eager",
        shapes=((1, 3, 4),),
        dtypes=("torch.float32",),
    )
    key_8 = RuntimeCacheKey.from_inputs(
        graph_signature="abc",
        split_id="after:act",
        mode="generated_eager",
        shapes=((8, 3, 4),),
        dtypes=("torch.float32",),
    )

    assert key_1 == key_8
