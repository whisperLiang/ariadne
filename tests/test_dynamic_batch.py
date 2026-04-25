from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from ariadne import SplitSpec, prepare_split
from ariadne.runtime.cache import RuntimeCacheKey
from ariadne.validation.equivalence import assert_forward_equivalent
from ariadne.validation.gradient import assert_gradient_equivalent


class ReshapeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6)
        self.act = nn.ReLU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.out(self.act(self.proj(x)))


class BatchOneAmbiguousSingletonNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6)
        self.act = nn.ReLU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 1, -1)
        x = x.reshape(x.shape[0], -1)
        return self.out(self.act(self.proj(x)))


class BatchStructuralVariantNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(12, 6)
        self.act = nn.GELU()
        self.out = nn.Linear(6, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 1:  # noqa: SIM108 - keeps the structural trace branch explicit.
            x = x.view(1, -1)
        else:
            x = x.clone().reshape(x.shape[0], -1)
        return self.out(self.act(self.proj(x)))


def test_dynamic_batch_reuses_trace_for_multiple_batch_sizes() -> None:
    model = ReshapeNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(4, 3, 4),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 16),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    for batch_size in (2, 8, 16):
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


def test_batch_one_trace_replays_with_static_singleton_dimensions() -> None:
    torch.manual_seed(0)
    model = BatchOneAmbiguousSingletonNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(1, 3, 4),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(1, 4),
            trainable=True,
            trace_batch_mode="batch_1",
        ),
    )

    for batch_size in (1, 2, 4):
        inputs = (torch.randn(batch_size, 3, 4),)
        assert_forward_equivalent(model, runtime, inputs)


def test_batch_one_trace_supports_split_training_with_larger_batch() -> None:
    torch.manual_seed(0)
    direct_model = BatchOneAmbiguousSingletonNet()
    split_model = copy.deepcopy(direct_model)
    runtime = prepare_split(
        split_model,
        example_inputs=(torch.randn(1, 3, 4, requires_grad=True),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(1, 4),
            trainable=True,
            trace_batch_mode="batch_1",
        ),
    )

    x_direct = torch.randn(4, 3, 4, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    targets = torch.randn(4, 2)
    assert_gradient_equivalent(
        direct_model,
        runtime,
        (x_direct,),
        (x_split,),
        targets,
        loss_fn=F.mse_loss,
    )


def test_batch_one_trace_uses_prepared_structural_variant_for_larger_batch() -> None:
    torch.manual_seed(0)
    model = BatchStructuralVariantNet()
    runtime = prepare_split(
        model,
        example_inputs=(torch.randn(1, 3, 4),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(1, 4),
            trainable=True,
            trace_batch_mode="batch_1",
        ),
    )

    assert len(runtime.variants) == 1
    for batch_size in (1, 2, 4):
        inputs = (torch.randn(batch_size, 3, 4),)
        assert_forward_equivalent(model, runtime, inputs)


def test_batch_gt1_trace_supports_cross_batch_training() -> None:
    torch.manual_seed(0)
    direct_model = ReshapeNet()
    split_model = copy.deepcopy(direct_model)
    runtime = prepare_split(
        split_model,
        example_inputs=(torch.randn(2, 3, 4, requires_grad=True),),
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 5),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    x_direct = torch.randn(5, 3, 4, requires_grad=True)
    x_split = x_direct.detach().clone().requires_grad_(True)
    targets = torch.randn(5, 2)
    assert_gradient_equivalent(
        direct_model,
        runtime,
        (x_direct,),
        (x_split,),
        targets,
        loss_fn=F.mse_loss,
    )
