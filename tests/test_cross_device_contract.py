from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from ariadne import SplitSpec, prepare_split, prepare_split_replay
from ariadne.planner.frontier import enumerate_frontier_splits
from ariadne.trace.tracer import trace_model


class ConvBnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.SiLU()
        self.out = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.bn(self.conv(x))))


class TrainSplitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 7)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(7, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_cpu_cuda_candidates_share_semantic_boundary_contracts() -> None:
    torch.manual_seed(0)
    cpu_model = ConvBnNet().eval()
    cuda_model = copy.deepcopy(cpu_model).eval().cuda()
    cpu_inputs = (torch.randn(2, 3, 8, 8),)
    cuda_inputs = (cpu_inputs[0].cuda(),)

    cpu_plan = trace_model(
        cpu_model,
        example_inputs=cpu_inputs,
        dynamic_batch=(2, 3),
        trace_batch_mode="batch_gt1",
    )
    cuda_plan = trace_model(
        cuda_model,
        example_inputs=cuda_inputs,
        dynamic_batch=(2, 3),
        trace_batch_mode="batch_gt1",
    )

    cpu_candidates = enumerate_frontier_splits(cpu_plan)
    cuda_candidates = enumerate_frontier_splits(cuda_plan)

    assert len(cpu_candidates) == len(cuda_candidates)
    assert any(
        cpu.boundary_nodes != cuda.boundary_nodes
        for cpu, cuda in zip(cpu_candidates, cuda_candidates, strict=True)
    )
    for cpu, cuda in zip(cpu_candidates, cuda_candidates, strict=True):
        assert cpu.semantic_split_id == cuda.semantic_split_id
        assert cpu.boundary_keys == cuda.boundary_keys
        assert cpu.contract_signature == cuda.contract_signature


def test_cpu_prefix_cuda_suffix_replay_uses_semantic_boundary_contract() -> None:
    torch.manual_seed(1)
    cpu_model = ConvBnNet().eval()
    cuda_model = copy.deepcopy(cpu_model).eval().cuda()
    example = torch.randn(2, 3, 8, 8)
    split = SplitSpec(
        boundary="after:bn",
        dynamic_batch=(2, 3),
        trace_batch_mode="batch_gt1",
    )
    cpu_runtime = prepare_split_replay(
        cpu_model,
        example_inputs=(example,),
        split=split,
        mode="generated_eager",
        validation="fast",
    )
    cuda_runtime = prepare_split_replay(
        cuda_model,
        example_inputs=(example.cuda(),),
        split=split,
        mode="generated_eager",
        validation="fast",
    )

    cpu_boundary = cpu_runtime.run_prefix(example)
    mixed_output = cuda_runtime.run_suffix(cpu_boundary)

    assert mixed_output.device.type == "cuda"
    assert cpu_boundary.contract_signature == cuda_runtime.contract_signature
    torch.testing.assert_close(
        mixed_output.cpu(),
        cpu_model(example),
        rtol=1e-4,
        atol=1e-5,
    )


def test_cpu_prefix_cuda_suffix_train_backpropagates_to_prefix() -> None:
    torch.manual_seed(2)
    cpu_model = TrainSplitNet()
    cuda_model = copy.deepcopy(cpu_model).cuda()
    example = torch.randn(2, 5)
    split = SplitSpec(
        boundary="after:act",
        dynamic_batch=(2, 4),
        trainable=True,
        trace_batch_mode="batch_gt1",
    )
    cpu_runtime = prepare_split(
        cpu_model,
        example_inputs=(example,),
        split=split,
        mode="generated_eager",
    )
    cuda_runtime = prepare_split(
        cuda_model,
        example_inputs=(example.cuda(),),
        split=split,
        mode="generated_eager",
    )

    boundary = cpu_runtime.run_training_prefix(torch.randn(3, 5, requires_grad=True))
    targets = torch.randn(3, 3, device="cuda")
    _loss, boundary_grads = cuda_runtime.train_suffix(boundary, targets)
    cpu_runtime.backward_prefix(boundary, boundary_grads)

    assert cpu_model.fc1.weight.grad is not None
    assert cuda_model.fc2.weight.grad is not None
