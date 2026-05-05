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


class PoolNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.pool(self.conv1(x)))


class NormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.out = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.bn(self.conv(x)))


class MaxValueNet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _indices = torch.max(x, dim=1)
        return values * 2


class DropoutNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(5, 8)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.fc1(x)))


def test_frontier_planner_selects_named_module_boundary() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    candidate = select_split(
        plan,
        split=SplitSpec(
            boundary="after:act",
            dynamic_batch=(2, 16),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    assert len(candidate.boundary_nodes) == 1
    assert candidate.trainable_suffix
    assert candidate.boundary_schema[candidate.boundary_nodes[0]].symbolic_shape == ("B", 8)


def test_auto_split_returns_lowest_boundary_candidate() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    candidate = select_split(
        plan,
        split="auto",
        objective={"minimize": "boundary_bytes", "constraints": {"trainable_suffix": True}},
    )

    assert candidate in enumerate_frontier_splits(plan)
    assert candidate.trainable_suffix


def test_tracer_marks_dropout_rng_sensitive_and_trainable_split_allows_prefix_rng() -> None:
    plan = trace_model(
        DropoutNet().train(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    assert any(node.module_path == "drop" and node.rng_sensitive for node in plan.nodes)

    candidate = select_split(
        plan,
        split=SplitSpec(
            boundary="after:drop",
            dynamic_batch=(2, 16),
            trainable=True,
            trace_batch_mode="batch_gt1",
        ),
    )

    assert candidate.boundary_after == "drop"
    assert any(plan.get_node(name).rng_sensitive for name in candidate.prefix_nodes)


def test_frontier_planner_ignores_unconsumed_detach_nodes() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    assert any(node.target == "detach.default" for node in plan.nodes)

    candidates = enumerate_frontier_splits(plan)

    assert any(candidate.boundary_after == "act" for candidate in candidates)
    assert all(
        plan.get_node(name).target != "detach.default"
        for candidate in candidates
        for name in (*candidate.prefix_nodes, *candidate.suffix_nodes)
    )


def test_frontier_planner_ignores_unused_multi_output_nodes() -> None:
    plan = trace_model(
        PoolNet(),
        example_inputs=(torch.randn(2, 3, 8, 8),),
        dynamic_batch=(2, 4),
        trace_batch_mode="batch_gt1",
    )
    consumer_counts: dict[str, int] = {}
    for node in plan.nodes:
        for parent in node.parents:
            consumer_counts[parent] = consumer_counts.get(parent, 0) + 1
    unused_pool_outputs = {
        node.name
        for node in plan.nodes
        if node.target == "max_pool2d_with_indices.default"
        and consumer_counts.get(node.name, 0) == 0
    }

    assert unused_pool_outputs

    candidates = enumerate_frontier_splits(plan)

    assert sum(candidate.boundary_after == "pool" for candidate in candidates) == 1
    assert all(
        name not in unused_pool_outputs
        for candidate in candidates
        for name in (*candidate.prefix_nodes, *candidate.suffix_nodes)
    )


def test_frontier_planner_ignores_generic_unused_multi_output_nodes() -> None:
    plan = trace_model(
        MaxValueNet(),
        example_inputs=(torch.randn(3, 5),),
        dynamic_batch=(2, 8),
        trace_batch_mode="batch_gt1",
    )
    consumer_counts: dict[str, int] = {}
    for node in plan.nodes:
        for parent in node.parents:
            consumer_counts[parent] = consumer_counts.get(parent, 0) + 1
    unused_multi_outputs = {
        node.name
        for node in plan.nodes
        if node.target == "max.dim" and consumer_counts.get(node.name, 0) == 0
    }

    assert unused_multi_outputs

    candidates = enumerate_frontier_splits(plan)

    assert sum(candidate.boundary_after == "node_0" for candidate in candidates) == 1
    assert all(
        name not in unused_multi_outputs
        for candidate in candidates
        for name in (*candidate.prefix_nodes, *candidate.suffix_nodes)
    )


def test_frontier_planner_ignores_unused_batchnorm_auxiliary_nodes() -> None:
    plan = trace_model(
        NormNet().eval(),
        example_inputs=(torch.randn(2, 3, 8, 8),),
        dynamic_batch=(2, 4),
        trace_batch_mode="batch_gt1",
    )
    auxiliary_targets = {"empty.memory_format", "native_batch_norm.default"}
    consumer_counts: dict[str, int] = {}
    for node in plan.nodes:
        for parent in node.parents:
            consumer_counts[parent] = consumer_counts.get(parent, 0) + 1
    unused_auxiliary_outputs = {
        node.name
        for node in plan.nodes
        if node.target in auxiliary_targets and consumer_counts.get(node.name, 0) == 0
    }

    assert unused_auxiliary_outputs

    candidates = enumerate_frontier_splits(plan)

    assert sum(candidate.boundary_after == "bn" for candidate in candidates) == 1
    assert all(
        name not in unused_auxiliary_outputs
        for candidate in candidates
        for name in (*candidate.prefix_nodes, *candidate.suffix_nodes)
    )


def test_frontier_planner_does_not_offer_parameter_transpose_as_boundary() -> None:
    plan = trace_model(
        TinyNet(),
        example_inputs=(torch.randn(4, 5),),
        dynamic_batch=(2, 16),
        trace_batch_mode="batch_gt1",
    )

    candidates = enumerate_frontier_splits(plan)

    assert all(
        plan.get_node(boundary).target != "t.default"
        for candidate in candidates
        for boundary in candidate.boundary_nodes
    )
