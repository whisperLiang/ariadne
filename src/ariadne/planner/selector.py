"""Split candidate selection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ariadne.pattern.split_spec import SplitSpec
from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits
from ariadne.trace.trace_plan import TraceNode, TracePlan


def select_split(
    plan: TracePlan,
    *,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None = None,
) -> SplitCandidate:
    candidates = list(enumerate_frontier_splits(plan))
    if not candidates:
        raise ValueError("No valid frontier split candidates were found.")

    if split == "auto":
        return _select_auto(
            candidates,
            objective,
            require_trainable_suffix=False,
        )
    if isinstance(split, SplitSpec) and split.boundary == "auto":
        return _select_auto(
            candidates,
            objective,
            require_trainable_suffix=split.trainable,
        )

    if not isinstance(split, SplitSpec):
        raise TypeError("split must be a SplitSpec or 'auto'.")

    requested = split.boundary.removeprefix("after:")
    matches = [
        candidate for candidate in candidates if _matches_boundary(plan, candidate, requested)
    ]
    if not matches:
        labels = ", ".join(f"after:{candidate.boundary_after}" for candidate in candidates)
        raise ValueError(f"No split matches {split.boundary!r}. Available split labels: {labels}.")
    selected = matches[-1]
    if split.trainable and not selected.trainable_suffix:
        raise ValueError(f"Split {split.boundary!r} does not have trainable suffix parameters.")
    return selected


def _select_auto(
    candidates: list[SplitCandidate],
    objective: Mapping[str, Any] | None,
    *,
    require_trainable_suffix: bool,
) -> SplitCandidate:
    filtered = candidates
    constraints = dict(objective.get("constraints", {})) if objective is not None else {}
    if constraints.get("trainable_suffix") or require_trainable_suffix:
        filtered = [candidate for candidate in filtered if candidate.trainable_suffix]
    if not filtered:
        raise ValueError("No split candidates satisfy the requested objective constraints.")

    minimize = objective.get("minimize") if objective is not None else "boundary_bytes"
    if minimize == "boundary_bytes":
        return min(filtered, key=lambda candidate: candidate.cost.boundary_bytes)
    return min(filtered, key=lambda candidate: candidate.cost.boundary_bytes)


def _matches_boundary(plan: TracePlan, candidate: SplitCandidate, requested: str) -> bool:
    if requested in {
        candidate.boundary_after,
        candidate.split_id,
        f"after:{candidate.boundary_after}",
    }:
        return True
    candidate_node = _node_for_candidate(plan, candidate)
    if requested == candidate_node.name:
        return True
    if candidate_node.module_path and (
        candidate_node.module_path == requested
        or candidate_node.module_path.startswith(f"{requested}.")
    ):
        return True
    requested_nodes = [
        node
        for node in plan.nodes
        if node.module_path == requested or (node.module_path or "").startswith(f"{requested}.")
    ]
    return bool(requested_nodes and requested_nodes[-1].name == candidate_node.name)


def _node_for_candidate(plan: TracePlan, candidate: SplitCandidate) -> TraceNode:
    boundary = candidate.boundary_after
    for node in reversed(plan.nodes):
        if node.module_path == boundary or node.name == boundary:
            return node
    for node in reversed(plan.nodes):
        if node.module_path and node.module_path.startswith(f"{boundary}."):
            return node
    raise KeyError(f"Could not resolve candidate boundary {boundary!r}.")
