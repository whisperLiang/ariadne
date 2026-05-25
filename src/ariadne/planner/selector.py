"""Split candidate selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose
from typing import Any

from ariadne.pattern.split_spec import SplitSpec, parse_boundary_percent
from ariadne.planner.frontier import SplitCandidate, enumerate_frontier_splits
from ariadne.trace.trace_plan import TraceNode, TracePlan


def select_split(
    plan: TracePlan,
    *,
    split: SplitSpec | str,
    objective: Mapping[str, Any] | None = None,
    candidates: Sequence[SplitCandidate] | None = None,
) -> SplitCandidate:
    all_candidates = list(candidates if candidates is not None else enumerate_frontier_splits(plan))
    valid_candidates = [
        candidate for candidate in all_candidates if candidate.rejection_reason is None
    ]
    if not all_candidates:
        raise ValueError("No valid frontier split candidates were found.")

    if split == "auto":
        return _select_auto(
            valid_candidates,
            objective,
            require_trainable_suffix=False,
        )
    if isinstance(split, SplitSpec) and split.boundary == "auto":
        return _select_auto(
            valid_candidates,
            objective,
            require_trainable_suffix=split.trainable,
        )

    if not isinstance(split, SplitSpec):
        raise TypeError("split must be a SplitSpec or 'auto'.")

    percent = parse_boundary_percent(split.boundary)
    if percent is not None:
        return _select_percent_candidate(
            valid_candidates,
            percent,
            require_trainable_suffix=split.trainable,
        )

    requested = split.boundary.removeprefix("after:")
    matches = [
        candidate for candidate in all_candidates if _matches_boundary(plan, candidate, requested)
    ]
    if not matches:
        labels = ", ".join(f"after:{candidate.boundary_after}" for candidate in all_candidates)
        raise ValueError(f"No split matches {split.boundary!r}. Available split labels: {labels}.")
    module_exits = [
        candidate
        for candidate in matches
        if _is_module_exit_candidate(plan, candidate, requested)
    ]
    selected = (
        min(
            module_exits,
            key=lambda candidate: _module_exit_rank(plan, candidate, requested),
        )
        if module_exits
        else matches[-1]
    )
    if selected.rejection_reason is not None:
        raise ValueError(
            f"Split {split.boundary!r} is not replayable: {selected.rejection_reason}"
        )
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


def _select_percent_candidate(
    candidates: list[SplitCandidate],
    percent: float,
    *,
    require_trainable_suffix: bool,
) -> SplitCandidate:
    ranked_candidates = list(enumerate(candidates))
    if require_trainable_suffix:
        ranked_candidates = [
            (index, candidate)
            for index, candidate in ranked_candidates
            if candidate.trainable_suffix
        ]
    if not ranked_candidates:
        raise ValueError("No split candidates satisfy the requested percentage constraints.")

    target_index = (percent / 100.0) * (len(candidates) - 1)
    return min(
        ranked_candidates,
        key=lambda item: (
            abs(item[0] - target_index),
            0 if isclose(float(item[0]), target_index) else 1,
            item[0],
        ),
    )[1]


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
    requested_nodes = _requested_module_nodes(plan, requested)
    if requested_nodes and requested_nodes[-1].name == candidate_node.name:
        return True
    if requested_nodes:
        last_requested_index = max(plan.index_of(node.name) for node in requested_nodes)
        candidate_index = plan.index_of(candidate_node.name)
        if candidate_index > last_requested_index:
            intervening = plan.nodes[last_requested_index + 1 : candidate_index]
            return all(not node.is_compute for node in intervening)
    return False


def _is_module_exit_candidate(
    plan: TracePlan,
    candidate: SplitCandidate,
    requested: str,
) -> bool:
    requested_nodes = _requested_module_nodes(plan, requested)
    if not requested_nodes:
        return False
    if any(
        _module_path_matches(plan.get_node(name).module_path, requested)
        for name in candidate.suffix_nodes
    ):
        return False
    candidate_node = _node_for_candidate(plan, candidate)
    candidate_index = plan.index_of(candidate_node.name)
    first_requested_index = min(plan.index_of(node.name) for node in requested_nodes)
    return candidate_index >= first_requested_index


def _module_exit_rank(
    plan: TracePlan,
    candidate: SplitCandidate,
    requested: str,
) -> tuple[int, int]:
    candidate_node = _node_for_candidate(plan, candidate)
    candidate_index = plan.index_of(candidate_node.name)
    requested_nodes = _requested_module_nodes(plan, requested)
    requested_indexes = [plan.index_of(node.name) for node in requested_nodes]
    last_requested_index = max(requested_indexes) if requested_indexes else candidate_index

    if candidate.boundary_after == requested or candidate_node.module_path == requested:
        return (0, candidate_index)
    if candidate_index > last_requested_index:
        intervening = plan.nodes[last_requested_index + 1 : candidate_index]
        if all(not node.is_compute for node in intervening):
            return (1, candidate_index)
    if _module_path_matches(candidate_node.module_path, requested):
        return (2, candidate_index)
    return (3, candidate_index)


def _requested_module_nodes(plan: TracePlan, requested: str) -> list[TraceNode]:
    return [
        node
        for node in plan.nodes
        if node.module_path == requested or (node.module_path or "").startswith(f"{requested}.")
    ]


def _module_path_matches(module_path: str | None, requested: str) -> bool:
    return module_path == requested or (module_path or "").startswith(f"{requested}.")


def _node_for_candidate(plan: TracePlan, candidate: SplitCandidate) -> TraceNode:
    boundary = candidate.boundary_after
    for node in reversed(plan.nodes):
        if node.module_path == boundary or node.name == boundary:
            return node
    for node in reversed(plan.nodes):
        if node.module_path and node.module_path.startswith(f"{boundary}."):
            return node
    raise KeyError(f"Could not resolve candidate boundary {boundary!r}.")
