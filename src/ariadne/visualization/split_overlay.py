"""Render-time split overlay classification."""

from __future__ import annotations

from dataclasses import dataclass

from ariadne.planner.frontier import SplitCandidate


@dataclass(frozen=True)
class SplitOverlay:
    split_id: str
    boundary_after: str
    prefix_nodes: frozenset[str]
    suffix_nodes: frozenset[str]
    boundary_nodes: frozenset[str]
    passthrough_inputs: frozenset[str]
    boundary_bytes: int
    prefix_node_count: int
    suffix_node_count: int
    trainable_suffix: bool


def build_split_overlay(candidate: SplitCandidate) -> SplitOverlay:
    return SplitOverlay(
        split_id=candidate.split_id,
        boundary_after=candidate.boundary_after,
        prefix_nodes=frozenset(candidate.prefix_nodes),
        suffix_nodes=frozenset(candidate.suffix_nodes),
        boundary_nodes=frozenset(candidate.boundary_nodes),
        passthrough_inputs=frozenset(candidate.passthrough_inputs),
        boundary_bytes=candidate.cost.boundary_bytes,
        prefix_node_count=candidate.cost.prefix_node_count,
        suffix_node_count=candidate.cost.suffix_node_count,
        trainable_suffix=candidate.trainable_suffix,
    )


def classify_node_for_split(node_name: str, overlay: SplitOverlay) -> str:
    if node_name in overlay.boundary_nodes:
        return "boundary"
    if node_name in overlay.prefix_nodes:
        return "prefix"
    if node_name in overlay.suffix_nodes:
        return "suffix"
    if node_name in overlay.passthrough_inputs:
        return "passthrough"
    return "neutral"
