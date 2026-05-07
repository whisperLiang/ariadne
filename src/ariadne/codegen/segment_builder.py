"""Build prefix and suffix segments from runtime-interception traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ariadne.codegen.interception_segments import (
    build_interception_prefix,
    build_interception_suffix,
)
from ariadne.planner.frontier import SplitCandidate
from ariadne.trace.interception import InterceptionTraceArtifact
from ariadne.trace.trace_plan import TracePlan


@dataclass(frozen=True)
class SegmentBundle:
    prefix: torch.nn.Module
    training_prefix: torch.nn.Module
    suffix: torch.nn.Module
    boundary_order: tuple[str, ...]
    passthrough_order: tuple[str, ...]


@dataclass(frozen=True)
class ReplaySegmentBundle:
    prefix: torch.nn.Module
    suffix: torch.nn.Module
    boundary_order: tuple[str, ...]
    passthrough_order: tuple[str, ...]


def build_segments(plan: TracePlan, candidate: SplitCandidate) -> SegmentBundle:
    """Build generated eager prefix/suffix callables from a captured runtime trace."""
    artifact = _interception_artifact(plan)
    return SegmentBundle(
        prefix=build_interception_prefix(
            root=plan.root_module,
            artifact=artifact,
            op_names=candidate.prefix_nodes,
            raw_input_names=plan.input_node_names,
            boundary_order=candidate.boundary_nodes,
            class_name="PrefixSegment",
        ),
        training_prefix=build_interception_prefix(
            root=plan.root_module,
            artifact=artifact,
            op_names=candidate.prefix_nodes,
            raw_input_names=plan.input_node_names,
            boundary_order=candidate.boundary_nodes,
            class_name="TrainingPrefixSegment",
        ),
        suffix=build_interception_suffix(
            root=plan.root_module,
            artifact=artifact,
            op_names=candidate.suffix_nodes,
            boundary_order=candidate.boundary_nodes,
            passthrough_order=candidate.passthrough_inputs,
        ),
        boundary_order=candidate.boundary_nodes,
        passthrough_order=candidate.passthrough_inputs,
    )


def build_replay_segments(plan: TracePlan, candidate: SplitCandidate) -> ReplaySegmentBundle:
    """Build only the prefix/suffix callables needed for inference replay."""
    artifact = _interception_artifact(plan)
    return ReplaySegmentBundle(
        prefix=build_interception_prefix(
            root=plan.root_module,
            artifact=artifact,
            op_names=candidate.prefix_nodes,
            raw_input_names=plan.input_node_names,
            boundary_order=candidate.boundary_nodes,
            class_name="ReplayPrefixSegment",
        ),
        suffix=build_interception_suffix(
            root=plan.root_module,
            artifact=artifact,
            op_names=candidate.suffix_nodes,
            boundary_order=candidate.boundary_nodes,
            passthrough_order=candidate.passthrough_inputs,
        ),
        boundary_order=candidate.boundary_nodes,
        passthrough_order=candidate.passthrough_inputs,
    )


def _interception_artifact(plan: TracePlan) -> InterceptionTraceArtifact:
    if not isinstance(plan.runtime_artifact, InterceptionTraceArtifact):
        raise TypeError("Ariadne now requires a runtime-interception TracePlan.")
    return plan.runtime_artifact
