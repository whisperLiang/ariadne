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


def build_segments(plan: TracePlan, candidate: SplitCandidate) -> SegmentBundle:
    """Build generated eager prefix/suffix callables from a captured runtime trace."""
    if not isinstance(plan.runtime_artifact, InterceptionTraceArtifact):
        raise TypeError("Ariadne now requires a runtime-interception TracePlan.")
    return SegmentBundle(
        prefix=build_interception_prefix(
            root=plan.root_module,
            artifact=plan.runtime_artifact,
            op_names=candidate.prefix_nodes,
            raw_input_names=plan.input_node_names,
            boundary_order=candidate.boundary_nodes,
            class_name="PrefixSegment",
        ),
        training_prefix=build_interception_prefix(
            root=plan.root_module,
            artifact=plan.runtime_artifact,
            op_names=candidate.prefix_nodes,
            raw_input_names=plan.input_node_names,
            boundary_order=candidate.boundary_nodes,
            class_name="TrainingPrefixSegment",
        ),
        suffix=build_interception_suffix(
            root=plan.root_module,
            artifact=plan.runtime_artifact,
            op_names=candidate.suffix_nodes,
            boundary_order=candidate.boundary_nodes,
            passthrough_order=candidate.passthrough_inputs,
        ),
        boundary_order=candidate.boundary_nodes,
        passthrough_order=candidate.passthrough_inputs,
    )
