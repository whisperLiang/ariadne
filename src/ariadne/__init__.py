"""Public package interface for Ariadne."""

from ariadne.api import prepare_split, prepare_split_replay
from ariadne.pattern.split_spec import SplitSpec, TraceBatchMode
from ariadne.runtime.boundary import BoundaryPayload, register_boundary_serializer
from ariadne.runtime.replay_runtime import (
    ReplayBoundary,
    SplitReplayRuntime,
)
from ariadne.runtime.segment_runtime import SplitRuntime
from ariadne.validation.dynamic_batch import validate_dynamic_batches

__all__ = [
    "BoundaryPayload",
    "ReplayBoundary",
    "SplitReplayRuntime",
    "SplitRuntime",
    "SplitSpec",
    "TraceBatchMode",
    "prepare_split",
    "prepare_split_replay",
    "register_boundary_serializer",
    "validate_dynamic_batches",
]


def main() -> None:
    """Small console entry point for the generated uv package script."""
    print("Ariadne: dynamic-batch split replay for PyTorch.")
