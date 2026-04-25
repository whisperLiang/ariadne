"""Public package interface for Ariadne."""

from ariadne.api import prepare_split
from ariadne.pattern.split_spec import SplitSpec, TraceBatchMode
from ariadne.runtime.boundary import BoundaryPayload
from ariadne.runtime.segment_runtime import SplitRuntime

__all__ = ["BoundaryPayload", "SplitRuntime", "SplitSpec", "TraceBatchMode", "prepare_split"]


def main() -> None:
    """Small console entry point for the generated uv package script."""
    print("Ariadne: dynamic-batch split replay for PyTorch.")
