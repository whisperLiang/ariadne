"""Helpers for inspecting generated segment code."""

from __future__ import annotations

from torch.fx import GraphModule


def segment_source(segment: GraphModule) -> str:
    """Return readable generated Python for a segment."""
    return str(segment.code)
