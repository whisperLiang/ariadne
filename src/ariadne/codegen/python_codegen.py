"""Helpers for inspecting generated segment code."""

from __future__ import annotations

import torch


def segment_source(segment: torch.nn.Module) -> str:
    """Return readable generated Python for a segment."""
    return str(getattr(segment, "generated_source", segment))
