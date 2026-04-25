"""Optional torch.compile acceleration for generated segments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from ariadne.codegen.segment_builder import SegmentBundle


def maybe_compile_segments(
    segments: SegmentBundle,
    *,
    mode: str,
    compile_options: Mapping[str, Any] | None,
) -> SegmentBundle:
    if mode != "compiled":
        return segments
    options = dict(compile_options or {})
    return SegmentBundle(
        prefix=cast(torch.nn.Module, torch.compile(segments.prefix, **options)),
        suffix=cast(torch.nn.Module, torch.compile(segments.suffix, **options)),
        boundary_order=segments.boundary_order,
        passthrough_order=segments.passthrough_order,
    )
