"""Optional torch.compile acceleration for generated segments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from ariadne.codegen.segment_builder import ReplaySegmentBundle, SegmentBundle

DEFAULT_REPLAY_COMPILE_OPTIONS: dict[str, Any] = {
    "backend": "inductor",
    "mode": "reduce-overhead",
    "dynamic": True,
    "fullgraph": True,
}

DEFAULT_TRAINING_COMPILE_OPTIONS: dict[str, Any] = {
    "backend": "inductor",
    "mode": "reduce-overhead",
    "dynamic": True,
}


def maybe_compile_segments(
    segments: SegmentBundle,
    *,
    mode: str,
    compile_options: Mapping[str, Any] | None,
) -> SegmentBundle:
    if mode != "compiled":
        return segments
    options = training_compile_options(compile_options)
    return SegmentBundle(
        prefix=cast(torch.nn.Module, torch.compile(segments.prefix, **options)),
        training_prefix=cast(
            torch.nn.Module,
            torch.compile(segments.training_prefix, **options),
        ),
        suffix=cast(torch.nn.Module, torch.compile(segments.suffix, **options)),
        boundary_order=segments.boundary_order,
        passthrough_order=segments.passthrough_order,
    )


def training_compile_options(compile_options: Mapping[str, Any] | None) -> dict[str, Any]:
    user_options = dict(compile_options or {})
    options = dict(DEFAULT_TRAINING_COMPILE_OPTIONS)
    options.update(user_options)
    if options.get("backend") != "inductor" and "mode" not in user_options:
        options.pop("mode", None)
    return options


def replay_compile_options(
    compile_options: Mapping[str, Any] | None,
    *,
    fullgraph: bool | None = None,
) -> dict[str, Any]:
    user_options = dict(compile_options or {})
    options = dict(DEFAULT_REPLAY_COMPILE_OPTIONS)
    options.update(user_options)
    if options.get("backend") != "inductor" and "mode" not in user_options:
        options.pop("mode", None)
    if fullgraph is not None:
        options["fullgraph"] = fullgraph
    return options


def compile_replay_segments(
    segments: ReplaySegmentBundle,
    *,
    compile_options: Mapping[str, Any] | None,
) -> ReplaySegmentBundle:
    options = dict(compile_options or {})
    return ReplaySegmentBundle(
        prefix=cast(torch.nn.Module, torch.compile(segments.prefix, **options)),
        suffix=cast(torch.nn.Module, torch.compile(segments.suffix, **options)),
        boundary_order=segments.boundary_order,
        passthrough_order=segments.passthrough_order,
    )
