"""Validation helpers for declarative patterns and boundary schemas."""

from __future__ import annotations

from ariadne.pattern.shape_pattern import BoundaryTensorSpec
from ariadne.pattern.split_spec import SplitSpec


def validate_split_spec(spec: SplitSpec) -> None:
    if spec.boundary != "auto" and not spec.boundary.startswith("after:"):
        raise ValueError("SplitSpec.boundary must be 'auto' or use the form 'after:<label>'.")
    if spec.dynamic_batch is not None:
        low, high = spec.dynamic_batch
        if low < 1 or high < low:
            raise ValueError("SplitSpec.dynamic_batch must be a positive inclusive range.")
        if spec.trace_batch_mode == "batch_1" and not low <= 1 <= high:
            raise ValueError("batch_1 mode requires dynamic_batch to include batch size 1.")
        if spec.trace_batch_mode == "batch_gt1" and low < 2:
            raise ValueError("batch_gt1 mode requires dynamic_batch to start at 2 or greater.")


def validate_schema_labels(schema: dict[str, BoundaryTensorSpec], labels: tuple[str, ...]) -> None:
    missing = [label for label in labels if label not in schema]
    if missing:
        raise ValueError(f"Boundary schema is missing labels: {', '.join(missing)}.")
