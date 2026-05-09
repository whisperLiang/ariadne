"""User-facing split declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TraceBatchMode: TypeAlias = Literal["batch_1", "batch_gt1"]


def parse_boundary_percent(boundary: str) -> float | None:
    """Return the requested split percentage, or ``None`` for non-percent specs."""
    text = boundary.strip().lower()
    is_percent_form = False
    if text.startswith("percent:"):
        text = text.removeprefix("percent:").strip()
        is_percent_form = True
    elif text.startswith("at:") and text.endswith("%"):
        text = text.removeprefix("at:").strip()
        is_percent_form = True
    elif text.endswith("%"):
        is_percent_form = True

    if not is_percent_form:
        return None

    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        percent = float(text)
    except ValueError as error:
        raise ValueError("SplitSpec.boundary percent must be a numeric value.") from error
    if not 0.0 < percent < 100.0:
        raise ValueError("SplitSpec.boundary percent must be greater than 0 and less than 100.")
    return percent


@dataclass(frozen=True)
class SplitSpec:
    """Declarative split request.

    ``boundary`` accepts strings such as ``"after:layer3"``, ``"auto"``, or
    ``"percent:50"``.
    """

    boundary: str
    batch_symbol: str = "B"
    dynamic_batch: tuple[int, int] | None = (1, 64)
    trainable: bool = False
    trace_batch_mode: TraceBatchMode = "batch_1"

    def __post_init__(self) -> None:
        if not self.boundary:
            raise ValueError("SplitSpec.boundary must be non-empty.")
        if not self.batch_symbol.isidentifier():
            raise ValueError("SplitSpec.batch_symbol must be a valid identifier.")
        if self.dynamic_batch is not None:
            low, high = self.dynamic_batch
            if low < 1 or high < low:
                raise ValueError("SplitSpec.dynamic_batch must be a positive inclusive range.")
        if self.trace_batch_mode not in {"batch_1", "batch_gt1"}:
            raise ValueError("SplitSpec.trace_batch_mode must be 'batch_1' or 'batch_gt1'.")
