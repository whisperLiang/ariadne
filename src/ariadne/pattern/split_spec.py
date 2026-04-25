"""User-facing split declarations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

TraceBatchMode: TypeAlias = Literal["batch_1", "batch_gt1"]


@dataclass(frozen=True)
class SplitSpec:
    """Declarative split request.

    ``boundary`` accepts strings such as ``"after:layer3"`` or ``"auto"``.
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
