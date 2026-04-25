"""User-facing split declarations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSpec:
    """Declarative split request.

    ``boundary`` accepts strings such as ``"after:layer3"`` or ``"auto"``.
    """

    boundary: str
    batch_symbol: str = "B"
    dynamic_batch: tuple[int, int] | None = (1, 64)
    trainable: bool = False

    def __post_init__(self) -> None:
        if not self.boundary:
            raise ValueError("SplitSpec.boundary must be non-empty.")
        if not self.batch_symbol.isidentifier():
            raise ValueError("SplitSpec.batch_symbol must be a valid identifier.")
        if self.dynamic_batch is not None:
            low, high = self.dynamic_batch
            if low < 1 or high < low:
                raise ValueError("SplitSpec.dynamic_batch must be a positive inclusive range.")
