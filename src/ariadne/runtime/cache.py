"""Runtime cache keys that exclude concrete batch size."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCacheKey:
    graph_signature: str
    split_id: str
    mode: str
    non_batch_shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        graph_signature: str,
        split_id: str,
        mode: str,
        shapes: tuple[tuple[int, ...], ...],
        dtypes: tuple[str, ...],
    ) -> RuntimeCacheKey:
        return cls(
            graph_signature=graph_signature,
            split_id=split_id,
            mode=mode,
            non_batch_shapes=tuple(shape[1:] for shape in shapes),
            dtypes=dtypes,
        )
