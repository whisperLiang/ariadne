"""Compile policy configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompilePolicy:
    enabled: bool = False
    options: dict[str, Any] = field(default_factory=dict)
