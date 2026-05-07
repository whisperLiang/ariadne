"""Optional compiler integration."""

from ariadne.compiler.compile_policy import CompilePolicy
from ariadne.compiler.torch_compile import (
    compile_replay_segments,
    maybe_compile_segments,
    replay_compile_options,
    training_compile_options,
)

__all__ = [
    "CompilePolicy",
    "compile_replay_segments",
    "maybe_compile_segments",
    "replay_compile_options",
    "training_compile_options",
]
