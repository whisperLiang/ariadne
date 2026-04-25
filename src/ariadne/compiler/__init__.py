"""Optional compiler integration."""

from ariadne.compiler.compile_policy import CompilePolicy
from ariadne.compiler.torch_compile import maybe_compile_segments

__all__ = ["CompilePolicy", "maybe_compile_segments"]
