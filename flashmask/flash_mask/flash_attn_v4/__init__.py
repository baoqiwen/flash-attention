"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

import cutlass.cute as cute

# Auto-detect framework: prefer torch, fall back to paddle.
try:
    import torch  # noqa: F401

    from flash_mask.flash_attn_v4.torch.interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_mask.flash_attn_v4.torch.cute_dsl_utils import cute_compile_patched
except ImportError:
    from flash_mask.flash_attn_v4.paddle.interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import cute_compile_patched

# Patch cute.compile to optionally dump SASS
cute.compile = cute_compile_patched


__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
