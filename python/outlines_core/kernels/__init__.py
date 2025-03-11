"""Kernel implementations for various backends."""

from .numpy import _apply_token_bitmask_inplace_kernel as numpy_kernel
from .torch import _apply_token_bitmask_inplace_kernel as torch_kernel

try:
    from .mlx import _apply_token_bitmask_kernel as mlx_kernel
except ImportError:
    pass
