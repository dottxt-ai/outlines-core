# Provides kernels for masking a logits tensor,
# using the write_into_mask method on the `Guide` object and the bitmask
# which it writes into a tensor.
#
# Kernels inspired by https://github.com/guidance-ai/llguidance/blob/main/python/llguidance/torch.py
from outlines_core import Guide

try:
    import torch
except Exception as e:
    raise ImportError(
        "`torch` is required to use the kernels from"
        "`outlines_core.kernels.torch. You can install "
        "`torch` using the official guide at https://pytorch.org/get-started/locally/"
    ) from e


def allocate_token_bitmask(vocab_size: int) -> torch.Tensor:
    """
    Allocate a token bitmask for use with the `Guide.write_into_mask` API and logits masking,
    based on the vocab_size.

    Arguments:
        - vocab_size: int
    Returns:
        -  torch.Tensor
    """
    return torch.full(
        (1, (vocab_size + 31) // 32),
        -1,
        dtype=torch.int32,
        pin_memory=torch.cuda.is_available(),
    )


# This takes roughly 23 microseconds per run, with a bitmask of
# 1k allowed tokens, and 128k logits tensor.
# Also compiles to one graph with no graph breaks
# Performance characteristics are:
# - Larger the logits array ( length ), the longer the kernel takes
# - Constant time for mask i.e. number of allowed tokens does not affect execution
#   time
@torch.compile(dynamic=True)
def _apply_token_bitmask_inplace_kernel(logits, mask):
    # This will set any logits beyond the mask
    # to -torch.inf
    cutoff = 32 * mask.shape[1]
    logits[:, cutoff:] = -torch.inf

    # Unpack mask so each bit is a boolean
    bit_masks = (
        (
            torch.bitwise_right_shift(
                mask.unsqueeze(-1),
                torch.arange(32, device=mask.device, dtype=torch.int32),
            )
            & 1
        )
        .bool()
        .view(mask.shape[0], -1)
        .narrow(1, 0, logits.shape[1])
    )

    logits.masked_fill_(~bit_masks, -torch.inf)


def apply_token_bitmask_inplace(logits: torch.Tensor, mask: torch.Tensor) -> None:
    """
    Apply a logits bitmask inplace, setting the probability of invalid tokens
    to -infinity.

    Arguments:
        logits (torch.Tensor): The logits tensor.

        mask (torch.Tensor): The token bitmask representing the validity of
          each token in the logits tensor.

    Raises:
        ValueError: If any of the following conditions are not met:
            - `mask.dtype` is not `torch.int32`
            - `mask` is not a 2D array
            - `logits` is not a 2D array
            - `mask.shape`shape does not match `logits.shape`

    Returns:
        None: Modifies the logits tensor in place
    """
    if mask.dtype != torch.int32:
        raise ValueError(
            f"Invalid mask dtype: Expected `torch.int32`, but got `{mask.dtype}`."
        )
    elif mask.dim() != 2:
        raise ValueError(
            f"Invalid mask dimensions: Expected a 2D array, but got {mask.dim()}D."
        )
    elif logits.dim() != 2:
        raise ValueError(
            f"Invalid logits dimensions: Expected a 2D array, but got {logits.dim()}D."
        )
    elif mask.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Invalid batch size: Expected `mask.shape[0]` ({mask.shape[0]}) to match `logits.shape[0]` ({logits.shape[0]})."
        )
    _apply_token_bitmask_inplace_kernel(logits, mask)


def fill_next_token_bitmask(guide: Guide, mask: torch.Tensor) -> None:
    """
    Writes a bitmask to represent the tokens permissible by the current state of the `guide`.
    Each bit in the bitmask corresponds to a token ID, with a bit value of 1 indicating that
    the token is allowed and 0 indicating that it is disallowed. This function directly modifies
    the `mask` tensor in-place.

    Arguments:
        guide (Guide): An instance of the `Guide` class that provides the current guidance state.
        mask (torch.Tensor): A 2D tensor of type `torch.int32` where the bitmask will be written.
                             The tensor must be contiguous, have a single batch dimension
                             (shape[0] == 1), and reside on the CPU.

    Raises:
        ValueError: If any of the following conditions are not met:
                    - `mask.dtype` is not `torch.int32`
                    - `mask` is not a 2D tensor
                    - `mask` does not have a single batch dimension (shape[0] != 1)
                    - `mask` is not contiguous in memory
                    - `mask` is not on the CPU device

    Returns:
        None: Modifies the `mask` tensor in-place.
    """
    if mask.dtype != torch.int32:
        raise ValueError(
            f"Invalid mask dtype: Expected `torch.int32`, but got `{mask.dtype}`."
        )
    elif mask.dim() != 2:
        raise ValueError(
            f"Invalid mask dimensions: Expected a 2D array, but got {mask.dim()}D."
        )
    elif mask.shape[0] != 1:
        raise ValueError(
            f"Invalid batch size: Batch mask writes are not supported. Expected shape[0] == 1, but got shape {mask.shape}."
        )
    elif not mask.is_contiguous():
        raise ValueError(
            "Mask array must be contiguous in memory. Use `mask.contiguous()` to fix it."
        )
    elif mask.device != torch.device("cpu"):
        raise ValueError(
            f"Invalid device: Expected `mask` tensor to be on device `cpu`, but found it on `{mask.device}`."
        )

    guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())
