from outlines_core import Guide

try:
    import mlx.core as mx
    import numpy as np
except ImportError as e:
    missing_dep = "numpy" if "numpy" in str(e) else "mlx"
    raise ImportError(
        f"To use the kernels in `outlines_core.kernels.mlx`, {missing_dep} must be installed. You can install it with `pip install {missing_dep}`"
    ) from e


def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
    return np.full(
        (1, (vocab_size + 31) // 32),
        -1,
        dtype=np.int32,
    )


_KERNEL_SOURCE = r"""
// Batch index
uint batch = thread_position_in_grid.y;
// Element index
uint elem = thread_position_in_grid.x;

uint bit = ((elem >> 5) < static_cast<uint>(mask_shape[1])) &&
            ((mask[batch * mask_shape[1] + (elem >> 5)] >> (elem & 31)) & 1);

out[batch * inp_shape[1] + elem] = bit ? inp[batch * inp_shape[1] + elem] : static_cast<T>(-INFINITY);
"""

_KERNEL = mx.fast.metal_kernel(
    name="bitmask_apply_batched",
    input_names=["inp", "mask"],
    output_names=["out"],
    source=_KERNEL_SOURCE,
)


@mx.compile
def _apply_token_bitmask_kernel(data: mx.array, mask: mx.array) -> mx.array:
    return _KERNEL(
        inputs=[data, mask],
        template=[("T", data.dtype)],
        grid=(data.shape[1], data.shape[0], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[data.shape],
        output_dtypes=[data.dtype],
    )[0]


def apply_token_bitmask(logits: mx.array, mask_np: np.ndarray) -> mx.array:
    """
    Apply a logits bitmask inplace, setting the probability of invalid tokens
    to -infinity.

    Arguments:
        logits (mx.array): The logits tensor.

        mask (mx.array): The token bitmask representing the validity of each
          token in the logits tensor.

    Raises:
        ValueError: If any of the following conditions are not met:
            - `mask.dtype` is not `mx.int32`
            - `mask` is not a 2D array
            - `logits` is not a 2D array
            - `mask.shape`shape does not match `logits.shape`

    Returns:
        None: Modifies the mask array in place.
    """
    # makes a copy - non consuming
    mask = mx.array(mask_np)

    logits = logits if len(logits.shape) != 1 else mx.expand_dims(logits, axis=0)
    mask = mask if len(mask.shape) != 1 else mx.expand_dims(mask, axis=0)

    if mask.dtype != mx.int32:
        raise ValueError(
            f"Invalid mask dtype: Expected `np.int32`, but got `{mask.dtype}`."
        )
    elif len(mask.shape) != 2:
        raise ValueError(
            f"Invalid mask dimensions: Expected a 2D array, but got {mask.ndim}D."
        )
    elif len(logits.shape) != 2:
        raise ValueError(
            f"Invalid logits dimensions: Expected a 2D array, but got {logits.ndim}D."
        )
    elif mask.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Invalid batch size: Expected `mask.shape[0]` ({mask.shape[0]}) to match `logits.shape[0]` ({logits.shape[0]})."
        )
    return _apply_token_bitmask_kernel(logits, mask)


def fill_next_token_bitmask(guide: Guide, mask: np.ndarray) -> None:
    """
    Writes a bitmask to represent the tokens permissible by the current state of the `guide`.
    Each bit in the bitmask corresponds to a token ID, with a bit value of 1 indicating that
    the token is allowed and 0 indicating that it is disallowed. This function directly modifies
    the `mask` array in-place.

    Arguments:
        guide (Guide): An instance of the `Guide` class that provides the current guidance state.
        mask (torch.Tensor): A 2D tensor of type `torch.int32` where the bitmask will be written.
                             The tensor must be contiguous, have a single batch dimension
                             (shape[0] == 1), and reside on the CPU.

    Raises:
        ValueError: If any of the following conditions are not met:
                    - `mask.dtype` is not `np.int32`
                    - `mask` is not a 2D tensor
                    - `mask` does not have a single batch dimension (shape[0] != 1)
                    - `mask` is not contiguous in memory
                    - `mask` is not on the CPU device

    Returns:
        None: Modifies the `mask` tensor in-place.
    """
    if mask.dtype != np.int32:
        raise ValueError(
            f"Invalid mask dtype: Expected `np.int32`, but got `{mask.dtype}`."
        )
    elif mask.ndim != 2:
        raise ValueError(
            f"Invalid mask dimensions: Expected a 2D array, but got {mask.ndim}D."
        )
    elif mask.shape[0] != 1:
        raise ValueError(
            f"Invalid batch size: Batch mask writes are not supported. Expected shape[0] == 1, but got shape {mask.shape}."
        )
    elif not mask.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "Mask array must be contiguous in memory. Use `np.ascontiguousarray(mask)`."
        )

    return guide.write_mask_into(mask.ctypes.data, mask.size, mask.itemsize)
