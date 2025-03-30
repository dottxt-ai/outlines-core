from outlines_core import Guide

try:
    import numba
    import numpy as np
except ImportError as e:
    missing_dep = "numba" if "numba" in str(e) else "numpy"
    raise ImportError(
        f"To use the kernels in `outlines_core.kernels.numpy`, `{missing_dep}` must be installed. You can install it with `pip install {missing_dep}`"
    ) from e


def allocate_token_bitmask(vocab_size: int) -> np.ndarray:
    return np.full(
        (1, (vocab_size + 31) // 32),
        -1,
        dtype=np.int32,
    )


@numba.njit
def _apply_token_bitmask_inplace_kernel(logits, mask):
    mask_len = mask.shape[1]
    cutoff = 32 * mask_len

    if logits.shape[1] > cutoff:
        logits[:, cutoff:] = -np.inf
        logits = logits[:, :cutoff]

    n_rows, n_cols = logits.shape

    for i in range(n_rows):
        for mi in range(mask_len):
            mval = mask[i, mi]
            base = mi * 32
            for bit in range(32):
                j = base + bit

                if j >= n_cols:
                    break

                if ((mval >> bit) & 1) == 0:
                    logits[i, j] = -np.inf


def apply_token_bitmask_inplace(logits: np.ndarray, mask: np.ndarray) -> None:
    """
    Apply a logits bitmask inplace, setting the probability of invalid tokens
    to -infinity.

    Arguments:
        logits (np.ndarray): The logits tensor.

        mask (np.ndarray): The token bitmask representing the validity of each
          token in the logits tensor.

    Raises:
        ValueError: If any of the following conditions are not met:
            - `mask.dtype` is not `np.int32`
            - `mask` is not a 2D array
            - `logits` is not a 2D array
            - `mask.shape`shape does not match `logits.shape`

    Returns:
        None: Modifies the mask array in place.
    """
    logits = logits if logits.ndim != 1 else np.expand_dims(logits, axis=0)
    mask = mask if mask.ndim != 1 else np.expand_dims(mask, axis=0)

    if mask.dtype != np.int32:
        raise ValueError(
            f"Invalid mask dtype: Expected `np.int32`, but got `{mask.dtype}`."
        )
    elif mask.ndim != 2:
        raise ValueError(
            f"Invalid mask dimensions: Expected a 2D array, but got {mask.ndim}D."
        )
    elif logits.ndim != 2:
        raise ValueError(
            f"Invalid logits dimensions: Expected a 2D array, but got {logits.ndim}D."
        )
    elif mask.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Invalid batch size: Expected `mask.shape[0]` ({mask.shape[0]}) to match `logits.shape[0]` ({logits.shape[0]})."
        )
    _apply_token_bitmask_inplace_kernel(logits, mask)


def fill_next_token_bitmask(guide: Guide, mask: np.ndarray) -> None:
    """
    Writes a bitmask to represent the tokens permissible by the current state of the `guide`.
    Each bit in the bitmask corresponds to a token ID, with a bit value of 1 indicating that
    the token is allowed and 0 indicating that it is disallowed. This function directly modifies
    the `mask` array in-place.

    Arguments:
        guide (Guide): An instance of the `Guide` class that provides the current guidance state.
        mask (np.ndarray): A 2D tensor of type `torch.int32` where the bitmask will be written.
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
        None: Modifies the `mask` array in-place.
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
