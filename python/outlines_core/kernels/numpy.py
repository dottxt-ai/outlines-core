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
def _apply_token_bitmask_kernel(logits, mask):
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
        - logits: np.ndarray: The logits tensor.

        - mask: np.ndarray: The token bitmask representing the validity of each
          token in the logits tensor.

    Raises:
        - ValueError: If mask.dtype is not `int32`, mask or logits are not 2D, or
          logits and mask batch sizes do not match.
    Returns: None
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
            f"Invalid logits dimensions: Expected a 2D array, but got {mask.ndim}D."
        )
    elif mask.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Invalid batch size: Expected `mask.shape[0]` ({mask.shape[0]}) to match `logits.shape[0]` ({logits.shape[0]})."
        )
    _apply_token_bitmask_kernel(logits, mask)


def fill_next_token_bitmask(guide: Guide, mask: np.ndarray) -> None:
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
