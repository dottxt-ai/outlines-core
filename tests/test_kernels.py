import importlib

import numpy as np
import pytest
import torch

from outlines_core import Guide, Index, Vocabulary

VOCAB = Vocabulary.from_pretrained("gpt2", None, None)
VOCAB_LEN = len(VOCAB)


@pytest.fixture(scope="session")
def guide() -> Guide:
    return Guide(Index("\\+?[1-9][0-9]{7,14}", VOCAB))


@pytest.mark.no_cover
def test_interface_torch():
    from outlines_core.kernels.torch import (
        allocate_token_bitmask,
        apply_token_bitmask_inplace,
        fill_next_token_bitmask,
    )

    mask = allocate_token_bitmask(VOCAB_LEN)
    assert mask.shape[1] == 1571, "Mask sized incorrectly."
    assert mask.dim() == 2, "Mask should be 2D"
    assert mask.is_contiguous(), "Mask must be contiguous"
    assert mask.dtype == torch.int32, "Mask must be dtype torch.int32"

    # test apply_token_bitmask_inplace
    logits = torch.randn(1, 100)

    wrong_dtype_mask = mask.to(torch.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        apply_token_bitmask_inplace(logits, wrong_dtype_mask)

    mask_1d = torch.full((mask.shape[1],), -1, dtype=torch.int32)
    with pytest.raises(
        ValueError, match="Invalid mask dimensions: Expected a 2D array"
    ):
        apply_token_bitmask_inplace(logits, mask_1d)

    logits_1d = torch.randn(100)
    with pytest.raises(
        ValueError, match="Invalid logits dimensions: Expected a 2D array, but got 1D."
    ):
        apply_token_bitmask_inplace(logits_1d, mask)

    mask_batch2 = torch.full((2, mask.shape[1]), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="Invalid batch size"):
        apply_token_bitmask_inplace(logits, mask_batch2)

    # test fill_next_token_bitmask
    wrong_dtype_mask = mask.to(torch.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        fill_next_token_bitmask(None, wrong_dtype_mask)

    mask_1d = torch.full((mask.shape[1],), -1, dtype=torch.int32)
    with pytest.raises(ValueError, match="Invalid mask dimensions"):
        fill_next_token_bitmask(None, mask_1d)

    mask_batch2 = torch.full((2, mask.shape[1]), -1, dtype=torch.int32)
    with pytest.raises(
        ValueError, match="Invalid batch size: Batch mask writes are not supported"
    ):
        fill_next_token_bitmask(None, mask_batch2)


@pytest.mark.no_cover
def test_interface_numpy():
    from outlines_core.kernels.numpy import (
        allocate_token_bitmask,
        apply_token_bitmask_inplace,
        fill_next_token_bitmask,
    )

    mask = allocate_token_bitmask(VOCAB_LEN)
    assert mask.shape[1] == 1571, "Mask sized incorrectly."
    assert mask.ndim == 2, "Mask should be 2D"
    assert mask.flags["C_CONTIGUOUS"], "Mask must be contiguous"
    assert mask.dtype == np.int32, "Mask must be dtype np.int32"

    logits = np.random.randn(1, 100).astype(np.float32)

    wrong_dtype_mask = mask.astype(np.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        apply_token_bitmask_inplace(logits, wrong_dtype_mask)

    # 3d because 1d will be fixed automatically
    mask_3d = np.full((mask.shape[1], 1, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(
        ValueError, match="Invalid mask dimensions: Expected a 2D array"
    ):
        apply_token_bitmask_inplace(logits, mask_3d)

    with pytest.raises(
        ValueError, match="Invalid logits dimensions: Expected a 2D array"
    ):
        apply_token_bitmask_inplace(mask_3d, mask)

    mask_batch2 = np.full((2, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid batch size"):
        apply_token_bitmask_inplace(logits, mask_batch2)

    wrong_dtype_mask = mask.astype(np.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        fill_next_token_bitmask(None, wrong_dtype_mask)

    mask_1d = np.full((mask.shape[1],), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid mask dimensions"):
        fill_next_token_bitmask(None, mask_1d)

    mask_batch2 = np.full((2, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(
        ValueError, match="Invalid batch size: Batch mask writes are not supported"
    ):
        fill_next_token_bitmask(None, mask_batch2)


@pytest.mark.no_cover
@pytest.mark.skipif(
    not importlib.util.find_spec("mlx"), reason="mlx is required to test mlx kernels"  # type: ignore
)
def test_interface_mlx():
    import mlx.core as mx
    import numpy as np
    import pytest

    from outlines_core.kernels.mlx import (
        allocate_token_bitmask,
        apply_token_bitmask,
        fill_next_token_bitmask,
    )

    mask = allocate_token_bitmask(VOCAB_LEN)
    assert mask.shape[1] == (VOCAB_LEN + 31) // 32, "Mask sized incorrectly."
    assert mask.ndim == 2, "Mask should be 2D"
    assert mask.flags["C_CONTIGUOUS"], "Mask must be contiguous"
    assert mask.dtype == np.int32, "Mask must be dtype np.int32"

    logits = mx.array(np.random.randn(1, 100).astype(np.float32))
    wrong_dtype_mask = mask.astype(np.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        apply_token_bitmask(logits, wrong_dtype_mask)

    # 3d because 1d will be fixed automatically
    mask_3d = np.full((mask.shape[1], 1, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(
        ValueError, match="Invalid mask dimensions: Expected a 2D array"
    ):
        apply_token_bitmask(logits, mask_3d)

    with pytest.raises(
        ValueError, match="Invalid logits dimensions: Expected a 2D array"
    ):
        apply_token_bitmask(mask_3d, mask)

    mask_batch2 = np.full((2, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid batch size"):
        apply_token_bitmask(logits, mask_batch2)

    wrong_dtype_mask = mask.astype(np.int64)
    with pytest.raises(ValueError, match="Invalid mask dtype"):
        fill_next_token_bitmask(None, wrong_dtype_mask)

    mask_1d = np.full((mask.shape[1],), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="Invalid mask dimensions"):
        fill_next_token_bitmask(None, mask_1d)

    mask_batch2 = np.full((2, mask.shape[1]), -1, dtype=np.int32)
    with pytest.raises(
        ValueError, match="Invalid batch size: Batch mask writes are not supported"
    ):
        fill_next_token_bitmask(None, mask_batch2)


@pytest.mark.no_cover
def test_torch_correctness(guide):
    from outlines_core.kernels.torch import _apply_token_bitmask_inplace_kernel

    allowed_tokens = set(guide.get_tokens())

    logits = torch.tensor(torch.randn(1, VOCAB_LEN))

    orig_logits = logits.clone()

    mask = torch.tensor(
        torch.full((1, ((VOCAB_LEN + 31) // 32)), -1, dtype=torch.int32)
    )

    guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())

    _apply_token_bitmask_inplace_kernel(logits, mask)

    for j in range(VOCAB_LEN):
        if j in allowed_tokens:
            assert torch.isclose(
                logits[0, j], orig_logits[0, j], equal_nan=True
            ), f"Token {j} should be allowed but was masked."
        else:
            assert logits[0, j] == -float(
                "inf"
            ), f"Token {j} should be masked but was {logits[0, j].item()}."


@pytest.mark.no_cover
def test_numpy_correctness(guide):
    from outlines_core.kernels.numpy import _apply_token_bitmask_inplace_kernel

    allowed_tokens = set(guide.get_tokens())

    logits = np.random.randn(1, VOCAB_LEN).astype(np.float32)
    orig_logits = logits.copy()

    mask = np.full((1, ((VOCAB_LEN + 31) // 32)), -1, dtype=np.int32)

    guide.write_mask_into(mask.ctypes.data, mask.size, mask.itemsize)

    _apply_token_bitmask_inplace_kernel(logits, mask)

    for j in range(VOCAB_LEN):
        if j in allowed_tokens:
            np.testing.assert_allclose(
                logits[0, j],
                orig_logits[0, j],
                err_msg=f"Token {j} should be allowed but was masked.",
            )
        else:
            assert (
                logits[0, j] == -np.inf
            ), f"Token {j} should be masked was got {logits[0, j]}."


@pytest.mark.no_cover
@pytest.mark.skipif(
    not importlib.util.find_spec("mlx"), reason="mlx is required to test mlx kernels"  # type: ignore
)
def test_mlx_correctness(guide):
    import mlx.core as mx

    from outlines_core.kernels.mlx import _apply_token_bitmask_kernel

    allowed_tokens = set(guide.get_tokens())

    np_logits = np.random.randn(1, VOCAB_LEN).astype(np.float32)

    orig_logits = np_logits.copy()

    logits_mlx = mx.array(np_logits)

    mask = np.full((1, (VOCAB_LEN + 31) // 32), -1, dtype=np.int32)

    guide.write_mask_into(mask.ctypes.data, mask.size, mask.itemsize)

    logits_mlx_out = _apply_token_bitmask_kernel(logits_mlx, mx.array(mask))

    logits_out = np.array(logits_mlx_out)

    for j in range(VOCAB_LEN):
        if j in allowed_tokens:
            np.testing.assert_allclose(
                logits_out[0, j],
                orig_logits[0, j],
                err_msg=f"Token {j} should be allowed but was masked.",
            )
        else:
            assert (
                logits_out[0, j] == -np.inf
            ), f"Token {j} should be masked was got {logits_out[0, j]}."
