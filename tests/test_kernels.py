import sys

import pytest
import numpy as np
import torch

from outlines_core import Guide, Index, Vocabulary

VOCAB = Vocabulary.from_pretrained("gpt2")
VOCAB_LEN = len(VOCAB)

@pytest.fixture(scope="session")
def guide() -> Guide:
    return Guide(Index("\\+?[1-9][0-9]{7,14}", VOCAB))

def test_torch_correctness(guide):
    from outlines_core.kernels.torch import _apply_token_bitmask_inplace_kernel
    allowed_tokens = set(guide.get_tokens())

    logits = torch.tensor(torch.randn(1, VOCAB_LEN))

    orig_logits = logits.clone()

    mask = torch.tensor(torch.full((1, ((VOCAB_LEN + 31) // 32)), -1, dtype=torch.int32))

    guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())

    _apply_token_bitmask_inplace_kernel(logits, mask)

    for j in range(VOCAB_LEN):
        if j in allowed_tokens:
            assert torch.isclose(
                logits[0, j],
                orig_logits[0, j],
                equal_nan=True
            ), f"Token {j} should be allowed but was masked."
        else:
            assert logits[0, j] == -float('inf'), (
                f"Token {j} should be masked but was {logits[0, j].item()}."
            )


def test_numpy_correctness(guide):
    from outlines_core.kernels.numpy import _apply_token_bitmask_inplace_kernel
    allowed_tokens = set(guide.get_tokens())

    # Create logits as a 2D numpy array (shape (1, logits_width))
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
                err_msg=f"Token {j} should be allowed but was masked."
            )
        else:
            assert logits[0, j] == -np.inf, (
                f"Token {j} should be masked was got {logits[0, j]}."
            )


import importlib

@pytest.mark.skipif(
    not importlib.util.find_spec("mlx"), reason="mlx is required to test mlx kernels"
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
    
    logits_mlx_out = _apply_token_bitmask_kernel(logits_mlx, mask)

    logits_out = np.array(logits_mlx_out)

    for j in range(VOCAB_LEN):
        if j in allowed_tokens:
            np.testing.assert_allclose(
                logits_out[0, j],
                orig_logits[0, j],
                err_msg=f"Token {j} should be allowed but was masked."
            )
        else:
            assert logits_out[0, j] == -np.inf, (
                f"Token {j} should be masked was got {logits_out[0, j]}."
            )
