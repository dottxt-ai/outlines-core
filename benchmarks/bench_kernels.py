import random

import numpy as np
import torch

from outlines_core.kernels.numpy import (
    _apply_token_bitmask_inplace_kernel as numpy_kernel,
)
from outlines_core.kernels.torch import (
    _apply_token_bitmask_inplace_kernel as torch_kernel,
)


def generate_sparse_mask(batch, vocab, allowed_count=1000):
    mask_shape = (batch, (vocab + 31) // 32)
    mask = np.zeros(mask_shape, dtype=np.uint32)
    allowed_indices = random.sample(range(vocab), allowed_count)
    for idx in allowed_indices:
        group = idx // 32
        shift = idx % 32
        bit_mask = np.uint32(1) << np.uint32(shift)
        mask[0, group] |= bit_mask
    return mask


class TorchBitmaskApplyBenchmark:
    params = [[10, 100, 1_000, 10_000, 100_000], [1, 2, 4, 8]]
    param_names = ["allowed_tokens", "batch"]
    number = 10

    def setup(self, allowed_tokens, batch):
        self.device = "cpu"
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000
        self.batch = batch

        self.logits = torch.randn(self.batch, self.vocab, device=self.device)

        mask = torch.from_numpy(
            generate_sparse_mask(
                self.batch, self.vocab, allowed_count=self.allowed_tokens
            )
        )
        self.mask = mask.to(self.device)

        self.kernel = torch_kernel

        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens, batch):
        self.kernel(self.logits, self.mask)


class NumpyBitmaskApplyBenchmark:
    params = [[10, 100, 1_000, 10_000, 100_000], [1, 2, 4, 8]]
    param_names = ["allowed_tokens", "batch"]
    number = 10

    def setup(self, allowed_tokens, batch):
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000
        self.batch = batch

        self.logits = np.random.randn(self.batch, self.vocab).astype(np.float32)

        self.mask = generate_sparse_mask(
            self.batch, self.vocab, allowed_count=self.allowed_tokens
        )

        self.kernel = numpy_kernel

        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens, batch):
        self.kernel(self.logits, self.mask)


class MlxBitmaskApplyBenchmark:
    params = [[10, 100, 1_000, 10_000, 100_000], [1, 2, 4, 8]]
    param_names = ["allowed_tokens", "batch"]
    number = 10

    def setup(self, allowed_tokens, batch):
        try:
            import mlx.core as mx

            from outlines_core.kernels.mlx import (
                _apply_token_bitmask_kernel as mlx_kernel,
            )
        except ImportError:
            raise NotImplementedError

        self.allowed_tokens = allowed_tokens
        self.vocab = 128000
        self.batch = batch

        self.logits = mx.array(
            np.random.randn(self.batch, self.vocab).astype(np.float32)
        )

        self.mask = mx.array(
            generate_sparse_mask(
                self.batch, self.vocab, allowed_count=self.allowed_tokens
            )
        )

        self.kernel = mlx_kernel

        # warm up / compile
        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens, batch):
        self.kernel(self.logits, self.mask)
