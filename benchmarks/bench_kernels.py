# Note that these benchmarks measure kernels to be slower then they are
# when benchmarked outside of ASV for whatever reason.
# the approximate times for each kernel measured outside asv
# are commented above the kernels.
import random

import numpy as np
import torch
from outlines_core.kernels import numpy_kernel, torch_kernel



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
    params = [10, 100, 1_000, 10_000, 100_000]
    param_names = ["allowed_tokens"]
    number = 10

    def setup(self, allowed_tokens):
        self.device = "cpu"
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000

        self.logits = torch.randn(1, self.vocab, device=self.device)

        mask = torch.from_numpy(
            generate_sparse_mask(1, self.vocab, allowed_count=self.allowed_tokens)
        )
        self.mask = mask.to(self.device)

        self.kernel = torch_kernel

        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens):
        self.kernel(self.logits, self.mask)


class NumpyBitmaskApplyBenchmark:
    params = [10, 100, 1_000, 10_000, 100_000]
    param_names = ["allowed_tokens"]
    number = 10

    def setup(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000

        self.logits = np.random.randn(1, self.vocab).astype(np.float32)

        self.mask = generate_sparse_mask(
            1, self.vocab, allowed_count=self.allowed_tokens
        )

        self.kernel = numpy_kernel

        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens):
        self.kernel(self.logits, self.mask)


class MlxBitmaskApplyBenchmark:
    params = [10, 100, 1_000, 10_000, 100_000]
    param_names = ["allowed_tokens"]
    number = 10

    def setup(self, allowed_tokens):
        try:
            import mlx.core as mx
            from outlines_core.kernels.mlx import mlx_kernel
        except ImportError:
            self.mlx_available = False

        self.allowed_tokens = allowed_tokens
        self.vocab = 128000

        self.logits = mx.array(np.random.randn(1, self.vocab).astype(np.float32))

        self.mask = mx.array(
            generate_sparse_mask(1, self.vocab, allowed_count=self.allowed_tokens)
        )

        self.kernel = mlx_kernel

        # warm up / compile
        for _ in range(4):
            self.kernel(self.logits, self.mask)

    def time_kernel(self, allowed_tokens):
        self.kernel(self.logits, self.mask)
