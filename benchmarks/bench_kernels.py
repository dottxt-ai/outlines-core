# Note that these benchmarks measure kernels to be slower then they are
# when benchmarked outside of ASV for whatever reason. 
# the approximate times for each kernel measured outside asv 
# are commented above the kernels.
import torch
import numpy as np
import random

from outlines_core.kernels.torch import _apply_token_bitmask_kernel as torch_kernel
from outlines_core.kernels.numpy import _apply_token_bitmask_kernel as np_kernel


class TorchBitmaskApplyBench:
    params = [10, 100, 1000, 10000]
    param_names = ["allowed_tokens"]
    number = 10

    def setup(self, allowed_tokens):
        self.device = "cpu"
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000

        self.logits = torch.randn(1, self.vocab, device=self.device)
        self.logits_ref = self.logits.clone()

        mask = self._generate_sparse_mask(1, self.vocab, allowed_count=self.allowed_tokens)
        self.mask = mask.to(self.device)

        self.kernel = torch_kernel

        self.input_tensor = self.logits_ref.clone()

        # warm up / compile
        for _ in range(4):
            self.kernel(self.input_tensor, self.mask)
            self.input_tensor.copy_(self.logits_ref)

    def _generate_sparse_mask(self, batch, vocab, allowed_count=1000):
        mask_shape = (batch, (vocab + 31) // 32)
        mask = torch.zeros(mask_shape, dtype=torch.int32)
        allowed_indices = random.sample(range(vocab), allowed_count)
        for idx in allowed_indices:
            group = idx // 32
            shift = idx % 32
            mask[0, group] |= (1 << shift)
        return mask

    def time_kernel(self, allowed_tokens):
        self.kernel(self.logits_ref, self.mask)


class NumpyBitmaskApplyBench:
    params = [10, 100, 1000, 10000]
    param_names = ["allowed_tokens"]
    number = 10

    def setup(self, allowed_tokens):
        self.allowed_tokens = allowed_tokens
        self.vocab = 128000

        self.logits = np.random.randn(1, self.vocab).astype(np.float32)
        self.logits_ref = self.logits.copy()

        self.mask = self._generate_sparse_mask(1, self.vocab, allowed_count=self.allowed_tokens)
        
        self.kernel = np_kernel

        self.input_array = self.logits_ref.copy()

        # warm up / compile
        for _ in range(4):
            self.kernel(self.input_array, self.mask)
            self.input_array[:] = self.logits_ref

    def _generate_sparse_mask(self, batch, vocab, allowed_count=1000):
        mask_shape = (batch, (vocab + 31) // 32)
        mask = np.zeros(mask_shape, dtype=np.int32)
        allowed_indices = random.sample(range(vocab), allowed_count)
        for idx in allowed_indices:
            group = idx // 32
            shift = idx % 32
            mask[0, group] |= (1 << shift)
        return mask

    def time_kernel(self, allowed_tokens):
        self.kernel(self.logits_ref, self.mask)
