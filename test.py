import torch
import time
import random

# -------------------------------
# Original Kernel (for reference)
# -------------------------------
@torch.compile(dynamic=True)
def apply_token_bitmask_inplace_kernel_original(logits: torch.Tensor, mask: torch.Tensor):
    # Having these if statements in here is faster than having it
    # run non-compiled in the parent function.
    # 5 microseconds for both checks
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    cutoff = 32 * mask.shape[1]
    # This should not happen, so long as the mask
    # is allocated at the correct size
    if logits.shape[1] > cutoff:
        logits[:, cutoff:] = float("-inf")
        logits = logits[:, :cutoff]

    mask_expanded = torch.repeat_interleave(mask, 32, dim=1)
    bit_indices = torch.arange(32, device=logits.device, dtype=torch.int32).repeat(mask.shape[1])
    bit_masks = (mask_expanded >> bit_indices) & 1
    bit_masks = bit_masks[:, :logits.shape[1]]
    logits.masked_fill_(bit_masks == 0, float("-inf"))

# -------------------------------
# Optimized Kernel (final version)
# -------------------------------
# @torch.compile(dynamic=True)
# def optimized_kernel(logits, mask):
#     logits = logits.unsqueeze(0) if logits.dim() == 1 else logits
#     mask = mask.unsqueeze(0) if mask.dim() == 1 else mask
    
#     # This should not modify, so long as the mask
#     # is allocated at the correct size
#     logits = torch.where(
#         torch.ge(
#             torch.arange(
#                 logits.shape[1],
#                 device=logits.device
#             ), 
#             32 * mask.shape[1]
#         ), 
#         float("-inf"), 
#         logits
#     )

#     bit_masks = ((mask.unsqueeze(-1) >> torch.arange(32, device=mask.device, dtype=torch.int32))
#                  & 1).bool().view(mask.shape[0], -1)

#     bit_masks = bit_masks[:, :logits.shape[1]]
#     logits.masked_fill_(~bit_masks, float("-inf"))

@torch.compile(dynamic=True)
def optimized_kernel(logits, mask):
    # This will set any logits beyond the mask
    # to -torch.inf
    cutoff = 32 * mask.shape[1]
    logits[:, cutoff:] = -torch.inf

    # Unpack each 32-bit mask value into 32 individual bits (as booleans)
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


# -------------------------------
# Bitmask Allocation & Sparse Mask Generation
# -------------------------------
def allocate_token_bitmask(batch: int, vocab: int) -> torch.Tensor:
    shape = (batch, (vocab + 31) // 32)
    return torch.full(shape, -1, dtype=torch.int32, pin_memory=torch.cuda.is_available())

def generate_sparse_mask(batch: int, vocab: int, allowed_count: int = 1000) -> torch.Tensor:
    mask_shape = (batch, (vocab + 31) // 32)
    mask = torch.zeros(mask_shape, dtype=torch.int32)
    allowed_indices = random.sample(range(vocab), allowed_count)
    for idx in allowed_indices:
        group = idx // 32
        shift = idx % 32
        mask[0, group] |= (1 << shift)
    return mask

# -------------------------------
# Benchmarking Function
# -------------------------------
def benchmark(func, logits: torch.Tensor, mask: torch.Tensor, iterations: int = 1000) -> float:
    # Warm-up iterations (clone is included here but these aren't timed)
    for _ in range(10):
        tmp = logits.clone()
        func(tmp, mask)
    
    # Precompute all clones outside the timed loop.
    clones = [logits.clone() for _ in range(iterations)]
    
    # Time only the function execution.
    start = time.time()
    for tmp in clones:
        func(tmp, mask)
    end = time.time()
    
    return (end - start) / iterations

# -------------------------------
# Main: Run Benchmark Tests
# -------------------------------
if __name__ == "__main__":
    # Always run on CPU.
    device = "cpu"
    batch = 1
    vocab = 128000  # 128k tokens
    allowed_tokens = 100000  # roughly 1k allowed tokens

    logits_original = torch.randn(batch, vocab, device=device)
    logits_optimized = logits_original.clone()

    # Create a sparse mask (replicating the original allocation conditions).
    mask = generate_sparse_mask(batch, vocab, allowed_count=allowed_tokens).to(device)

    time_orig = benchmark(apply_token_bitmask_inplace_kernel_original, logits_original, mask)
    time_opt = benchmark(optimized_kernel, logits_optimized, mask)

    print(f"Original kernel average time: {time_orig*1e3:.3f} ms")
    print(f"Optimized kernel average time: {time_opt*1e3:.3f} ms")
