# benchmarks/benchmark_mask_write.py
"""
This benchmark uses a vocabulary of string representations of integers 0-10000
where token IDs match their string representations, and measures the overall
execution time for different numbers of allowed tokens.
"""
from outlines_core import Guide, Index, Vocabulary
from outlines_core.kernels.torch import allocate_token_bitmask, fill_next_token_bitmask

class MaskWriteBenchmark:
    """
    Benchmark for measuring the raw performance of write_mask_into with various token range sizes.
    """
    
    # Define patterns for specific token counts
    # Each tuple contains (token_count, regex_pattern)
    TOKEN_PATTERNS = [
        (1, "0"),                                     # 1 token
        (10, "[0-9]"),                                # 10 tokens (0-9)
        (100, "[0-9]|[1-9][0-9]"),                    # 100 tokens (0-99)
        (500, "[0-9]|[1-9][0-9]|[1-4][0-9][0-9]"),    # 500 tokens (0-499)
        (1000, "[0-9]|[1-9][0-9]|[1-9][0-9][0-9]"),   # 1000 tokens (0-999)
        (2500, "[0-9]|[1-9][0-9]|[1-9][0-9][0-9]|1[0-9][0-9][0-9]|2[0-4][0-9][0-9]"), # 2500 tokens (0-2499)
        (5000, "[0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-4][0-9][0-9][0-9]"), # 5000 tokens (0-4999)
        (7500, "[0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-6][0-9][0-9][0-9]|7[0-4][0-9][0-9]"), # 7500 tokens (0-7499)
        (10000, "[0-9]|[1-9][0-9]|[1-9][0-9][0-9]|[1-9][0-9][0-9][0-9]") # 10000 tokens (0-9999)
    ]
    
    # Extract token counts for params
    params = [
        [count for count, _ in TOKEN_PATTERNS]
    ]
    param_names = ['allowed_tokens']
    
    def setup(self, allowed_tokens):
        """
        Set up the benchmark with a vocabulary of string integers 0-10000
        where token IDs match their string representations.
        """
        # Fixed vocabulary size at 128256
        self.vocab_size = 128256
        
        # Create a vocabulary with string representations of integers as tokens
        # where token IDs match their string representations
        self.vocabulary = self._create_integer_vocabulary()
        
        # Find the regex pattern for this token count
        self.regex = next(pattern for count, pattern in self.TOKEN_PATTERNS if count == allowed_tokens)
        
        # Create the index and guide
        self.index = Index(self.regex, self.vocabulary)
        self.guide = Guide(self.index)
        
        # Pre-allocate the mask tensor
        self.mask = allocate_token_bitmask(self.vocab_size)
        
        # Verify the number of allowed tokens
        actual_tokens = len(self.guide.get_tokens())
        if actual_tokens != allowed_tokens:
            print(f"Warning: Expected {allowed_tokens} tokens, but got {actual_tokens} for regex: {self.regex}")
    
    def _create_integer_vocabulary(self) -> Vocabulary:
        """
        Create a vocabulary where:
        - Tokens are string representations of integers from 0 to 10000
        - Token IDs are the same as the integer values
        - EOS token ID is 10001 with string representation "eos"
        """
        # Create a map of string integers to token IDs
        token_map = {}
        for i in range(10001):  # 0 to 10000 inclusive
            token_map[str(i)] = [i]  # Token ID matches the integer
        
        # Add EOS token
        token_map["eos"] = [10001]
        
        # Create and return the vocabulary with EOS token ID 10001
        return Vocabulary(10001, token_map)
    
    def setup_benchmark(self, allowed_tokens):
        """
        Prepare for each benchmark iteration by resetting the mask.
        This runs before each timing and is not included in the measured time.
        """
        # Reset the mask to initial state
        self.mask.fill_(-1)
    
    def time_write_mask_into(self, allowed_tokens):
        """
        Benchmark the write_mask_into method with a specific range of allowed tokens.
        Only the fill_next_token_bitmask call is timed, not the mask reset.
        """
        # Only this operation is timed
        fill_next_token_bitmask(self.guide, self.mask)