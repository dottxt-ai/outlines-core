from typing import List, Tuple, Union

import torch
from datasets.fingerprint import Hasher
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer:
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerTokenizer(tokenizer)
