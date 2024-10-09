from typing import Dict, Hashable, List, Protocol, Set, Tuple, TypeVar, Union

Array = TypeVar("Array")


class Tokenizer(Hashable, Protocol):
    eos_token: str
    eos_token_id: int
    pad_token_id: int
    vocabulary: Dict[str, int]
    special_tokens: Set[str]

    def encode(self, prompt: Union[str, List[str]]) -> Tuple[Array, Array]:
        """Translate the input prompts into arrays of token ids and attention mask."""
        ...

    def decode(self, token_ids: Array) -> List[str]:
        """Translate an array of token ids to a string or list of strings."""
        ...

    def convert_token_to_string(self, token: str) -> str:
        """Convert a token to its equivalent string.

        This is for instance useful for BPE tokenizers where whitespaces are
        represented by the special characted `Ġ`. This prevents matching a raw
        token that includes `Ġ` with a string.
        """
        ...
