from typing import Dict, List, Optional, Set, Tuple

def build_regex_from_schema(
    json: str, whitespace_pattern: Optional[str] = None
) -> str: ...
def to_regex(json: Dict, whitespace_pattern: Optional[str] = None) -> str: ...

BOOLEAN: str
DATE: str
DATE_TIME: str
INTEGER: str
NULL: str
NUMBER: str
STRING: str
STRING_INNER: str
TIME: str
UUID: str
WHITESPACE: str
EMAIL: str
URI: str

class Guide:
    def __init__(self, index: Index):
        """
        Defines a guide object an index an initializes it in its start state.
        """
    def get_start_tokens(self) -> List[int]:
        """
        Gets the list of allowed tokens from the start state.
        """
        ...
    def read_next_token(self, token_id: int) -> List[int]:
        """
        Reads the next token according to the model and returns a list of allowable tokens.
        """
        ...
    def is_finished(self) -> bool:
        """
        Checks if the automaton is in a final state.
        """
        ...

class Vocabulary:
    """
    Vocabulary of an LLM.
    """

    @staticmethod
    def from_dict(eos_token_id: int, map: Dict[str, List[int]]) -> "Vocabulary":
        """
        Creates a vocabulary from a map of tokens to token ids and eos token id.
        """
        ...
    @staticmethod
    def from_pretrained(model: str) -> "Vocabulary":
        """
        Creates the vocabulary of a pre-trained model.
        """
        ...
    def __repr__(self) -> str:
        """
        Gets the debug string representation of the vocabulary.
        """
        ...
    def __str__(self) -> str:
        """
        Gets the string representation of the vocabulary.
        """
        ...
    def __eq__(self, other: object) -> bool:
        """
        Gets whether two vocabularies are the same.
        """
        ...
    def get_eos_token_id(self) -> Optional[int]:
        """
        Gets the end of sentence token id.
        """
        ...

class Index:
    @staticmethod
    def from_regex(regex: str, vocabulary: "Vocabulary") -> "Index":
        """
        Creates an index from a regex and vocabulary.
        """
        ...
    def get_allowed_tokens(self, state: int) -> Optional[List[int]]:
        """Returns allowed tokens in this state."""
        ...
    def get_next_state(self, state: int, token_id: int) -> Optional[int]:
        """Updates the state."""
        ...
    def is_final_state(self, state: int) -> bool:
        """Determines whether the current state is a final state."""
        ...
    def final_states(self) -> List[int]:
        """Get all final states."""
        ...
    def get_index_dict(self) -> Dict[int, Dict[int, int]]:
        """Returns the Index as a Python Dict object."""
        ...
    def get_initial_state(self) -> int:
        """Returns the ID of the initial state of the input FSM automata."""
        ...
