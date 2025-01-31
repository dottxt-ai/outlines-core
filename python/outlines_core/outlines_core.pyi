from typing import Dict, List, Optional, Set, Tuple, Union

def build_regex_from_schema(
    json_schema: str, whitespace_pattern: Optional[str] = None
) -> str:
    """Creates regex string from JSON schema with optional whitespace pattern."""
    ...

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
        """Creates a Guide object based on Index or statefull Index."""
    def get_state(self) -> int:
        """Retrieves current state id of the Guide."""
        ...
    def get_tokens(self) -> List[int]:
        """Gets the list of allowed tokens for the current state."""
        ...
    def advance(self, token_id: int) -> List[int]:
        """Guide moves to the next state provided by the token id and returns a list of allowed tokens."""
        ...
    def is_finished(self) -> bool:
        """Checks if the automaton is in a final state."""
        ...
    def __repr__(self) -> str:
        """Gets the debug string representation of the guide."""
        ...
    def __str__(self) -> str:
        """Gets the string representation of the guide."""
    def __eq__(self, other: object) -> bool:
        """Compares whether two guides are the same."""
        ...

class Vocabulary:
    def __init__(self, eos_token_id: int, map: Dict[Union[str, bytes], List[int]]):
        """Creates a vocabulary from a map of tokens to token ids and eos token id."""
        ...
    @staticmethod
    def from_pretrained(
        model: str, revision: Optional[str], token: Optional[str]
    ) -> "Vocabulary":
        """Creates the vocabulary of a pre-trained model."""
        ...
    def insert(self, token: Union[str, bytes], token_id: int):
        """Inserts new token with token_id or extends list of token_ids if token already present."""
        ...
    def remove(self, token: Union[str, bytes]):
        """Removes a token from vocabulary."""
        ...
    def get_eos_token_id(self) -> Optional[int]:
        """Gets the end of sentence token id."""
        ...
    def get(self, token: Union[str, bytes]) -> Optional[List[int]]:
        """Gets the end of sentence token id."""
        ...
    def __repr__(self) -> str:
        """Gets the debug string representation of the vocabulary."""
        ...
    def __str__(self) -> str:
        """Gets the string representation of the vocabulary."""
        ...
    def __eq__(self, other: object) -> bool:
        """Compares whether two vocabularies are the same."""
        ...
    def __len__(self) -> int:
        """Returns length of Vocabulary's tokens, excluding EOS token."""
        ...
    def __deepcopy__(self, memo: dict) -> "Vocabulary":
        """Makes a deep copy of the Vocabulary."""
        ...

class Index:
    def __init__(self, regex: str, vocabulary: "Vocabulary"):
        """Creates an index from a regex and vocabulary."""
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
    def get_final_states(self) -> List[int]:
        """Get all final states."""
        ...
    def get_transitions(self) -> Dict[int, Dict[int, int]]:
        """Returns the Index as a Python Dict object."""
        ...
    def get_initial_state(self) -> int:
        """Returns the ID of the initial state of the input FSM automata."""
        ...
    def __repr__(self) -> str:
        """Gets the debug string representation of the index."""
        ...
    def __str__(self) -> str:
        """Gets the string representation of the index."""
    def __eq__(self, other: object) -> bool:
        """Compares whether two indexes are the same."""
        ...
    def __deepcopy__(self, memo: dict) -> "Index":
        """Makes a deep copy of the Index."""
        ...
