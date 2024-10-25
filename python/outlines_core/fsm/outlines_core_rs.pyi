from typing import Dict, List, Optional, Set, Tuple

class FSMInfo:
    initial: int
    finals: Set[int]
    transitions: Dict[Tuple[int, int], int]
    alphabet_anything_value: int
    alphabet_symbol_mapping: Dict[str, int]

    def __init__(
        self,
        initial: int,
        finals: Set[int],
        transitions: Dict[Tuple[int, int], int],
        alphabet_anything_value: int,
        alphabet_symbol_mapping: Dict[str, int],
    ) -> None: ...

def build_regex_from_schema(
    json: str, whitespace_pattern: Optional[str] = None
) -> str: ...
def to_regex(json: Dict, whitespace_pattern: Optional[str] = None) -> str: ...
def _walk_fsm(
    fsm_transitions: Dict[Tuple[int, int], int],
    fsm_initial: int,
    fsm_finals: Set[int],
    token_transition_keys: List[int],
    start_state: int,
    full_match: bool,
) -> List[int]: ...
def state_scan_tokens(
    fsm_transitions: Dict[Tuple[int, int], int],
    fsm_initial: int,
    fsm_finals: Set[int],
    vocabulary: Vocabulary,
    vocabulary_transition_keys: Dict[str, List[int]],
    start_state: int,
) -> Set[Tuple[int, int]]: ...
def get_token_transition_keys(
    alphabet_symbol_mapping: Dict[str, int],
    alphabet_anything_value: int,
    token_str: str,
) -> List[int]: ...
def get_vocabulary_transition_keys(
    alphabet_symbol_mapping: Dict[str, int],
    alphabet_anything_value: int,
    vocabulary: Vocabulary,
    frozen_tokens: Set[str],
) -> Dict[str, List[int]]: ...
def create_fsm_index_end_to_end(
    fsm_info: FSMInfo,
    vocabulary: Vocabulary,
    frozen_tokens: frozenset[str],
) -> Dict[int, Dict[int, int]]: ...

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

class Vocabulary:
    """
    Vocabulary of an LLM.
    """

    @staticmethod
    def from_dict(map: Dict[str, List[int]]) -> "Vocabulary":
        """
        Creates a vocabulary from a dictionary of tokens to token IDs.
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

class Index:
    def get_allowed_tokens(self, state: int) -> Optional[List[int]]:
        """Returns allowed tokens in this state."""
        ...
    def get_next_state(self, state: int, token_id: int) -> Optional[int]:
        """Updates the state."""
        ...
    def is_final_state(self, state: int) -> bool:
        """Determines whether the current state is a final state."""
        ...
    def get_index_dict(self) -> Dict[int, Dict[int, int]]:
        """Returns the Index as a Python Dict object."""
        ...
    def get_initial_state(self) -> int:
        """Returns the ID of the initial state of the input FSM automata."""
        ...
