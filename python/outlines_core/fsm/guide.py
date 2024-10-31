from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Set, Tuple, Union

import interegular
import torch
from outlines_core.fsm.regex import (
    create_fsm_index_tokenizer,
    make_byte_level_fsm,
    make_deterministic_fsm,
)

from .outlines_core_rs import Index


@dataclass(frozen=True)
class Write:
    """Write instruction.

    Attributes
    ----------
    tokens
        The sequence of tokens to be added to the current sequence by the
        generation process.

    """

    tokens: List[int]


@dataclass(frozen=True)
class Generate:
    """Generate instruction

    Attributes
    ----------
    tokens
        The tokens that lead to a valid completion if generated.  A value
        of ``None`` indicates that all tokens are allowed.
    """

    tokens: Optional[List[int]]


Instruction = Union[Write, Generate]


class Guide(Protocol):
    """Base definition of a generation guide.

    A generation guide defines the behavior of a finite-state machine that guides
    a text generation procedure. Unlike the DFAs built from regular expressions
    guides can also emit a `Write` instructions which tells the model that it can
    append a sequence of tokens (or token word) instead of generating it.

    """

    initial_state: Any

    def get_next_instruction(self, state: Any) -> Instruction:
        ...

    def get_next_state(self, state: Any, token_id: int) -> Any:
        ...

    def is_final_state(self, state: Any) -> bool:
        ...

    def copy(self) -> "Guide":
        ...


class StopAtEOSGuide(Guide):
    """Guide to generate tokens until the EOS token has been generated."""

    final_state = 1
    start_state = 0  # TODO: remove start_state, use only initial_state
    initial_state = 0

    def __init__(self, tokenizer):
        """Initialize the generation guide.

        model
            The logit generator used to generate the next token.

        """
        self.eos_token_id = tokenizer.eos_token_id
        self.vocabulary = tokenizer.vocabulary.values()

    def get_next_instruction(self, state: int) -> Instruction:
        if self.is_final_state(state):
            return Write([self.eos_token_id])
        return Generate(None)

    def get_next_state(self, state: int, token_id: int) -> int:
        if token_id == self.eos_token_id or state == self.final_state:
            return self.final_state

        return self.initial_state

    def is_final_state(self, state: int):
        return state == self.final_state

    def copy(self):
        return self


def create_states_mapping(
    regex_string: str,
    tokenizer,
    regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
    frozen_tokens: List[str] = [],
) -> Tuple[Index, Set[int], Set[int]]:
    """Create the variables related to the mapping between states and tokens from a regex string.

    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    regex_string:
        The regular expression string to generate a states mapping for.
    tokenizer:
        The model's tokenizer.
    regex_parser:
        A function that parses a regex string into an `interegular` Pattern object.
    frozen_tokens:
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps:
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids:
        A set of token ids that correspond to empty strings.
    final_states:
        A set of final states in the FSM.
    """
    regex_fsm = regex_parser(regex_string).to_fsm()
    return create_states_mapping_from_fsm(regex_fsm, tokenizer, frozen_tokens)


def create_states_mapping_from_fsm(
    fsm: interegular.fsm.FSM,
    tokenizer,
    frozen_tokens: List[str] = [],
) -> Tuple[Index, Set[int], Set[int]]:
    """Create the variables related to the mapping between states and tokens from an FSM.

    The parameters of the function are used for caching purpose.

    Parameters
    ----------
    fsm:
        An FSM for the regular expression.
    tokenizer:
        The model's tokenizer.
    frozen_tokens:
        A list of tokens that should be kept as-is when expanding the token-level FSM
        into a byte-level FSM. Defaults to an empty list.

    Returns
    -------
    states_to_token_maps:
        A mapping from states to a mapping from token ids originating from that state
        to the next state to transition to given that token. The structure is as follows:
        (origin_state -> (token_id -> next_state))
    empty_token_ids:
        A set of token ids that correspond to empty strings.
    final_states:
        A set of final states in the FSM.
    """
    byte_fsm = make_byte_level_fsm(
        fsm.reduce(), keep_utf8=True, frozen_tokens=frozen_tokens
    )
    regex_fsm, _ = make_deterministic_fsm(byte_fsm)
    states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(
        regex_fsm, tokenizer
    )

    return states_to_token_maps, empty_token_ids, regex_fsm.finals


class RegexGuide(Guide):
    """Guide to generate text in the language of a regular expression."""

    initial_state = 0

    def __init__(
        self, states_to_token_maps, empty_token_ids, eos_tensor, initial_state
    ):
        self.states_to_token_maps = states_to_token_maps
        self.empty_token_ids = empty_token_ids
        self.eos_tensor = eos_tensor
        self.initial_state = initial_state

    @classmethod
    def from_regex(
        cls,
        regex_string: str,
        tokenizer,
        _create_states_mapping=create_states_mapping,
        device=None,
        regex_parser: Callable[[str], interegular.Pattern] = interegular.parse_pattern,
        frozen_tokens: List[str] = [],
    ):
        (
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
        ) = _create_states_mapping(
            regex_string,
            tokenizer,
            regex_parser=regex_parser,
            frozen_tokens=frozen_tokens,
        )
        eos_tensor = torch.tensor([tokenizer.eos_token_id], device=device)
        initial_state = states_to_token_maps.get_initial_state()
        return cls(states_to_token_maps, empty_token_ids, eos_tensor, initial_state)

    @classmethod
    def from_interegular_fsm(
        cls,
        interegular_fsm: interegular.fsm.FSM,
        tokenizer,
        _create_states_mapping_from_fsm=create_states_mapping_from_fsm,
        device=None,
        frozen_tokens: List[str] = [],
    ):
        (
            states_to_token_maps,
            empty_token_ids,
            fsm_finals,
        ) = _create_states_mapping_from_fsm(
            interegular_fsm, tokenizer, frozen_tokens=frozen_tokens
        )
        eos_tensor = torch.tensor([tokenizer.eos_token_id], device=device)
        initial_state = states_to_token_maps.get_initial_state()
        return cls(states_to_token_maps, empty_token_ids, eos_tensor, initial_state)

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.

        Parameters
        ----------
        state
            The current state of the guide.

        Returns
        -------
        A `Generate` instance that contains the model and the allowed token ids.

        """
        if state == -1:
            return Write(self.eos_tensor)
        next_tokens_mask = self.states_to_token_maps.get_allowed_tokens(state)
        # TODO: Create the Write and Generate objects within Rust instead?
        if next_tokens_mask is None:
            return Write(self.eos_tensor)

        return Generate(torch.tensor(next_tokens_mask))

    def get_next_state(self, state: int, token_id: int) -> int:
        """Update the state of the guide.

        We use the index to determine to which state the guide should transition
        given the token that was just generated.

        Parameters
        ----------
        state
            The current state of the guide.
        token_id
            The id of the token that was just generated.

        Returns
        -------
        The new state of the guide.

        """
        if state == -1:
            return -1
        next_state = self.states_to_token_maps.get_next_state(state, token_id)
        if next_state is None:
            return -1
        else:
            return next_state

    def is_final_state(self, state: int) -> bool:
        """Determine whether the current state of the guide is a final state."""
        return state == -1 or self.states_to_token_maps.is_final_state(state)

    def copy(self):
        return self

    def get_index_dict(self):
        """Returns the Index as a Python Dict object."""
        return self.states_to_token_maps.get_transitions()
