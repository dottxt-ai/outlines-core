import copy
import gc
import pickle
from typing import Dict, List, Union

import pytest
from outlines_core import Index, Vocabulary


@pytest.fixture(scope="session")
def index() -> Index:
    eos_token_id = 2
    # types here only to please mypy checks
    tokens: Dict[Union[str, bytes], List[int]] = {"0": [0], "1": [1]}
    regex = r"[0-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    return Index(regex, vocabulary)


def test_basic_interface(index):
    init_state = index.get_initial_state()
    assert init_state == 0
    assert index.is_final_state(init_state) is False

    allowed_tokens = index.get_allowed_tokens(init_state)
    assert allowed_tokens == [3] # BIT 0 and 1 Activated

    next_state = index.get_next_state(init_state, 0)
    assert next_state == 1
    assert index.is_final_state(next_state) is True
    assert index.get_final_states() == {1}

    expected_transitions = {
        0: {
            0: 1,
            1: 1,
        },
        1: {
            2: 1,
        },
    }
    assert index.get_transitions() == expected_transitions


def test_pickling(index):
    serialized = pickle.dumps(index)
    deserialized = pickle.loads(serialized)
    assert deserialized == index


def test_deepcopy(index):
    index2 = copy.deepcopy(index)
    assert index2 == index

    copy_index2 = copy.deepcopy(index2)
    assert copy_index2 == index2

    index2_id = id(index2)
    del index2
    gc.collect()
    is_deleted = not any(id(o) == index2_id for o in gc.get_objects())
    assert is_deleted

    assert copy_index2 == index
