import copy
import gc
import pickle

import pytest

from outlines_core import Index, Vocabulary


@pytest.fixture(scope="session")
def index() -> Index:
    eos_token_id = 3
    tokens = {"1": [1], "2": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    return Index(regex, vocabulary)


def test_basic_interface(index):
    init_state = index.get_initial_state()
    assert init_state == 12
    assert index.is_final_state(init_state) is False

    allowed_tokens = index.get_allowed_tokens(init_state)
    assert allowed_tokens == [1, 2]

    next_state = index.get_next_state(init_state, allowed_tokens[-1])
    assert next_state == 20
    assert index.is_final_state(next_state) is True
    assert index.get_final_states() == {20}

    expected_transitions = {
        12: {
            1: 20,
            2: 20,
        },
        20: {
            3: 20,
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
