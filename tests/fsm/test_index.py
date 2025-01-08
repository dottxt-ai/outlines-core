import copy
import gc
import pickle
from typing import Dict, List, Union

import pytest
from outlines_core.fsm import Index, Vocabulary


@pytest.fixture(scope="session")
def index() -> Index:
    eos_token_id = 3
    # types here only to please mypy checks
    tokens: Dict[Union[str, bytes], List[int]] = {"1": [1], "2": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    return Index(regex, vocabulary)


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
