import pickle
import pytest

from outlines_core.fsm import Vocabulary

def test_supports_strings_as_keys():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    vocabulary = Vocabulary.from_dict(eos_token_id, tokens)
    
    assert vocabulary.get_eos_token_id() == eos_token_id
    assert vocabulary.get("1") == [1]
    assert vocabulary.get(b"1") == [1]
    assert len(vocabulary) == 2

def test_supports_bytes_as_keys():
    eos_token_id = 3
    tokens = {b"1": [1], b"a": [2]}
    vocabulary = Vocabulary.from_dict(eos_token_id, tokens)

    assert vocabulary.get_eos_token_id() == eos_token_id
    assert vocabulary.get(b"1") == [1]
    assert vocabulary.get("1") == [1]
    assert len(vocabulary) == 2

def test_do_not_supports_other_types_as_keys():
    eos_token_id = 3
    tokens = {1: [1], 2: [2]}

    with pytest.raises(
        TypeError,
        match="Expected a dictionary with keys of type String or Bytes"
    ):
        Vocabulary.from_dict(eos_token_id, tokens)

def test_pickling():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    vocabulary = Vocabulary.from_dict(eos_token_id, tokens)

    serialized = pickle.dumps(vocabulary)
    deserialized = pickle.loads(serialized)
    assert deserialized == vocabulary