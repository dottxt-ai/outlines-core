import pickle

import pytest
from outlines_core.fsm import Vocabulary


def test_supports_strings_as_keys():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    vocabulary = Vocabulary(eos_token_id, tokens)

    assert vocabulary.get_eos_token_id() == eos_token_id
    assert vocabulary.get("1") == [1]
    assert vocabulary.get(b"1") == [1]
    assert len(vocabulary) == 2


def test_supports_bytes_as_keys():
    eos_token_id = 3
    tokens = {b"1": [1], b"a": [2]}
    vocabulary = Vocabulary(eos_token_id, tokens)

    assert vocabulary.get_eos_token_id() == eos_token_id
    assert vocabulary.get(b"1") == [1]
    assert vocabulary.get("1") == [1]
    assert len(vocabulary) == 2


def test_do_not_supports_other_types():
    eos_token_id = 0

    with pytest.raises(
        TypeError,
        match=r"Expected a dict with keys of type str or bytes and values of type list\[int\], got",
    ):
        Vocabulary(eos_token_id, 1)

    with pytest.raises(
        TypeError,
        match="Dict keys or/and values of the wrong types",
    ):
        Vocabulary(eos_token_id, {1: [1], 2: [2]})


def test_get_bad_type():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    vocabulary = Vocabulary(eos_token_id, tokens)

    with pytest.raises(
        TypeError,
        match="Expected a token of type str or bytes, got",
    ):
        vocabulary.get(1)


def test_from_pretrained():
    vocabulary = Vocabulary.from_pretrained("gpt2")
    assert vocabulary.get_eos_token_id() == 50256


def test_pickling():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    vocabulary = Vocabulary(eos_token_id, tokens)

    serialized = pickle.dumps(vocabulary)
    deserialized = pickle.loads(serialized)
    assert deserialized == vocabulary
