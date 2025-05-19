import copy
import pickle
from typing import Dict, List, Union

import pytest

from outlines_core import Guide, Index, Vocabulary


@pytest.fixture(scope="session")
def index() -> Index:
    eos_token_id = 3
    # types here only to please mypy checks
    tokens: Dict[Union[str, bytes], List[int]] = {"1": [1], "2": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    return Index(regex, vocabulary)


def test_interface():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    assert guide.get_state() == index.get_initial_state() == 12
    assert guide.get_tokens() == [1]

    assert guide.advance(1) == [vocabulary.get_eos_token_id()]
    assert guide.is_finished()
    assert guide.get_state() == 20
    assert guide.get_tokens() == [eos_token_id]

    with pytest.raises(
        ValueError,
        match="No next state found for the current state",
    ):
        # No advancement is possible for state with allowed tokens == eos
        assert guide.advance(eos_token_id)
        # As well as with any other random token id
        assert guide.advance(4)


def test_regex_final_state_walk():
    # Make sure that the Guide can walk to the final state correctly.
    eos_token_id = 104
    tokens = {b"\n": [103], b".": [102], b"`": [101]}
    regex = r"`\n(\.\n)?`\n"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    assert guide.get_tokens() == [101]
    assert guide.advance(101) == [103]
    assert sorted(guide.advance(103)) == [101, 102]
    assert guide.advance(101) == [103]
    assert guide.advance(103) == [vocabulary.get_eos_token_id()]
    assert guide.is_finished()


def test_token_trans_keys_identical():
    tokens = {"a": [1], "b": [2], "z": [3]}
    eos_token_id = 4
    regex = r"z[ab]z"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)

    guide1 = Guide(index)
    guide2 = Guide(index)

    assert sorted(guide1.advance(3)) == sorted(guide2.advance(3))
    # `a` and `b` have similar transitions to `z`
    assert sorted(guide1.advance(1)) == sorted(guide2.advance(2))
    assert guide1.advance(3) == guide2.advance(3) == [eos_token_id]
    assert guide1.is_finished()
    assert guide2.is_finished()


def test_str_and_bytes_produce_the_same():
    tokens1 = {"a": [1], "b": [2], "z": [3]}
    tokens2 = {b"a": [1], b"b": [2], b"z": [3]}
    eos_token_id = 4
    regex = r"z[ab]z"

    vocabulary1 = Vocabulary(eos_token_id, tokens1)
    vocabulary2 = Vocabulary(eos_token_id, tokens2)
    index1 = Index(regex, vocabulary1)
    index2 = Index(regex, vocabulary2)
    guide1 = Guide(index1)
    guide2 = Guide(index2)

    assert sorted(guide1.advance(3)) == sorted(guide2.advance(3))
    # `a` and `b` have similar transitions to `z`
    assert sorted(guide1.advance(1)) == sorted(guide2.advance(2))
    assert guide1.advance(3) == guide2.advance(3) == [eos_token_id]
    assert guide1.is_finished()
    assert guide2.is_finished()


def test_pickling(index):
    guide = Guide(index)
    serialized = pickle.dumps(guide)
    deserialized = pickle.loads(serialized)
    assert sorted(deserialized.get_tokens()) == sorted(guide.get_tokens())


@pytest.mark.parametrize(
    "model, revision",
    [
        ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        ("microsoft/phi-2", "ef382358ec9e382308935a992d908de099b64c23"),
        ("Qwen/Qwen1.5-0.5B-Chat", "4d14e384a4b037942bb3f3016665157c8bcb70ea"),
        (
            "NousResearch/Hermes-2-Pro-Llama-3-8B",
            "783fd50eb82d7f57758de033861f54d62dde234f",
        ),
    ],
)
def test_pickling_from_pretrained_with_revision(model, revision):
    regex = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    vocabulary = Vocabulary.from_pretrained(model, revision=revision)
    index = Index(regex, vocabulary)
    assert len(index.get_transitions()) == 810

    guide = Guide(index)
    serialized = pickle.dumps(guide)
    deserialized = pickle.loads(serialized)
    assert sorted(deserialized.get_tokens()) == sorted(guide.get_tokens())


def test_equality(index):
    guide1 = Guide(index)
    guide2 = Guide(index)
    assert guide1 == guide2

    # confirm that equality is about inner index, not reference difference
    index2 = copy.deepcopy(index)
    guide3 = Guide(index2)
    assert guide3 == guide2 == guide1

    # progress one of the guides, confirm different state == different guide
    guide1.advance(guide1.get_tokens()[-1])
    assert guide1 != guide2
    assert guide3 == guide2


def test_write_mask_into(index):
    import torch

    guide = Guide(index)
    mask = torch.tensor(torch.tensor([-1], dtype=torch.uint32))
    guide.write_mask_into(mask.data_ptr(), mask.numel(), mask.element_size())

    expected_mask = 0
    for token in guide.get_tokens():
        expected_mask |= 1 << (token % 32)

    assert (
        mask[0] == expected_mask
    ), f"mask and expected mask do not match. Mask: {mask[0]}, Expected: {expected_mask}"


def test_write_mask_into_interface(index):
    import torch

    guide = Guide(index)
    mask = torch.tensor(torch.tensor([-1], dtype=torch.uint32))

    with pytest.raises(ValueError, match="Invalid buffer size"):
        guide.write_mask_into(mask.data_ptr(), 0, mask.element_size())
    with pytest.raises(ValueError, match="Invalid element size"):
        guide.write_mask_into(mask.data_ptr(), mask.numel(), 5)
    with pytest.raises(ValueError, match="Invalid data pointer"):
        guide.write_mask_into(0, mask.numel(), mask.element_size())
    with pytest.raises(ValueError, match="Invalid data pointer alignment"):
        guide.write_mask_into(5, mask.numel(), mask.element_size())


def test_rollback(index):
    guide = Guide(index, max_rollback=3)

    first_state = guide.get_state()
    guide.advance(1)

    # Roll back one token to initial state
    guide.rollback_state(1)
    assert not guide.is_finished()
    # we should now be in the initial state
    assert guide.get_state() == first_state


def test_rollback_interface(index):
    guide = Guide(index, max_rollback=3)

    # Rolling back more than recorded history must raise
    with pytest.raises(ValueError, match="Cannot roll back"):
        guide.rollback_state(5)


@pytest.mark.parametrize(
    "seq, expected",
    [
        ([1], True),  # single allowed token: accept
        ([2], True),  # different allowed token: accept
        ([1, 1], False),  # too long for r"[1-9]": reject
        ([2, 3], False),  # extra token: reject
    ],
)
def test_accepts_tokens_correctness(index, seq, expected):
    guide = Guide(index)
    assert guide.accepts_tokens(seq) is expected
