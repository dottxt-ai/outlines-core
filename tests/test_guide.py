import copy
import pickle
import time
from typing import Dict, List, Union

import pytest
from outlines_core import Guide, Index, Vocabulary, create_mask, mask_to_list


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


def test_get_tokens_into_mask_valid_size():
    tokens = {"a": [0], "b": [1], "z": [2]}
    eos_token_id = 3
    regex = r"z[ab]z"

    vocabulary = Vocabulary(eos_token_id, tokens)

    index = Index(regex, vocabulary)

    guide = Guide(index)
    vocab_size = 3
    mask_buffer = create_mask(vocab_size + 1)

    guide.get_tokens_into_mask(mask_buffer)

    allowed_tokens = mask_to_list(mask_buffer)
    assert len(allowed_tokens) == 1
    assert allowed_tokens[0] == 2


def test_get_tokens_into_mask_invalid_size():
    tokens = {
        "a": [0],
        "b": [1],
        "z": [2],
        "c": [3],
        "d": [4],
        "e": [5],
        "f": [6],
        "g": [7],
    }
    eos_token_id = 8
    regex = r"z[ab]z"

    vocabulary = Vocabulary(eos_token_id, tokens)

    index = Index(regex, vocabulary)
    guide = Guide(index)
    mask = create_mask(1)
    with pytest.raises(
        ValueError,
        match=r"Mask size \(8 bytes\) lower than required size \(9 bytes\)",
    ):
        guide.get_tokens_into_mask(mask)


def test_advance_with_mask_valid_transition():
    tokens = {"a": [1], "b": [2], "z": [3]}
    eos_token_id = 4
    regex = r"z[ab]z"
    vocab_size = 4

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    mask = create_mask(vocab_size)

    # Step 1 :Initial step, waits "z" (token_id=3)
    guide.get_tokens_into_mask(mask)
    allowed_token = mask_to_list(mask)
    assert len(allowed_token) == 1 and allowed_token[0] == 3

    # Step 2 : Advance with "z" (token_id=3), waits "a" or "b" (token_id=1 ou 2)
    guide.advance_with_mask(3, mask)
    allowed_token = mask_to_list(mask)
    assert (
        len(allowed_token) == 2
        and (allowed_token[0] == 1 and allowed_token[1] == 2)
        or (allowed_token[0] == 2 and allowed_token[1] == 1)
    )

    # Step 3 : Advance with "a" (token_id=1), waits "z" (token_id=3)
    guide.advance_with_mask(1, mask)
    allowed_token = mask_to_list(mask)
    assert len(allowed_token) == 1 and allowed_token[0] == 3

    # Step 4 : Advance with "z" (token_id=3), waits final state (eos=4)
    guide.advance_with_mask(3, mask)
    allowed_token = mask_to_list(mask)
    assert len(allowed_token) == 1 and allowed_token[0] == 4 and guide.is_finished()


def test_advance_with_mask_invalid_transition(index):
    guide = Guide(index)
    mask = create_mask(3)

    with pytest.raises(
        ValueError,
        match="No next state found for the current state",
    ):
        guide.advance_with_mask(2, mask)
        guide.advance_with_mask(2, mask)


def test_TFM_standard():
    regexes = [
        {
            "name": "email",
            "regex": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        },
        {"name": "phone", "regex": "\\+?[1-9][0-9]{7,14}"},
        {
            "name": "date",
            "regex": "([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\\.|-|/)([1-9]|0[1-9]|1[0-2])(\\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\\.|-|/)([1-9]|0[1-9]|1[0-2])(\\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
        },
        {
            "name": "ip",
            "regex": "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
        },
        {
            "name": "url",
            "regex": "(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?",
        },
        {"name": "ssn", "regex": "\\d{3}-\\d{2}-\\d{4}"},
    ]
    vocab = Vocabulary.from_pretrained("gpt2")
    print("\n> Current Behavior :")
    for regex in regexes:
        index = Index(regex["regex"], vocab)
        guide = Guide(index)

        start = time.perf_counter()

        guide.get_tokens()

        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6  # Conversion en microsecondes
        print(f"{regex['name']}: TFM: {elapsed_us:.2f} µs")


def test_TFM_optimized():
    regexes = [
        {
            "name": "email",
            "regex": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
        },
        {"name": "phone", "regex": "\\+?[1-9][0-9]{7,14}"},
        {
            "name": "date",
            "regex": "([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\\.|-|/)([1-9]|0[1-9]|1[0-2])(\\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\\.|-|/)([1-9]|0[1-9]|1[0-2])(\\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
        },
        {
            "name": "ip",
            "regex": "(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
        },
        {
            "name": "url",
            "regex": "(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?",
        },
        {"name": "ssn", "regex": "\\d{3}-\\d{2}-\\d{4}"},
    ]
    vocab = Vocabulary.from_pretrained("gpt2")
    print("\n> Optimized Behavior :")
    for regex in regexes:
        index = Index(regex["regex"], vocab)
        guide = Guide(index)
        mask = create_mask(len(vocab) + 1)

        start = time.perf_counter()

        guide.get_tokens_into_mask(mask)

        end = time.perf_counter()

        elapsed_us = (end - start) * 1e6  # Conversion en microsecondes
        print(f"{regex['name']}: TFM: {elapsed_us:.2f} µs")
