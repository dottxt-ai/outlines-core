import copy
import json
import pickle

import pytest

from outlines_core import Vocabulary


@pytest.fixture(scope="session")
def vocabulary():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    return Vocabulary(eos_token_id, tokens)


@pytest.mark.thread_unsafe(reason="mutably borrows a shared Vocabulary")
def test_basic_vocabulary_interface(vocabulary):
    assert vocabulary.get_eos_token_id() == 3
    assert vocabulary.get("1") == vocabulary.get(b"1") == [1]
    assert len(vocabulary) == 3

    vocabulary.insert("b", 4)
    assert vocabulary.get("b") == [4]
    assert len(vocabulary) == 4

    vocabulary.insert(b"b", 5)
    assert vocabulary.get("b") == vocabulary.get(b"b") == [4, 5]
    assert len(vocabulary) == 5

    vocabulary.remove("b")
    assert vocabulary.get("b") is None

    # second remove doesn't fail too
    vocabulary.remove("b")
    assert vocabulary.get("b") is None

    assert vocabulary.get("a") == [2]
    vocabulary.remove(b"a")
    assert vocabulary.get("a") is None


def test_string_and_bytes_as_tokens():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    btokens = {b"1": [1], b"a": [2]}
    vocabulary = Vocabulary(eos_token_id, tokens)
    bvocabulary = Vocabulary(eos_token_id, btokens)

    assert (
        vocabulary.get_eos_token_id() == bvocabulary.get_eos_token_id() == eos_token_id
    )
    assert vocabulary.get(b"1") == vocabulary.get("1") == [1]
    assert bvocabulary.get(b"1") == bvocabulary.get("1") == [1]
    assert len(vocabulary) == len(bvocabulary) == 3


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


def test_get_bad_type(vocabulary):
    with pytest.raises(
        TypeError,
        match="Expected a token of type str or bytes, got",
    ):
        vocabulary.get(1)


@pytest.mark.thread_unsafe(reason="mutably borrows a shared Vocabulary")
def test_insert_bad_type(vocabulary):
    with pytest.raises(
        TypeError,
        match="Expected a token of type str or bytes, got",
    ):
        vocabulary.insert(1, 6)


@pytest.mark.thread_unsafe(reason="mutably borrows a shared Vocabulary")
def test_insert_eos_token(vocabulary):
    with pytest.raises(
        ValueError, match="EOS token should not be inserted into Vocabulary"
    ):
        vocabulary.insert("eos-token", 3)


def test_from_pretrained():
    vocabulary = Vocabulary.from_pretrained("gpt2")
    assert vocabulary.get_eos_token_id() == 50256


@pytest.fixture
def local_tokenizer(tmp_path):
    directory = tmp_path / "local tokenizer"
    directory.mkdir()
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "vocab": {"<eos>": 0, "a": 1, "Ġ": 2},
            "merges": [],
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    return directory


@pytest.mark.parametrize("eos_token", ["<eos>", {"content": "<eos>"}])
def test_from_local_tokenizer_config(local_tokenizer, eos_token):
    (local_tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": eos_token}), encoding="utf-8"
    )
    vocabulary = Vocabulary.from_pretrained(str(local_tokenizer))
    assert vocabulary.get_eos_token_id() == 0
    assert vocabulary.get("<eos>") is None
    assert vocabulary.get("a") == [1]
    assert vocabulary.get(" ") == [2]


def test_local_generation_config_has_priority(local_tokenizer):
    (local_tokenizer / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 0}), encoding="utf-8"
    )
    (local_tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "a"}), encoding="utf-8"
    )
    vocabulary = Vocabulary.from_pretrained(str(local_tokenizer), revision="local only")
    assert vocabulary.get_eos_token_id() == 0
    assert vocabulary.get("a") == [1]


def test_local_tokenizer_without_eos_metadata(local_tokenizer):
    with pytest.raises(ValueError, match="EOS token id"):
        Vocabulary.from_pretrained(str(local_tokenizer))


def test_local_invalid_generation_config_falls_back(local_tokenizer):
    (local_tokenizer / "generation_config.json").write_text("{", encoding="utf-8")
    (local_tokenizer / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": "<eos>"}), encoding="utf-8"
    )
    assert Vocabulary.from_pretrained(str(local_tokenizer)).get_eos_token_id() == 0


def test_local_directory_without_tokenizer(tmp_path):
    with pytest.raises(ValueError):
        Vocabulary.from_pretrained(str(tmp_path))


def test_pickling(vocabulary):
    serialized = pickle.dumps(vocabulary)
    deserialized = pickle.loads(serialized)
    assert deserialized == vocabulary


def test_deepcopy(vocabulary):
    vocabulary2 = copy.deepcopy(vocabulary)
    assert vocabulary2 == vocabulary

    copy_vocabulary2 = copy.deepcopy(vocabulary2)
    assert copy_vocabulary2 == vocabulary2

    vocabulary2.insert("new", 4)
    assert vocabulary2 != copy_vocabulary2
    assert len(vocabulary2) - 1 == len(copy_vocabulary2)
    assert copy_vocabulary2 == vocabulary
    assert len(copy_vocabulary2) == len(vocabulary)
