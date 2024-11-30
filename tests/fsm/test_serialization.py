import pickle

import pytest
from outlines_core.fsm.guide import RegexGuide
from transformers import AutoTokenizer

from tests.fsm.test_regex import TransformerTokenizer


def test_serialization():
    class MockTokenizer:
        vocabulary = {"1": 1, "a": 2, "eos": 3}
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return token

    regex_str = "[1-9]"
    tokenizer = MockTokenizer()

    fsm = RegexGuide.from_regex(regex_str, tokenizer)

    serialized = pickle.dumps(fsm)
    deserialized = pickle.loads(serialized)

    assert fsm.eos_tensor == deserialized.eos_tensor
    assert fsm.initial_state == deserialized.initial_state


@pytest.mark.parametrize(
    "hf_tokenizer_uri, revision",
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
def test_complex_serialization(hf_tokenizer_uri, revision):
    # The combined regular expressions of a lexer state in a Python grammar
    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_uri, revision=revision)
    tokenizer = TransformerTokenizer(tokenizer)

    fsm = RegexGuide.from_regex(regex_str, tokenizer)

    serialized = pickle.dumps(fsm)
    deserialized = pickle.loads(serialized)

    assert fsm.eos_tensor == deserialized.eos_tensor
    assert fsm.initial_state == deserialized.initial_state
