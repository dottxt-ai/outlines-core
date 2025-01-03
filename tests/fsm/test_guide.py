import pickle

import pytest
from outlines_core.fsm import Guide, Index, Vocabulary


def test_stop_at_eos():
    eos_token_id = 3
    tokens = {"1": [1], "a": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    assert guide.get_start_tokens() == [1]
    assert guide.read_next_token(1) == [vocabulary.get_eos_token_id()]
    assert guide.is_finished()

    with pytest.raises(
        ValueError,
        match="No next state found for the current state",
    ):
        assert guide.read_next_token(4) == []


def test_regex_final_state_walk():
    # Make sure that the Guide can walk to the final state correctly.
    eos_token_id = 104
    tokens = {b"\n": [103], b".": [102], b"`": [101]}
    regex = r"`\n(\.\n)?`\n"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    assert guide.get_start_tokens() == [101]
    assert guide.read_next_token(101) == [103]
    assert sorted(guide.read_next_token(103)) == [101, 102]
    assert guide.read_next_token(101) == [103]
    assert guide.read_next_token(103) == [vocabulary.get_eos_token_id()]
    assert guide.is_finished()


def test_token_trans_keys_identical():
    tokens = {"a": [1], "b": [2], "z": [3]}
    eos_token_id = 4
    regex = r"z[ab]z"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)

    guide1 = Guide(index)
    guide2 = Guide(index)

    assert guide1.read_next_token(3) == guide2.read_next_token(3)
    # `a` and `b` have similar transitions to `z`
    assert guide1.read_next_token(1) == guide2.read_next_token(2)
    assert guide1.read_next_token(3) == guide2.read_next_token(3) == [eos_token_id]
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

    assert guide1.read_next_token(3) == guide2.read_next_token(3)
    # `a` and `b` have similar transitions to `z`
    assert guide1.read_next_token(1) == guide2.read_next_token(2)
    assert guide1.read_next_token(3) == guide2.read_next_token(3) == [eos_token_id]
    assert guide1.is_finished()
    assert guide2.is_finished()


def test_pickling():
    eos_token_id = 3
    tokens = {"1": [1], "2": [2]}
    regex = r"[1-9]"

    vocabulary = Vocabulary(eos_token_id, tokens)
    index = Index(regex, vocabulary)
    guide = Guide(index)

    serialized = pickle.dumps(guide)
    deserialized = pickle.loads(serialized)
    assert sorted(deserialized.get_start_tokens()) == sorted(guide.get_start_tokens())


#   @pytest.mark.parametrize(
#       "hf_tokenizer_uri, revision",
#       [
#           ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"),
#           ("microsoft/phi-2", "ef382358ec9e382308935a992d908de099b64c23"),
#           ("Qwen/Qwen1.5-0.5B-Chat", "4d14e384a4b037942bb3f3016665157c8bcb70ea"),
#           (
#               "NousResearch/Hermes-2-Pro-Llama-3-8B",
#               "783fd50eb82d7f57758de033861f54d62dde234f",
#           ),
#       ],
#   )
#   def test_create_fsm_index_tokenizer(hf_tokenizer_uri, revision):
#       # The combined regular expressions of a lexer state in a Python grammar
#       regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

#       regex_pattern = interegular.parse_pattern(regex_str)
#       # Not reduced, so that there are many states
#       regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())
#       bytes_fsm = make_byte_level_better_fsm(regex_fsm, keep_utf8=True)

#       num_fsm_states = len(regex_fsm.states)
#       assert num_fsm_states == 220

#       num_bytes_fsm_states = len(bytes_fsm.states)
#       assert num_bytes_fsm_states == 235

#       tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_uri, revision=revision)
#       tokenizer = TransformerTokenizer(tokenizer)

#       states_to_token_subsets, empty_token_ids = create_fsm_index_tokenizer(
#           bytes_fsm, tokenizer
#       )

#       assert not empty_token_ids
#       assert len(states_to_token_subsets.get_transitions()) / num_fsm_states > 0.94
