from typing import List, Tuple, Union

import interegular
import pytest
import torch
from datasets.fingerprint import Hasher
from outlines_core.fsm.outlines_core_rs import Vocabulary
from transformers import AutoTokenizer, PreTrainedTokenizer

# Could be adapted

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


#   @pytest.mark.parametrize(
#       "regex,string,should_accept",
#       [
#           ("[a-c]+", "😀", False),
#           ("[^a-c]+", "😀", True),
#           ("😀+", "😀😀😀", True),
#           ("😀+", "a", False),
#           ("[😀-😍]{2}", "😈😈", True),
#           ("[😀-😍]{2}", "aa", False),
#           ("[^😀-😍]{2}", "aa", True),
#           ("[^😀-😍]{2}", "😈😈", False),
#           ("[^😀-😍]{2}", "😎😎", True),
#           ("[^😀-😍]{2}", "😎😓", True),
#           ("[^😀-😍]{2}", "😎😈", False),
#           ("[😀-🙌]{2}", "😎😈", True),
#           ("[^😀-🙌]{2}", "😎😈", False),
#           ("[^😀-🙌]{2}", "🙏🙏", True),
#           ("[^😀-🙌]{2}", "🙏😎", False),
#       ],
#   )
#   def test_make_byte_level_fsm(regex, string, should_accept):
#       str_fsm = interegular.parse_pattern(regex).to_fsm()
#       str_accepts = str_fsm.accepts(string)
#       assert str_accepts == should_accept

#       byte_fsm = make_byte_level_fsm(str_fsm)
#       byte_accepts = byte_fsm.accepts(to_bytes(string))  # type: ignore
#       assert byte_accepts == str_accepts

#       mix_fsm = make_byte_level_fsm(str_fsm, keep_utf8=True)
#       mix_accepts = mix_fsm.accepts(to_bytes(string))  # type: ignore
#       assert mix_accepts == str_accepts

#       mix_accepts_utf8 = mix_fsm.accepts(string)  # type: ignore
#       assert mix_accepts_utf8 == str_accepts

#       def advance(fsm, state, seq):
#           for symbol in seq:
#               if state is None:
#                   return None
#               key = fsm.alphabet[symbol]
#               state = fsm.map[state].get(key)
#           return state

#       # verify each state along the pattern
#       str_state = str_fsm.initial
#       byte_state = byte_fsm.initial
#       mix_state = byte_fsm.initial
#       for symbol in string:
#           str_state = advance(str_fsm, str_state, symbol)
#           byte_state = advance(byte_fsm, byte_state, to_bytes(symbol))
#           mix_state_utf8 = advance(mix_fsm, mix_state, symbol)
#           mix_state = advance(mix_fsm, mix_state, to_bytes(symbol))
#           assert byte_state == str_state
#           assert mix_state == str_state
#           assert mix_state_utf8 == str_state


#   @pytest.mark.skip(reason="Only for local profiling")
#   def test_regex_index_performance():
#       from line_profiler import LineProfiler  # type: ignore [import]

#       regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

#       regex_pattern = interegular.parse_pattern(regex_str)
#       # Not reduced, so that there are many states
#       regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())

#       num_fsm_states = len(regex_fsm.states)
#       assert num_fsm_states == 220

#       tokenizer = AutoTokenizer.from_pretrained("gpt2")
#       tokenizer = TransformerTokenizer(tokenizer)

#       res, _ = create_fsm_index_tokenizer(regex_fsm, tokenizer)
#       assert len(res) > 1

#       profiler = LineProfiler(create_fsm_index_end_to_end)

#       profiler.runctx(
#           "create_fsm_index_tokenizer(regex_fsm, tokenizer)",
#           globals(),
#           locals(),
#       )
#       profiler.dump_stats("line-profiler-create_fsm_index.pkl")
#       profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)


#   def test_token_trans_keys_identical():
#       """assert two tokens w/ identical behavior wrt FSM have same trans key seq"""

#       class MockTokenizer:
#           vocabulary = {"a": 1, "b": 2, "z": 3, "eos": 4}
#           special_tokens = {"eos"}
#           eos_token_id = 4

#           def convert_token_to_string(self, token):
#               return token

#       tokenizer = MockTokenizer()

#       pattern = r"z[ab]z"
#       regex_pattern = interegular.parse_pattern(pattern)
#       interegular_fsm = regex_pattern.to_fsm().reduce()
#       regex_fsm, _ = make_deterministic_fsm(interegular_fsm)
#       tokens_to_token_ids, _ = reduced_vocabulary(tokenizer)
#       token_str_to_tranition_keys = get_vocabulary_transition_keys(
#           regex_fsm.fsm_info.alphabet_symbol_mapping,
#           regex_fsm.fsm_info.alphabet_anything_value,
#           Vocabulary.from_dict(tokens_to_token_ids),
#           frozenset(),
#       )

#       # `a` and `b` both are workable, but `z` has distinct transition rules
#       assert interegular_fsm.accepts("zaz")
#       assert interegular_fsm.accepts("zbz")
#       assert token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["b"]
#       assert not token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["z"]