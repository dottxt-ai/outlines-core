from typing import List, Tuple, Union

import interegular
import pytest
import torch
from datasets.fingerprint import Hasher
from outlines_core.fsm.outlines_core_rs import Vocabulary
from outlines_core.fsm.regex import (
    BetterAlphabet,
    BetterFSM,
    _walk_fsm,
    create_fsm_index_end_to_end,
    create_fsm_index_tokenizer,
    get_token_transition_keys,
    get_vocabulary_transition_keys,
    make_byte_level_fsm,
    make_deterministic_fsm,
    reduced_vocabulary,
)
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )


class TransformerTokenizer:
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs["padding"] = True
        kwargs["return_tensors"] = "pt"
        output = self.tokenizer(prompt, **kwargs)
        return output["input_ids"], output["attention_mask"]

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, "model_name") and hasattr(self, "kwargs"):
                return (
                    other.model_name == self.model_name and other.kwargs == self.kwargs
                )
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __getstate__(self):
        state = {"tokenizer": self.tokenizer}
        return state

    def __setstate__(self, state):
        self.__init__(state["tokenizer"])


def identity(s):
    return s


def to_bytes(s):
    return [chr(b) if b < 0x80 else f"\x00{b:02X}" for b in s.encode("utf-8")]


def merge_symbols(byte_hexs):
    return "".join(["\x00" + b if len(b) == 2 else b for b in byte_hexs])


def token_str_to_trans_key(fsm, input_string):
    return get_token_transition_keys(
        fsm.fsm_info.alphabet_symbol_mapping,
        fsm.fsm_info.alphabet_anything_value,
        input_string,
    )


def walk_fsm_from_token_str_rust(
    fsm,
    input_string: str,
    start_state: int,
    full_match: bool = True,
):
    return _walk_fsm(
        fsm.fsm_info.transitions,
        fsm.fsm_info.initial,
        fsm.fsm_info.finals,
        token_str_to_trans_key(fsm, input_string),
        start_state,
        full_match=full_match,
    )


def make_byte_level_better_fsm(fsm: BetterFSM, keep_utf8=False) -> BetterFSM:
    new_fsm = make_byte_level_fsm(fsm, keep_utf8)
    return BetterFSM(
        alphabet=BetterAlphabet(new_fsm.alphabet._symbol_mapping),
        states=new_fsm.states,
        initial=new_fsm.initial,
        finals=new_fsm.finals,
        map=new_fsm.map,
    )


def test_walk_fsm():
    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]*")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(
        walk_fsm_from_token_str_rust(regex_fsm, "0", regex_fsm.initial, full_match=True)
    )
    assert res == (1,)

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, "00", regex_fsm.initial, full_match=False
        )
    )
    assert res == (1,)

    res = tuple(
        walk_fsm_from_token_str_rust(regex_fsm, "!", regex_fsm.initial, full_match=True)
    )
    assert res == tuple()

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, "00", regex_fsm.initial, full_match=True
        )
    )
    assert res == tuple()

    # This should fail, because state `1` reads nothing
    res = tuple(walk_fsm_from_token_str_rust(regex_fsm, "0", 1, full_match=True))
    assert res == tuple()

    regex_pattern = interegular.parse_pattern("0|[1-9][2-9]+")
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    res = tuple(
        walk_fsm_from_token_str_rust(regex_fsm, "1", regex_fsm.initial, full_match=True)
    )
    assert res == tuple()

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, "1", regex_fsm.initial, full_match=False
        )
    )
    assert res == (2,)

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, "12", regex_fsm.initial, full_match=True
        )
    )
    assert res == (2, 3)

    pattern = interegular.parse_pattern(r"(?:[^\W\d]\w*|[\t \x0c]+)")
    fsm, _ = make_deterministic_fsm(pattern.to_fsm().reduce())

    res = tuple(walk_fsm_from_token_str_rust(fsm, "x ", fsm.initial, full_match=False))
    assert res == (2,)

    start_state = list(fsm.finals)[0]
    res = tuple(walk_fsm_from_token_str_rust(fsm, "!", start_state, full_match=False))
    assert res == tuple()


@pytest.mark.parametrize(
    "transform",
    [
        identity,
        to_bytes,
    ],
)
def test_walk_fsm_multi_bytes(transform):
    regex_pattern = interegular.parse_pattern("😂|[😇-😍][😈-😍]*")
    str_regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
    regex_fsm = make_byte_level_better_fsm(str_regex_fsm, keep_utf8=True)

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, merge_symbols(transform("😂")), regex_fsm.initial, full_match=True
        )
    )
    assert res[-1:] == (1,)

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm,
            merge_symbols(transform("😂😂")),
            regex_fsm.initial,
            full_match=False,
        )
    )
    assert res[-1:] == (1,)

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm, merge_symbols(transform("!")), regex_fsm.initial, full_match=True
        )
    )
    assert res == tuple()

    res = tuple(
        walk_fsm_from_token_str_rust(
            regex_fsm,
            merge_symbols(transform("😂😂")),
            regex_fsm.initial,
            full_match=True,
        )
    )
    assert res == tuple()


def test_create_fsm_index_end_to_end():
    regex_str = "0|[1-9][0-9]*"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

    tokens_to_token_ids = {
        "blah": [0],
        "1a": [1],
        "2": [2],
        "0": [3],
        "<EOS>": [4],
    }

    res = create_fsm_index_end_to_end(
        regex_fsm.fsm_info,
        Vocabulary.from_dict(tokens_to_token_ids),
        frozenset(),
    )

    assert res == {0: {2: 2, 3: 1}, 2: {2: 2, 3: 2}}


def test_create_fsm_index_end_to_end_multi_byte():
    regex_str = "😇| [😈-😍][😇-😎]*"

    regex_pattern = interegular.parse_pattern(regex_str)
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())
    byte_fsm = make_byte_level_better_fsm(regex_fsm, keep_utf8=True)

    tokens_to_token_ids = {
        "blah": [0],
        "😈a": [1],
        "😇": [2],
        "😍": [3],
        merge_symbols(("F0", "9F", "98", "8D")): [4],  # '😍'
        " 😍": [5],
        merge_symbols((" ", "F0", "9F", "98", "8D")): [6],  # ' 😍'
        merge_symbols((" ", "F0", "9F", "98")): [7],  # ' 😍' incomplete
        "<EOS>": [8],
    }

    res = create_fsm_index_end_to_end(
        byte_fsm.fsm_info,
        Vocabulary.from_dict(tokens_to_token_ids),
        frozenset(),
    )

    assert res == {0: {5: 3, 6: 3, 7: 7, 2: 2}, 3: {2: 3, 3: 3, 4: 3}}


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
def test_create_fsm_index_tokenizer(hf_tokenizer_uri, revision):
    # The combined regular expressions of a lexer state in a Python grammar
    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())
    bytes_fsm = make_byte_level_better_fsm(regex_fsm, keep_utf8=True)

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    num_bytes_fsm_states = len(bytes_fsm.states)
    assert num_bytes_fsm_states == 235

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_uri, revision=revision)
    tokenizer = TransformerTokenizer(tokenizer)

    states_to_token_subsets, empty_token_ids = create_fsm_index_tokenizer(
        bytes_fsm, tokenizer
    )

    assert not empty_token_ids
    assert len(states_to_token_subsets.get_transitions()) / num_fsm_states > 0.94


@pytest.mark.parametrize(
    "regex,string,should_accept",
    [
        ("[a-c]+", "😀", False),
        ("[^a-c]+", "😀", True),
        ("😀+", "😀😀😀", True),
        ("😀+", "a", False),
        ("[😀-😍]{2}", "😈😈", True),
        ("[😀-😍]{2}", "aa", False),
        ("[^😀-😍]{2}", "aa", True),
        ("[^😀-😍]{2}", "😈😈", False),
        ("[^😀-😍]{2}", "😎😎", True),
        ("[^😀-😍]{2}", "😎😓", True),
        ("[^😀-😍]{2}", "😎😈", False),
        ("[😀-🙌]{2}", "😎😈", True),
        ("[^😀-🙌]{2}", "😎😈", False),
        ("[^😀-🙌]{2}", "🙏🙏", True),
        ("[^😀-🙌]{2}", "🙏😎", False),
    ],
)
def test_make_byte_level_fsm(regex, string, should_accept):
    str_fsm = interegular.parse_pattern(regex).to_fsm()
    str_accepts = str_fsm.accepts(string)
    assert str_accepts == should_accept

    byte_fsm = make_byte_level_fsm(str_fsm)
    byte_accepts = byte_fsm.accepts(to_bytes(string))  # type: ignore
    assert byte_accepts == str_accepts

    mix_fsm = make_byte_level_fsm(str_fsm, keep_utf8=True)
    mix_accepts = mix_fsm.accepts(to_bytes(string))  # type: ignore
    assert mix_accepts == str_accepts

    mix_accepts_utf8 = mix_fsm.accepts(string)  # type: ignore
    assert mix_accepts_utf8 == str_accepts

    def advance(fsm, state, seq):
        for symbol in seq:
            if state is None:
                return None
            key = fsm.alphabet[symbol]
            state = fsm.map[state].get(key)
        return state

    # verify each state along the pattern
    str_state = str_fsm.initial
    byte_state = byte_fsm.initial
    mix_state = byte_fsm.initial
    for symbol in string:
        str_state = advance(str_fsm, str_state, symbol)
        byte_state = advance(byte_fsm, byte_state, to_bytes(symbol))
        mix_state_utf8 = advance(mix_fsm, mix_state, symbol)
        mix_state = advance(mix_fsm, mix_state, to_bytes(symbol))
        assert byte_state == str_state
        assert mix_state == str_state
        assert mix_state_utf8 == str_state


@pytest.mark.skip(reason="Only for local profiling")
def test_regex_index_performance():
    from line_profiler import LineProfiler  # type: ignore [import]

    regex_str = "(?:(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|[0-9](?:(?:_)?[0-9])*)(?:J|j)|(?:[0-9](?:(?:_)?[0-9])*(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*|(?:[0-9](?:(?:_)?[0-9])*\\.(?:[0-9](?:(?:_)?[0-9])*)?|\\.[0-9](?:(?:_)?[0-9])*)(?:(?:e|E)(?:(?:\\+|\\-))?[0-9](?:(?:_)?[0-9])*)?)|0(?:x|X)(?:(?:_)?(?:[0-9]|[a-f]|[A-F]))+|0(?:b|B)(?:(?:_)?[0-1])+|0(?:o|O)(?:(?:_)?[0-7])+|(?:(?i:([ubf]?r?|r[ubf])('([^\\\\']|.)*?'))|(?i:([ubf]?r?|r[ubf])(\"([^\\\"]|.)*?\")))|(?:(?:\r?\n[\t ]*|#[^\n]*))+|[1-9](?:(?:_)?[0-9])*|\\\\[\t \x0c]*\r?\n|continue|nonlocal|assert|global|import|lambda|return|async|await|break|class|False|match|raise|while|yield|case|from|None|pass|True|with|def|del|for|not|try|if|[^\\W\\d]\\w*|#[^\n]*|[\t \x0c]+|\\.\\.\\.|@|\\{|\\(|\\[|\\-|\\+|\\*|\\~"

    regex_pattern = interegular.parse_pattern(regex_str)
    # Not reduced, so that there are many states
    regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm())

    num_fsm_states = len(regex_fsm.states)
    assert num_fsm_states == 220

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer = TransformerTokenizer(tokenizer)

    res, _ = create_fsm_index_tokenizer(regex_fsm, tokenizer)
    assert len(res) > 1

    profiler = LineProfiler(create_fsm_index_end_to_end)

    profiler.runctx(
        "create_fsm_index_tokenizer(regex_fsm, tokenizer)",
        globals(),
        locals(),
    )
    profiler.dump_stats("line-profiler-create_fsm_index.pkl")
    profiler.print_stats(output_unit=1e-3, summarize=True, stripzeros=True)


def test_token_trans_keys_identical():
    """assert two tokens w/ identical behavior wrt FSM have same trans key seq"""

    class MockTokenizer:
        vocabulary = {"a": 1, "b": 2, "z": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

    tokenizer = MockTokenizer()

    pattern = r"z[ab]z"
    regex_pattern = interegular.parse_pattern(pattern)
    interegular_fsm = regex_pattern.to_fsm().reduce()
    regex_fsm, _ = make_deterministic_fsm(interegular_fsm)
    tokens_to_token_ids, _ = reduced_vocabulary(tokenizer)
    token_str_to_tranition_keys = get_vocabulary_transition_keys(
        regex_fsm.fsm_info.alphabet_symbol_mapping,
        regex_fsm.fsm_info.alphabet_anything_value,
        Vocabulary.from_dict(tokens_to_token_ids),
        frozenset(),
    )

    # `a` and `b` both are workable, but `z` has distinct transition rules
    assert interegular_fsm.accepts("zaz")
    assert interegular_fsm.accepts("zbz")
    assert token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["b"]
    assert not token_str_to_tranition_keys["a"] == token_str_to_tranition_keys["z"]


def test_token_trans_keys_walk_fsm():
    """assert _walk_fsm works using transition keys"""

    class MockTokenizer:
        vocabulary = {"ab": 1, "ac": 2, "az": 3, "eos": 4}
        special_tokens = {"eos"}
        eos_token_id = 4

        def convert_token_to_string(self, token):
            return token

    tokenizer = MockTokenizer()

    pattern = r"a[bc]z"
    regex_pattern = interegular.parse_pattern(pattern)
    interegular_fsm = regex_pattern.to_fsm().reduce()
    regex_fsm, _ = make_deterministic_fsm(interegular_fsm)
    tokens_to_token_ids, _ = reduced_vocabulary(tokenizer)
    token_str_to_tranition_keys = get_vocabulary_transition_keys(
        regex_fsm.fsm_info.alphabet_symbol_mapping,
        regex_fsm.fsm_info.alphabet_anything_value,
        Vocabulary.from_dict(tokens_to_token_ids),
        frozenset(),
    )

    # verify initial state valid only for "ab" and "ac" using transition key seq
    token_acceptance = {"ab": True, "ac": True, "az": False}
    for token, should_accept in token_acceptance.items():
        token_trans_key_seq = token_str_to_tranition_keys[token]
        state_seq = _walk_fsm(
            regex_fsm.fsm_info.transitions,
            regex_fsm.fsm_info.initial,
            regex_fsm.fsm_info.finals,
            token_trans_key_seq,
            regex_fsm.initial,
            False,
        )
        is_accepted = len(state_seq) >= len(token_trans_key_seq)
        assert should_accept == is_accepted


@pytest.mark.parametrize(
    "rare_token",
    [
        "�",
        "��",
        "�.",
        "�..",
        "▁�",
        "▁▁�",
        "▁�.",
        "▁�.",
        "▁▁�..",
    ],
)
def test_reduced_vocabulary_with_rare_tokens(rare_token):
    """Assert reduced_vocabulary works with rare tokens.

    See [1] and [2] for context.

    [1]: https://github.com/dottxt-ai/outlines/pull/763
    [2]: https://github.com/dottxt-ai/outlines/pull/948
    [3]: https://github.com/dottxt-ai/outlines/pull/1153
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer = TransformerTokenizer(tokenizer=tokenizer)
    tokenizer.vocabulary[rare_token] = max(tokenizer.vocabulary.values()) + 1
    reduced_vocabulary(tokenizer)


def test_reduced_vocabulary_with_byte_tokens():
    class MockTokenizer:
        vocabulary = {
            "string": 1,
            b"\xa1": 2,  # Qwen-Style
            "eos": 3,
        }
        special_tokens = {"eos"}
        eos_token_id = 3

        def convert_token_to_string(self, token):
            return b"\xef\xbf\xbd".decode()

    tokens_to_token_ids = reduced_vocabulary(MockTokenizer())

    # See fsm.regex.get_token_transition_keys()
    # FSM transition keys represents bytes as <null_prefix><hex_byte>
    assert tokens_to_token_ids[0] == {"string": [1], "\x00A1": [2]}
