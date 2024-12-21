import pytest
from outlines_core.fsm import Guide, Index, Vocabulary


def test_stop_at_eos_txt():
    eos_token_id = 3
    # TODO: support bytes from python
    # tokens = {b"1": {1}, b"a": {2}}
    tokens = {"1": [1], "a": [2]}

    regex = r"[1-9]"
    vocabulary = Vocabulary.from_dict(eos_token_id, tokens)

    index = Index(regex, vocabulary)
    guide = Guide(index)

    assert list(guide.get_start_tokens()) == [1]
    assert list(guide.read_next_token(1)) == [vocabulary.get_eos_token_id()]
    assert guide.is_finished()

    with pytest.raises(
        ValueError,
        match="No next state found for the current state",
    ):
        assert list(guide.read_next_token(4)) == []
