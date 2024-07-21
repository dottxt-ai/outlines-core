import pytest
from outlines_core.models.tokenizer import Tokenizer


def test_tokenizer():
    with pytest.raises(TypeError, match="instantiate abstract"):
        Tokenizer()
