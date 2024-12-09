import json
import re
from typing import Literal, Union

import interegular
import pytest
from outlines_core.fsm.json_schema import build_regex_from_schema, to_regex
from pydantic import BaseModel, Field


@pytest.mark.parametrize("whitespace_pattern", [None, r"[\n ]*", "abc"])
def test_json_schema_custom_whitespace_pattern(whitespace_pattern):
    """assert whitespace_pattern setting respected"""

    class MockModel(BaseModel):
        foo: int
        bar: str

    schema = json.dumps(MockModel.model_json_schema())

    # assert any ws pattern can be used
    if whitespace_pattern == "abc":
        build_regex_from_schema(schema, whitespace_pattern)
        return

    pattern = build_regex_from_schema(schema, whitespace_pattern)

    mock_result_mult_ws = (
        """{     "foo"   :   4, \n\n\n   "bar": "baz    baz baz bar"\n\n}"""
    )
    mock_result_maybe_ws = """{"foo" : 4 ,"bar":"baz    baz baz bar"}"""

    match_default_ws = re.fullmatch(pattern, mock_result_maybe_ws)
    if whitespace_pattern is None:
        assert match_default_ws
    else:
        assert re.fullmatch(pattern, mock_result_mult_ws)


def test_one_of_doesnt_produce_illegal_lookaround():
    """Reproduces failure in https://github.com/dottxt-ai/outlines/issues/823"""

    class Cat(BaseModel):
        pet_type: Literal["cat"]
        meows: int

    class Dog(BaseModel):
        pet_type: Literal["dog"]
        barks: float

    class Model(BaseModel):
        pet: Union[Cat, Dog] = Field(..., discriminator="pet_type")
        n: int

    json_schema = json.dumps(Model.model_json_schema())
    pattern = build_regex_from_schema(json_schema, whitespace_pattern=None)

    # check if the pattern uses lookarounds incompatible with interegular.Pattern.to_fsm()
    interegular.parse_pattern(pattern).to_fsm()


def test_match_object():
    test_regex = to_regex(
        {
            "type": "object",
            "maxProperties": 0,
        }
    )
    assert test_regex == r"\{[ ]?\}"
