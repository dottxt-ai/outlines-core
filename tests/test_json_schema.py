import json
import re

import pytest
from pydantic import BaseModel

from outlines_core.json_schema import build_regex_from_schema


def test_build_regex_from_json_schema():
    class FooBar(BaseModel):
        foo: int
        bar: str

    schema = json.dumps(FooBar.model_json_schema())

    regex = build_regex_from_schema(schema)
    expected = """{"foo" : 4 ,"bar":"baz    baz baz bar"}"""
    assert re.fullmatch(regex, expected)

    # any whitespace pattern can be used
    regex = build_regex_from_schema(schema, r"[\n ]*")
    expected = """{     "foo"   :   4, \n\n\n   "bar": "baz    baz baz bar"\n\n}"""
    assert re.fullmatch(regex, expected)


def test_invalid_json():
    with pytest.raises(
        TypeError,
        match="Expected a valid JSON string.",
    ):
        build_regex_from_schema("{'name':")


def test_types_presence_and_not_emptyness():
    from outlines_core.json_schema import (
        BOOLEAN,
        DATE,
        DATE_TIME,
        EMAIL,
        INTEGER,
        NULL,
        NUMBER,
        STRING,
        STRING_INNER,
        TIME,
        URI,
        UUID,
        WHITESPACE,
    )

    assert BOOLEAN
    assert DATE
    assert DATE_TIME
    assert EMAIL
    assert INTEGER
    assert NULL
    assert NUMBER
    assert STRING
    assert STRING_INNER
    assert TIME
    assert URI
    assert UUID
    assert WHITESPACE


@pytest.mark.parametrize(
    "additional_properties, accepted, rejected",
    [
        (
            True,
            ['{"a": 1}', '{"a": 1, "b": "x"}', '{"a": 1, "b": {"c": [1]}}'],
            ['{"b": "x"}', '{"a": 1, "a": "x"}'],
        ),
        (
            {"type": "integer"},
            ['{"a": 1}', '{"a": 1, "b": 2, "c": 3}'],
            ['{"a": 1, "b": "x"}', '{"a": 1, "a": 2}'],
        ),
        (False, ['{"a": 1}'], ['{"a": 1, "b": 2}']),
    ],
)
def test_additional_properties_with_properties(
    additional_properties, accepted, rejected
):
    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "additionalProperties": additional_properties,
    }
    regex = build_regex_from_schema(json.dumps(schema))
    for value in accepted:
        assert re.fullmatch(regex, value), value
    for value in rejected:
        assert not re.fullmatch(regex, value), value
