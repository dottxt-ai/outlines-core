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


def test_required_only_anyof_with_properties():
    schema = {
        "type": "object",
        "properties": {"modifier": {"enum": ["", "dark"]}},
        "anyOf": [{"required": ["modifier"]}],
        "additionalProperties": False,
    }
    regex = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(regex, '{"modifier":"dark"}')
    assert re.fullmatch(regex, '{"modifier":""}')
    assert not re.fullmatch(regex, "{}")
    assert not re.fullmatch(regex, '{"modifier":"light"}')


def test_required_only_anyof_combines_parent_and_alternatives():
    schema = {
        "properties": {
            "base": {"const": 1},
            "a": {"const": 2},
            "b": {"const": 3},
        },
        "required": ["base"],
        "anyOf": [{"required": ["a", "a"]}, {"required": ["b"]}],
    }
    regex = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(regex, '{"base":1,"a":2}')
    assert re.fullmatch(regex, '{"base":1,"b":3}')
    assert re.fullmatch(regex, '{"base":1,"a":2,"b":3}')
    assert not re.fullmatch(regex, '{"base":1}')
    assert not re.fullmatch(regex, '{"a":2}')


def test_required_only_anyof_reference_depth_is_independent():
    schema = {
        "$id": "urn:fixture",
        "$defs": {"v": {"const": 1}},
        "properties": {"a": {"$ref": "urn:fixture#/$defs/v"}},
        "anyOf": [{"required": ["a"]} for _ in range(5)],
    }
    regex = build_regex_from_schema(json.dumps(schema))
    assert re.fullmatch(regex, '{"a":1}')
    assert not re.fullmatch(regex, "{}")
    assert not re.fullmatch(regex, '{"a":2}')
