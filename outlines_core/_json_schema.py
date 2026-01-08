# Re-export from Rust extension
from .outlines_core import json_schema as _json_schema

BOOLEAN = _json_schema.BOOLEAN
DATE = _json_schema.DATE
DATE_TIME = _json_schema.DATE_TIME
EMAIL = _json_schema.EMAIL
INTEGER = _json_schema.INTEGER
NULL = _json_schema.NULL
NUMBER = _json_schema.NUMBER
STRING = _json_schema.STRING
STRING_INNER = _json_schema.STRING_INNER
TIME = _json_schema.TIME
URI = _json_schema.URI
UUID = _json_schema.UUID
WHITESPACE = _json_schema.WHITESPACE
build_regex_from_schema = _json_schema.build_regex_from_schema

__all__ = [
    "BOOLEAN",
    "DATE",
    "DATE_TIME",
    "EMAIL",
    "INTEGER",
    "NULL",
    "NUMBER",
    "STRING",
    "STRING_INNER",
    "TIME",
    "URI",
    "UUID",
    "WHITESPACE",
    "build_regex_from_schema",
]
