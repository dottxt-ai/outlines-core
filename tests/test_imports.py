"""Tests for package imports to catch import/module registration issues."""


def test_import_outlines_core():
    import outlines_core

    assert hasattr(outlines_core, "Guide")
    assert hasattr(outlines_core, "Index")
    assert hasattr(outlines_core, "Vocabulary")
    assert hasattr(outlines_core, "json_schema")


def test_import_json_schema_module():
    from outlines_core import json_schema

    assert hasattr(json_schema, "BOOLEAN")
    assert hasattr(json_schema, "build_regex_from_schema")


def test_import_from_json_schema():
    from outlines_core.json_schema import (  # noqa: F401
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
        build_regex_from_schema,
    )

    assert BOOLEAN == "(true|false)"
    assert callable(build_regex_from_schema)


def test_import_main_classes():
    from outlines_core import Guide, Index, Vocabulary

    assert Guide is not None
    assert Index is not None
    assert Vocabulary is not None


# The tests below ensure that users can import using
# `outlines_core.outlines_core`. This is not the intended usage but we should
# keep supporting it as users may have fallen back to it when the regular
# Python imports were buggy (fixed in v0.2.14)


def test_import_rust_extension_directly():
    from outlines_core import outlines_core

    assert hasattr(outlines_core, "Guide")
    assert hasattr(outlines_core, "Index")
    assert hasattr(outlines_core, "Vocabulary")
    assert hasattr(outlines_core, "json_schema")


def test_import_from_rust_extension():
    from outlines_core.outlines_core import Guide, Index, Vocabulary

    assert Guide is not None
    assert Index is not None
    assert Vocabulary is not None


def test_import_json_schema_from_rust_extension():
    from outlines_core.outlines_core import json_schema

    assert hasattr(json_schema, "BOOLEAN")
    assert hasattr(json_schema, "build_regex_from_schema")
