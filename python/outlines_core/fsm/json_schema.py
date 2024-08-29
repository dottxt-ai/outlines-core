import inspect
import warnings
from typing import Callable

from pydantic import create_model

from .outlines_core_rs import (  # noqa: F401
    BOOLEAN,
    DATE,
    DATE_TIME,
    INTEGER,
    NULL,
    NUMBER,
    STRING,
    STRING_INNER,
    TIME,
    UUID,
    WHITESPACE,
    build_regex_from_schema,
    to_regex,
)


def get_schema_from_signature(fn: Callable) -> str:
    """Turn a function signature into a JSON schema.

    Every JSON object valid to the output JSON Schema can be passed
    to `fn` using the ** unpacking syntax.

    """
    signature = inspect.signature(fn)
    arguments = {}
    for name, arg in signature.parameters.items():
        if arg.annotation == inspect._empty:
            raise ValueError("Each argument must have a type annotation")
        else:
            arguments[name] = (arg.annotation, ...)

    try:
        fn_name = fn.__name__
    except Exception as e:
        fn_name = "Arguments"
        warnings.warn(
            f"The function name could not be determined. Using default name 'Arguments' instead. For debugging, here is exact error:\n{e}",
            category=UserWarning,
        )
    model = create_model(fn_name, **arguments)  # type: ignore

    return model.model_json_schema()
