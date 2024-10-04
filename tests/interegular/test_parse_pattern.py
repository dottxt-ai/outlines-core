import pytest
from interegular.patterns import Pattern as InteregularPattern
from interegular.patterns import _CharGroup, _Concatenation, _Repeated
from outlines_core.fsm.regex import parse_pattern

import interegular


def convert_to_interegular(element):
    if isinstance(element, InteregularPattern):
        return element

    element_type = type(element).__name__

    if element_type == "PyLiteral":
        # TODO: handle the negated case if needed
        return _CharGroup(frozenset(element.value), negated=False)

    elif element_type == "PyCharGroup":
        return _CharGroup(frozenset(element.chars), negated=element.inverted)

    elif element_type == "PyRepeated":
        base = convert_to_interegular(element.element)
        return _Repeated(base, element.min, element.max)

    elif element_type == "PyConcatenation":
        parts = [convert_to_interegular(e) for e in element.elements]
        return _Concatenation(parts)

    elif element_type == "PyAlternation":
        options = [convert_to_interegular(e) for e in element.elements]
        return InteregularPattern(options)

    elif element_type == "PyCapture":
        # interegular doesn't have a direct equivalent for Capture
        # we'll just convert the inner element
        return convert_to_interegular(element.element)

    elif element_type == "PyGroup":
        # similar to Capture, we'll just convert the inner element
        return convert_to_interegular(element.element)

    elif element_type == "PyAnchor":
        # TODO: handle the different types of anchors if needed
        # interegular doesn't have a direct equivalent for Anchor either
        # in this case, we'll just raise an error
        raise NotImplementedError("Anchors are not supported in interegular")

    elif element_type == "PyFlag":
        return convert_to_interegular(element.element)

    else:
        raise ValueError(f"Unhandled element type: {element_type}")


def deep_compare(pattern1, pattern2):
    if type(pattern1) != type(pattern2):
        return False

    if isinstance(pattern1, InteregularPattern):
        if len(pattern1.options) != len(pattern2.options):
            return False
        return all(
            deep_compare(opt1, opt2)
            for opt1, opt2 in zip(pattern1.options, pattern2.options)
        )

    elif isinstance(pattern1, _Concatenation):
        if len(pattern1.parts) != len(pattern2.parts):
            return False
        return all(
            deep_compare(elem1, elem2)
            for elem1, elem2 in zip(pattern1.parts, pattern2.parts)
        )

    elif isinstance(pattern1, _CharGroup):
        return pattern1.chars == pattern2.chars and pattern1.negated == pattern2.negated

    elif isinstance(pattern1, _Repeated):
        return (
            deep_compare(pattern1.base, pattern2.base)
            and pattern1.min == pattern2.min
            and pattern1.max == pattern2.max
        )

    else:
        raise ValueError(f"Unhandled pattern type: {type(pattern1)}")


@pytest.mark.parametrize(
    "regex_string",
    [
        "ab",
        "a|b",
        "[ab]",
        "a*b+c?",
        "(ab|cd)*",
        "[a-z0-9]+",
        "foo(bar|baz)*qux",
        "(a|b|c){1,3}",
        "[^aeiou]{2,4}",
    ],
)
def test_parse_pattern(regex_string):
    ref_pattern = interegular.parse_pattern(regex_string)
    custom_pattern = parse_pattern(regex_string)
    converted_pattern = convert_to_interegular(custom_pattern)

    print(f"\nRegex: {regex_string}")
    print(f"Reference pattern: {ref_pattern}")
    print(f"Converted pattern: {converted_pattern}")

    are_equal = deep_compare(ref_pattern, converted_pattern)

    return are_equal


# TODO: remove if not needed
# tests copied so they can be run as a standalone script
if __name__ == "__main__":
    test_cases = [
        "ab",
        "a|b",
        "[ab]",
        "a*b+c?",
        "(ab|cd)*",
        "[a-z0-9]+",
        "foo(bar|baz)*qux",
        "(a|b|c){1,3}",
        "[^aeiou]{2,4}",
    ]

    all_passed = all(test_parse_pattern(case) for case in test_cases)
    print(f"All tests passed: {all_passed}")
