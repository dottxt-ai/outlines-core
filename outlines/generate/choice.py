from functools import singledispatch
from typing import Callable, List

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.samplers import Sampler, multinomial

from .regex import regex


@singledispatch
def choice(
    model, choices: List[str], sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    regex_str = r"(" + r"|".join(choices) + r")"

    generator = regex(model, regex_str, sampler)
    generator.format_sequence = lambda x: x

    return generator
