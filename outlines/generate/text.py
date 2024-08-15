from functools import singledispatch

from outlines.fsm.guide import StopAtEOSGuide
from outlines.generate.api import (
    SequenceGenerator,
    SequenceGeneratorAdapter,
    VisionSequenceGeneratorAdapter,
)
from outlines.samplers import Sampler, multinomial


@singledispatch
def text(model, sampler: Sampler = multinomial()) -> SequenceGeneratorAdapter:
    """Generate text with a `Transformer` model.

    Note
    ----
    Python 3.11 allows dispatching on Union types and
    this should greatly simplify the code.

    Arguments
    ---------
    model:
        An instance of `Transformer` that represents a model from the
        `transformers` library.
    sampler:
        The sampling algorithm to use to generate token ids from the logits
        distribution.

    Returns
    -------
    A `SequenceGeneratorAdapter` instance that generates text.

    """
    return SequenceGeneratorAdapter(model, None, sampler)
