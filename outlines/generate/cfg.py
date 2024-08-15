from functools import singledispatch

from outlines.generate.api import SequenceGeneratorAdapter
from outlines.samplers import Sampler, multinomial


@singledispatch
def cfg(
    model, cfg_str: str, sampler: Sampler = multinomial()
) -> SequenceGeneratorAdapter:
    """Generate text in the language of a Context-Free Grammar

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
    raise NotImplementedError(
        f"The CFG Logits processor is not available for {type(model)}. "
        + "Please subscribe to https://github.com/outlines-dev/outlines/issues/684"
        + " for updates on the fix."
    )
