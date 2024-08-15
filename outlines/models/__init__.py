"""Module that contains all the models integrated in outlines.

We group the models in submodules by provider instead of theme (completion, chat
completion, diffusers, etc.) and use routing functions everywhere else in the
codebase.

"""

from typing import Union

from .transformers import Transformers, TransformerTokenizer, mamba, transformers

LogitsGenerator = Union[Transformers]
