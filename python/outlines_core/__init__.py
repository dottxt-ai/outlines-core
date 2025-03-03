"""This package provides core functionality for structured generation, formerly implemented in Outlines."""
from importlib.metadata import PackageNotFoundError, version

from .outlines_core_rs import Guide, Index, Vocabulary
from .utils import create_mask, first_token_id_from_mask, mask_to_list

try:
    __version__ = version("outlines_core")
except PackageNotFoundError:
    pass
