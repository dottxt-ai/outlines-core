"""This package provides core functionality for structured generation, formerly implemented in Outlines."""
from importlib.metadata import PackageNotFoundError, version

from .outlines_core_rs import Guide, Index, Vocabulary

try:
    __version__ = version("outlines_core")
except PackageNotFoundError:
    pass
