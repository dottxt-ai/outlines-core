"""This package provides core functionality for structured generation, formerly implemented in Outlines."""

from importlib.metadata import PackageNotFoundError, version

from .outlines_core_rs import Guide as Guide
from .outlines_core_rs import Index as Index
from .outlines_core_rs import Vocabulary as Vocabulary

try:
    __version__ = version("outlines_core")
except PackageNotFoundError:
    pass
