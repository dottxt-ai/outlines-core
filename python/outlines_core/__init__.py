"""Outlines is a Generative Model Programming Framework."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("outlines_core")
except PackageNotFoundError:
    pass


__all__ = ["models"]
