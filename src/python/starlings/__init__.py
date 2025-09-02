"""Starlings: high-performance entity resolution evaluation for Python."""

from importlib.metadata import version  # noqa: PLC0415

from .starlings import PyCollection as Collection  # noqa: F401
from .starlings import PyPartition as Partition  # noqa: F401

__version__ = version("starlings")

__all__ = ["Collection", "Partition"]
