"""
starlings - A Python package for comparing entity resolutions from different processes.

This package provides high-performance entity resolution evaluation using a three-layer
architecture: string interning, roaring bitmaps, and entity hashing.
"""

from .starlings import StringInternerCore, EntityCore, CollectionCore
from .frame import EntityFrame
from .entity import Entity
from .collection import Collection

# Clean user-facing names
StringInterner = StringInternerCore

__all__ = [
    "StringInterner",
    "Entity",
    "Collection",
    "EntityFrame",
    # Core objects available for testing
    "StringInternerCore",
    "EntityCore",
    "CollectionCore",
]
