"""
EntityFrame - A Python package for comparing entity resolutions from different processes.

This package provides high-performance entity resolution evaluation using a three-layer
architecture: string interning, roaring bitmaps, and entity hashing.
"""

from .entityframe import StringInternerCore, EntityCore, CollectionCore
from .frame import EntityFrame
from .entity import Entity as EntityWrapper
from .collection import Collection

# Export clean names for users
StringInterner = StringInternerCore
Entity = EntityCore  # For backward compatibility with tests
EntityCollection = CollectionCore

__all__ = [
    "StringInterner",
    "Entity",
    "Collection",
    "EntityFrame",
    "EntityCollection",
    "EntityWrapper",  # For tests that need the wrapper
]
