"""
EntityFrame - A Python package for comparing entity resolutions from different processes.

This package provides high-performance entity resolution evaluation using a three-layer
architecture: string interning, roaring bitmaps, and entity hashing.
"""

from .entityframe import StringInterner, Entity, EntityCollection
from .frame import EntityFrame
from .entity import EntityWrapper
from .collection import CollectionWrapper


__all__ = [
    "StringInterner",
    "Entity",
    "EntityFrame",
    "EntityCollection",
    "EntityWrapper",
    "CollectionWrapper",
]
