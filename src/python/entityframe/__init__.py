"""
EntityFrame - A Python package for comparing entity resolutions from different processes.

This package provides high-performance entity resolution evaluation using a three-layer
architecture: string interning, roaring bitmaps, and entity hashing.
"""

from ._rust import StringInterner, Entity, EntityCollection, EntityFrame


__all__ = [
    "StringInterner",
    "Entity",
    "EntityFrame",
    "EntityCollection",
]
