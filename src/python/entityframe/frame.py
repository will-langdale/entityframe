"""
Python wrapper for EntityFrame to provide convenient collection attribute access.
"""

from __future__ import annotations
from typing import Any, Dict, List
from .collection import CollectionWrapper
from .entityframe import EntityFrame as RustEntityFrame


class EntityFrame:
    """Enhanced EntityFrame wrapper with collection attribute access."""

    def __init__(self) -> None:
        """Initialize EntityFrame."""
        self._frame = RustEntityFrame()
        self._collection_wrappers: Dict[str, CollectionWrapper] = {}

    def declare_dataset(self, name: str) -> None:
        """
        Declare a dataset upfront for efficiency.

        Args:
            name: Dataset name
        """
        # This would need to be implemented in Rust if not already available
        pass  # For now, datasets are auto-declared

    def add_method(self, name: str, entities: List[Dict[str, Any]]) -> None:
        """
        Add a collection of entities from one method.

        Args:
            name: Method/collection name
            entities: List of entity dictionaries (may include 'metadata' key)
        """
        # Convert entities to format expected by Rust
        # The Rust function now expects PyObject (dict) format to handle metadata
        self._frame.add_method(name, entities)
        # Clear cached wrapper so it gets recreated with new data
        if name in self._collection_wrappers:
            del self._collection_wrappers[name]

    def __getattr__(self, name: str) -> Any:
        """
        Get collection as an attribute (e.g., frame.conservative).

        Args:
            name: Collection name

        Returns:
            CollectionWrapper for the named collection

        Raises:
            AttributeError: If collection doesn't exist
        """
        # First, try to get attribute from the underlying frame
        if hasattr(self._frame, name):
            return getattr(self._frame, name)

        # Check if this is a collection name
        try:
            if name not in self._collection_wrappers:
                # Verify collection exists
                _ = self._frame.get_collection(name)
                # Cache the wrapper
                self._collection_wrappers[name] = CollectionWrapper(self._frame, name)
            return self._collection_wrappers[name]
        except Exception:
            # If collection doesn't exist, raise AttributeError for proper Python behavior
            raise AttributeError(f"'EntityFrame' object has no attribute '{name}'")

    def compare_collections(self, name1: str, name2: str) -> List[Dict[str, Any]]:
        """
        Compare two collections and return similarity metrics.

        Args:
            name1: First collection name
            name2: Second collection name

        Returns:
            List of comparison results
        """
        return list(self._frame.compare_collections(name1, name2))

    def get_collection_names(self) -> List[str]:
        """Get names of all collections."""
        return list(self._frame.get_collection_names())

    def total_entities(self) -> int:
        """Get total number of entities across all collections."""
        return int(self._frame.total_entities())

    def get_dataset_names(self) -> List[str]:
        """Get names of all datasets."""
        return list(self._frame.get_dataset_names())

    @property
    def interner(self) -> Any:
        """Access to the internal string interner."""
        return self._frame.interner
