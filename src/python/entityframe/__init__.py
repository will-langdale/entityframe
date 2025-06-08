"""
EntityFrame - A Python package for comparing entity resolutions from different processes.

This package provides high-performance entity resolution evaluation using a three-layer
architecture: string interning, roaring bitmaps, and entity hashing.
"""

from ._rust import hello_rust, StringInterner, Entity
from typing import List, Dict, Any, Optional


class EntityCollection:
    """High-level API for managing and comparing entity resolution methods."""

    def __init__(self):
        self.interner = StringInterner()
        self.methods: Dict[str, List[Entity]] = {}

    def add_method(
        self, method_name: str, entities_data: List[Dict[str, List[str]]]
    ) -> None:
        """Add entities from a method's output.

        Args:
            method_name: Name of the entity resolution method
            entities_data: List of entity dictionaries, where each dict maps
                         dataset names to lists of record ID strings
        """
        entities = []
        for entity_data in entities_data:
            entity = Entity()
            for dataset, record_ids in entity_data.items():
                # Intern all record IDs and add to entity
                interned_ids = [self.interner.intern(rid) for rid in record_ids]
                entity.add_records(dataset, interned_ids)
            entities.append(entity)

        self.methods[method_name] = entities

    def get_method_names(self) -> List[str]:
        """Get names of all registered methods."""
        return list(self.methods.keys())

    def get_entities(self, method_name: str) -> Optional[List[Entity]]:
        """Get entities for a specific method."""
        return self.methods.get(method_name)

    def compare_methods(self, method1: str, method2: str) -> List[Dict[str, Any]]:
        """Compare two methods and return similarity metrics.

        Returns:
            List of comparison results, one per entity pair
        """
        entities1 = self.methods.get(method1, [])
        entities2 = self.methods.get(method2, [])

        results = []

        # Simple comparison: assume same ordering for now
        # In real implementation, would need entity matching logic
        min_len = min(len(entities1), len(entities2))

        for i in range(min_len):
            jaccard = entities1[i].jaccard_similarity(entities2[i])
            results.append(
                {
                    "entity_index": i,
                    "method1": method1,
                    "method2": method2,
                    "jaccard": jaccard,
                    "method1_records": entities1[i].total_records(),
                    "method2_records": entities2[i].total_records(),
                }
            )

        return results


def hello_python() -> str:
    """Return a hello message from Python."""
    return "Hello from Python!"


def hello_world() -> str:
    """Return combined hello messages from both Python and Rust."""
    return f"{hello_python()} {hello_rust()}"


__all__ = [
    "hello_python",
    "hello_rust",
    "hello_world",
    "StringInterner",
    "Entity",
    "EntityCollection",
]
