"""
Python wrapper for EntityCore to provide convenient access to metadata and hashing.
"""

from typing import Any, Optional


class Entity:
    """Wrapper for EntityCore with convenient metadata and hashing methods."""

    def __init__(self, frame: Any, collection_name: str, entity_index: int) -> None:
        self.frame = frame
        self.collection_name = collection_name
        self.entity_index = entity_index

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata on this entity."""
        self.frame.set_entity_metadata(
            self.collection_name, self.entity_index, key, value
        )

    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata from this entity."""
        return self.frame.get_entity_metadata(
            self.collection_name, self.entity_index, key
        )

    def hash(self, algorithm: str = "sha256") -> bytes:
        """
        Compute deterministic hash of this entity.

        Args:
            algorithm: Hash algorithm to use. Options: sha256, sha512, sha3-256, sha3-512, blake3

        Returns:
            Hash as bytes
        """
        result = self.frame.hash_entity(
            self.collection_name, self.entity_index, algorithm
        )
        return bytes(result)

    def hexdigest(self, algorithm: str = "sha256") -> str:
        """
        Compute deterministic hash and return as hex string.

        Args:
            algorithm: Hash algorithm to use. Options: sha256, sha512, sha3-256, sha3-512, blake3

        Returns:
            Hash as hex string
        """
        return self.hash(algorithm).hex()
