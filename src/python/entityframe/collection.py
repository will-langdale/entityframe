"""
Python wrapper for EntityCollection to provide convenient hash and metadata methods.
"""

from typing import Any, Dict, Iterator


class CollectionWrapper:
    """Wrapper for EntityCollection with convenient hashing and access methods."""

    def __init__(self, frame: Any, collection_name: str) -> None:
        self.frame = frame
        self.collection_name = collection_name
        self._collection = frame.get_collection(collection_name)

    def add_hash(self, algorithm: str = "blake3") -> None:
        """
        Add hashes to all entities in this collection using optimized batch processing.

        Args:
            algorithm: Hash algorithm to use. Options: sha256, sha512, sha3-256, sha3-512, blake3
        """
        # Call the frame method to modify the collection in-place
        self.frame.add_collection_hash(self.collection_name, algorithm)

    def verify_hashes(self, algorithm: str = "blake3") -> bool:
        """
        Verify hashes for all entities in this collection.

        Args:
            algorithm: Hash algorithm to use for verification

        Returns:
            True if all hashes verified successfully, False otherwise
        """
        # Call the frame method to verify hashes
        return bool(
            self.frame.verify_collection_hashes(self.collection_name, algorithm)
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get entity at index with metadata included.

        Args:
            index: Entity index

        Returns:
            Dictionary with entity data and metadata
        """
        # Get the current collection from the frame (not the cached copy)
        current_collection = self.frame.get_collection(self.collection_name)
        if current_collection is None:
            raise KeyError(f"Collection '{self.collection_name}' not found")

        # Get the raw entity
        entity = current_collection.get_entity(index)

        # Get datasets for this entity using the frame's dataset registry
        datasets = {}
        dataset_names = self.frame.get_dataset_names()

        for dataset_name in dataset_names:
            try:
                records = entity.get_records(dataset_name)
                if records:  # Only include datasets that have records
                    # Convert from record IDs to strings - this is a simplified approach
                    # In a full implementation, we'd need access to the interner
                    datasets[dataset_name] = [f"record_{r}" for r in records]
            except Exception:
                continue  # Dataset doesn't exist for this entity

        # Get metadata for this entity
        metadata = {}

        # Get all metadata keys and retrieve their values
        metadata_keys = entity.get_metadata_keys()
        for key_id in metadata_keys:
            # Convert key ID back to string using interner
            try:
                key_name = self.frame.interner.get_string(key_id)
                # Use the frame API to get metadata which returns PyObject
                value = self.frame.get_entity_metadata(
                    self.collection_name, index, key_name
                )
                if value is not None:
                    metadata[key_name] = value
            except Exception:
                # Skip if we can't resolve the key name
                continue

        result: Dict[str, Any] = {"metadata": metadata}
        result.update(datasets)

        return result

    def __len__(self) -> int:
        """Get number of entities in this collection."""
        current_collection = self.frame.get_collection(self.collection_name)
        if current_collection is None:
            return 0
        return int(current_collection.len())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over entities in this collection."""
        for i in range(len(self)):
            yield self[i]
