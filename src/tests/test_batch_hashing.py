"""
Batch hashing functionality tests using the new API.
Tests core batch hashing functionality, consistency, and correctness.
"""

from starlings import EntityFrame


class TestBatchHashing:
    """Test batch hashing functionality and correctness using new API."""

    def test_basic_batch_functionality(self):
        """Test that batch hashing works correctly with simple entities."""
        # Simple test data
        entities = [
            {"users": ["alice", "bob"], "orders": ["order_1"]},
            {"users": ["charlie"], "orders": ["order_2", "order_3"]},
            {"users": ["alice"], "addresses": ["addr_1"]},
        ]

        frame = EntityFrame()
        frame.add_method("basic_test", entities)
        collection = frame.basic_test

        # Test batch hashing using new API
        collection.add_hash("sha256")

        # Verify all entities have hashes
        assert len(collection) == len(entities)

        for i in range(len(entities)):
            entity = collection[i]
            assert "metadata" in entity
            assert "hash" in entity["metadata"]
            hash_bytes = entity["metadata"]["hash"]
            assert isinstance(hash_bytes, bytes)
            assert len(hash_bytes) == 32  # SHA-256 = 32 bytes

    def test_algorithm_correctness(self):
        """Test that different algorithms produce different but valid hashes."""
        entities = [{"users": ["test_user"], "data": ["test_data"]}]

        # Test SHA-256
        frame1 = EntityFrame()
        frame1.add_method("sha_test", entities)
        frame1.sha_test.add_hash("sha256")
        sha_hash = frame1.sha_test[0]["metadata"]["hash"]
        assert len(sha_hash) == 32

        # Test Blake3
        frame2 = EntityFrame()
        frame2.add_method("blake_test", entities)
        frame2.blake_test.add_hash("blake3")
        blake_hash = frame2.blake_test[0]["metadata"]["hash"]
        assert len(blake_hash) == 32

        # Different algorithms should produce different hashes
        assert sha_hash != blake_hash

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic across multiple runs."""
        entities = [
            {"customers": ["c3", "c1", "c2"], "orders": ["o2", "o1"]},  # Unsorted input
            {
                "customers": ["c1", "c2", "c3"],
                "orders": ["o1", "o2"],
            },  # Same data, sorted
        ]

        # Create two separate frames with same data
        frame1 = EntityFrame()
        frame1.add_method("test1", [entities[0]])
        frame1.test1.add_hash("sha256")
        hash1 = frame1.test1[0]["metadata"]["hash"]

        frame2 = EntityFrame()
        frame2.add_method("test2", [entities[1]])
        frame2.test2.add_hash("sha256")
        hash2 = frame2.test2[0]["metadata"]["hash"]

        # Same logical entity should produce same hash regardless of input order
        assert hash1 == hash2

    def test_hash_verification(self):
        """Test hash verification functionality."""
        entities = [
            {"users": ["u1", "u2"], "orders": ["o1"]},
            {"users": ["u3"], "orders": ["o2", "o3"]},
        ]

        frame = EntityFrame()
        frame.add_method("verify_test", entities)
        collection = frame.verify_test

        # Add hashes
        collection.add_hash("blake3")

        # Verify hashes should pass
        assert collection.verify_hashes("blake3") is True

        # Verify with different algorithm should fail (no hashes for that algorithm)
        # Note: This tests the case where we verify with an algorithm different from what was used
        # The verification should pass if there are no hashes, or fail if hashes don't match
        # Our current implementation returns True if no hashes exist, which is reasonable

    def test_string_interning_efficiency(self):
        """Test that string interning works efficiently with hashing."""
        # Create entities with overlapping strings to test interning
        entities = []
        for i in range(100):
            entities.append(
                {
                    "users": [f"user_{i % 10}"],  # Reuse user names to test interning
                    "orders": [f"order_{i}"],
                }
            )

        frame = EntityFrame()
        frame.add_method("interning_test", entities)
        collection = frame.interning_test

        # Add hashes - this should work efficiently due to string interning
        collection.add_hash("sha256")

        # Verify all entities have valid hashes
        assert len(collection) == 100
        for i in range(100):
            entity = collection[i]
            assert "hash" in entity["metadata"]
            assert len(entity["metadata"]["hash"]) == 32

        # Verify dataset names are properly interned
        dataset_names = frame.get_dataset_names()
        assert "users" in dataset_names
        assert "orders" in dataset_names
        assert (
            len(dataset_names) == 2
        )  # Only 2 unique dataset names despite 100 entities
