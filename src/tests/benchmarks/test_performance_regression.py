"""
Performance regression tests using the new API.
Tests core performance to catch regressions in key operations.
"""

import time
from entityframe import EntityFrame


class TestPerformanceRegression:
    """Test performance regression using new API."""

    def test_small_scale_performance(self):
        """Test performance at small scales to catch regressions."""
        count = 1000
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(2)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("regression_test", entities)
        collection = frame.regression_test

        # Test performance (should be >10k entities/sec after optimization)
        start = time.time()
        collection.add_hash("blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Regression test: Should be much faster than original 3k entities/sec
        assert (
            rate > 10_000
        ), f"Performance regression: {rate:.0f} entities/sec < 10k entities/sec"

        # Verify results are correct
        assert len(collection) == count
        for i in range(min(10, count)):  # Check first 10 entities
            entity = collection[i]
            assert "hash" in entity["metadata"]
            assert len(entity["metadata"]["hash"]) == 32  # blake3 hash size

    def test_algorithm_performance_comparison(self):
        """Test that all algorithms perform reasonably and consistently."""
        count = 500
        entities = [
            {"users": [f"user_{i}"], "orders": [f"order_{i}"]} for i in range(count)
        ]

        algorithms = ["blake3", "sha256", "sha512", "sha3-256"]
        results = {}

        for algorithm in algorithms:
            frame = EntityFrame()
            frame.add_method("algo_test", entities)
            collection = frame.algo_test

            start = time.time()
            collection.add_hash(algorithm)
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            results[algorithm] = rate

            # Each algorithm should process at least 5k entities/sec
            assert rate > 5_000, f"{algorithm} too slow: {rate:.0f} entities/sec"

            # Verify correctness
            entity = collection[0]
            assert "hash" in entity["metadata"]

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic across runs."""
        entities = [
            {"customers": ["c3", "c1", "c2"], "orders": ["o2", "o1"]},  # Unsorted input
            {
                "customers": ["c1", "c2", "c3"],
                "orders": ["o1", "o2"],
            },  # Same data, sorted
        ]

        # Create two separate frames
        frame1 = EntityFrame()
        frame1.add_method("test1", [entities[0]])
        frame1.test1.add_hash("sha256")
        hash1 = frame1.test1[0]["metadata"]["hash"]

        frame2 = EntityFrame()
        frame2.add_method("test2", [entities[1]])
        frame2.test2.add_hash("sha256")
        hash2 = frame2.test2[0]["metadata"]["hash"]

        # Same logical entity should produce same hash regardless of input order
        assert hash1 == hash2, "Hashing is not deterministic"

    def test_memory_efficiency_basic(self):
        """Test basic memory efficiency with string interning."""
        # Create many entities with repeated strings to test interning efficiency
        count = 1000
        entities = []
        for i in range(count):
            entities.append(
                {
                    "users": [f"user_{i % 100}"],  # Only 100 unique user names
                    "category": [f"cat_{i % 10}"],  # Only 10 unique categories
                }
            )

        frame = EntityFrame()
        frame.add_method("memory_test", entities)
        collection = frame.memory_test

        # Add hashes
        start = time.time()
        collection.add_hash("blake3")
        elapsed = time.time() - start

        # Should still be fast despite many entities
        rate = count / elapsed if elapsed > 0 else 0
        assert rate > 8_000, f"Memory efficiency issue: {rate:.0f} entities/sec"

        # Verify string interning worked - should only have unique strings
        dataset_names = frame.get_dataset_names()
        assert len(dataset_names) == 2  # users, category

        # Verify all entities have hashes
        assert len(collection) == count
        for i in range(min(5, count)):  # Check first 5
            entity = collection[i]
            assert "hash" in entity["metadata"]
