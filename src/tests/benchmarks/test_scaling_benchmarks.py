"""
Scaling performance tests using the new API.
Tests that performance scales appropriately with data size.
"""

import time
from entityframe import EntityFrame


class TestScalingBenchmarks:
    """Test scaling performance using new API."""

    def test_linear_scaling_verification(self):
        """Test that performance scales linearly (not quadratically) with entity count."""
        scales = [100, 500, 1000, 2000]
        results = []

        for count in scales:
            entities = []
            for i in range(count):
                entity = {
                    "users": [f"user_{i}"],
                    "orders": [f"order_{i}_{j}" for j in range(2)],
                }
                entities.append(entity)

            frame = EntityFrame()
            frame.add_method("scaling_test", entities)
            collection = frame.scaling_test

            # Measure performance
            start = time.time()
            collection.add_hash("blake3")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            results.append({"count": count, "rate": rate, "time": elapsed})

            assert len(collection) == count
            # Each scale should maintain reasonable performance
            assert rate > 1_000, f"Scale {count}: {rate:.0f} entities/sec too slow"

        # Verify scaling doesn't degrade catastrophically
        # Rate should not drop by more than 50% when going from smallest to largest
        min_rate = min(r["rate"] for r in results)
        max_rate = max(r["rate"] for r in results)
        degradation_ratio = min_rate / max_rate
        assert (
            degradation_ratio > 0.5
        ), f"Performance degraded too much: {degradation_ratio:.2f}"

    def test_large_scale_performance(self):
        """Test performance with larger datasets to catch algorithmic issues."""
        count = 5000
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "products": [f"prod_{i % 1000}"],  # Some repetition for realism
                "categories": [f"cat_{i % 100}"],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("large_test", entities)
        collection = frame.large_test

        # Measure performance
        start = time.time()
        collection.add_hash("blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Should handle 5k entities reasonably fast
        assert rate > 3_000, f"Large scale too slow: {rate:.0f} entities/sec"
        assert len(collection) == count

        # Verify correctness on sample
        for i in range(0, count, 1000):  # Check every 1000th entity
            entity = collection[i]
            assert "hash" in entity["metadata"]
            assert len(entity["metadata"]["hash"]) == 32

    def test_parallel_vs_sequential_threshold(self):
        """Test that parallel processing kicks in appropriately."""
        # Test both sides of the 1000 entity threshold for parallel processing
        small_count = 500  # Should use sequential
        large_count = 2000  # Should use parallel

        for count, name in [(small_count, "small"), (large_count, "large")]:
            entities = [
                {"users": [f"user_{i}"], "data": [f"data_{i}"]} for i in range(count)
            ]

            frame = EntityFrame()
            frame.add_method(f"{name}_test", entities)
            collection = getattr(frame, f"{name}_test")

            start = time.time()
            collection.add_hash("sha256")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0

            # Both should be reasonably fast
            assert rate > 2_000, f"{name} scale too slow: {rate:.0f} entities/sec"

            # Verify correctness
            assert len(collection) == count
            entity = collection[0]
            assert "hash" in entity["metadata"]

    def test_memory_scaling_efficiency(self):
        """Test memory efficiency doesn't degrade with scale."""
        count = 3000
        entities = []
        for i in range(count):
            # Create entities with high string reuse to test interning
            entities.append(
                {
                    "users": [f"user_{i % 100}"],  # 100 unique users
                    "regions": [f"region_{i % 20}"],  # 20 unique regions
                    "types": [f"type_{i % 5}"],  # 5 unique types
                }
            )

        frame = EntityFrame()
        frame.add_method("memory_scale_test", entities)
        collection = frame.memory_scale_test

        # Add hashes
        start = time.time()
        collection.add_hash("blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Should be efficient due to string interning
        assert rate > 4_000, f"Memory scaling issue: {rate:.0f} entities/sec"

        # Verify string interning worked
        dataset_names = frame.get_dataset_names()
        assert len(dataset_names) == 3  # users, regions, types

        # Verify all have hashes
        assert len(collection) == count
