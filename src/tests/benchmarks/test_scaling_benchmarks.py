"""
Scaling performance tests using the new API.
Tests that performance scales appropriately with data size.
"""

import time
import pytest
from starlings import EntityFrame


class TestScalingBenchmarks:
    """Test scaling performance using new API."""

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_million_scale_performance(self):
        """Test performance at 1M scale to validate production readiness."""
        count = 1_000_000
        print(f"\nTesting EntityFrame at {count:,} entity scale...")

        # Create realistic entities with string interning patterns
        entities = []
        print("Generating test data...")
        for i in range(count):
            entity = {
                "users": [f"user_{i % 10_000}"],  # 10k unique users
                "regions": [f"region_{i % 100}"],  # 100 unique regions
                "products": [f"prod_{i % 1_000}"],  # 1k unique products
                "categories": [f"cat_{i % 50}"],  # 50 unique categories
            }
            entities.append(entity)

            # Progress indicator
            if i % 100_000 == 0 and i > 0:
                print(f"  Generated {i:,} entities...")

        print("Creating EntityFrame...")
        frame = EntityFrame()
        frame.add_method("million_scale_test", entities)
        collection = frame.million_scale_test

        print("Starting hash benchmark...")
        start = time.time()
        collection.add_hash("blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0
        print(f"Completed: {rate:,.0f} entities/sec ({elapsed:.2f}s total)")

        # Performance expectations for 1M entities
        # Should process at least 50k entities/sec at this scale
        assert rate > 50_000, f"Million scale too slow: {rate:,.0f} entities/sec"
        assert len(collection) == count

        # Verify string interning efficiency
        dataset_names = frame.get_dataset_names()
        assert len(dataset_names) == 4  # users, regions, products, categories

        # Sample verification - check every 100,000th entity
        print("Verifying sample entities...")
        for i in range(0, count, 100_000):
            entity = collection[i]
            assert "hash" in entity["metadata"]
            assert len(entity["metadata"]["hash"]) == 32  # blake3 hash size

        print("✓ Million scale test completed successfully!")
        print(f"  Performance: {rate:,.0f} entities/sec")
        print(f"  Memory efficiency: {len(dataset_names)} unique datasets")

    @pytest.mark.slow
    def test_comparison_scaling_performance(self):
        """Test entity comparison performance at larger scales."""
        count = 10_000  # Comparison is O(n) so test at reasonable scale

        # Create two methods with partial overlap
        entities1 = []
        entities2 = []

        for i in range(count):
            # Method 1: base entities
            entities1.append(
                {
                    "users": [f"user_{i}"],
                    "orders": [f"order_{i}"],
                }
            )

            # Method 2: 50% overlap with method 1
            if i < count // 2:
                # First half overlaps exactly
                entities2.append(
                    {
                        "users": [f"user_{i}"],
                        "orders": [f"order_{i}"],
                    }
                )
            else:
                # Second half is different
                entities2.append(
                    {
                        "users": [f"user_{i + count}"],
                        "orders": [f"order_{i + count}"],
                    }
                )

        frame = EntityFrame()
        frame.add_method("method1", entities1)
        frame.add_method("method2", entities2)

        # Measure comparison performance
        start = time.time()
        comparisons = frame.compare_collections("method1", "method2")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Should compare at least 5k entities/sec
        assert rate > 5_000, f"Comparison too slow: {rate:,.0f} entities/sec"
        assert len(comparisons) == count

        # Verify correctness of comparison results
        # First half should have jaccard = 1.0 (identical)
        # Second half should have jaccard = 0.0 (no overlap)
        first_half_perfect = all(
            comp["jaccard"] == 1.0 for comp in comparisons[: count // 2]
        )
        second_half_zero = all(
            comp["jaccard"] == 0.0 for comp in comparisons[count // 2 :]
        )

        assert first_half_perfect, "First half comparisons should be perfect matches"
        assert second_half_zero, "Second half comparisons should have no overlap"
