"""
Scaling benchmarks to verify performance scales appropriately.

These tests verify that EntityFrame scales well with entity count
and maintains expected performance characteristics at larger scales.
"""

import time
from entityframe import EntityFrame


class TestScalingBenchmarks:
    """Scaling benchmarks to verify performance scales appropriately."""

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

            # Measure performance
            start = time.time()
            hashes = frame.hash_collection_hex("scaling_test", "blake3")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            results.append({"count": count, "rate": rate, "time": elapsed})

            assert len(hashes) == count
            # Each scale should maintain reasonable performance
            assert rate > 1_000, f"Scale {count}: {rate:.0f} entities/sec too slow"

        # Verify scaling doesn't degrade catastrophically
        # Performance should not drop by more than 10x from smallest to largest
        smallest_rate = results[0]["rate"]
        largest_rate = results[-1]["rate"]
        degradation = smallest_rate / largest_rate

        assert (
            degradation < 10.0
        ), f"Performance degrades too much: {degradation:.1f}x from {scales[0]} to {scales[-1]} entities"

        # Log scaling results for visibility
        print("\nScaling benchmark results:")
        for result in results:
            print(
                f"  {result['count']:4d} entities: {result['rate']:8.0f} entities/sec ({result['time']:.3f}s)"
            )

    def test_large_scale_performance(self):
        """Test performance at larger scales to verify production readiness."""
        large_scales = [5000, 10000]

        for count in large_scales:
            entities = []
            for i in range(count):
                entity = {
                    "users": [f"user_{i}"],
                    "orders": [f"order_{i}_{j}" for j in range(3)],
                    "events": [f"event_{i}_{j}" for j in range(2)],
                }
                entities.append(entity)

            frame = EntityFrame()
            frame.add_method("large_test", entities)

            # Test with BLAKE3 (fastest algorithm)
            start = time.time()
            hashes = frame.hash_collection_hex("large_test", "blake3")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0

            # Should maintain good performance even at large scale
            assert (
                rate > 5_000
            ), f"Large scale {count}: {rate:.0f} entities/sec too slow"
            assert len(hashes) == count

            print(f"Large scale {count}: {rate:,.0f} entities/sec ({elapsed:.2f}s)")

            # Don't run tests that take too long
            if elapsed > 30:
                print(f"Stopping large scale tests (took {elapsed:.1f}s)")
                break

    def test_parallel_vs_sequential_threshold(self):
        """Test that parallel processing kicks in appropriately."""
        # Test around the 1000 entity threshold where parallelization begins
        test_counts = [800, 999, 1000, 1001, 1200]

        for count in test_counts:
            entities = [
                {"users": [f"user_{i}"], "orders": [f"order_{i}"]} for i in range(count)
            ]

            frame = EntityFrame()
            frame.add_method("threshold_test", entities)

            start = time.time()
            hashes = frame.hash_collection_hex("threshold_test", "blake3")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            mode = "SINGLE" if count < 1000 else "PARALLEL"

            # Both modes should perform well
            assert (
                rate > 5_000
            ), f"Threshold test {count} ({mode}): {rate:.0f} entities/sec too slow"
            assert len(hashes) == count

            print(f"  {count:4d} entities ({mode:8s}): {rate:8.0f} entities/sec")

    def test_memory_scaling_efficiency(self):
        """Test that memory usage scales efficiently with entity count."""
        scales = [500, 1000, 2000]

        for count in scales:
            # Create entities with controlled string overlap
            entities = []
            for i in range(count):
                # Create some overlap to test string interning efficiency
                base_user = i % 100  # Cycle through 100 user patterns
                entity = {
                    "users": [f"user_{base_user}"],
                    "orders": [f"order_{i}"],  # Unique orders
                }
                entities.append(entity)

            frame = EntityFrame()
            frame.add_method("memory_scaling_test", entities)

            # Measure memory efficiency through string interning
            interner_size = frame.interner_size()

            # Should achieve significant compression due to user overlap
            # Theoretical: count orders + ~100 users + 2 datasets
            expected_strings = count + 100 + 2
            actual_compression = (
                expected_strings / interner_size if interner_size > 0 else 1
            )

            # Verify reasonable memory efficiency
            assert (
                actual_compression > 0.8
            ), f"Memory scaling inefficient at {count}: {actual_compression:.2f}"

            # Performance should remain good despite memory sharing
            start = time.time()
            hashes = frame.hash_collection_hex("memory_scaling_test", "blake3")
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0

            assert (
                rate > 5_000
            ), f"Memory scaling test {count}: {rate:.0f} entities/sec too slow"
            assert len(hashes) == count

            print(
                f"  {count:4d} entities: {interner_size:4d} strings ({actual_compression:.2f} efficiency), {rate:8.0f} entities/sec"
            )

    def test_realistic_workload_scaling(self):
        """Test with realistic entity resolution workloads."""
        count = 3000

        # Create realistic mixed workload
        entities = []
        for i in range(count):
            # 70% small entities, 20% medium, 10% large
            if i < count * 0.7:
                # Small entities (typical case)
                entity = {
                    "customers": [f"cust_{i}"],
                    "emails": [f"user{i}@example.com"],
                }
            elif i < count * 0.9:
                # Medium entities
                entity = {
                    "customers": [f"cust_{i}", f"customer_{i}"],
                    "emails": [f"user{i}@example.com", f"alt{i}@example.com"],
                    "phones": [f"phone_{i}"],
                }
            else:
                # Large entities (outliers)
                entity = {
                    "customers": [f"cust_{i}_{j}" for j in range(5)],
                    "emails": [f"email_{i}_{j}@example.com" for j in range(3)],
                    "phones": [f"phone_{i}_{j}" for j in range(2)],
                }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("realistic_workload", entities)

        # Test performance with mixed entity sizes
        start = time.time()
        hashes = frame.hash_collection_hex("realistic_workload", "blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Should handle realistic mixed workloads efficiently
        assert rate > 5_000, f"Realistic workload: {rate:.0f} entities/sec too slow"
        assert len(hashes) == count

        print(f"Realistic workload ({count} entities): {rate:,.0f} entities/sec")
        print(f"  Memory efficiency: {frame.interner_size()} unique strings")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
