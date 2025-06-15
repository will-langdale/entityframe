"""
Performance regression tests to ensure optimizations don't break.

These tests verify that EntityFrame maintains expected performance levels
and that different algorithms and output formats work correctly.
"""

import time
from entityframe import EntityFrame


class TestPerformanceRegression:
    """Performance regression tests to catch performance degradation."""

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

        # Test performance (should be >10k entities/sec after optimization)
        start = time.time()
        hashes = frame.hash_collection_hex("regression_test", "blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Regression test: Should be much faster than original 3k entities/sec
        assert (
            rate > 10_000
        ), f"Performance regression detected: {rate:.0f} entities/sec < 10k threshold"
        assert len(hashes) == count

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

            start = time.time()
            hashes = frame.hash_collection_hex("algo_test", algorithm)
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            results[algorithm] = {"rate": rate, "hashes": hashes}

            # Each algorithm should be reasonably fast
            assert rate > 1_000, f"{algorithm} too slow: {rate:.0f} entities/sec"
            assert len(hashes) == count

        # Verify algorithms produce different results
        blake3_hashes = results["blake3"]["hashes"]
        sha256_hashes = results["sha256"]["hashes"]
        assert (
            blake3_hashes != sha256_hashes
        ), "Different algorithms should produce different hashes"

        # BLAKE3 should generally be fastest (though not required)
        # This is informational, not a hard requirement
        fastest_algo = max(results.keys(), key=lambda k: results[k]["rate"])
        print(
            f"Fastest algorithm: {fastest_algo} ({results[fastest_algo]['rate']:.0f} entities/sec)"
        )

    def test_output_format_consistency(self):
        """Test that different output formats work correctly and consistently."""
        count = 200
        entities = [
            {"users": [f"user_{i}"], "orders": [f"order_{i}"]} for i in range(count)
        ]

        frame = EntityFrame()
        frame.add_method("format_test", entities)

        # Test PyBytes format
        pybytes_hashes = frame.hash_collection("format_test", "sha256")

        # Test hex format
        hex_hashes = frame.hash_collection_hex("format_test", "sha256")

        # Test raw concatenated format
        raw_bytes = frame.hash_collection_raw("format_test", "sha256")

        # Verify consistency
        assert len(pybytes_hashes) == count
        assert len(hex_hashes) == count

        # Verify hex format correctness (SHA-256 = 32 bytes = 64 hex chars)
        for hex_hash in hex_hashes:
            assert (
                len(hex_hash) == 64
            ), f"SHA-256 hex should be 64 chars, got {len(hex_hash)}"
            assert all(
                c in "0123456789abcdef" for c in hex_hash
            ), "Invalid hex characters"

        # Verify hex and bytes consistency
        for pybytes_hash, hex_hash in zip(pybytes_hashes, hex_hashes):
            assert bytes(pybytes_hash).hex() == hex_hash, "Hex/bytes mismatch"

        # Verify raw bytes format (basic check)
        raw_data = bytes(raw_bytes)
        assert (
            len(raw_data) > count * 30
        ), "Raw bytes should contain at least hash size * count"

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

        frame2 = EntityFrame()
        frame2.add_method("test2", [entities[1]])

        # Hash the same logical entity (different order) multiple times
        hash1_run1 = frame1.hash_collection_hex("test1", "sha256")
        hash1_run2 = frame1.hash_collection_hex("test1", "sha256")
        hash2_run1 = frame2.hash_collection_hex("test2", "sha256")

        # All should be identical (deterministic and order-independent)
        assert hash1_run1 == hash1_run2, "Hashing should be deterministic"
        assert hash1_run1 == hash2_run1, "Hashing should be order-independent"

    def test_memory_efficiency_basic(self):
        """Test basic memory efficiency with string interning."""
        count = 1000
        overlap_ratio = 0.7  # 70% shared strings

        entities = []
        for i in range(count):
            if i < count * overlap_ratio:
                # Shared strings (cycle through patterns for overlap)
                base = i % 50
                entity = {
                    "users": [f"shared_user_{base}"],
                    "data": [f"shared_data_{base}"],
                }
            else:
                # Unique strings
                entity = {"users": [f"unique_user_{i}"], "data": [f"unique_data_{i}"]}
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("memory_test", entities)

        # Test that string interning provides compression
        interner_size = frame.interner_size()
        theoretical_strings = count * 2 + 2  # 2 strings per entity + 2 datasets
        compression_ratio = (
            theoretical_strings / interner_size if interner_size > 0 else 1
        )

        # Should achieve some compression due to string overlap
        assert (
            compression_ratio > 1.2
        ), f"Insufficient string compression: {compression_ratio:.1f}x"

        # Performance should still be good
        start = time.time()
        hashes = frame.hash_collection_hex("memory_test", "blake3")
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0

        assert (
            rate > 5_000
        ), f"Performance with string overlap too slow: {rate:.0f} entities/sec"
        assert len(hashes) == count


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
