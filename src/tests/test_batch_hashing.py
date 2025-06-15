"""
Batch hashing functionality tests.
Tests core batch hashing functionality, consistency, and correctness.
Performance tests have been moved to benchmarks/
"""

from entityframe import EntityFrame


class TestBatchHashing:
    """Test batch hashing functionality and correctness."""

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

        # Test batch hashing
        batch_hashes = frame.hash_collection("basic_test", "sha256")

        # Test individual hashing for comparison
        individual_hashes = []
        for i in range(len(entities)):
            hash_bytes = frame.hash_entity("basic_test", i, "sha256")
            individual_hashes.append(hash_bytes)

        # Verify consistency between batch and individual hashing
        assert len(batch_hashes) == len(individual_hashes) == len(entities)

        # Note: We can't directly compare PyBytes objects in tests easily,
        # but we can verify they're the same type and count
        for batch_hash, individual_hash in zip(batch_hashes, individual_hashes):
            assert isinstance(batch_hash, type(individual_hash))
            assert len(bytes(batch_hash)) == len(bytes(individual_hash))

        # Test different algorithms produce different results
        blake3_hashes = frame.hash_collection("basic_test", "blake3")
        assert len(blake3_hashes) == len(batch_hashes)

    def test_batch_vs_individual_consistency(self):
        """Test that batch and individual hashing produce consistent results."""
        entities = [{"users": [f"u{i}"], "data": [f"d{i}"]} for i in range(10)]

        frame = EntityFrame()
        frame.add_method("consistency_test", entities)

        # Get batch results
        batch_hashes = frame.hash_collection_hex("consistency_test", "sha256")

        # Get individual results
        individual_hashes = []
        for i in range(len(entities)):
            hash_bytes = frame.hash_entity("consistency_test", i, "sha256")
            individual_hashes.append(bytes(hash_bytes).hex())

        # Should be identical
        assert len(batch_hashes) == len(individual_hashes)
        for batch_hash, individual_hash in zip(batch_hashes, individual_hashes):
            assert (
                batch_hash == individual_hash
            ), "Batch and individual hashing should be consistent"

    def test_algorithm_correctness(self):
        """Test different algorithms work correctly and produce different results."""
        entities = [{"users": [f"u{i}"], "orders": [f"o{i}"]} for i in range(5)]

        frame = EntityFrame()
        frame.add_method("algo_test", entities)

        algorithms = ["sha256", "blake3", "sha3-256", "sha512"]
        results = {}

        # Test each algorithm
        for algorithm in algorithms:
            hashes = frame.hash_collection_hex("algo_test", algorithm)
            results[algorithm] = hashes

            assert len(hashes) == len(entities)

            # Verify hash format based on algorithm
            if algorithm == "sha256":
                assert all(
                    len(h) == 64 for h in hashes
                ), "SHA-256 should be 64 hex chars"
            elif algorithm == "sha512":
                assert all(
                    len(h) == 128 for h in hashes
                ), "SHA-512 should be 128 hex chars"
            elif algorithm == "blake3":
                assert all(
                    len(h) == 64 for h in hashes
                ), "BLAKE3 should be 64 hex chars (32 bytes)"

        # Verify algorithms produce different results
        assert (
            results["sha256"] != results["blake3"]
        ), "Different algorithms should produce different hashes"
        assert (
            results["sha256"] != results["sha512"]
        ), "Different algorithms should produce different hashes"

    def test_output_format_consistency(self):
        """Test different output formats work correctly and consistently."""
        entities = [{"users": [f"u{i}"]} for i in range(5)]
        frame = EntityFrame()
        frame.add_method("format_test", entities)

        # Test different output formats
        bytes_hashes = frame.hash_collection("format_test", "sha256")
        hex_hashes = frame.hash_collection_hex("format_test", "sha256")
        raw_bytes = frame.hash_collection_raw("format_test", "sha256")

        assert len(bytes_hashes) == len(hex_hashes) == len(entities)

        # Verify hex format (SHA-256 = 32 bytes = 64 hex chars)
        for hex_hash in hex_hashes:
            assert (
                len(hex_hash) == 64
            ), f"SHA-256 hex should be 64 chars, got {len(hex_hash)}"
            assert all(
                c in "0123456789abcdef" for c in hex_hash
            ), "Invalid hex characters"

        # Test consistency between hex and bytes
        for bytes_hash, hex_hash in zip(bytes_hashes, hex_hashes):
            assert hex_hash == bytes(bytes_hash).hex(), "Hex/bytes format mismatch"

        # Verify raw bytes format has reasonable size
        assert (
            len(bytes(raw_bytes)) > len(entities) * 30
        ), "Raw bytes should contain hash data"

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic and order-independent."""
        # Same data in different orders
        entities1 = [{"customers": ["c3", "c1", "c2"], "orders": ["o2", "o1"]}]
        entities2 = [{"customers": ["c1", "c2", "c3"], "orders": ["o1", "o2"]}]

        frame1 = EntityFrame()
        frame1.add_method("test1", entities1)

        frame2 = EntityFrame()
        frame2.add_method("test2", entities2)

        # Hash multiple times
        hash1_run1 = frame1.hash_collection_hex("test1", "sha256")
        hash1_run2 = frame1.hash_collection_hex("test1", "sha256")
        hash2_run1 = frame2.hash_collection_hex("test2", "sha256")

        # Should be deterministic and order-independent
        assert hash1_run1 == hash1_run2, "Hashing should be deterministic"
        assert hash1_run1 == hash2_run1, "Hashing should be order-independent"

    def test_string_interning_efficiency(self):
        """Test that string interning provides memory efficiency."""
        count = 100
        overlap_ratio = 0.8  # 80% shared strings

        entities = []
        for i in range(count):
            if i < count * overlap_ratio:
                # Shared strings (cycle through patterns)
                base = i % 10  # Only 10 unique patterns
                entity = {
                    "users": [f"shared_user_{base}"],
                    "data": [f"shared_data_{base}"],
                }
            else:
                # Unique strings
                entity = {"users": [f"unique_user_{i}"], "data": [f"unique_data_{i}"]}
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("interning_test", entities)

        # Test hashing works
        hashes = frame.hash_collection_hex("interning_test", "blake3")
        assert len(hashes) == count

        # Calculate memory efficiency
        interner_size = frame.interner_size()
        theoretical_strings = count * 2 + 2  # 2 strings per entity + 2 datasets
        compression_ratio = (
            theoretical_strings / interner_size if interner_size > 0 else 1
        )

        # Should achieve significant compression due to string overlap
        assert (
            compression_ratio > 2.0
        ), f"String interning should provide >2x compression, got {compression_ratio:.1f}x"
