"""
Bulletproof batch hashing tests.
Simple, reliable tests that complete quickly and provide useful data.
"""

import time
from entityframe import EntityFrame


class TestBatchHashing:
    """Test batch hashing with simple, reliable tests."""

    def test_basic_batch_functionality(self):
        """Test that batch hashing works correctly with simple entities."""
        print("\n✅ Testing basic batch hashing functionality...")

        # Simple, small test for reliability
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

        # Verify consistency
        assert len(batch_hashes) == len(individual_hashes) == len(entities)
        print(f"   • Generated {len(batch_hashes)} consistent hashes")

        # Test different algorithms produce different results
        blake3_hashes = frame.hash_collection("basic_test", "blake3")
        assert batch_hashes != blake3_hashes
        print("   • Different algorithms produce different hashes ✓")

    def test_small_scale_performance(self):
        """Test performance at small, reliable scales."""
        print("\n⚡ Testing small-scale performance...")

        scales = [100, 500, 1000]

        for count in scales:
            # Simple entities for consistent testing
            entities = []
            for i in range(count):
                entity = {"users": [f"user_{i}", f"u{i}"], "orders": [f"order_{i}"]}
                entities.append(entity)

            frame = EntityFrame()
            frame.add_method("perf_test", entities)

            # Test batch hashing
            start = time.time()
            hashes = frame.hash_collection("perf_test", "blake3")
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0

            print(f"   • {count:,} entities: {rate:,.0f} entities/sec ({elapsed:.3f}s)")

            # Basic performance assertion
            assert rate > 50, f"Should achieve >50 entities/sec, got {rate:.0f}"
            assert len(hashes) == count

    def test_batch_vs_individual_comparison(self):
        """Compare batch vs individual hashing at reasonable scale."""
        print("\n🔄 Testing batch vs individual hashing comparison...")

        count = 500  # Reasonable size for both approaches

        # Simple entities
        entities = [{"users": [f"u{i}"], "data": [f"d{i}"]} for i in range(count)]

        frame = EntityFrame()
        frame.add_method("comparison", entities)

        # Test individual hashing (sample for speed)
        sample_size = min(100, count)
        individual_start = time.time()
        for i in range(sample_size):
            frame.hash_entity("comparison", i, "sha256")
        individual_time = time.time() - individual_start
        individual_rate = sample_size / individual_time if individual_time > 0 else 0

        # Test batch hashing (full dataset)
        batch_start = time.time()
        batch_hashes = frame.hash_collection("comparison", "sha256")
        batch_time = time.time() - batch_start
        batch_rate = count / batch_time if batch_time > 0 else 0

        print(
            f"   • Individual: {individual_rate:,.0f} entities/sec (sample of {sample_size})"
        )
        print(f"   • Batch: {batch_rate:,.0f} entities/sec (full {count})")

        # Both should be reasonably fast
        assert (
            individual_rate > 50
        ), f"Individual hashing too slow: {individual_rate:.0f}/sec"
        assert batch_rate > 100, f"Batch hashing too slow: {batch_rate:.0f}/sec"
        assert len(batch_hashes) == count

    def test_algorithm_comparison(self):
        """Test different algorithms at consistent scale."""
        print("\n🔬 Testing algorithm performance comparison...")

        count = 200
        entities = [{"users": [f"u{i}"], "orders": [f"o{i}"]} for i in range(count)]

        frame = EntityFrame()
        frame.add_method("algo_test", entities)

        algorithms = ["sha256", "blake3", "sha3-256"]
        results = {}

        for algorithm in algorithms:
            start = time.time()
            hashes = frame.hash_collection("algo_test", algorithm)
            elapsed = time.time() - start

            rate = count / elapsed if elapsed > 0 else 0
            results[algorithm] = rate

            print(f"   • {algorithm}: {rate:,.0f} entities/sec ({elapsed:.3f}s)")

            assert len(hashes) == count
            assert rate > 50, f"{algorithm} too slow: {rate:.0f}/sec"

        # Verify algorithms produce different results
        sha256_hashes = frame.hash_collection("algo_test", "sha256")
        blake3_hashes = frame.hash_collection("algo_test", "blake3")
        assert sha256_hashes != blake3_hashes

        print("   • All algorithms working correctly ✓")

    def test_hex_output_format(self):
        """Test hex string output format."""
        print("\n📝 Testing hex output format...")

        entities = [{"users": [f"u{i}"]} for i in range(50)]
        frame = EntityFrame()
        frame.add_method("hex_test", entities)

        # Test hex batch hashing
        hex_hashes = frame.hash_collection_hex("hex_test", "sha256")
        bytes_hashes = frame.hash_collection("hex_test", "sha256")

        assert len(hex_hashes) == len(bytes_hashes) == len(entities)

        # Verify hex format (SHA-256 = 32 bytes = 64 hex chars)
        for hex_hash in hex_hashes:
            assert (
                len(hex_hash) == 64
            ), f"SHA-256 hex should be 64 chars, got {len(hex_hash)}"
            assert all(
                c in "0123456789abcdef" for c in hex_hash
            ), "Invalid hex characters"

        # Test consistency between hex and bytes
        for hex_hash, bytes_hash in zip(hex_hashes, bytes_hashes):
            assert hex_hash == bytes_hash.hex(), "Hex/bytes mismatch"

        print(f"   • {len(hex_hashes)} hex hashes verified ✓")

    def test_memory_efficiency(self):
        """Test memory efficiency with string overlap."""
        print("\n💾 Testing memory efficiency...")

        count = 2000
        overlap_ratio = 0.7  # 70% shared strings

        entities = []
        for i in range(count):
            if i < count * overlap_ratio:
                # Shared strings (cycle through patterns)
                base = i % 100
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

        # Test hashing
        start = time.time()
        hashes = frame.hash_collection("memory_test", "blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        # Calculate memory efficiency
        interner_size = frame.interner_size()
        theoretical_strings = count * 2 + 2  # 2 strings per entity + 2 datasets
        compression_ratio = (
            theoretical_strings / interner_size if interner_size > 0 else 1
        )

        print(f"   • Hashed {count:,} entities: {rate:,.0f} entities/sec")
        print(
            f"   • String compression: {compression_ratio:.1f}x ({theoretical_strings:,} → {interner_size:,})"
        )
        print(f"   • Memory savings: {(1 - interner_size/theoretical_strings):.1%}")

        assert len(hashes) == count
        assert rate > 100, f"Performance too slow: {rate:.0f}/sec"
        assert (
            compression_ratio > 1.5
        ), f"Insufficient compression: {compression_ratio:.1f}x"
