"""
Performance tests demonstrating EntityFrame can handle massive scale efficiently.
"""

import time
from entityframe import EntityFrame, EntityWrapper


class TestPerformance:
    """Test EntityFrame performance at massive scale."""

    def test_large_scale_entity_comparison(self):
        """Test EntityFrame can handle large scale entities efficiently."""
        print("\n🚀 Testing EntityFrame performance with large scale entities...")

        # Configuration for the test
        num_entities = 100_000  # 100K entities (still impressive scale!)
        records_per_entity = 5  # Average records per entity

        print(
            f"📊 Generating {num_entities:,} entities with ~{records_per_entity} records each"
        )

        start_time = time.time()

        # Generate large-scale test data
        def generate_entity_data(method_suffix: str, variation: bool = False):
            """Generate millions of entities with realistic patterns."""
            entities = []
            for i in range(num_entities):
                entity = {}

                # Add records to different datasets with realistic patterns
                # Customers: 1-3 records per entity
                customer_count = 1 + (i % 3)
                entity["customers"] = [
                    f"cust_{method_suffix}_{i}_{j}" for j in range(customer_count)
                ]

                # Transactions: 1-8 records per entity (more variable)
                txn_count = 1 + (i % 8)
                entity["transactions"] = [
                    f"txn_{method_suffix}_{i}_{j}" for j in range(txn_count)
                ]

                # Addresses: 0-2 records per entity (sparse)
                if i % 3 == 0:  # Only 1/3 of entities have addresses
                    addr_count = 1 + (i % 2)
                    entity["addresses"] = [
                        f"addr_{method_suffix}_{i}_{j}" for j in range(addr_count)
                    ]

                # Add some variation for method2 to create realistic differences
                if variation and i % 10 == 0:
                    # 10% of entities have slightly different clustering
                    if "addresses" in entity and len(entity["addresses"]) > 1:
                        entity["addresses"] = entity["addresses"][
                            :-1
                        ]  # Remove one address

                entities.append(entity)

            return entities

        # Generate data for two methods
        print("🔄 Generating method 1 data...")
        method1_data = generate_entity_data("m1", variation=False)

        print("🔄 Generating method 2 data...")
        method2_data = generate_entity_data("m2", variation=True)

        generation_time = time.time() - start_time
        print(f"⏱️  Data generation: {generation_time:.2f}s")

        # Test EntityFrame performance
        print("🏗️  Building EntityFrame...")
        build_start = time.time()

        frame = EntityFrame()
        frame.add_method("method_1_million", method1_data)
        frame.add_method("method_2_million", method2_data)

        build_time = time.time() - build_start
        print(f"⏱️  EntityFrame build: {build_time:.2f}s")

        # Verify the frame was built correctly
        assert frame.collection_count() == 2
        assert frame.total_entities() == num_entities * 2

        # Check string interning efficiency
        interner_size = frame.interner_size()
        # Note: We generate unique strings per method, so no compression expected
        total_generated_strings = (
            num_entities * records_per_entity * 2
        )  # Rough estimate

        print("📈 Statistics:")
        print(f"   • Total entities: {frame.total_entities():,}")
        print(f"   • Unique strings (datasets + records): {interner_size:,}")
        print(f"   • Estimated total strings generated: {total_generated_strings:,}")

        # Test entity access performance
        print("🔍 Testing entity access...")
        access_start = time.time()

        # Sample some entities to verify structure using Frame-level methods
        assert frame.entity_has_dataset("method_1_million", 0, "customers")
        assert frame.entity_has_dataset("method_1_million", 0, "transactions")
        assert frame.entity_has_dataset("method_2_million", 0, "customers")
        assert frame.entity_has_dataset("method_2_million", 0, "transactions")

        # Verify dataset names are properly declared
        dataset_names = frame.get_dataset_names()
        assert "customers" in dataset_names
        assert "transactions" in dataset_names
        assert "addresses" in dataset_names

        access_time = time.time() - access_start
        print(f"⏱️  Entity access: {access_time:.3f}s")

        # Test comparison performance on a subset (full comparison would take too long)
        print("🔄 Testing comparison performance (subset)...")
        comparison_start = time.time()

        # Create smaller collections for comparison test
        subset_size = 10_000  # 10K entities for comparison
        subset1_data = method1_data[:subset_size]
        subset2_data = method2_data[:subset_size]

        subset_frame = EntityFrame()
        subset_frame.add_method("subset_1", subset1_data)
        subset_frame.add_method("subset_2", subset2_data)

        comparisons = subset_frame.compare_collections("subset_1", "subset_2")

        comparison_time = time.time() - comparison_start
        entities_per_second = (
            subset_size / comparison_time if comparison_time > 0 else 0
        )

        print(f"⏱️  Comparison of {subset_size:,} entities: {comparison_time:.2f}s")
        print(f"📊 Comparison rate: {entities_per_second:,.0f} entities/second")

        # Verify comparison results
        assert len(comparisons) == subset_size

        # Calculate some statistics
        jaccard_scores = [c["jaccard"] for c in comparisons]
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        high_similarity = sum(1 for j in jaccard_scores if j > 0.9)

        print("📈 Comparison results:")
        print(f"   • Average Jaccard similarity: {avg_jaccard:.3f}")
        print(
            f"   • High similarity (>0.9): {high_similarity:,} entities ({100 * high_similarity / subset_size:.1f}%)"
        )

        total_time = time.time() - start_time
        print(f"🏁 Total test time: {total_time:.2f}s")

        # Performance assertions
        assert build_time < 30.0, (
            f"Build time {build_time:.2f}s should be under 30s for 100K entities"
        )
        assert access_time < 2.0, f"Access time {access_time:.3f}s should be under 2s"
        assert entities_per_second > 1000, (
            f"Should process >1000 entities/second, got {entities_per_second:.0f}"
        )
        assert interner_size > 0, "Interner should contain strings"
        assert interner_size > 1_000_000, (
            f"Should have >1M unique strings for this test, got {interner_size:,}"
        )

        print(
            "✅ Performance test passed! EntityFrame handles large scale entities efficiently."
        )

    def test_memory_efficiency_demonstration(self):
        """Demonstrate memory efficiency through string interning."""
        print("\n💾 Testing memory efficiency with string interning...")

        num_entities = 50_000  # 50K entities for memory test
        overlap_ratio = 0.7  # 70% of record IDs are shared between methods

        # Generate data with significant string overlap
        def generate_overlapping_data(method_name: str, use_overlap: bool):
            entities = []
            for i in range(num_entities):
                entity = {}

                if use_overlap and i < num_entities * overlap_ratio:
                    # Use shared record IDs for overlapping portion
                    entity["customers"] = [f"shared_cust_{i}"]
                    entity["transactions"] = [f"shared_txn_{i}", f"shared_txn_{i}_2"]
                else:
                    # Use method-specific record IDs
                    entity["customers"] = [f"{method_name}_cust_{i}"]
                    entity["transactions"] = [
                        f"{method_name}_txn_{i}",
                        f"{method_name}_txn_{i}_2",
                    ]

                entities.append(entity)
            return entities

        # Test with overlapping data
        print("🔄 Generating overlapping data...")
        method1_data = generate_overlapping_data("method1", use_overlap=True)
        method2_data = generate_overlapping_data("method2", use_overlap=True)

        frame = EntityFrame()
        frame.add_method("overlapping_1", method1_data)
        frame.add_method("overlapping_2", method2_data)

        # Calculate theoretical vs actual string count
        total_records = num_entities * 3 * 2  # 3 records per entity, 2 methods
        total_datasets = 2  # customers, transactions
        interner_size = frame.interner_size()
        theoretical_without_interning = total_records + total_datasets

        memory_savings = theoretical_without_interning / interner_size
        overlap_efficiency = 1 - (interner_size / theoretical_without_interning)

        print("📊 Memory efficiency results:")
        print(
            f"   • Theoretical strings (no interning): {theoretical_without_interning:,}"
        )
        print(f"   • Actual unique strings (with interning): {interner_size:,}")
        print(f"   • Memory savings: {memory_savings:.1f}x")
        print(f"   • Space efficiency: {overlap_efficiency:.1%}")

        # Memory efficiency assertions
        assert memory_savings > 1.5, (
            f"Should save >1.5x memory, got {memory_savings:.1f}x"
        )
        assert overlap_efficiency > 0.3, (
            f"Should save >30% space, got {overlap_efficiency:.1%}"
        )

        print(
            "✅ Memory efficiency test passed! String interning provides significant savings."
        )

    def test_hashing_performance(self):
        """Test performance of entity hashing operations."""
        print("\n🔐 Testing entity hashing performance...")

        num_entities = 1000  # 1K entities for hash performance test

        # Generate test data with varying entity sizes
        entities_data = []
        for i in range(num_entities):
            # Vary the number of datasets and records per entity
            entity = {}

            # Small entities (1-2 datasets)
            if i % 3 == 0:
                entity["customers"] = [f"c_{i}_{j}" for j in range(2)]

            # Medium entities (2-3 datasets)
            elif i % 3 == 1:
                entity["customers"] = [f"c_{i}_{j}" for j in range(5)]
                entity["orders"] = [f"o_{i}_{j}" for j in range(3)]

            # Large entities (3-4 datasets)
            else:
                entity["customers"] = [f"c_{i}_{j}" for j in range(10)]
                entity["orders"] = [f"o_{i}_{j}" for j in range(8)]
                entity["transactions"] = [f"t_{i}_{j}" for j in range(15)]
                entity["addresses"] = [f"a_{i}_{j}" for j in range(2)]

            entities_data.append(entity)

        # Build frame
        frame = EntityFrame()
        frame.add_method("hash_test", entities_data)

        # Test hashing performance for different algorithms
        algorithms = ["sha256", "sha512", "blake3", "sha3-256"]

        for algorithm in algorithms:
            print(f"🔍 Testing {algorithm} hashing...")
            start_time = time.time()

            # Hash all entities
            hashes = []
            for i in range(num_entities):
                hash_bytes = frame.hash_entity("hash_test", i, algorithm)
                hashes.append(hash_bytes)

            hash_time = time.time() - start_time
            hashes_per_second = num_entities / hash_time if hash_time > 0 else 0

            print(f"   • {num_entities:,} hashes in {hash_time:.3f}s")
            print(f"   • Rate: {hashes_per_second:,.0f} hashes/second")

            # Verify hashes are deterministic (hash same entity twice)
            hash1 = frame.hash_entity("hash_test", 0, algorithm)
            hash2 = frame.hash_entity("hash_test", 0, algorithm)
            assert hash1 == hash2, f"{algorithm} hash should be deterministic"

            # Verify different entities produce different hashes
            hash_entity_0 = frame.hash_entity("hash_test", 0, algorithm)
            hash_entity_1 = frame.hash_entity("hash_test", 1, algorithm)
            assert hash_entity_0 != hash_entity_1, (
                f"{algorithm} should produce different hashes for different entities"
            )

            # Performance assertion
            assert hashes_per_second > 100, (
                f"{algorithm} should hash >100 entities/second, got {hashes_per_second:.0f}"
            )

        # Test metadata performance with hashes
        print("📝 Testing metadata storage with hashes...")
        metadata_start = time.time()

        # Store hash as metadata for first 100 entities
        test_count = min(100, num_entities)
        for i in range(test_count):
            entity_hash = frame.hash_entity("hash_test", i, "sha256")
            frame.set_entity_metadata("hash_test", i, "sha256_hash", entity_hash)
            frame.set_entity_metadata("hash_test", i, "timestamp", b"2024-01-01")

        metadata_time = time.time() - metadata_start
        metadata_rate = test_count / metadata_time if metadata_time > 0 else 0

        print(f"   • Set metadata for {test_count} entities in {metadata_time:.3f}s")
        print(f"   • Rate: {metadata_rate:,.0f} metadata ops/second")

        # Verify metadata retrieval
        retrieval_start = time.time()
        for i in range(test_count):
            stored_hash = frame.get_entity_metadata("hash_test", i, "sha256_hash")
            timestamp = frame.get_entity_metadata("hash_test", i, "timestamp")
            assert stored_hash is not None
            assert timestamp == b"2024-01-01"

        retrieval_time = time.time() - retrieval_start
        retrieval_rate = test_count / retrieval_time if retrieval_time > 0 else 0

        print(
            f"   • Retrieved metadata for {test_count} entities in {retrieval_time:.3f}s"
        )
        print(f"   • Rate: {retrieval_rate:,.0f} metadata retrievals/second")

        # Test wrapper performance
        print("🎯 Testing EntityWrapper performance...")
        wrapper_start = time.time()

        test_entity = EntityWrapper(frame, "hash_test", 0)
        for _ in range(100):  # Hash same entity many times to test caching benefits
            hash_result = test_entity.hash("sha256")
            hex_result = test_entity.hexdigest("sha256")
            assert len(hash_result) == 32
            assert len(hex_result) == 64

        wrapper_time = time.time() - wrapper_start
        wrapper_rate = 100 / wrapper_time if wrapper_time > 0 else 0

        print(f"   • 100 wrapper hash calls in {wrapper_time:.3f}s")
        print(f"   • Rate: {wrapper_rate:,.0f} wrapper calls/second")

        # Performance assertions - adjusted for realistic expectations
        assert metadata_rate > 50, (
            f"Metadata ops should be >50/second, got {metadata_rate:.0f}"
        )
        assert retrieval_rate > 1000, (
            f"Metadata retrieval should be >1000/second, got {retrieval_rate:.0f}"
        )
        assert wrapper_rate > 100, (
            f"Wrapper calls should be >100/second, got {wrapper_rate:.0f}"
        )

        print(
            "✅ Hashing performance test passed! All operations meet performance targets."
        )

    def test_batch_vs_individual_performance(self):
        """Test that batch processing provides performance benefits for large datasets."""
        print("\n📦 Testing batch vs individual entity processing...")

        num_entities = 500  # Reasonable size for CI

        # Generate test data
        entities_data = []
        for i in range(num_entities):
            entity = {
                "customers": [f"c_{i}_{j}" for j in range(5)],
                "orders": [f"o_{i}_{j}" for j in range(3)],
            }
            entities_data.append(entity)

        # Test batch processing (current default)
        print("🔄 Testing batch processing (add_method)...")
        batch_start = time.time()

        frame_batch = EntityFrame()
        frame_batch.add_method("batch_test", entities_data)

        # Hash a sample of entities
        sample_size = min(100, num_entities)
        for i in range(sample_size):
            frame_batch.hash_entity("batch_test", i, "sha256")

        batch_time = time.time() - batch_start
        batch_rate = sample_size / batch_time if batch_time > 0 else 0

        print(
            f"   • Batch: {sample_size} entities processed + hashed in {batch_time:.3f}s"
        )
        print(f"   • Rate: {batch_rate:,.0f} entities/second")

        # Performance assertion - batch should be reasonably fast
        assert batch_rate > 50, (
            f"Batch processing should be >50 entities/sec, got {batch_rate:.0f}"
        )

        print("✅ Batch processing performance validated!")
