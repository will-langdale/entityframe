"""
Performance tests demonstrating EntityFrame can handle massive scale efficiently.
"""

import time
from entityframe import EntityFrame


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
        print(f"   • Unique strings: {interner_size:,}")
        print(f"   • Estimated total strings generated: {total_generated_strings:,}")

        # Test entity access performance
        print("🔍 Testing entity access...")
        access_start = time.time()

        collection1 = frame.get_collection("method_1_million")
        collection2 = frame.get_collection("method_2_million")

        # Sample some entities to verify structure
        sample_entity1 = collection1.get_entity(0)
        sample_entity2 = collection2.get_entity(0)

        assert sample_entity1.has_dataset("customers")
        assert sample_entity1.has_dataset("transactions")
        assert sample_entity2.has_dataset("customers")
        assert sample_entity2.has_dataset("transactions")

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
        assert interner_size > 0, "String interner should contain strings"
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
        unique_strings = frame.interner_size()
        theoretical_without_interning = total_records

        memory_savings = theoretical_without_interning / unique_strings
        overlap_efficiency = 1 - (unique_strings / theoretical_without_interning)

        print("📊 Memory efficiency results:")
        print(
            f"   • Theoretical strings (no interning): {theoretical_without_interning:,}"
        )
        print(f"   • Actual unique strings (with interning): {unique_strings:,}")
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
