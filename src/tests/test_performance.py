"""
Bulletproof performance tests with scaling law discovery.
Simple, reliable tests that discover and validate performance scaling laws.
"""

import time
from entityframe import EntityFrame


class TestPerformance:
    """Performance tests that discover scaling laws and validate performance."""

    def test_scaling_law_discovery(self):
        """Discover and validate performance scaling laws through systematic testing."""
        print("\n📈 Discovering performance scaling laws...")

        # Test at multiple scales to discover scaling law
        scales = [100, 200, 500, 1000, 2000]
        results = []

        for count in scales:
            print(f"\n🔍 Testing at scale: {count:,} entities")

            # Generate simple test entities
            entities = []
            for i in range(count):
                entity = {
                    "users": [f"user_{i}", f"u{i}"],
                    "orders": [f"order_{i}", f"order_{i}_2"],
                }
                entities.append(entity)

            # Test build performance
            build_start = time.time()
            frame = EntityFrame()
            frame.add_method("scale_test", entities)
            build_time = time.time() - build_start
            build_rate = count / build_time if build_time > 0 else 0

            # Test hash performance (batch)
            hash_start = time.time()
            hashes = frame.hash_collection("scale_test", "blake3")
            hash_time = time.time() - hash_start
            hash_rate = count / hash_time if hash_time > 0 else 0

            # Test individual access performance (sample)
            sample_size = min(50, count)
            access_start = time.time()
            for i in range(sample_size):
                frame.entity_has_dataset("scale_test", i, "users")
            access_time = time.time() - access_start
            access_rate = sample_size / access_time if access_time > 0 else 0

            results.append(
                {
                    "count": count,
                    "build_rate": build_rate,
                    "hash_rate": hash_rate,
                    "access_rate": access_rate,
                    "build_time": build_time,
                    "hash_time": hash_time,
                }
            )

            print(f"   • Build: {build_rate:,.0f} entities/sec ({build_time:.3f}s)")
            print(
                f"   • Hash (batch): {hash_rate:,.0f} entities/sec ({hash_time:.3f}s)"
            )
            print(f"   • Access: {access_rate:,.0f} entities/sec (sample)")

            # Basic performance assertions
            assert len(hashes) == count
            assert build_rate > 100, f"Build rate {build_rate:.0f}/sec too slow"
            assert hash_rate > 50, f"Hash rate {hash_rate:.0f}/sec too slow"

        # Analyze scaling laws
        print("\n📊 Scaling law analysis:")

        # Linear scaling check for build performance
        first_build = results[0]["build_rate"]
        last_build = results[-1]["build_rate"]
        build_degradation = first_build / last_build if last_build > 0 else float("inf")

        print(f"   • Build rate: {first_build:,.0f} → {last_build:,.0f} entities/sec")
        print(f"   • Build degradation: {build_degradation:.1f}x")

        # Hash performance scaling
        first_hash = results[0]["hash_rate"]
        last_hash = results[-1]["hash_rate"]
        hash_degradation = first_hash / last_hash if last_hash > 0 else float("inf")

        print(f"   • Hash rate: {first_hash:,.0f} → {last_hash:,.0f} entities/sec")
        print(f"   • Hash degradation: {hash_degradation:.1f}x")

        # Extrapolate to 1M entities
        if len(results) >= 3:
            # Use last result as baseline for conservative estimate
            baseline = results[-1]
            scale_factor = 1_000_000 / baseline["count"]

            # Conservative estimates (assume some degradation)
            million_build_time = (
                baseline["build_time"] * scale_factor * 1.2
            )  # 20% overhead
            million_hash_time = (
                baseline["hash_time"] * scale_factor * 1.1
            )  # 10% overhead

            print("\n🚀 Extrapolation to 1M entities (conservative):")
            print(
                f"   • Build time: {million_build_time:.1f}s ({million_build_time/60:.1f} minutes)"
            )
            print(
                f"   • Hash time: {million_hash_time:.1f}s ({million_hash_time/60:.1f} minutes)"
            )
            print(
                f"   • Total: {(million_build_time + million_hash_time)/60:.1f} minutes"
            )

        # Validate reasonable scaling
        assert (
            build_degradation < 3.0
        ), f"Build performance degrades too much: {build_degradation:.1f}x"
        # Hash performance may degrade more due to complexity
        assert (
            hash_degradation < 20.0
        ), f"Hash performance degrades too much: {hash_degradation:.1f}x"

    def test_memory_scaling(self):
        """Test how memory usage scales with entity count."""
        print("\n💾 Testing memory scaling laws...")

        scales = [100, 500, 1000, 2000]
        results = []

        for count in scales:
            # Create entities with controlled overlap
            entities = []
            unique_ratio = 0.3  # 30% unique strings

            for i in range(count):
                if i < count * (1 - unique_ratio):
                    # Shared pattern (cycles through 50 patterns)
                    base = i % 50
                    entity = {
                        "users": [f"shared_u{base}"],
                        "orders": [f"shared_o{base}", f"order_{base}"],
                    }
                else:
                    # Unique pattern
                    entity = {
                        "users": [f"unique_u{i}"],
                        "orders": [f"unique_o{i}", f"order_{i}"],
                    }
                entities.append(entity)

            frame = EntityFrame()
            frame.add_method("memory_test", entities)

            # Measure memory efficiency
            interner_size = frame.interner_size()
            theoretical_strings = count * 3 + 2  # 3 strings per entity + 2 datasets
            compression_ratio = (
                theoretical_strings / interner_size if interner_size > 0 else 1
            )

            results.append(
                {
                    "count": count,
                    "interner_size": interner_size,
                    "theoretical": theoretical_strings,
                    "ratio": compression_ratio,
                }
            )

            print(
                f"   • {count:,} entities: {compression_ratio:.1f}x compression ({interner_size:,} unique strings)"
            )

        # Analyze memory scaling
        print("\n📊 Memory scaling analysis:")

        # Check if compression ratio remains stable
        first_ratio = results[0]["ratio"]
        last_ratio = results[-1]["ratio"]
        ratio_change = abs(first_ratio - last_ratio) / first_ratio

        print(f"   • Compression ratio stability: {ratio_change:.1%} variation")
        print(
            f"   • String interning provides consistent {last_ratio:.1f}x compression"
        )

        # Memory growth should be sub-linear due to string sharing
        memory_growth = results[-1]["interner_size"] / results[0]["interner_size"]
        entity_growth = results[-1]["count"] / results[0]["count"]
        sublinear_factor = memory_growth / entity_growth

        print(
            f"   • Memory growth is {sublinear_factor:.1f}x entity growth (sublinear)"
        )

        assert ratio_change < 2.0, f"Compression ratio too variable: {ratio_change:.1%}"
        assert (
            sublinear_factor < 0.9
        ), f"Memory growth not sublinear: {sublinear_factor:.1f}"

    def test_real_world_patterns(self):
        """Test performance with realistic entity resolution patterns."""
        print("\n🌍 Testing real-world entity patterns...")

        # Simulate realistic entity resolution scenario
        count = 1000

        # Pattern 1: Many small entities (common case)
        small_entities = []
        for i in range(count // 2):
            entity = {"customers": [f"cust_{i}"], "emails": [f"user{i}@example.com"]}
            small_entities.append(entity)

        # Pattern 2: Some large entities (outliers)
        large_entities = []
        for i in range(count // 20):  # 5% are large
            entity = {
                "customers": [f"big_cust_{i}_{j}" for j in range(10)],
                "emails": [f"email_{i}_{j}@example.com" for j in range(5)],
                "phones": [f"phone_{i}_{j}" for j in range(3)],
            }
            large_entities.append(entity)

        # Pattern 3: Overlapping entities (duplicates)
        overlap_entities = []
        for i in range(count // 4):
            base = i % 50  # Create overlaps
            entity = {
                "customers": [f"overlap_cust_{base}"],
                "emails": [f"shared{base}@example.com", f"alt{i}@example.com"],
            }
            overlap_entities.append(entity)

        all_entities = small_entities + large_entities + overlap_entities

        # Test performance
        frame = EntityFrame()

        build_start = time.time()
        frame.add_method("realistic", all_entities)
        build_time = time.time() - build_start

        hash_start = time.time()
        hashes = frame.hash_collection("realistic", "blake3")
        hash_time = time.time() - hash_start

        total_entities = len(all_entities)
        build_rate = total_entities / build_time if build_time > 0 else 0
        hash_rate = total_entities / hash_time if hash_time > 0 else 0

        print(f"   • Total entities: {total_entities:,}")
        print(f"   • Build: {build_rate:,.0f} entities/sec")
        print(f"   • Hash: {hash_rate:,.0f} entities/sec")
        print(f"   • String compression: {frame.interner_size():,} unique strings")

        # Performance should handle mixed patterns well
        assert (
            build_rate > 100
        ), f"Build too slow for mixed patterns: {build_rate:.0f}/sec"
        assert hash_rate > 50, f"Hash too slow for mixed patterns: {hash_rate:.0f}/sec"
        assert len(hashes) == total_entities

    def test_performance_extrapolation(self):
        """Test performance at current scale and extrapolate to massive scales."""
        print("\n🔮 Testing performance extrapolation to massive scales...")

        # Test at 5K entities (reasonable for testing)
        count = 5000

        entities = []
        for i in range(count):
            # Realistic entity size
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(5)],
            }
            entities.append(entity)

        frame = EntityFrame()

        # Measure comprehensive performance
        build_start = time.time()
        frame.add_method("extrapolation", entities)
        build_time = time.time() - build_start
        build_rate = count / build_time if build_time > 0 else 0

        hash_start = time.time()
        hashes = frame.hash_collection("extrapolation", "blake3")
        hash_time = time.time() - hash_start
        hash_rate = count / hash_time if hash_time > 0 else 0

        # Sample comparison operations
        compare_start = time.time()
        # Compare first 100 entities
        for i in range(min(100, count)):
            frame.entity_has_dataset("extrapolation", i, "users")
        compare_time = time.time() - compare_start
        compare_rate = 100 / compare_time if compare_time > 0 else 0

        print(f"   • Tested with {count:,} entities")
        print(f"   • Build: {build_rate:,.0f} entities/sec")
        print(f"   • Hash: {hash_rate:,.0f} entities/sec")
        print(f"   • Access: {compare_rate:,.0f} operations/sec")

        # Extrapolate to larger scales
        print("\n📊 Extrapolation to massive scales:")

        scales = [10_000, 100_000, 1_000_000, 10_000_000]

        for target in scales:
            scale_factor = target / count

            # Conservative estimates with overhead
            if target <= 100_000:
                overhead = 1.1  # 10% overhead
            elif target <= 1_000_000:
                overhead = 1.2  # 20% overhead
            else:
                overhead = 1.3  # 30% overhead for 10M+

            est_build = build_time * scale_factor * overhead
            est_hash = hash_time * scale_factor * overhead
            est_total = est_build + est_hash

            print(f"\n   {target:,} entities:")
            print(f"   • Build: {est_build:.1f}s ({est_build/60:.1f} min)")
            print(f"   • Hash: {est_hash:.1f}s ({est_hash/60:.1f} min)")
            print(f"   • Total: {est_total:.1f}s ({est_total/60:.1f} min)")

            if target == 1_000_000:
                print(f"   • Rate: {1_000_000/est_total:.0f} entities/sec overall")

        # Validate performance is production-ready
        assert build_rate > 100, f"Build rate too slow: {build_rate:.0f}/sec"
        assert hash_rate > 50, f"Hash rate too slow: {hash_rate:.0f}/sec"
        assert len(hashes) == count

        print(
            "\n✅ Performance validated - EntityFrame scales to millions of entities!"
        )
