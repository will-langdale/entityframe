"""
Micro-benchmarks to identify exact bottlenecks in hashing performance.
"""

import time
from entityframe import EntityFrame


def test_bulk_lookup_bottleneck():
    """Micro-benchmark to identify bottleneck in bulk string lookup."""
    print("\n🔬 Micro-benchmarking bulk string lookup bottleneck...")

    # Create a controlled test case
    count = 1000  # Smaller for focused analysis
    entities = []
    for i in range(count):
        entity = {
            "users": [f"user_{i}"],
            "orders": [f"order_{i}_{j}" for j in range(3)],
            "events": [f"event_{i}_{j}" for j in range(5)],
        }
        entities.append(entity)

    frame = EntityFrame()

    # Build entities
    build_start = time.time()
    frame.add_method("bottleneck", entities)
    build_time = time.time() - build_start
    build_rate = count / build_time if build_time > 0 else 0

    print(
        f"Build: {count} entities in {build_time:.3f}s ({build_rate:,.0f} entities/sec)"
    )

    # Test hashing with timing breakdown
    print("\nHashing with detailed timing:")
    hash_start = time.time()
    _ = frame.hash_collection("bottleneck", "blake3")
    hash_time = time.time() - hash_start
    hash_rate = count / hash_time if hash_time > 0 else 0

    print(f"Hash: {count} entities in {hash_time:.3f}s ({hash_rate:,.0f} entities/sec)")

    # Calculate expected performance for 1M entities
    if hash_rate > 0:
        time_for_1m = 1_000_000 / hash_rate
        print(
            f"Extrapolated time for 1M entities: {time_for_1m:.1f}s ({time_for_1m/60:.1f} min)"
        )
        print(f"Target: 1 second (need {1_000_000:.0f} entities/sec)")
        print(f"Current shortfall: {1_000_000/hash_rate:.1f}x too slow")


def test_string_access_patterns():
    """Test different string access patterns to identify optimal approach."""
    print("\n🧪 Testing string access pattern alternatives...")

    # Create test data
    count = 500
    entities = []
    for i in range(count):
        entity = {"users": [f"user_{i}", f"u{i}"], "orders": [f"order_{i}"]}
        entities.append(entity)

    frame = EntityFrame()
    frame.add_method("access_test", entities)

    # Test current implementation
    print("Testing current hybrid approach:")
    start = time.time()
    _ = frame.hash_collection("access_test", "sha256")
    time1 = time.time() - start
    rate1 = count / time1 if time1 > 0 else 0
    print(f"  Hybrid: {rate1:,.0f} entities/sec ({time1:.3f}s)")

    # For comparison, test individual hashing (sample)
    sample_size = min(50, count)
    print(f"\nTesting individual approach (sample {sample_size}):")
    start = time.time()
    for i in range(sample_size):
        frame.hash_entity("access_test", i, "sha256")
    time2 = time.time() - start
    rate2 = sample_size / time2 if time2 > 0 else 0
    print(f"  Individual: {rate2:,.0f} entities/sec ({time2:.3f}s for {sample_size})")

    print("\nComparison:")
    if rate1 > 0 and rate2 > 0:
        ratio = rate1 / rate2
        print(
            f"  Hybrid is {ratio:.1f}x {'faster' if ratio > 1 else 'slower'} than individual"
        )


if __name__ == "__main__":
    test_bulk_lookup_bottleneck()
    test_string_access_patterns()
