"""
Focused micro-benchmarks to identify remaining scaling bottlenecks.
"""

import time
from entityframe import EntityFrame


def test_scaling_pattern():
    """Test performance at multiple scales to identify scaling bottleneck."""
    print("\n🔍 Scaling bottleneck analysis...")

    scales = [100, 250, 500, 1000, 2000, 5000]

    for count in scales:
        # Simple entities for consistent testing
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(5)],
            }
            entities.append(entity)

        frame = EntityFrame()

        # Build
        frame.add_method("scaling", entities)

        # Hash
        hash_start = time.time()
        _ = frame.hash_collection("scaling", "blake3")
        hash_time = time.time() - hash_start
        hash_rate = count / hash_time if hash_time > 0 else 0

        print(f"{count:5} entities: {hash_rate:8.0f} entities/sec ({hash_time:.3f}s)")

        # Calculate degradation from smallest scale
        if count == scales[0]:
            baseline_rate = hash_rate
        else:
            degradation = baseline_rate / hash_rate if hash_rate > 0 else float("inf")
            print(f"                  {degradation:.1f}x degradation from {scales[0]}")


def test_memory_vs_cpu():
    """Test if the bottleneck is memory or CPU."""
    print("\n🧠 Memory vs CPU bottleneck test...")

    # Test with small entities (less memory pressure)
    small_entities = []
    for i in range(2000):
        entity = {"users": [f"u{i}"], "data": [f"d{i}"]}
        small_entities.append(entity)

    # Test with large entities (more memory pressure)
    large_entities = []
    for i in range(2000):
        entity = {
            "users": [f"user_{i}_{j}" for j in range(5)],
            "orders": [f"order_{i}_{j}" for j in range(10)],
            "events": [f"event_{i}_{j}" for j in range(15)],
        }
        large_entities.append(entity)

    frame1 = EntityFrame()
    frame1.add_method("small", small_entities)

    frame2 = EntityFrame()
    frame2.add_method("large", large_entities)

    # Test small entities
    start = time.time()
    _ = frame1.hash_collection("small", "blake3")
    time1 = time.time() - start
    rate1 = len(small_entities) / time1 if time1 > 0 else 0

    # Test large entities
    start = time.time()
    _ = frame2.hash_collection("large", "blake3")
    time2 = time.time() - start
    rate2 = len(large_entities) / time2 if time2 > 0 else 0

    print(f"Small entities (2 records each): {rate1:,.0f} entities/sec")
    print(f"Large entities (30 records each): {rate2:,.0f} entities/sec")
    print(f"Large entities are {rate1/rate2:.1f}x slower per entity")

    # Calculate per-record rates
    small_records_per_entity = 2
    large_records_per_entity = 30

    small_record_rate = rate1 * small_records_per_entity
    large_record_rate = rate2 * large_records_per_entity

    print(f"Small: {small_record_rate:,.0f} records/sec")
    print(f"Large: {large_record_rate:,.0f} records/sec")
    print(
        f"Per-record processing: {small_record_rate/large_record_rate:.1f}x difference"
    )


if __name__ == "__main__":
    test_scaling_pattern()
    test_memory_vs_cpu()
