"""
Test comparing different hashing approaches for performance optimization.
"""

import time
from entityframe import EntityFrame


def test_optimization_comparison():
    """Compare different hashing approaches to measure optimization gains."""
    print("\n🚀 Testing hash output format performance comparison...")

    # Create test data
    count = 1000
    entities = []
    for i in range(count):
        entity = {
            "users": [f"user_{i}"],
            "orders": [f"order_{i}_{j}" for j in range(3)],
            "events": [f"event_{i}_{j}" for j in range(5)],
        }
        entities.append(entity)

    frame = EntityFrame()
    frame.add_method("optimization_test", entities)

    print(f"\nTesting with {count} entities:")

    # Test 1: Original PyBytes approach
    start = time.time()
    pybytes_hashes = frame.hash_collection("optimization_test", "blake3")
    pybytes_time = time.time() - start
    pybytes_rate = count / pybytes_time

    print(f"  PyBytes approach: {pybytes_rate:,.0f} entities/sec ({pybytes_time:.3f}s)")

    # Test 2: Hex string approach
    start = time.time()
    hex_hashes = frame.hash_collection_hex("optimization_test", "blake3")
    hex_time = time.time() - start
    hex_rate = count / hex_time

    print(f"  Hex string approach: {hex_rate:,.0f} entities/sec ({hex_time:.3f}s)")

    # Test 3: Raw concatenated bytes approach
    start = time.time()
    raw_bytes = frame.hash_collection_raw("optimization_test", "blake3")
    raw_time = time.time() - start
    raw_rate = count / raw_time

    print(f"  Raw bytes approach: {raw_rate:,.0f} entities/sec ({raw_time:.3f}s)")

    # Calculate improvements
    hex_improvement = hex_rate / pybytes_rate
    raw_improvement = raw_rate / pybytes_rate

    print("\nPerformance improvements:")
    print(f"  Hex vs PyBytes: {hex_improvement:.1f}x faster")
    print(f"  Raw vs PyBytes: {raw_improvement:.1f}x faster")

    # Verify correctness - all should have same number of hashes
    assert len(pybytes_hashes) == count
    assert len(hex_hashes) == count

    # For raw bytes, parse the concatenated format
    parsed_count = 0
    offset = 0
    raw_data = bytes(raw_bytes)
    while offset < len(raw_data):
        # Read hash length (4 bytes, little-endian)
        if offset + 4 > len(raw_data):
            break
        hash_len = int.from_bytes(raw_data[offset : offset + 4], "little")
        offset += 4

        # Skip hash bytes
        if offset + hash_len > len(raw_data):
            break
        offset += hash_len
        parsed_count += 1

    assert parsed_count == count
    print(f"  ✓ All approaches returned {count} hashes correctly")

    # Performance targets
    if hex_rate > 50000:
        print("  🎯 Hex approach achieves >50k entities/sec target!")
    if raw_rate > 100000:
        print("  🚀 Raw approach achieves >100k entities/sec target!")

    # Current progress toward 1M/sec target
    print("\nProgress toward 1M entities/sec target:")
    print(f"  PyBytes: {pybytes_rate/1000000*100:.1f}% of target")
    print(f"  Hex: {hex_rate/1000000*100:.1f}% of target")
    print(f"  Raw: {raw_rate/1000000*100:.1f}% of target")


if __name__ == "__main__":
    test_optimization_comparison()
