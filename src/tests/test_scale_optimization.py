"""
Test performance optimization across different scales.
"""

import time
from entityframe import EntityFrame


def test_scale_optimization():
    """Test optimization effectiveness across different entity counts."""
    print("\n📈 Testing optimization across different scales...")

    scales = [50, 100, 500, 1000, 2000]

    for count in scales:
        print(f"\n--- SCALE: {count} entities ---")

        # Create test data
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(5)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("scale_test", entities)

        # Test PyBytes approach
        start = time.time()
        pybytes_hashes = frame.hash_collection("scale_test", "blake3")
        pybytes_time = time.time() - start
        pybytes_rate = count / pybytes_time if pybytes_time > 0 else 0

        # Test Hex approach
        start = time.time()
        hex_hashes = frame.hash_collection_hex("scale_test", "blake3")
        hex_time = time.time() - start
        hex_rate = count / hex_time if hex_time > 0 else 0

        # Test Raw bytes approach
        start = time.time()
        raw_bytes = frame.hash_collection_raw("scale_test", "blake3")
        raw_time = time.time() - start
        raw_rate = count / raw_time if raw_time > 0 else 0

        # Calculate improvement ratios
        hex_improvement = hex_rate / pybytes_rate if pybytes_rate > 0 else 0
        raw_improvement = raw_rate / pybytes_rate if pybytes_rate > 0 else 0

        print(f"  PyBytes: {pybytes_rate:8.0f} entities/sec ({pybytes_time:.3f}s)")
        print(
            f"  Hex:     {hex_rate:8.0f} entities/sec ({hex_time:.3f}s) - {hex_improvement:.1f}x"
        )
        print(
            f"  Raw:     {raw_rate:8.0f} entities/sec ({raw_time:.3f}s) - {raw_improvement:.1f}x"
        )

        # Verify correctness
        assert len(pybytes_hashes) == count
        assert len(hex_hashes) == count

        # Quick check for raw bytes format
        raw_data = bytes(raw_bytes)
        assert len(raw_data) > count * 30  # Should have at least hash size * count


if __name__ == "__main__":
    test_scale_optimization()
