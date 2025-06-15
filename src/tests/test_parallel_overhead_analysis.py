"""
Test to identify parallel processing overhead.
"""

import time
from entityframe import EntityFrame


def test_parallel_overhead():
    """Test if parallel processing is actually hurting performance."""
    print("\n🔄 Testing parallel processing overhead hypothesis...")

    # Test different sizes to see where parallelization kicks in
    scales = [50, 100, 200, 500, 800, 999, 1000, 1001, 1500, 2000]

    print("\nPerformance across scales (looking for threshold effects):")
    print("Entities | Rate (entities/sec) | Time (ms) | Threading")
    print("-" * 55)

    for count in scales:
        # Create test data
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(2)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("parallel_test", entities)

        # Time the operation
        start = time.time()
        hashes = frame.hash_collection_hex("parallel_test", "blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0
        threading_mode = "SINGLE" if count < 1000 else "PARALLEL"

        print(f"{count:8d} | {rate:13.0f} | {elapsed*1000:8.1f} | {threading_mode}")

        # Verify correctness
        assert len(hashes) == count

    print("\nKey observations:")
    print(
        "- Performance should drop significantly at 1000+ entities if parallelization is the issue"
    )
    print("- Single-threaded (< 1000) should be much faster than parallel (>= 1000)")


if __name__ == "__main__":
    test_parallel_overhead()
