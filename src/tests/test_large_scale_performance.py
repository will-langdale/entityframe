"""
Test large-scale performance with the optimization.
"""

import time
from entityframe import EntityFrame


def test_large_scale_performance():
    """Test performance at larger scales to approach 1M entities/sec."""
    print("\n🎯 Testing large-scale performance toward 1M entities/sec target...")

    scales = [1000, 2000, 5000, 10000, 20000]

    print("\nLarge-scale performance test:")
    print("Entities | Rate (entities/sec) | Time (ms) | Progress to 1M")
    print("-" * 65)

    for count in scales:
        # Create test data
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(2)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("large_scale_test", entities)

        # Time the operation using hex output for best performance
        start = time.time()
        hashes = frame.hash_collection_hex("large_scale_test", "blake3")
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0
        progress = rate / 1_000_000 * 100

        print(f"{count:8d} | {rate:15.0f} | {elapsed*1000:8.1f} | {progress:8.1f}%")

        # Verify correctness
        assert len(hashes) == count

        # Stop if we're taking too long per test
        if elapsed > 10:
            print(f"   (Stopping here - test took {elapsed:.1f}s)")
            break

    print("\nKey insights:")
    print("- Looking for linear or constant scaling with entity count")
    print("- Target: 1,000,000 entities/sec")
    print("- Current best appears to be ~250k+ entities/sec")


def test_algorithm_comparison_optimized():
    """Test different algorithms with the optimization."""
    print("\n🧪 Testing algorithm performance with optimization...")

    count = 5000
    entities = []
    for i in range(count):
        entity = {
            "users": [f"user_{i}"],
            "orders": [f"order_{i}_{j}" for j in range(2)],
        }
        entities.append(entity)

    algorithms = ["blake3", "sha256", "sha512", "sha3-256"]

    print(f"\nAlgorithm comparison with {count} entities:")
    print("Algorithm | Rate (entities/sec) | Time (ms)")
    print("-" * 45)

    for algorithm in algorithms:
        frame = EntityFrame()
        frame.add_method("algo_test", entities)

        start = time.time()
        hashes = frame.hash_collection_hex("algo_test", algorithm)
        elapsed = time.time() - start

        rate = count / elapsed if elapsed > 0 else 0

        print(f"{algorithm:9s} | {rate:15.0f} | {elapsed*1000:8.1f}")

        # Verify correctness
        assert len(hashes) == count


if __name__ == "__main__":
    test_large_scale_performance()
    test_algorithm_comparison_optimized()
