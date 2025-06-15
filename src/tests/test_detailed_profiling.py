"""
Detailed profiling tests using the new Rust profiling infrastructure.
"""

import time
from entityframe import EntityFrame


def test_detailed_profiling():
    """Test the new detailed profiling functionality."""
    print("\n🔬 Testing detailed profiling functionality...")

    # Create test data with varied complexity
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
    frame.add_method("profiling_test", entities)

    # Test the new profiling method
    print("\nRunning hash collection with detailed profiling:")
    start = time.time()
    hashes = frame.hash_collection_profiling("profiling_test", "blake3")
    elapsed = time.time() - start

    print("\nPython timing validation:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Hash count: {len(hashes)}")
    print(f"  Python-measured rate: {count/elapsed:.0f} entities/sec")


def test_profiling_comparison():
    """Compare profiling across different scales to identify scaling issues."""
    print("\n📊 Profiling comparison across scales...")

    scales = [250, 500, 1000, 2000]

    for count in scales:
        print(f"\n--- SCALE: {count} entities ---")

        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(5)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("scaling_profiling", entities)

        # Run with profiling
        _ = frame.hash_collection_profiling("scaling_profiling", "blake3")


def test_algorithm_profiling():
    """Profile different hash algorithms to identify best performer."""
    print("\n🧪 Algorithm profiling comparison...")

    count = 1000
    entities = []
    for i in range(count):
        entity = {
            "users": [f"user_{i}"],
            "orders": [f"order_{i}_{j}" for j in range(2)],
        }
        entities.append(entity)

    algorithms = ["blake3", "sha256", "sha3-256"]

    for algo in algorithms:
        print(f"\n--- ALGORITHM: {algo} ---")

        frame = EntityFrame()
        frame.add_method("algo_profiling", entities)

        _ = frame.hash_collection_profiling("algo_profiling", algo)


if __name__ == "__main__":
    test_detailed_profiling()
    test_profiling_comparison()
    test_algorithm_profiling()
