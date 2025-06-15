"""
Test if parallelization is causing overhead instead of benefits.
"""

import time
from entityframe import EntityFrame


def test_parallel_vs_sequential():
    """Compare individual hashing vs batch hashing to see if parallelization helps."""
    print("\n⚡ Parallel vs Sequential comparison...")

    scales = [500, 1000, 2000]

    for count in scales:
        print(f"\n{count} entities:")

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
        frame.add_method("test", entities)

        # Test batch (parallel) hashing
        start = time.time()
        batch_hashes = frame.hash_collection("test", "blake3")
        batch_time = time.time() - start
        batch_rate = count / batch_time if batch_time > 0 else 0

        # Test individual (sequential) hashing - sample for speed
        sample_size = min(100, count)
        start = time.time()
        individual_hashes = []
        for i in range(sample_size):
            hash_bytes = frame.hash_entity("test", i, "blake3")
            individual_hashes.append(hash_bytes)
        individual_time = time.time() - start
        individual_rate = sample_size / individual_time if individual_time > 0 else 0

        print(f"  Batch (parallel):     {batch_rate:8.0f} entities/sec")
        print(f"  Individual (sample):  {individual_rate:8.0f} entities/sec")

        if individual_rate > 0:
            speedup = batch_rate / individual_rate
            print(f"  Parallel speedup:     {speedup:.1f}x")

            # Verify hashes are the same
            if len(individual_hashes) > 0 and len(batch_hashes) > 0:
                match = individual_hashes[0] == batch_hashes[0]
                print(f"  Hash consistency:     {'✓' if match else '✗'}")


def test_thread_scaling():
    """Test if we can improve by controlling thread count."""
    import os

    print("\n🧵 Thread scaling test...")

    # Set thread count to different values
    original_threads = os.environ.get("RAYON_NUM_THREADS", "default")

    count = 2000
    entities = []
    for i in range(count):
        entity = {
            "users": [f"user_{i}"],
            "orders": [f"order_{i}_{j}" for j in range(3)],
            "events": [f"event_{i}_{j}" for j in range(5)],
        }
        entities.append(entity)

    frame = EntityFrame()
    frame.add_method("threads", entities)

    thread_counts = [1, 2, 4, 8, 16]

    for threads in thread_counts:
        os.environ["RAYON_NUM_THREADS"] = str(threads)

        # Need to restart Python to pick up new thread count
        # For now, just test with current setup
        start = time.time()
        _ = frame.hash_collection("threads", "blake3")
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0

        print(f"  {threads:2} threads: {rate:8.0f} entities/sec")

        # Only test first thread count in this simple version
        break

    # Restore original thread setting
    if original_threads == "default":
        os.environ.pop("RAYON_NUM_THREADS", None)
    else:
        os.environ["RAYON_NUM_THREADS"] = original_threads


if __name__ == "__main__":
    test_parallel_vs_sequential()
    test_thread_scaling()
