"""
Test to identify dataset ID scaling issue.
"""

from entityframe import EntityFrame


def test_dataset_scaling():
    """Test if the number of dataset IDs is causing quadratic scaling."""
    print("\n🔍 Testing dataset ID scaling hypothesis...")

    scales = [50, 100, 200, 500, 1000]

    print("\nDataset scaling analysis:")
    print("Entities | Datasets | Records | Est Iterations | Rate")
    print("-" * 60)

    for count in scales:
        # Create test data - each entity has unique dataset names
        entities = []
        for i in range(count):
            entity = {
                "users": [f"user_{i}"],
                "orders": [f"order_{i}_{j}" for j in range(3)],
                "events": [f"event_{i}_{j}" for j in range(5)],
            }
            entities.append(entity)

        frame = EntityFrame()
        frame.add_method("scaling_test", entities)

        # Get information about the frame
        dataset_count = len(frame.get_dataset_names())
        interner_size = frame.interner_size()

        # Estimate total iterations (entities × dataset_ids)
        estimated_iterations = count * dataset_count

        # Quick performance test
        import time

        start = time.time()
        hashes = frame.hash_collection_hex("scaling_test", "blake3")
        elapsed = time.time() - start
        rate = count / elapsed if elapsed > 0 else 0

        print(
            f"{count:8d} | {dataset_count:8d} | {interner_size:7d} | {estimated_iterations:14d} | {rate:8.0f}"
        )

        # Verify correctness
        assert len(hashes) == count

    print("\nKey insight:")
    print(
        "- If 'Est Iterations' grows quadratically with entities, that's our bottleneck"
    )
    print("- Each entity loops through ALL dataset IDs, even unused ones")
    print("- Performance should be inversely proportional to 'Est Iterations'")


def test_dataset_efficiency():
    """Test the efficiency of dataset lookup per entity."""
    print("\n📊 Testing dataset lookup efficiency...")

    # Create a test case where we know the dataset distribution
    entities = []
    for i in range(100):
        entity = {
            f"dataset_{i}": [f"record_{i}"],  # Each entity gets a unique dataset
        }
        entities.append(entity)

    frame = EntityFrame()
    frame.add_method("efficiency_test", entities)

    dataset_count = len(frame.get_dataset_names())
    interner_size = frame.interner_size()

    print("  100 entities with unique datasets:")
    print(f"  Total dataset IDs: {dataset_count}")
    print(f"  Total interner size: {interner_size}")
    print(f"  Each entity loops through {dataset_count} dataset IDs")
    print("  But each entity only uses 1 dataset ID")
    print(f"  Efficiency: 1/{dataset_count} = {1/dataset_count:.1%} useful iterations")

    # Quick performance test
    import time

    start = time.time()
    frame.hash_collection_hex("efficiency_test", "blake3")
    elapsed = time.time() - start
    rate = 100 / elapsed if elapsed > 0 else 0

    print(f"  Performance: {rate:.0f} entities/sec")


if __name__ == "__main__":
    test_dataset_scaling()
    test_dataset_efficiency()
