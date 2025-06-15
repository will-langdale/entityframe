"""
Tests for EntityFrame functionality (multi-collection container).
"""

import pytest
from entityframe import StringInterner, EntityCollection, EntityFrame


class TestEntityFrame:
    """Test the EntityFrame functionality (like pandas DataFrame)."""

    def test_frame_creation(self):
        """Test basic EntityFrame creation."""
        frame = EntityFrame()

        assert frame.collection_count() == 0
        assert frame.total_entities() == 0
        assert frame.get_collection_names() == []
        assert frame.interner_size() == 0

    def test_add_collection_manually(self):
        """Test manually creating and adding an empty collection to the frame."""
        frame = EntityFrame()
        collection = EntityCollection("splink")

        frame.add_collection("splink", collection)

        assert frame.collection_count() == 1
        assert frame.total_entities() == 0  # Empty collection
        assert "splink" in frame.get_collection_names()

        retrieved_collection = frame.get_collection("splink")
        assert retrieved_collection is not None
        assert retrieved_collection.len() == 0
        assert retrieved_collection.process_name == "splink"

    def test_add_method_convenience(self):
        """Test the convenience method for adding entity data directly."""
        frame = EntityFrame()

        entity_data = [
            {
                "customers": ["cust_001", "cust_002"],
                "transactions": ["txn_100"],
            },
            {
                "customers": ["cust_003"],
                "transactions": ["txn_101", "txn_102"],
            },
        ]

        frame.add_method("splink", entity_data)

        assert frame.collection_count() == 1
        assert frame.total_entities() == 2
        # Interner: customers, transactions, cust_001, cust_002, cust_003, txn_100, txn_101, txn_102 (8 items)
        assert frame.interner_size() == 8

        collection = frame.get_collection("splink")
        assert collection is not None
        assert collection.len() == 2
        assert collection.process_name == "splink"

    def test_shared_interner_across_collections(self):
        """Test that collections share the frame's interner for efficiency."""
        frame = EntityFrame()

        # Add two methods with overlapping record IDs
        method1_data = [{"customers": ["cust_001", "cust_002"]}]
        method2_data = [{"customers": ["cust_001", "cust_003"]}]  # cust_001 overlaps

        frame.add_method("splink", method1_data)
        frame.add_method("dedupe", method2_data)

        # The interner should have deduplicated "cust_001"
        assert frame.interner_size() == 4  # customers, cust_001, cust_002, cust_003
        assert frame.collection_count() == 2
        assert frame.total_entities() == 2

    def test_compare_collections(self):
        """Test comparing two collections within a frame."""
        frame = EntityFrame()

        # Add two methods with known similarity
        method1_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"customers": ["cust_003", "cust_004"]},
        ]

        method2_data = [
            {"customers": ["cust_001", "cust_002"]},  # Identical
            {"customers": ["cust_003", "cust_005"]},  # Partial overlap
        ]

        frame.add_method("splink", method1_data)
        frame.add_method("dedupe", method2_data)

        comparisons = frame.compare_collections("splink", "dedupe")

        assert len(comparisons) == 2

        # First entity should be identical
        assert comparisons[0]["jaccard"] == 1.0
        assert comparisons[0]["process1"] == "splink"
        assert comparisons[0]["process2"] == "dedupe"

        # Second entity should have partial overlap: {cust_003} / {cust_003, cust_004, cust_005} = 1/3
        expected_jaccard = 1.0 / 3.0
        assert abs(comparisons[1]["jaccard"] - expected_jaccard) < 1e-10

    def test_compare_nonexistent_collections(self):
        """Test error handling when comparing nonexistent collections."""
        frame = EntityFrame()
        frame.add_method("splink", [{"customers": ["cust_001"]}])

        # Try to compare with nonexistent collection
        with pytest.raises(KeyError, match="Collection 'nonexistent' not found"):
            frame.compare_collections("splink", "nonexistent")

        with pytest.raises(KeyError, match="Collection 'also_missing' not found"):
            frame.compare_collections("also_missing", "splink")

    def test_get_nonexistent_collection(self):
        """Test that getting a nonexistent collection returns None."""
        frame = EntityFrame()

        result = frame.get_collection("nonexistent")
        assert result is None

    def test_multiple_collections_independence(self):
        """Test that multiple collections in a frame are independent."""
        frame = EntityFrame()

        # Add different types of entity data
        splink_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"transactions": ["txn_100", "txn_101"]},
        ]

        dedupe_data = [
            {"customers": ["cust_003"]},
            {"addresses": ["addr_001", "addr_002", "addr_003"]},
        ]

        frame.add_method("splink", splink_data)
        frame.add_method("dedupe", dedupe_data)

        assert frame.collection_count() == 2
        assert frame.total_entities() == 4  # 2 + 2

        # Check splink collection
        splink_collection = frame.get_collection("splink")
        assert splink_collection.len() == 2
        assert splink_collection.process_name == "splink"

        # Check dedupe collection
        dedupe_collection = frame.get_collection("dedupe")
        assert dedupe_collection.len() == 2
        assert dedupe_collection.process_name == "dedupe"

        # Collections should be independent - test using Frame methods
        # Splink entities should have different datasets than dedupe
        assert frame.entity_has_dataset("splink", 0, "customers")
        assert frame.entity_has_dataset("splink", 1, "transactions")
        assert frame.entity_has_dataset("dedupe", 0, "customers")
        assert frame.entity_has_dataset("dedupe", 1, "addresses")

    def test_interner_property_access(self):
        """Test accessing the frame's simplified interner."""
        frame = EntityFrame()

        interner = frame.interner
        assert isinstance(interner, StringInterner)
        assert interner.len() == 0

        # Add some data and check interner grows
        frame.add_method("splink", [{"customers": ["cust_001"]}])
        assert frame.interner_size() == 2  # "customers" + "cust_001"

    def test_frame_with_empty_collections(self):
        """Test frame behaviour with empty collections."""
        frame = EntityFrame()
        empty_collection = EntityCollection("empty_process")

        frame.add_collection("empty", empty_collection)

        assert frame.collection_count() == 1
        assert frame.total_entities() == 0  # Empty collection contributes 0 entities

        retrieved = frame.get_collection("empty")
        assert retrieved is not None
        assert retrieved.len() == 0
        assert retrieved.is_empty()

    def test_frame_statistics(self):
        """Test various statistical methods on the frame."""
        frame = EntityFrame()

        # Add multiple collections with different sizes
        frame.add_method(
            "method1", [{"customers": ["c1"]}, {"customers": ["c2"]}]
        )  # 2 entities
        frame.add_method("method2", [{"customers": ["c3"]}])  # 1 entity
        frame.add_method("method3", [])  # 0 entities

        assert frame.collection_count() == 3
        assert frame.total_entities() == 3  # 2 + 1 + 0
        assert frame.interner_size() == 4  # customers, c1, c2, c3

        assert sorted(frame.get_collection_names()) == ["method1", "method2", "method3"]

    def test_frame_dataset_declaration_and_usage(self):
        """Test that datasets can be declared and used correctly."""

        frame = EntityFrame()

        # Declare some datasets upfront
        frame.declare_dataset("dataset_a")
        frame.declare_dataset("dataset_b")

        # Add a method with those datasets
        frame.add_method(
            "test_method",
            [
                {"dataset_a": ["rec1", "rec2"], "dataset_b": ["rec3", "rec4"]},
                {
                    "dataset_a": ["rec5"],
                    "dataset_c": ["rec6", "rec7"],  # This dataset gets auto-declared
                },
            ],
        )

        # Verify the collection was added
        assert "test_method" in frame.get_collection_names()

        # Get the collection back and verify it works correctly
        retrieved_collection = frame.get_collection("test_method")
        assert retrieved_collection is not None
        assert retrieved_collection.len() == 2

        # Check that entities can be accessed with the frame's dataset names
        assert frame.entity_has_dataset("test_method", 0, "dataset_a")
        assert frame.entity_has_dataset("test_method", 0, "dataset_b")
        assert frame.entity_has_dataset("test_method", 1, "dataset_a")
        assert frame.entity_has_dataset("test_method", 1, "dataset_c")

        # Verify all datasets are known to the frame
        dataset_names = frame.get_dataset_names()
        assert "dataset_a" in dataset_names
        assert "dataset_b" in dataset_names
        assert "dataset_c" in dataset_names

    def test_multiple_methods_share_datasets(self):
        """Test that multiple methods can share dataset names efficiently."""

        frame = EntityFrame()

        # Add first method
        frame.add_method(
            "method1", [{"dataset_a": ["rec1", "rec2"], "dataset_b": ["rec3"]}]
        )

        # Add second method with overlapping dataset names
        frame.add_method(
            "method2",
            [
                {
                    "dataset_a": ["rec4", "rec5"],  # Same dataset name
                    "dataset_c": ["rec6"],  # New dataset
                }
            ],
        )

        # Verify both collections exist
        assert set(frame.get_collection_names()) == {"method1", "method2"}

        # Verify datasets are properly registered
        assert "dataset_a" in frame.get_dataset_names()
        assert "dataset_b" in frame.get_dataset_names()
        assert "dataset_c" in frame.get_dataset_names()

        # Verify we can compare the collections
        comparisons = frame.compare_collections("method1", "method2")
        assert len(comparisons) == 1
        assert comparisons[0]["jaccard"] == 0.0  # No overlap in records

    def test_add_method_creates_collection_with_entities(self):
        """Test that add_method creates a collection with the correct entities."""

        frame = EntityFrame()

        # Add a method with entity data
        frame.add_method("test", [{"data": ["id1", "id2", "id3"]}])

        # Verify the collection was created
        assert frame.collection_count() == 1
        assert "test" in frame.get_collection_names()

        # Verify the entity has the correct number of records
        retrieved = frame.get_collection("test")
        entity = retrieved.get_entity(0)
        assert entity.total_records() == 3

        # Verify interner has dataset name + records
        assert frame.interner_size() == 4  # data, id1, id2, id3

    def test_multiple_methods_with_diverse_datasets(self):
        """Test multiple methods with different dataset combinations."""

        # Create frame and pre-declare datasets for efficiency
        frame = EntityFrame()
        frame.declare_dataset("customers")
        frame.declare_dataset("orders")
        frame.declare_dataset("products")

        # Add first method
        frame.add_method(
            "method1",
            [
                {"customers": ["c1", "c2"], "orders": ["o1"]},
                {"customers": ["c3"], "products": ["p1", "p2"]},
            ],
        )

        # Add second method with different dataset combinations
        frame.add_method(
            "method2",
            [
                {"customers": ["c4", "c5"], "products": ["p3"]},
                {"orders": ["o2", "o3"], "products": ["p4"]},
            ],
        )

        # Verify both work correctly
        assert len(frame.get_collection_names()) == 2

        # Compare them
        comparisons = frame.compare_collections("method1", "method2")
        assert len(comparisons) == 2

        # Entity 0 comparison:
        # method1: {"customers": ["c1", "c2"], "orders": ["o1"]}
        # method2: {"customers": ["c4", "c5"], "products": ["p3"]}
        # No record overlap, so Jaccard should be 0.0

        # Entity 1 comparison:
        # method1: {"customers": ["c3"], "products": ["p1", "p2"]}
        # method2: {"orders": ["o2", "o3"], "products": ["p4"]}
        # No record overlap, so Jaccard should be 0.0

        # Both should be 0.0 since record IDs are different even though datasets overlap
        assert all(comp["jaccard"] == 0.0 for comp in comparisons)
