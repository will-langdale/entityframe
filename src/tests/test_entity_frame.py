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
        """Test manually creating and adding a collection to the frame."""
        frame = EntityFrame()
        collection = EntityCollection("splink")
        interner = frame.interner  # Get the frame's interner

        entity_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"customers": ["cust_003"]},
        ]

        collection.add_entities(entity_data, interner)
        frame.add_collection("splink", collection)

        assert frame.collection_count() == 1
        assert frame.total_entities() == 2
        assert "splink" in frame.get_collection_names()

        retrieved_collection = frame.get_collection("splink")
        assert retrieved_collection is not None
        assert retrieved_collection.len() == 2
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
        assert (
            frame.interner_size() == 6
        )  # cust_001, cust_002, cust_003, txn_100, txn_101, txn_102

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
        assert frame.interner_size() == 3  # cust_001, cust_002, cust_003
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

        # Collections should be independent
        splink_entities = splink_collection.get_entities()
        dedupe_entities = dedupe_collection.get_entities()

        # Splink entities should have different datasets than dedupe
        assert splink_entities[0].has_dataset("customers")
        assert splink_entities[1].has_dataset("transactions")
        assert dedupe_entities[0].has_dataset("customers")
        assert dedupe_entities[1].has_dataset("addresses")

    def test_legacy_method_compatibility(self):
        """Test that legacy methods still work for backward compatibility."""
        frame = EntityFrame()

        entity_data = [{"customers": ["cust_001", "cust_002"]}]
        frame.add_method("splink", entity_data)

        # Test legacy method aliases
        assert frame.get_method_names() == frame.get_collection_names()
        assert frame.method_count() == frame.collection_count()
        assert frame.get_entities("splink") is not None
        assert frame.compare_methods("splink", "splink") == frame.compare_collections(
            "splink", "splink"
        )

    def test_interner_property_access(self):
        """Test accessing the frame's interner."""
        frame = EntityFrame()

        interner = frame.interner
        assert isinstance(interner, StringInterner)
        assert interner.len() == 0

        # Add some data and check interner grows
        frame.add_method("splink", [{"customers": ["cust_001"]}])
        assert frame.interner_size() == 1

    def test_frame_with_empty_collections(self):
        """Test frame behavior with empty collections."""
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
        assert frame.interner_size() == 3  # c1, c2, c3

        assert sorted(frame.get_collection_names()) == ["method1", "method2", "method3"]
