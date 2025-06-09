"""
Tests for EntityCollection functionality (single process entity resolution).
"""

import pytest
from entityframe import EntityCollection, EntityFrame


class TestEntityCollection:
    """Test the EntityCollection functionality (like pandas Series)."""

    def test_collection_creation(self):
        """Test basic EntityCollection creation."""
        collection = EntityCollection("splink")

        assert collection.process_name == "splink"
        assert collection.len() == 0
        assert collection.is_empty()
        assert collection.total_records() == 0

    def test_collection_creation_through_frame(self):
        """Test creating collections with entities through EntityFrame."""
        # Add entity data through frame (the proper way in simplified API)
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

        frame = EntityFrame()
        frame.add_method("splink", entity_data)
        collection = frame.get_collection("splink")

        assert collection.len() == 2
        assert not collection.is_empty()
        assert collection.total_records() == 6  # 3 + 3 records

        # Check that frame's interner was used
        assert frame.interner_size() >= 7  # customers, transactions + 5 record strings

    def test_get_entity_by_index(self):
        """Test retrieving individual entities by index."""
        frame = EntityFrame()
        entity_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"customers": ["cust_003", "cust_004"]},
        ]

        frame.add_method("splink", entity_data)
        collection = frame.get_collection("splink")

        # Test valid indices
        entity1 = collection.get_entity(0)
        assert entity1.total_records() == 2

        entity2 = collection.get_entity(1)
        assert entity2.total_records() == 2

        # Test invalid index
        with pytest.raises(IndexError):
            collection.get_entity(2)

    def test_get_all_entities(self):
        """Test retrieving all entities from collection."""
        frame = EntityFrame()
        entity_data = [
            {"customers": ["cust_001"]},
            {"customers": ["cust_002"]},
            {"customers": ["cust_003"]},
        ]

        frame.add_method("dedupe", entity_data)
        collection = frame.get_collection("dedupe")

        entities = collection.get_entities()
        assert len(entities) == 3

        for i, entity in enumerate(entities):
            assert entity.total_records() == 1
            # Test through frame since collections don't expose dataset checking in simplified API
            assert frame.entity_has_dataset("dedupe", i, "customers")

    def test_compare_collections(self):
        """Test comparing two collections with same number of entities."""
        frame = EntityFrame()

        # Add identical entities to both collections using frame
        identical_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"customers": ["cust_003", "cust_004"]},
        ]

        frame.add_method("splink", identical_data.copy())
        frame.add_method("dedupe", identical_data.copy())

        collection1 = frame.get_collection("splink")
        collection2 = frame.get_collection("dedupe")

        comparisons = collection1.compare_with(collection2)

        assert len(comparisons) == 2

        # Check first comparison
        comp1 = comparisons[0]
        assert comp1["entity_index"] == 0
        assert comp1["process1"] == "splink"
        assert comp1["process2"] == "dedupe"
        assert comp1["jaccard"] == 1.0  # Identical entities

        # Check second comparison
        comp2 = comparisons[1]
        assert comp2["entity_index"] == 1
        assert comp2["jaccard"] == 1.0  # Identical entities

    def test_compare_collections_partial_overlap(self):
        """Test comparing collections with partially overlapping entities."""
        frame = EntityFrame()

        # Add data with partial overlap
        data1 = [{"customers": ["cust_001", "cust_002", "cust_003"]}]
        data2 = [{"customers": ["cust_002", "cust_003", "cust_004"]}]

        frame.add_method("splink", data1)
        frame.add_method("dedupe", data2)

        collection1 = frame.get_collection("splink")
        collection2 = frame.get_collection("dedupe")

        comparisons = collection1.compare_with(collection2)

        assert len(comparisons) == 1

        # Intersection: {cust_002, cust_003}, Union: {cust_001, cust_002, cust_003, cust_004}
        # Jaccard = 2/4 = 0.5
        expected_jaccard = 2.0 / 4.0
        assert abs(comparisons[0]["jaccard"] - expected_jaccard) < 1e-10

    def test_compare_collections_different_sizes(self):
        """Test that comparing collections with different sizes raises error."""
        frame = EntityFrame()

        data1 = [{"customers": ["cust_001"]}, {"customers": ["cust_002"]}]
        data2 = [{"customers": ["cust_003"]}]  # Different size

        frame.add_method("splink", data1)
        frame.add_method("dedupe", data2)

        collection1 = frame.get_collection("splink")
        collection2 = frame.get_collection("dedupe")

        with pytest.raises(
            ValueError, match="Collections must have the same number of entities"
        ):
            collection1.compare_with(collection2)

    def test_empty_collection_operations(self):
        """Test operations on empty collections."""
        collection = EntityCollection("empty_process")

        assert collection.len() == 0
        assert collection.is_empty()
        assert collection.total_records() == 0
        assert collection.get_entities() == []

        with pytest.raises(IndexError):
            collection.get_entity(0)

    def test_process_name_immutability(self):
        """Test that process name is set at creation and accessible."""
        collection = EntityCollection("my_process")
        assert collection.process_name == "my_process"

        # Process name should be read-only (getter property)
        # We can't directly test immutability in Python, but the Rust implementation enforces it
