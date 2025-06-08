"""
Tests for core EntityFrame functionality.
"""

import pytest
from entityframe import StringInterner, Entity, EntityCollection


class TestStringInterner:
    """Test the string interning functionality."""

    def test_basic_interning(self):
        """Test basic string interning operations."""
        interner = StringInterner()

        # Test initial state
        assert interner.is_empty()
        assert interner.len() == 0

        # Test interning
        id1 = interner.intern("hello")
        id2 = interner.intern("world")
        id3 = interner.intern("hello")  # Should return same ID

        assert id1 == 0
        assert id2 == 1
        assert id3 == 0  # Same as id1

        assert interner.len() == 2
        assert not interner.is_empty()

    def test_string_retrieval(self):
        """Test retrieving strings by ID."""
        interner = StringInterner()

        # Intern some strings
        id1 = interner.intern("customer_001")
        id2 = interner.intern("transaction_555")

        # Retrieve strings
        assert interner.get_string(id1) == "customer_001"
        assert interner.get_string(id2) == "transaction_555"

        # Test invalid ID
        with pytest.raises(IndexError):
            interner.get_string(999)

    def test_deduplication(self):
        """Test that identical strings are deduplicated."""
        interner = StringInterner()

        # Intern same string multiple times
        ids = [interner.intern("duplicate") for _ in range(10)]

        # All should return the same ID
        assert all(id == ids[0] for id in ids)
        assert interner.len() == 1


class TestEntity:
    """Test the Entity functionality."""

    def test_entity_creation(self):
        """Test basic entity creation and record addition."""
        entity = Entity()

        # Test adding individual records
        entity.add_record("customers", 1)
        entity.add_record("customers", 2)
        entity.add_record("transactions", 10)

        # Test adding multiple records
        entity.add_records("transactions", [11, 12, 13])

        # Test retrieval
        customers = entity.get_records("customers")
        transactions = entity.get_records("transactions")

        assert set(customers) == {1, 2}
        assert set(transactions) == {10, 11, 12, 13}

    def test_dataset_operations(self):
        """Test dataset-related operations."""
        entity = Entity()

        entity.add_records("customers", [1, 2, 3])
        entity.add_records("transactions", [10, 11])

        # Test dataset listing
        datasets = entity.get_datasets()
        assert set(datasets) == {"customers", "transactions"}

        # Test dataset checking
        assert entity.has_dataset("customers")
        assert entity.has_dataset("transactions")
        assert not entity.has_dataset("addresses")

        # Test total records
        assert entity.total_records() == 5

    def test_empty_dataset_retrieval(self):
        """Test retrieving records from non-existent datasets."""
        entity = Entity()

        # Should return empty list for non-existent dataset
        records = entity.get_records("nonexistent")
        assert records == []

    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical entities."""
        entity1 = Entity()
        entity1.add_records("customers", [1, 2, 3])
        entity1.add_records("transactions", [10, 11])

        entity2 = Entity()
        entity2.add_records("customers", [1, 2, 3])
        entity2.add_records("transactions", [10, 11])

        similarity = entity1.jaccard_similarity(entity2)
        assert similarity == 1.0

    def test_jaccard_similarity_disjoint(self):
        """Test Jaccard similarity for completely disjoint entities."""
        entity1 = Entity()
        entity1.add_records("customers", [1, 2, 3])

        entity2 = Entity()
        entity2.add_records("customers", [4, 5, 6])

        similarity = entity1.jaccard_similarity(entity2)
        assert similarity == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity for partially overlapping entities."""
        entity1 = Entity()
        entity1.add_records("customers", [1, 2, 3])
        entity1.add_records("transactions", [10, 11])

        entity2 = Entity()
        entity2.add_records("customers", [2, 3, 4])
        entity2.add_records("transactions", [11, 12])

        # Intersection: customers {2, 3}, transactions {11} = 3 total
        # Union: customers {1, 2, 3, 4}, transactions {10, 11, 12} = 7 total
        # Jaccard = 3/7
        similarity = entity1.jaccard_similarity(entity2)
        expected = 3.0 / 7.0
        assert abs(similarity - expected) < 1e-10

    def test_jaccard_similarity_different_datasets(self):
        """Test Jaccard similarity for entities with different datasets."""
        entity1 = Entity()
        entity1.add_records("customers", [1, 2, 3])

        entity2 = Entity()
        entity2.add_records("transactions", [1, 2, 3])

        # No overlap in datasets, so Jaccard = 0
        similarity = entity1.jaccard_similarity(entity2)
        assert similarity == 0.0

    def test_jaccard_similarity_empty_entities(self):
        """Test Jaccard similarity for empty entities."""
        entity1 = Entity()
        entity2 = Entity()

        similarity = entity1.jaccard_similarity(entity2)
        assert similarity == 1.0  # Both empty, so considered identical


class TestEntityCollection:
    """Test the EntityCollection high-level API."""

    def test_entity_collection_creation(self):
        """Test basic EntityCollection operations."""
        collection = EntityCollection()

        assert collection.get_method_names() == []
        assert collection.get_entities("nonexistent") is None

    def test_add_method(self):
        """Test adding method results to collection."""
        collection = EntityCollection()

        # Create some test entity data
        splink_data = [
            {
                "customers": ["cust_001", "cust_002"],
                "transactions": ["txn_100", "txn_101"],
            },
            {
                "customers": ["cust_003"],
                "transactions": ["txn_102", "txn_103", "txn_104"],
            },
        ]

        collection.add_method("splink", splink_data)

        # Check that method was added
        assert "splink" in collection.get_method_names()

        entities = collection.get_entities("splink")
        assert entities is not None
        assert len(entities) == 2

        # Check first entity
        entity1 = entities[0]
        assert entity1.has_dataset("customers")
        assert entity1.has_dataset("transactions")
        assert entity1.total_records() == 4

        # Check second entity
        entity2 = entities[1]
        assert entity2.total_records() == 4

    def test_compare_methods(self):
        """Test comparing two methods."""
        collection = EntityCollection()

        # Add two methods with identical data
        method1_data = [
            {"customers": ["cust_001", "cust_002"]},
            {"customers": ["cust_003", "cust_004"]},
        ]

        method2_data = [
            {"customers": ["cust_001", "cust_002"]},  # Identical
            {"customers": ["cust_003", "cust_005"]},  # Partial overlap
        ]

        collection.add_method("method1", method1_data)
        collection.add_method("method2", method2_data)

        # Compare methods
        comparison = collection.compare_methods("method1", "method2")

        assert len(comparison) == 2

        # First entity should be identical
        assert comparison[0]["jaccard"] == 1.0
        assert comparison[0]["method1"] == "method1"
        assert comparison[0]["method2"] == "method2"

        # Second entity should have partial overlap (2/3 = 0.666...)
        expected_jaccard = (
            1.0 / 3.0
        )  # Intersection: {cust_003}, Union: {cust_003, cust_004, cust_005}
        assert abs(comparison[1]["jaccard"] - expected_jaccard) < 1e-10

    def test_string_interner_reuse(self):
        """Test that string interner is reused across entities."""
        collection = EntityCollection()

        # Add methods with overlapping record IDs
        method1_data = [{"customers": ["cust_001", "cust_002"]}]
        method2_data = [{"customers": ["cust_001", "cust_003"]}]

        collection.add_method("method1", method1_data)
        collection.add_method("method2", method2_data)

        # The interner should have deduplicated "cust_001"
        assert collection.interner.len() == 3  # cust_001, cust_002, cust_003

        # Verify the strings are interned correctly
        assert collection.interner.get_string(0) in {"cust_001", "cust_002", "cust_003"}
        assert collection.interner.get_string(1) in {"cust_001", "cust_002", "cust_003"}
        assert collection.interner.get_string(2) in {"cust_001", "cust_002", "cust_003"}
