"""
Tests for core EntityFrame components: StringInterner and Entity.
"""

import pytest
from entityframe import StringInterner, Entity


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

    def test_large_scale_interning(self):
        """Test interning performance with many strings."""
        interner = StringInterner()

        # Intern 1000 unique strings
        ids = []
        for i in range(1000):
            id = interner.intern(f"string_{i}")
            ids.append(id)

        assert interner.len() == 1000
        assert len(set(ids)) == 1000  # All IDs should be unique

        # Test retrieval
        for i, id in enumerate(ids):
            assert interner.get_string(id) == f"string_{i}"


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

    def test_duplicate_record_handling(self):
        """Test that duplicate records are handled correctly by roaring bitmaps."""
        entity = Entity()

        # Add the same record multiple times
        entity.add_record("customers", 1)
        entity.add_record("customers", 1)
        entity.add_record("customers", 1)

        customers = entity.get_records("customers")
        assert customers == [1]  # Should only appear once
        assert entity.total_records() == 1

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

    def test_entity_with_multiple_datasets(self):
        """Test entity behavior with multiple diverse datasets."""
        entity = Entity()

        # Add records to various datasets
        entity.add_records("customers", [100, 101, 102])
        entity.add_records("transactions", [200, 201])
        entity.add_records("addresses", [300])
        entity.add_records("phone_numbers", [400, 401, 402, 403])

        assert entity.total_records() == 10
        assert len(entity.get_datasets()) == 4
        assert entity.has_dataset("customers")
        assert entity.has_dataset("transactions")
        assert entity.has_dataset("addresses")
        assert entity.has_dataset("phone_numbers")

        # Check individual datasets
        assert len(entity.get_records("customers")) == 3
        assert len(entity.get_records("transactions")) == 2
        assert len(entity.get_records("addresses")) == 1
        assert len(entity.get_records("phone_numbers")) == 4

    def test_entity_large_scale(self):
        """Test entity performance with large numbers of records."""
        entity = Entity()

        # Add 10,000 records to a dataset
        large_record_set = list(range(10000))
        entity.add_records("large_dataset", large_record_set)

        assert entity.total_records() == 10000
        assert len(entity.get_records("large_dataset")) == 10000
        assert entity.has_dataset("large_dataset")
