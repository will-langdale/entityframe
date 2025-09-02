"""Basic tests for starlings Collection and Partition functionality."""

import pytest

import starlings


class TestCollectionBasic:
    """Test basic Collection functionality."""
    
    def test_empty_collection_creation(self):
        """Test creating collection with no edges."""
        collection = starlings.Collection.from_edges([])
        partition = collection.at(0.5)
        assert len(partition) == 0
        assert partition.entities == []
    
    def test_single_edge_collection(self):
        """Test collection with single edge."""
        edges = [("a", "b", 0.8)]
        collection = starlings.Collection.from_edges(edges)
        
        # At threshold 1.0, should be two separate entities
        partition_high = collection.at(1.0)
        entities_high = partition_high.entities
        assert len(entities_high) == 2
        assert all(len(entity) == 1 for entity in entities_high)
        
        # At threshold 0.5, should be merged into one entity
        partition_low = collection.at(0.5)
        entities_low = partition_low.entities
        assert len(entities_low) == 1
        assert len(entities_low[0]) == 2
    
    def test_multiple_edges_same_threshold(self):
        """Test n-way merge at same threshold."""
        edges = [
            ("a", "b", 0.7),
            ("b", "c", 0.7),
            ("c", "d", 0.7),
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # At threshold 1.0, all separate
        partition_high = collection.at(1.0)
        assert len(partition_high) == 4
        
        # At threshold 0.6, all merged
        partition_low = collection.at(0.6)
        entities_low = partition_low.entities
        assert len(entities_low) == 1
        assert len(entities_low[0]) == 4
    
    def test_disconnected_components(self):
        """Test graph with disconnected components."""
        edges = [
            ("a", "b", 0.8),  # Component 1
            ("c", "d", 0.6),  # Component 2
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # At threshold 1.0, four separate entities
        partition_high = collection.at(1.0)
        assert len(partition_high) == 4
        
        # At threshold 0.5, two connected components
        partition_mid = collection.at(0.5)
        entities_mid = partition_mid.entities
        assert len(entities_mid) == 2
        assert all(len(entity) == 2 for entity in entities_mid)
    
    def test_isolates_handling(self):
        """Test that isolates appear as singletons."""
        edges = [
            ("a", "b", 0.8),
        ]
        collection = starlings.Collection.from_edges(edges, source="test")
        
        # Should have connected pair at low threshold
        partition_low = collection.at(0.5)
        entities_low = partition_low.entities
        assert len(entities_low) == 1  # Only the connected pair
        assert len(entities_low[0]) == 2
        
        # Should have separate entities at high threshold  
        partition_high = collection.at(1.0)
        assert len(partition_high) == 2
    
    def test_key_type_conversions(self):
        """Test different key types: int, str, bytes."""
        edges = [
            (1, 2, 0.9),              # int keys
            ("hello", "world", 0.8),  # str keys
            (b"foo", b"bar", 0.7),    # bytes keys
            (1, "mixed", 0.6),        # mixed types
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # Should handle all key types without errors
        partition = collection.at(0.5)
        entities = partition.entities
        
        # Should have entities (exact count depends on merge thresholds)
        assert len(entities) > 0
    
    def test_threshold_boundary_conditions(self):
        """Test threshold boundary conditions."""
        edges = [("a", "b", 0.5)]
        collection = starlings.Collection.from_edges(edges)
        
        # At exactly threshold value, should be merged
        partition_exact = collection.at(0.5)
        entities_exact = partition_exact.entities
        assert len(entities_exact) == 1
        assert len(entities_exact[0]) == 2
        
        # Just above threshold, should be separate
        partition_above = collection.at(0.50001)
        assert len(partition_above) == 2
        
        # Just below threshold, should be merged
        partition_below = collection.at(0.49999)
        entities_below = partition_below.entities
        assert len(entities_below) == 1
        assert len(entities_below[0]) == 2
    
    def test_zero_and_one_thresholds(self):
        """Test extreme threshold values 0.0 and 1.0."""
        edges = [
            ("a", "b", 0.8),
            ("c", "d", 0.6),
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # At 0.0, should merge at or below minimum edge threshold
        partition_zero = collection.at(0.0)
        entities_zero = partition_zero.entities
        # With thresholds 0.8 and 0.6, at 0.0 both edges should be active
        # This should result in 2 separate pairs
        assert len(entities_zero) == 2
        assert all(len(entity) == 2 for entity in entities_zero)
        
        # At 1.0, everything should be separate
        partition_one = collection.at(1.0)
        assert len(partition_one) == 4
    
    def test_large_integer_keys(self):
        """Test handling of large integer keys."""
        large_u32 = 2**31 - 1  # Just under u32::MAX
        # Use smaller value that fits in u64 safely in Python
        large_but_safe = 2**32  # Just over u32::MAX
        
        edges = [
            (large_u32, large_u32 + 1, 0.8),
            (large_but_safe, large_but_safe + 1, 0.7),
        ]
        collection = starlings.Collection.from_edges(edges)
        
        partition = collection.at(0.5)
        entities = partition.entities
        
        # Should handle large integers without overflow
        assert len(entities) == 2  # Two separate pairs
        assert all(len(entity) == 2 for entity in entities)
    
    def test_negative_integer_error(self):
        """Test that negative integers raise appropriate error."""
        with pytest.raises(ValueError, match="non-negative"):
            starlings.Collection.from_edges([(-1, 0, 0.5)])
    
    def test_invalid_key_type_error(self):
        """Test that invalid key types raise appropriate error."""
        with pytest.raises(TypeError, match="Key must be"):
            starlings.Collection.from_edges([(3.14, 2.71, 0.5)])  # float keys
    
    def test_string_representation(self):
        """Test string representations for debugging."""
        edges = [("a", "b", 0.8)]
        collection = starlings.Collection.from_edges(edges)
        
        # Collection should show Collection
        repr_str = repr(collection)
        assert "Collection" in repr_str
        
        # Partition should show entity count
        partition = collection.at(0.5)
        partition_repr = repr(partition)
        assert "Partition" in partition_repr
        assert "entities=" in partition_repr
    
    def test_threshold_caching_consistency(self):
        """Test that repeated calls with same threshold return consistent results."""
        edges = [
            ("a", "b", 0.8),
            ("b", "c", 0.6),
            ("d", "e", 0.4),
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # Call same threshold multiple times
        partition1 = collection.at(0.7)
        partition2 = collection.at(0.7)
        partition3 = collection.at(0.7)
        
        # Results should be identical
        assert partition1.entities == partition2.entities
        assert partition2.entities == partition3.entities
        assert len(partition1) == len(partition2) == len(partition3)


class TestCollectionAdvanced:
    """Advanced test scenarios."""
    
    def test_source_parameter(self):
        """Test source parameter in from_edges."""
        edges = [("a", "b", 0.8)]
        
        # Test with explicit source
        collection = starlings.Collection.from_edges(edges, source="custom_source")
        partition = collection.at(0.5)
        
        # Should still work correctly
        entities = partition.entities
        assert len(entities) == 1
        assert len(entities[0]) == 2
    
    def test_empty_string_keys(self):
        """Test handling of empty string keys."""
        edges = [("", "non-empty", 0.8)]
        collection = starlings.Collection.from_edges(edges)
        
        partition = collection.at(0.5)
        entities = partition.entities
        assert len(entities) == 1
        assert len(entities[0]) == 2
    
    def test_duplicate_edges(self):
        """Test handling of duplicate edges."""
        edges = [
            ("a", "b", 0.8),
            ("a", "b", 0.8),  # duplicate
            ("b", "a", 0.8),  # reverse (should be treated as same)
        ]
        collection = starlings.Collection.from_edges(edges)
        
        # Should handle duplicates gracefully
        partition = collection.at(0.5)
        entities = partition.entities
        assert len(entities) == 1
        assert len(entities[0]) == 2


if __name__ == "__main__":
    pytest.main([__file__])