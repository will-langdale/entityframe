use roaring::RoaringBitmap;

/// Represents a partition at a specific threshold level
///
/// A partition is a collection of entities (disjoint sets of records) at a given threshold.
/// Each entity is represented as a RoaringBitmap containing the indices of records that
/// belong together at this threshold level.
///
/// This structure includes pre-computed statistics like entity sizes for efficient access
/// to common metrics without recomputation.
#[derive(Debug, Clone)]
pub struct PartitionLevel {
    /// The threshold value for this partition
    threshold: f64,

    /// Entities as sets of record indices (RoaringBitmaps)
    /// Each bitmap represents one entity containing the record indices that belong to it
    entities: Vec<RoaringBitmap>,

    /// Pre-computed entity sizes for quick access
    /// Corresponds 1:1 with the entities vector
    entity_sizes: Vec<u32>,
}

impl PartitionLevel {
    /// Create a new partition level
    ///
    /// # Arguments
    /// * `threshold` - The threshold value for this partition
    /// * `entities` - Vector of RoaringBitmaps, each representing an entity
    ///
    /// # Returns
    /// A new PartitionLevel with pre-computed entity sizes
    pub fn new(threshold: f64, entities: Vec<RoaringBitmap>) -> Self {
        let entity_sizes = entities.iter().map(|e| e.len() as u32).collect();
        Self {
            threshold,
            entities,
            entity_sizes,
        }
    }

    /// Get the threshold for this partition
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the entities in this partition
    pub fn entities(&self) -> &[RoaringBitmap] {
        &self.entities
    }

    /// Get the number of entities in this partition
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Get pre-computed entity sizes
    pub fn entity_sizes(&self) -> &[u32] {
        &self.entity_sizes
    }

    /// Get the total number of records across all entities
    pub fn total_records(&self) -> u32 {
        self.entity_sizes.iter().sum()
    }

    /// Check if a specific record belongs to any entity
    pub fn contains_record(&self, record_id: u32) -> bool {
        self.entities
            .iter()
            .any(|entity| entity.contains(record_id))
    }

    /// Find which entity (if any) contains a specific record
    pub fn find_entity_for_record(&self, record_id: u32) -> Option<usize> {
        self.entities
            .iter()
            .position(|entity| entity.contains(record_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_level_creation() {
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);

        let mut entity2 = RoaringBitmap::new();
        entity2.insert(2);
        entity2.insert(3);
        entity2.insert(4);

        let partition = PartitionLevel::new(0.7, vec![entity1, entity2]);

        assert_eq!(partition.threshold(), 0.7);
        assert_eq!(partition.num_entities(), 2);
        assert_eq!(partition.entity_sizes(), &[2, 3]);
        assert_eq!(partition.total_records(), 5);
    }

    #[test]
    fn test_contains_record() {
        let mut entity = RoaringBitmap::new();
        entity.insert(1);
        entity.insert(2);

        let partition = PartitionLevel::new(0.5, vec![entity]);

        assert!(partition.contains_record(1));
        assert!(partition.contains_record(2));
        assert!(!partition.contains_record(0));
        assert!(!partition.contains_record(3));
    }

    #[test]
    fn test_find_entity_for_record() {
        let mut entity1 = RoaringBitmap::new();
        entity1.insert(0);
        entity1.insert(1);

        let mut entity2 = RoaringBitmap::new();
        entity2.insert(2);

        let partition = PartitionLevel::new(0.8, vec![entity1, entity2]);

        assert_eq!(partition.find_entity_for_record(0), Some(0));
        assert_eq!(partition.find_entity_for_record(1), Some(0));
        assert_eq!(partition.find_entity_for_record(2), Some(1));
        assert_eq!(partition.find_entity_for_record(3), None);
    }
}
