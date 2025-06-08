use pyo3::prelude::*;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Core entity representation using roaring bitmaps for record ID sets.
/// Uses interned dataset IDs (u32) instead of strings for massive memory savings.
#[pyclass]
#[derive(Clone)]
pub struct Entity {
    datasets: HashMap<u32, RoaringBitmap>,
}

impl Default for Entity {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl Entity {
    #[new]
    pub fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    /// Add a record ID to a dataset using interned dataset ID (internal method).
    pub fn add_record_by_id(&mut self, dataset_id: u32, record_id: u32) {
        self.datasets
            .entry(dataset_id)
            .or_default()
            .insert(record_id);
    }

    /// Add multiple record IDs to a dataset using interned dataset ID (internal method).
    pub fn add_records_by_id(&mut self, dataset_id: u32, record_ids: Vec<u32>) {
        let bitmap = self.datasets.entry(dataset_id).or_default();
        // Use RoaringBitmap's bulk insertion method for better performance
        bitmap.extend(&record_ids);
    }

    /// Add a record ID to a dataset (Python API - requires string lookup).
    pub fn add_record(&mut self, dataset: &str, record_id: u32) {
        // For Python API compatibility, we use a simple hash of the dataset name
        // In practice, this should be used with proper interner context
        let dataset_id = self.hash_dataset_name(dataset);
        self.add_record_by_id(dataset_id, record_id);
    }

    /// Add multiple record IDs to a dataset (Python API - requires string lookup).
    pub fn add_records(&mut self, dataset: &str, record_ids: Vec<u32>) {
        let dataset_id = self.hash_dataset_name(dataset);
        self.add_records_by_id(dataset_id, record_ids);
    }

    /// Simple hash function for dataset names (used in Python API).
    pub fn hash_dataset_name(&self, dataset: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        dataset.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get record IDs for a dataset as a list (Python API).
    pub fn get_records(&self, dataset: &str) -> PyResult<Vec<u32>> {
        let dataset_id = self.hash_dataset_name(dataset);
        match self.datasets.get(&dataset_id) {
            Some(bitmap) => Ok(bitmap.iter().collect()),
            None => Ok(Vec::new()),
        }
    }

    /// Get record IDs for a dataset by ID (internal method).
    pub fn get_records_by_id(&self, dataset_id: u32) -> Vec<u32> {
        match self.datasets.get(&dataset_id) {
            Some(bitmap) => bitmap.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Get all dataset IDs (internal method for performance).
    pub fn get_dataset_ids(&self) -> Vec<u32> {
        self.datasets.keys().copied().collect()
    }

    /// Get all dataset names (Python API - limited without interner context).
    pub fn get_datasets(&self) -> Vec<String> {
        // For Python API compatibility, return placeholder names
        // In practice, this should be called through EntityCollection which has interner access
        self.datasets
            .keys()
            .map(|id| format!("dataset_{}", id))
            .collect()
    }

    /// Check if the entity contains records in a given dataset (Python API).
    pub fn has_dataset(&self, dataset: &str) -> bool {
        let dataset_id = self.hash_dataset_name(dataset);
        self.datasets.contains_key(&dataset_id)
    }

    /// Check if the entity contains records for a dataset ID (internal method).
    pub fn has_dataset_id(&self, dataset_id: u32) -> bool {
        self.datasets.contains_key(&dataset_id)
    }

    /// Get the total number of records across all datasets.
    pub fn total_records(&self) -> u64 {
        self.datasets.values().map(|bitmap| bitmap.len()).sum()
    }

    /// Compute Jaccard similarity with another entity.
    pub fn jaccard_similarity(&self, other: &Entity) -> f64 {
        let mut union_size = 0u64;
        let mut intersection_size = 0u64;

        // Process datasets from self first
        for (dataset_id, self_bitmap) in &self.datasets {
            if let Some(other_bitmap) = other.datasets.get(dataset_id) {
                // Both entities have this dataset
                intersection_size += (self_bitmap & other_bitmap).len();
                union_size += (self_bitmap | other_bitmap).len();
            } else {
                // Only self has this dataset
                union_size += self_bitmap.len();
            }
        }

        // Process remaining datasets from other (that self doesn't have)
        for (dataset_id, other_bitmap) in &other.datasets {
            if !self.datasets.contains_key(dataset_id) {
                // Only other has this dataset
                union_size += other_bitmap.len();
            }
        }

        if union_size == 0 {
            1.0 // Both entities are empty
        } else {
            intersection_size as f64 / union_size as f64
        }
    }
}

impl Entity {
    /// Remap dataset IDs according to the provided mapping (internal method)
    pub fn remap_dataset_ids(&mut self, remapping: &HashMap<u32, u32>) {
        // Create a new HashMap with remapped dataset IDs
        let mut new_datasets = HashMap::new();

        for (old_id, bitmap) in self.datasets.drain() {
            let new_id = remapping.get(&old_id).copied().unwrap_or(old_id);

            // If the new_id already exists (collision), merge the bitmaps
            match new_datasets.get_mut(&new_id) {
                Some(existing_bitmap) => {
                    // Merge the bitmaps using bitwise OR
                    *existing_bitmap |= &bitmap;
                }
                None => {
                    new_datasets.insert(new_id, bitmap);
                }
            }
        }

        self.datasets = new_datasets;
    }

    /// Remap both dataset and record IDs according to the provided mappings (internal method)
    pub fn remap_all_ids(
        &mut self,
        dataset_remapping: &HashMap<u32, u32>,
        record_remapping: &HashMap<u32, u32>,
    ) {
        use roaring::RoaringBitmap;

        // Create a new HashMap with remapped dataset and record IDs
        let mut new_datasets = HashMap::new();

        for (old_dataset_id, bitmap) in self.datasets.drain() {
            let new_dataset_id = dataset_remapping
                .get(&old_dataset_id)
                .copied()
                .unwrap_or(old_dataset_id);

            // Remap record IDs within the bitmap if needed
            let new_bitmap = if record_remapping.is_empty() {
                bitmap
            } else {
                let mut remapped_bitmap = RoaringBitmap::new();
                for old_record_id in bitmap.iter() {
                    let new_record_id = record_remapping
                        .get(&old_record_id)
                        .copied()
                        .unwrap_or(old_record_id);
                    remapped_bitmap.insert(new_record_id);
                }
                remapped_bitmap
            };

            // If the new_dataset_id already exists (collision), merge the bitmaps
            match new_datasets.get_mut(&new_dataset_id) {
                Some(existing_bitmap) => {
                    // Merge the bitmaps using bitwise OR
                    *existing_bitmap |= &new_bitmap;
                }
                None => {
                    new_datasets.insert(new_dataset_id, new_bitmap);
                }
            }
        }

        self.datasets = new_datasets;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_entity_basic() {
        let mut entity = Entity::new();

        // Test adding records
        entity.add_record("customers", 1);
        entity.add_record("customers", 2);
        entity.add_records("transactions", vec![10, 11, 12]);

        // Test retrieval
        let customers = entity.get_records("customers").unwrap();
        assert_eq!(customers, vec![1, 2]);

        let transactions = entity.get_records("transactions").unwrap();
        assert_eq!(transactions, vec![10, 11, 12]);

        // Test datasets - get_datasets() returns placeholder names since Entity doesn't have interner access
        let datasets = entity.get_datasets();
        assert_eq!(datasets.len(), 2); // Should have 2 datasets

        // Test has_dataset method which can work with string names
        assert!(entity.has_dataset("customers"));
        assert!(entity.has_dataset("transactions"));

        // Test total records
        assert_eq!(entity.total_records(), 5);
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut entity1 = Entity::new();
        entity1.add_records("customers", vec![1, 2, 3]);
        entity1.add_records("transactions", vec![10, 11]);

        let mut entity2 = Entity::new();
        entity2.add_records("customers", vec![2, 3, 4]);
        entity2.add_records("transactions", vec![11, 12]);

        // Intersection: customers {2, 3}, transactions {11} = 3 total
        // Union: customers {1, 2, 3, 4}, transactions {10, 11, 12} = 7 total
        // Jaccard = 3/7 ≈ 0.428
        let similarity = entity1.jaccard_similarity(&entity2);
        assert!((similarity - 3.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_with_same_datasets_different_records() {
        let mut entity1 = Entity::new();
        let mut entity2 = Entity::new();

        // Both entities have "customers" dataset but different records
        entity1.add_records_by_id(0, vec![10, 11]);
        entity2.add_records_by_id(0, vec![20, 21]);

        // Jaccard should be 0.0 since no record IDs overlap
        let jaccard = entity1.jaccard_similarity(&entity2);
        assert_eq!(jaccard, 0.0);
    }

    #[test]
    fn test_jaccard_with_overlapping_records() {
        let mut entity1 = Entity::new();
        let mut entity2 = Entity::new();

        // Both entities have "customers" dataset with some overlapping records
        entity1.add_records_by_id(0, vec![10, 11]);
        entity2.add_records_by_id(0, vec![11, 12]);

        // Intersection: {11} = 1 record
        // Union: {10, 11, 12} = 3 records
        // Jaccard = 1/3 ≈ 0.333
        let jaccard = entity1.jaccard_similarity(&entity2);
        assert!((jaccard - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_entity_remap_basic() {
        let mut entity = Entity::new();

        // Add records using public API
        entity.add_records_by_id(0, vec![10, 11]);
        entity.add_records_by_id(1, vec![20, 21]);

        // Create remapping tables
        let mut dataset_remapping = HashMap::new();
        dataset_remapping.insert(0, 5); // Dataset 0 -> 5
        dataset_remapping.insert(1, 6); // Dataset 1 -> 6

        let mut record_remapping = HashMap::new();
        record_remapping.insert(10, 100); // Record 10 -> 100
        record_remapping.insert(11, 101); // Record 11 -> 101
        record_remapping.insert(20, 200); // Record 20 -> 200
        record_remapping.insert(21, 201); // Record 21 -> 201

        // Apply remapping
        entity.remap_all_ids(&dataset_remapping, &record_remapping);

        // Check results using public API
        let dataset5_records = entity.get_records_by_id(5);
        let dataset6_records = entity.get_records_by_id(6);

        assert_eq!(dataset5_records, vec![100, 101]);
        assert_eq!(dataset6_records, vec![200, 201]);
    }

    #[test]
    fn test_entity_remap_dataset_collision() {
        let mut entity = Entity::new();

        // Create two datasets that will be remapped to the same ID
        entity.add_records_by_id(0, vec![10, 11]);
        entity.add_records_by_id(1, vec![20, 21]);

        // Remap both to the same dataset ID (collision)
        let mut dataset_remapping = HashMap::new();
        dataset_remapping.insert(0, 5);
        dataset_remapping.insert(1, 5); // Both map to 5

        let record_remapping = HashMap::new(); // No record remapping

        entity.remap_all_ids(&dataset_remapping, &record_remapping);

        // Should have merged the bitmaps
        let merged_records = entity.get_records_by_id(5);
        assert_eq!(merged_records, vec![10, 11, 20, 21]);

        // Should have no records in the old dataset IDs
        assert_eq!(entity.get_records_by_id(0), Vec::<u32>::new());
        assert_eq!(entity.get_records_by_id(1), Vec::<u32>::new());
    }
}
