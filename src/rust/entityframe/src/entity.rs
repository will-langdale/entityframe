use pyo3::prelude::*;
use roaring::RoaringBitmap;
use std::collections::HashMap;

/// Core entity representation using roaring bitmaps for record ID sets.
/// Uses interned dataset IDs (u32) instead of strings for massive memory savings.
#[pyclass]
pub struct Entity {
    datasets: HashMap<u32, RoaringBitmap>,
    metadata: Option<HashMap<u32, PyObject>>,
    /// Pre-computed sorted record order for each dataset (for fast hashing)
    sorted_records: HashMap<u32, Vec<u32>>,
}

impl Clone for Entity {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let metadata = self
                .metadata
                .as_ref()
                .map(|map| map.iter().map(|(k, v)| (*k, v.clone_ref(py))).collect());

            Self {
                datasets: self.datasets.clone(),
                metadata,
                sorted_records: self.sorted_records.clone(),
            }
        })
    }
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
            metadata: None,
            sorted_records: HashMap::new(),
        }
    }

    /// Add a record ID to a dataset using interned dataset ID (internal method).
    pub fn add_record_by_id(&mut self, dataset_id: u32, record_id: u32) {
        self.datasets
            .entry(dataset_id)
            .or_default()
            .insert(record_id);
        // Invalidate sorted records since we modified the entity
        self.sorted_records.clear();
    }

    /// Add multiple record IDs to a dataset using interned dataset ID (internal method).
    pub fn add_records_by_id(&mut self, dataset_id: u32, record_ids: Vec<u32>) {
        let bitmap = self.datasets.entry(dataset_id).or_default();
        // Use RoaringBitmap's bulk insertion method for better performance
        bitmap.extend(&record_ids);
        // Invalidate sorted records since we modified the entity
        self.sorted_records.clear();
    }

    /// Add a record ID to a dataset (Python API - requires string lookup).
    pub fn add_record(&mut self, dataset: &str, record_id: u32) {
        // For Python API compatibility, we use a simple hash of the dataset name
        // In practise, this should be used with proper interner context
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
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(dataset.as_bytes());
        let hash = hasher.finalize();
        // Use first 4 bytes of Blake3 hash
        u32::from_le_bytes([
            hash.as_bytes()[0],
            hash.as_bytes()[1],
            hash.as_bytes()[2],
            hash.as_bytes()[3],
        ])
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
        // In practise, this should be called through EntityCollection which has interner access
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

    /// Set metadata for a key (requires interner for key interning).
    pub fn set_metadata(&mut self, key_id: u32, value: PyObject) {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key_id, value);
    }

    /// Get metadata by key ID (internal method).
    pub fn get_metadata_by_id(&self, key_id: u32) -> Option<&PyObject> {
        self.metadata.as_ref().and_then(|m| m.get(&key_id))
    }

    /// Check if metadata key exists.
    pub fn has_metadata(&self, key_id: u32) -> bool {
        self.metadata
            .as_ref()
            .map(|m| m.contains_key(&key_id))
            .unwrap_or(false)
    }

    /// Clear all metadata.
    pub fn clear_metadata(&mut self) {
        self.metadata = None;
    }

    /// Get all metadata keys.
    pub fn get_metadata_keys(&self) -> Vec<u32> {
        self.metadata
            .as_ref()
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default()
    }
}

impl Entity {
    /// Create a new entity with pre-computed sorted order.
    /// Avoids the O(R log R) sorting cost per entity.
    pub fn from_sorted_data(
        dataset_bitmaps: HashMap<u32, RoaringBitmap>,
        sorted_records: HashMap<u32, Vec<u32>>,
    ) -> Self {
        Self {
            datasets: dataset_bitmaps,
            metadata: None,
            sorted_records,
        }
    }

    /// Get pre-computed sorted record order for a dataset.
    /// Returns None if not available (entity created without pre-computed sorted order).
    pub fn get_sorted_records(&self, dataset_id: u32) -> Option<&[u32]> {
        self.sorted_records.get(&dataset_id).map(|v| v.as_slice())
    }

    /// Check if this entity has pre-computed sorted record order.
    pub fn has_sorted_records(&self) -> bool {
        !self.sorted_records.is_empty()
    }

    /// Invalidate sorted records cache (used when entity is modified after batch creation).
    pub fn invalidate_sorted_records(&mut self) {
        self.sorted_records.clear();
    }

    /// Get reference to datasets for hash computation (internal method)
    pub fn get_datasets_map(&self) -> &HashMap<u32, RoaringBitmap> {
        &self.datasets
    }

    /// Get reference to sorted records for hash computation (internal method)
    pub fn get_sorted_records_map(&self) -> &HashMap<u32, Vec<u32>> {
        &self.sorted_records
    }

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

        // Invalidate sorted records since dataset IDs changed
        self.sorted_records.clear();
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

        // Invalidate sorted records since IDs changed
        self.sorted_records.clear();
    }

    /// Compute deterministic hash of the entity using specified algorithm.
    /// Uses pre-computed sorted order for optimal performance.
    pub fn deterministic_hash(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<u8>> {
        // Use unified hasher from hash module
        let mut hasher = crate::hash::create_hasher(algorithm)?;

        // Get sorted dataset IDs from interner for deterministic order
        let sorted_dataset_ids = interner.get_sorted_ids().to_vec();

        // Process datasets in sorted order
        for &dataset_id in &sorted_dataset_ids {
            if let Some(bitmap) = self.datasets.get(&dataset_id) {
                // Hash dataset name
                let dataset_str = interner.get_string(dataset_id)?;
                hasher.update(dataset_str.as_bytes());

                // Use pre-computed sorted record order if available
                if let Some(sorted_record_ids) = self.get_sorted_records(dataset_id) {
                    // Fast path: use pre-sorted records
                    for &record_id in sorted_record_ids {
                        let record_str = interner.get_string(record_id)?;
                        hasher.update(record_str.as_bytes());
                    }
                } else {
                    // Fallback: sort records on demand
                    let mut record_ids: Vec<u32> = bitmap.iter().collect();
                    record_ids.sort_by(|&a, &b| {
                        let str_a = interner.get_string_internal(a).unwrap_or("");
                        let str_b = interner.get_string_internal(b).unwrap_or("");
                        str_a.cmp(str_b)
                    });

                    for record_id in record_ids {
                        let record_str = interner.get_string(record_id)?;
                        hasher.update(record_str.as_bytes());
                    }
                }
            }
        }

        // Finalize and return hash
        Ok(hasher.finalize().to_vec())
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

    #[test]
    fn test_entity_batch_creation() {
        use roaring::RoaringBitmap;

        // Create batch data
        let mut dataset_bitmaps = HashMap::new();
        let mut bitmap1 = RoaringBitmap::new();
        bitmap1.insert(10);
        bitmap1.insert(20);
        bitmap1.insert(30);
        dataset_bitmaps.insert(0, bitmap1);

        let mut bitmap2 = RoaringBitmap::new();
        bitmap2.insert(100);
        bitmap2.insert(200);
        dataset_bitmaps.insert(1, bitmap2);

        // Create sorted records
        let mut sorted_records = HashMap::new();
        sorted_records.insert(0, vec![10, 20, 30]); // Already sorted
        sorted_records.insert(1, vec![100, 200]); // Already sorted

        // Create entity from batch data
        let entity = Entity::from_sorted_data(dataset_bitmaps, sorted_records);

        // Verify entity has sorted records
        assert!(entity.has_sorted_records());
        assert_eq!(entity.get_sorted_records(0), Some([10, 20, 30].as_slice()));
        assert_eq!(entity.get_sorted_records(1), Some([100, 200].as_slice()));
        assert_eq!(entity.get_sorted_records(999), None); // Non-existent dataset

        // Verify regular functionality still works
        assert_eq!(entity.get_records_by_id(0), vec![10, 20, 30]);
        assert_eq!(entity.get_records_by_id(1), vec![100, 200]);
        assert_eq!(entity.total_records(), 5);
    }

    #[test]
    fn test_entity_sorted_records_invalidation() {
        let mut entity = Entity::new();

        // Add some records through normal API (should not have sorted records)
        entity.add_records_by_id(0, vec![10, 20]);
        assert!(!entity.has_sorted_records());

        // Manually set sorted records to test invalidation
        let mut sorted_records = HashMap::new();
        sorted_records.insert(0, vec![10, 20]);
        entity.sorted_records = sorted_records;
        assert!(entity.has_sorted_records());

        // Modify entity - should invalidate sorted records
        entity.add_record_by_id(0, 30);
        assert!(!entity.has_sorted_records());

        // Test invalidation on add_records_by_id
        entity.sorted_records = HashMap::new(); // Reset
        entity.sorted_records.insert(0, vec![10, 20]); // Add some data to make it non-empty
        assert!(entity.has_sorted_records());
        entity.add_records_by_id(1, vec![100]);
        assert!(!entity.has_sorted_records());

        // Test invalidation on remap
        entity.sorted_records = HashMap::new(); // Reset
        entity.sorted_records.insert(0, vec![10, 20]); // Add some data to make it non-empty
        assert!(entity.has_sorted_records());
        let mut remapping = HashMap::new();
        remapping.insert(0, 5);
        entity.remap_dataset_ids(&remapping);
        assert!(!entity.has_sorted_records());
    }

    #[test]
    fn test_optimised_hash_consistency() {
        use crate::interner::StringInterner;
        use roaring::RoaringBitmap;

        let mut interner = StringInterner::new();

        // Create test data with specific strings for deterministic testing
        let dataset1_name = "customers";
        let dataset2_name = "orders";
        let record1 = "customer_z";
        let record2 = "customer_a";
        let record3 = "customer_m";
        let record4 = "order_b";
        let record5 = "order_a";

        // Intern all strings
        let dataset1_id = interner.intern(dataset1_name);
        let dataset2_id = interner.intern(dataset2_name);
        let record1_id = interner.intern(record1);
        let record2_id = interner.intern(record2);
        let record3_id = interner.intern(record3);
        let record4_id = interner.intern(record4);
        let record5_id = interner.intern(record5);

        // Create entity with batch processing (has sorted records)
        let mut dataset_bitmaps = HashMap::new();
        let mut bitmap1 = RoaringBitmap::new();
        bitmap1.insert(record1_id);
        bitmap1.insert(record2_id);
        bitmap1.insert(record3_id);
        dataset_bitmaps.insert(dataset1_id, bitmap1);

        let mut bitmap2 = RoaringBitmap::new();
        bitmap2.insert(record4_id);
        bitmap2.insert(record5_id);
        dataset_bitmaps.insert(dataset2_id, bitmap2);

        // Create sorted records (alphabetically sorted by string value)
        let mut sorted_records = HashMap::new();
        sorted_records.insert(dataset1_id, vec![record2_id, record3_id, record1_id]); // customer_a, customer_m, customer_z
        sorted_records.insert(dataset2_id, vec![record5_id, record4_id]); // order_a, order_b

        let entity_optimised = Entity::from_sorted_data(dataset_bitmaps.clone(), sorted_records);

        // Create entity without batch processing (no sorted records)
        let mut entity_fallback = Entity::new();
        entity_fallback.datasets = dataset_bitmaps;

        // Both entities should produce the same hash
        let hash_optimised = entity_optimised
            .deterministic_hash(&mut interner, "sha256")
            .unwrap();
        let hash_fallback = entity_fallback
            .deterministic_hash(&mut interner, "sha256")
            .unwrap();

        assert_eq!(hash_optimised, hash_fallback);
        assert!(entity_optimised.has_sorted_records());
        assert!(!entity_fallback.has_sorted_records());

        // Test with different algorithms
        let hash_optimised_blake3 = entity_optimised
            .deterministic_hash(&mut interner, "blake3")
            .unwrap();
        let hash_fallback_blake3 = entity_fallback
            .deterministic_hash(&mut interner, "blake3")
            .unwrap();

        assert_eq!(hash_optimised_blake3, hash_fallback_blake3);
        assert_ne!(hash_optimised, hash_optimised_blake3); // Different algorithms should produce different hashes
    }

    #[test]
    fn test_hash_performance_paths() {
        use crate::interner::StringInterner;
        use roaring::RoaringBitmap;

        let mut interner = StringInterner::new();

        // Create a larger test case to verify performance benefits
        let dataset_id = interner.intern("test_dataset");
        let mut record_ids = Vec::new();
        for i in 0..100 {
            record_ids.push(interner.intern(&format!("record_{:03}", i)));
        }

        // Create entity with batch processing (fast path)
        let mut dataset_bitmaps = HashMap::new();
        let mut bitmap = RoaringBitmap::new();
        bitmap.extend(&record_ids);
        dataset_bitmaps.insert(dataset_id, bitmap.clone());

        let mut sorted_records = HashMap::new();
        let mut sorted_record_ids = record_ids.clone();
        sorted_record_ids.sort_by(|&a, &b| {
            interner
                .get_string_internal(a)
                .unwrap()
                .cmp(interner.get_string_internal(b).unwrap())
        });
        sorted_records.insert(dataset_id, sorted_record_ids);

        let entity_fast = Entity::from_sorted_data(dataset_bitmaps, sorted_records);

        // Create entity without batch processing (slow path)
        let mut entity_slow = Entity::new();
        entity_slow.datasets.insert(dataset_id, bitmap);

        // Both should produce the same hash
        let hash_fast = entity_fast
            .deterministic_hash(&mut interner, "sha256")
            .unwrap();
        let hash_slow = entity_slow
            .deterministic_hash(&mut interner, "sha256")
            .unwrap();

        assert_eq!(hash_fast, hash_slow);
        assert!(entity_fast.has_sorted_records());
        assert!(!entity_slow.has_sorted_records());
    }
}
