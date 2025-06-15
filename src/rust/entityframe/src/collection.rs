use pyo3::prelude::*;
use std::collections::HashMap;

use crate::entity::Entity;
use crate::interner::StringInterner;

/// EntityCollection: A collection of entities from a single process (like pandas Series)
/// Collections should only be created through EntityFrame.create_collection() to ensure shared interner
#[pyclass]
#[derive(Clone)]
pub struct EntityCollection {
    pub entities: Vec<Entity>,
    process_name: String,
    // Simple design: collections don't own interners, they get them from the frame
}

#[pymethods]
impl EntityCollection {
    /// Create EntityCollection (internal constructor)
    /// Users should use EntityFrame.create_collection() instead
    #[new]
    pub fn new(process_name: &str) -> Self {
        Self {
            entities: Vec::new(),
            process_name: process_name.to_string(),
        }
    }

    /// Get all entities in this collection
    pub fn get_entities(&self) -> Vec<Entity> {
        self.entities.clone()
    }

    /// Get the process name for this collection
    #[getter]
    pub fn process_name(&self) -> &str {
        &self.process_name
    }

    /// Get the number of entities in this collection
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the total number of records across all entities
    pub fn total_records(&self) -> u64 {
        self.entities
            .iter()
            .map(|entity| entity.total_records())
            .sum()
    }

    /// Get an entity by index
    pub fn get_entity(&self, index: usize) -> PyResult<Entity> {
        self.entities.get(index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })
    }

    /// Compare this collection with another collection entity-by-entity
    pub fn compare_with(
        &self,
        other: &EntityCollection,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        if self.entities.len() != other.entities.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Collections must have the same number of entities to compare",
            ));
        }

        Python::with_gil(|py| {
            // Pre-allocate the comparisons vector
            let mut comparisons = Vec::with_capacity(self.entities.len());

            for (i, (entity1, entity2)) in
                self.entities.iter().zip(other.entities.iter()).enumerate()
            {
                let jaccard = entity1.jaccard_similarity(entity2);

                let mut comparison = HashMap::new();
                comparison.insert(
                    "entity_index".to_string(),
                    i.into_pyobject(py).unwrap().into_any().unbind(),
                );
                comparison.insert(
                    "process1".to_string(),
                    self.process_name
                        .clone()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
                comparison.insert(
                    "process2".to_string(),
                    other
                        .process_name
                        .clone()
                        .into_pyobject(py)
                        .unwrap()
                        .into_any()
                        .unbind(),
                );
                comparison.insert(
                    "jaccard".to_string(),
                    jaccard.into_pyobject(py).unwrap().into_any().unbind(),
                );

                comparisons.push(comparison);
            }

            Ok(comparisons)
        })
    }
}

impl EntityCollection {
    /// Add entities to this collection using shared interner from frame
    /// This is the main method used by EntityFrame.add_method()
    /// This is a regular Rust method, not exposed to Python
    pub fn add_entities(
        &mut self,
        entity_data: Vec<HashMap<String, Vec<String>>>,
        interner: &mut StringInterner,
        dataset_name_to_id: &mut HashMap<String, u32>,
    ) {
        use roaring::RoaringBitmap;

        // Pre-allocate space for entities
        self.entities.reserve(entity_data.len());

        // Step 1: Process all dataset names first
        for entity_dict in &entity_data {
            for dataset_name in entity_dict.keys() {
                if !dataset_name_to_id.contains_key(dataset_name) {
                    let new_id = interner.intern(dataset_name);
                    dataset_name_to_id.insert(dataset_name.clone(), new_id);
                }
            }
        }

        // Step 2: Process each entity using batch processing per dataset
        for entity_dict in entity_data {
            let mut dataset_bitmaps = HashMap::new();
            let mut dataset_sorted_records = HashMap::new();

            // Group records by dataset for batch processing
            let mut dataset_records_raw = HashMap::new();
            for (dataset_name, record_ids) in entity_dict {
                let dataset_id = dataset_name_to_id[&dataset_name];
                dataset_records_raw.insert(dataset_id, record_ids);
            }

            // Batch intern and sort all records for this entity
            let batch_results = interner.batch_intern_by_dataset(&dataset_records_raw);

            // Create roaring bitmaps and store sorted order
            for (dataset_id, (record_ids, sorted_record_ids)) in batch_results {
                let mut bitmap = RoaringBitmap::new();
                bitmap.extend(&record_ids);
                dataset_bitmaps.insert(dataset_id, bitmap);
                dataset_sorted_records.insert(dataset_id, sorted_record_ids);
            }

            // Create entity with pre-computed sorted order
            let entity = Entity::from_sorted_data(dataset_bitmaps, dataset_sorted_records);
            self.entities.push(entity);
        }
    }

    /// Batch hash all entities in this collection for optimal performance
    pub fn hash_all_entities(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<Vec<u8>>> {
        if self.entities.is_empty() {
            return Ok(Vec::new());
        }

        // Get sorted dataset IDs from interner
        let sorted_dataset_ids = interner.get_sorted_ids().to_vec();

        // Collect all string IDs needed for ALL entities (bulk optimisation)
        let entity_refs: Vec<_> = self.entities.iter().collect();
        let string_ids = crate::hash::collect_string_ids(&entity_refs, &sorted_dataset_ids);

        // Single bulk string lookup for ALL entities
        let string_cache = interner.bulk_get_strings(&string_ids);

        // Process each entity using cached strings
        let mut results = Vec::with_capacity(self.entities.len());
        for entity in &self.entities {
            let hash = crate::hash::hash_entity_with_cache(
                entity.get_datasets_map(),
                entity.get_sorted_records_map(),
                &sorted_dataset_ids,
                &string_cache,
                algorithm,
            )?;
            results.push(hash);
        }

        Ok(results)
    }

    /// Batch hash all entities returning hex strings for convenience
    pub fn hash_all_entities_hex(
        &self,
        interner: &mut crate::interner::StringInterner,
        algorithm: &str,
    ) -> PyResult<Vec<String>> {
        let hashes = self.hash_all_entities(interner, algorithm)?;
        Ok(hashes.into_iter().map(hex::encode).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_creation() {
        let collection = EntityCollection::new("test_process");
        assert_eq!(collection.process_name(), "test_process");
        assert_eq!(collection.len(), 0);
        assert!(collection.is_empty());
        assert_eq!(collection.total_records(), 0);
    }

    #[test]
    fn test_add_entities_with_shared_interner() {
        let mut collection = EntityCollection::new("test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c1".to_string(), "c2".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data
            },
        ];

        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        assert_eq!(collection.len(), 2);
        assert_eq!(collection.total_records(), 4); // c1, c2, o1, c3
        assert_eq!(dataset_name_to_id.len(), 2); // customers, orders
        assert_eq!(interner.len(), 6); // customers, orders, c1, c2, o1, c3

        // Check first entity
        let entity1 = collection.get_entity(0).unwrap();
        assert_eq!(entity1.total_records(), 3); // c1, c2, o1

        // Check second entity
        let entity2 = collection.get_entity(1).unwrap();
        assert_eq!(entity2.total_records(), 1); // c3
    }

    #[test]
    fn test_batch_processing_with_sorted_records() {
        let mut collection = EntityCollection::new("batch_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Create test data with unsorted records to verify sorting works
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c3".to_string(), "c1".to_string(), "c2".to_string()],
                );
                data.insert(
                    "orders".to_string(),
                    vec!["o2".to_string(), "o1".to_string()],
                );
                data
            },
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c5".to_string(), "c4".to_string()],
                );
                data
            },
        ];

        // Use optimised entity processing
        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        assert_eq!(collection.len(), 2);

        // Check that entities have sorted records cached
        let entity1 = collection.get_entity(0).unwrap();
        let entity2 = collection.get_entity(1).unwrap();

        assert!(entity1.has_sorted_records());
        assert!(entity2.has_sorted_records());

        // Get dataset IDs
        let customers_id = dataset_name_to_id["customers"];
        let orders_id = dataset_name_to_id["orders"];

        // Verify sorted order for entity 1
        if let Some(sorted_customers) = entity1.get_sorted_records(customers_id) {
            let sorted_strings: Vec<&str> = sorted_customers
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["c1", "c2", "c3"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 1 should have sorted customers records");
        }

        if let Some(sorted_orders) = entity1.get_sorted_records(orders_id) {
            let sorted_strings: Vec<&str> = sorted_orders
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["o1", "o2"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 1 should have sorted orders records");
        }

        // Verify sorted order for entity 2
        if let Some(sorted_customers) = entity2.get_sorted_records(customers_id) {
            let sorted_strings: Vec<&str> = sorted_customers
                .iter()
                .map(|&id| interner.get_string_internal(id).unwrap())
                .collect();
            assert_eq!(sorted_strings, vec!["c4", "c5"]); // Should be alphabetically sorted
        } else {
            panic!("Entity 2 should have sorted customers records");
        }
    }

    #[test]
    fn test_batch_processing_efficiency() {
        let mut collection = EntityCollection::new("batch_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Same test data for verification
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c2".to_string(), "c1".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data
            },
        ];

        // Process with batch method (now the only method)
        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        // Verify results
        assert_eq!(collection.len(), 2);
        assert_eq!(collection.total_records(), 4); // c1, c2, o1, c3

        // Check entities have expected data
        for i in 0..collection.len() {
            let entity = collection.get_entity(i).unwrap();

            // All entities should have sorted records with optimised processing
            assert!(entity.has_sorted_records());
        }
    }

    #[test]
    fn test_batch_hashing() {
        let mut collection = EntityCollection::new("hash_test");
        let mut interner = StringInterner::new();
        let mut dataset_name_to_id = HashMap::new();

        // Create test data
        let entity_data = vec![
            {
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["c1".to_string(), "c2".to_string()],
                );
                data.insert("orders".to_string(), vec!["o1".to_string()]);
                data
            },
            {
                let mut data = HashMap::new();
                data.insert("customers".to_string(), vec!["c3".to_string()]);
                data.insert(
                    "orders".to_string(),
                    vec!["o2".to_string(), "o3".to_string()],
                );
                data
            },
        ];

        collection.add_entities(entity_data, &mut interner, &mut dataset_name_to_id);

        // Test batch hashing
        let hashes = collection
            .hash_all_entities(&mut interner, "sha256")
            .unwrap();
        assert_eq!(hashes.len(), 2);

        // Verify hash sizes (SHA-256 = 32 bytes)
        for hash in &hashes {
            assert_eq!(hash.len(), 32);
        }

        // Test hex batch hashing
        let hex_hashes = collection
            .hash_all_entities_hex(&mut interner, "blake3")
            .unwrap();
        assert_eq!(hex_hashes.len(), 2);

        // Verify hex strings (32 bytes = 64 hex chars)
        for hex_hash in &hex_hashes {
            assert_eq!(hex_hash.len(), 64);
        }

        // Test consistency - same algorithm should produce same results
        let hashes2 = collection
            .hash_all_entities(&mut interner, "sha256")
            .unwrap();
        assert_eq!(hashes, hashes2);

        // Test different algorithms produce different results
        let blake_hashes = collection
            .hash_all_entities(&mut interner, "blake3")
            .unwrap();
        assert_ne!(hashes, blake_hashes);
    }
}
