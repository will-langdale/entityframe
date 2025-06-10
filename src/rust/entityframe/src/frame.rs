use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;

use crate::collection::EntityCollection;
use crate::interner::StringInterner;

/// EntityFrame: A collection of EntityCollections with shared interner (like pandas DataFrame)
#[pyclass]
pub struct EntityFrame {
    collections: HashMap<String, EntityCollection>,
    // Single interner for all strings (datasets and records)
    interner: StringInterner,
    // Dataset name tracking for API convenience
    dataset_name_to_id: HashMap<String, u32>,
}

impl Default for EntityFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl EntityFrame {
    #[new]
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
            interner: StringInterner::new(),
            dataset_name_to_id: HashMap::new(),
        }
    }

    /// Create EntityFrame with pre-declared datasets for better performance.
    #[staticmethod]
    pub fn with_datasets(dataset_names: Vec<String>) -> Self {
        let mut frame = Self::new();
        for name in dataset_names {
            frame.declare_dataset(&name);
        }
        frame
    }

    /// Declare a dataset name upfront for efficient interning.
    pub fn declare_dataset(&mut self, dataset_name: &str) -> u32 {
        let dataset_id = self.interner.intern(dataset_name);
        self.dataset_name_to_id
            .insert(dataset_name.to_string(), dataset_id);
        dataset_id
    }

    /// Get dataset ID from name (internal method).
    pub fn get_dataset_id(&mut self, dataset_name: &str) -> u32 {
        if let Some(&existing_id) = self.dataset_name_to_id.get(dataset_name) {
            existing_id
        } else {
            self.declare_dataset(dataset_name)
        }
    }

    /// Get dataset name from ID (internal method).
    pub fn get_dataset_name(&self, dataset_id: u32) -> PyResult<&str> {
        self.interner.get_string(dataset_id)
    }

    /// Get a reference to the shared interner
    #[getter]
    pub fn interner(&self) -> StringInterner {
        self.interner.clone()
    }

    /// Create a new collection that will use this frame's shared interner
    pub fn create_collection(&self, name: &str) -> EntityCollection {
        EntityCollection::new(name)
    }

    /// Add a collection to the frame (simple - no ID remapping needed)
    pub fn add_collection(&mut self, name: &str, collection: EntityCollection) {
        self.collections.insert(name.to_string(), collection);
    }

    /// Create and add a collection with entity data in one step
    pub fn add_method(
        &mut self,
        method_name: &str,
        entity_data: Vec<HashMap<String, Vec<String>>>,
    ) {
        // Create a collection and add entities using frame's shared interner
        let mut collection = self.create_collection(method_name);
        collection.add_entities(
            entity_data,
            &mut self.interner,
            &mut self.dataset_name_to_id,
        );
        self.collections.insert(method_name.to_string(), collection);
    }

    /// Get collection names in this frame
    pub fn get_collection_names(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    /// Get a collection by name
    pub fn get_collection(&self, name: &str) -> Option<EntityCollection> {
        self.collections.get(name).cloned()
    }

    /// Compare two collections and return similarities
    pub fn compare_collections(
        &self,
        name1: &str,
        name2: &str,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let collection1 = self.collections.get(name1).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                name1
            ))
        })?;

        let collection2 = self.collections.get(name2).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                name2
            ))
        })?;

        collection1.compare_with(collection2)
    }

    /// Get the number of collections in this frame
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Get the total number of entities across all collections
    pub fn total_entities(&self) -> usize {
        self.collections
            .values()
            .map(|collection| collection.len())
            .sum()
    }

    /// Get the total number of unique strings in the interner
    pub fn interner_size(&self) -> usize {
        self.interner.len()
    }

    /// Get all declared dataset names
    pub fn get_dataset_names(&self) -> Vec<String> {
        self.dataset_name_to_id.keys().cloned().collect()
    }

    /// Check if an entity in a collection has a dataset (with proper name resolution)
    pub fn entity_has_dataset(
        &self,
        collection_name: &str,
        entity_index: usize,
        dataset_name: &str,
    ) -> PyResult<bool> {
        let collection = self.collections.get(collection_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                collection_name
            ))
        })?;

        let entity = collection.entities.get(entity_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })?;

        // Look up dataset ID from name
        if let Some(&dataset_id) = self.dataset_name_to_id.get(dataset_name) {
            Ok(entity.has_dataset_id(dataset_id))
        } else {
            // Dataset not found
            Ok(false)
        }
    }

    /// Set metadata on an entity
    pub fn set_entity_metadata(
        &mut self,
        collection_name: &str,
        entity_index: usize,
        key: &str,
        value: &[u8],
    ) -> PyResult<()> {
        let collection = self.collections.get_mut(collection_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                collection_name
            ))
        })?;

        let entity = collection.entities.get_mut(entity_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })?;

        // Intern the metadata key
        let key_id = self.interner.intern(key);
        entity.set_metadata(key_id, value.to_vec());
        Ok(())
    }

    /// Get metadata from an entity
    pub fn get_entity_metadata(
        &mut self,
        collection_name: &str,
        entity_index: usize,
        key: &str,
    ) -> PyResult<Option<Vec<u8>>> {
        let collection = self.collections.get(collection_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                collection_name
            ))
        })?;

        let entity = collection.entities.get(entity_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })?;

        // Try to intern the key to get its ID (this won't add it if it doesn't exist)
        let key_id = self.interner.intern(key);
        Ok(entity.get_metadata_by_id(key_id).map(|v| v.to_vec()))
    }

    /// Compute hash of an entity
    #[pyo3(signature = (collection_name, entity_index, algorithm = "sha256"))]
    pub fn hash_entity(
        &mut self,
        collection_name: &str,
        entity_index: usize,
        algorithm: &str,
    ) -> PyResult<Py<PyBytes>> {
        let collection = self.collections.get(collection_name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!(
                "Collection '{}' not found",
                collection_name
            ))
        })?;

        let entity = collection.entities.get(entity_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })?;

        // Compute hash
        let hash_bytes = entity.deterministic_hash(&mut self.interner, algorithm)?;

        // Return as PyBytes
        Python::with_gil(|py| Ok(PyBytes::new(py, &hash_bytes).into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_frame_creation() {
        let frame = EntityFrame::new();
        assert_eq!(frame.collection_count(), 0);
        assert_eq!(frame.total_entities(), 0);
        assert!(frame.get_collection_names().is_empty());
        assert_eq!(frame.interner_size(), 0);
    }

    #[test]
    fn test_single_interner_system_basic() {
        let mut frame = EntityFrame::new();

        // Test dataset declaration
        let dataset_id1 = frame.declare_dataset("customers");
        let dataset_id2 = frame.declare_dataset("orders");
        let dataset_id3 = frame.declare_dataset("customers"); // Same as dataset_id1

        assert_eq!(dataset_id1, 0);
        assert_eq!(dataset_id2, 1);
        assert_eq!(dataset_id3, 0); // Should be same as first
        assert_eq!(frame.get_dataset_names().len(), 2);

        // Test shared interner
        let mut interner = frame.interner();
        let record_id1 = interner.intern("rec1");
        let record_id2 = interner.intern("rec2");
        let record_id3 = interner.intern("rec1"); // Same as record_id1

        assert_eq!(record_id1, 2); // After customers=0, orders=1
        assert_eq!(record_id2, 3);
        assert_eq!(record_id3, 2); // Should be same as first
        assert_eq!(interner.len(), 4); // customers, orders, rec1, rec2
    }

    #[test]
    fn test_entity_frame_add_collection() {
        let mut frame = EntityFrame::new();
        let collection = EntityCollection::new("splink");

        frame.add_collection("splink", collection);

        assert_eq!(frame.collection_count(), 1);
        assert_eq!(frame.total_entities(), 0); // Empty collection
        assert!(frame.get_collection_names().contains(&"splink".to_string()));

        let retrieved_collection = frame.get_collection("splink").unwrap();
        assert_eq!(retrieved_collection.len(), 0);
        assert_eq!(retrieved_collection.process_name(), "splink");
    }

    #[test]
    fn test_add_method_collections_different_record_ids() {
        use std::collections::HashMap;

        let mut frame = EntityFrame::new();

        // Add two methods with different records using the simplified API
        frame.add_method(
            "coll1",
            vec![{
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["rec1".to_string(), "rec2".to_string()],
                );
                data
            }],
        );

        frame.add_method(
            "coll2",
            vec![{
                let mut data = HashMap::new();
                data.insert(
                    "customers".to_string(),
                    vec!["rec3".to_string(), "rec4".to_string()],
                );
                data
            }],
        );

        // Get entities and check they have different record IDs
        let c1 = frame.get_collection("coll1").unwrap();
        let c2 = frame.get_collection("coll2").unwrap();

        let e1 = c1.get_entity(0).unwrap();
        let e2 = c2.get_entity(0).unwrap();

        let e1_records = e1.get_records_by_id(0);
        let e2_records = e2.get_records_by_id(0);

        println!("Entity 1 records: {:?}", e1_records);
        println!("Entity 2 records: {:?}", e2_records);

        // They should have different record IDs since they're different strings
        assert_ne!(e1_records, e2_records);

        // Jaccard should be 0.0 since no record overlap
        let jaccard = e1.jaccard_similarity(&e2);
        println!("Jaccard: {}", jaccard);
        assert_eq!(jaccard, 0.0);
    }
}
