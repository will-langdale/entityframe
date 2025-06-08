use pyo3::prelude::*;
use std::collections::HashMap;

use crate::collection::EntityCollection;
use crate::interner::StringInterner;

/// EntityFrame: A collection of EntityCollections with shared interner (like pandas DataFrame)
#[pyclass]
pub struct EntityFrame {
    collections: HashMap<String, EntityCollection>,
    interner: StringInterner,
    // Dataset name interning for massive memory savings
    dataset_names: HashMap<u32, String>,      // ID -> name
    dataset_name_to_id: HashMap<String, u32>, // name -> ID
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
            dataset_names: HashMap::new(),
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
        if let Some(&existing_id) = self.dataset_name_to_id.get(dataset_name) {
            return existing_id;
        }

        let dataset_id = self.interner.intern(dataset_name);
        self.dataset_names
            .insert(dataset_id, dataset_name.to_string());
        self.dataset_name_to_id
            .insert(dataset_name.to_string(), dataset_id);
        dataset_id
    }

    /// Get dataset ID from name (internal method).
    pub fn get_dataset_id(&self, dataset_name: &str) -> Option<u32> {
        self.dataset_name_to_id.get(dataset_name).copied()
    }

    /// Get dataset name from ID (internal method).
    pub fn get_dataset_name(&self, dataset_id: u32) -> Option<&String> {
        self.dataset_names.get(&dataset_id)
    }

    /// Get a mutable reference to the interner for adding entities
    #[getter]
    pub fn interner(&self) -> StringInterner {
        self.interner.clone()
    }

    /// Add a collection to the frame, migrating its interner and dataset mappings
    pub fn add_collection(&mut self, name: &str, collection: EntityCollection) {
        // Migrate collection's dataset mappings to frame's centralized system
        for dataset_name in collection.dataset_name_to_id().keys() {
            self.declare_dataset(dataset_name);
        }

        // TODO: Ideally, we'd update the collection's entities to use the frame's dataset IDs
        // For now, assuming the collection's entities are already using consistent IDs

        self.collections.insert(name.to_string(), collection);
    }

    /// Create and add a collection with entity data in one step
    pub fn add_method(
        &mut self,
        method_name: &str,
        entity_data: Vec<HashMap<String, Vec<String>>>,
    ) {
        // Auto-declare any new datasets found in the data
        for entity_dict in &entity_data {
            for dataset_name in entity_dict.keys() {
                self.declare_dataset(dataset_name);
            }
        }

        // Use with_capacity to pre-allocate based on entity count
        let mut collection =
            EntityCollection::with_capacity(method_name.to_string(), entity_data.len());
        collection.add_entities_with_datasets(
            entity_data,
            &mut self.interner,
            &self.dataset_name_to_id,
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
        self.dataset_names.values().cloned().collect()
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

        if let Some(&dataset_id) = self.dataset_name_to_id.get(dataset_name) {
            Ok(entity.has_dataset_id(dataset_id))
        } else {
            Ok(false)
        }
    }
}
