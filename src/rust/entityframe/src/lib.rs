use pyo3::prelude::*;
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
use std::collections::{hash_map::Entry, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// String interning for efficient record ID storage and lookup.
#[pyclass]
#[derive(Clone)]
pub struct StringInterner {
    strings: Vec<Arc<str>>,
    string_to_id: FxHashMap<Arc<str>, u32>,
}

#[pymethods]
impl StringInterner {
    #[new]
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            string_to_id: FxHashMap::default(),
        }
    }

    /// Intern a string and return its ID.
    fn intern(&mut self, s: &str) -> u32 {
        // Use entry API to avoid double hashing
        let next_id = self.strings.len() as u32;
        match self.string_to_id.entry(Arc::from(s)) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let arc_str = entry.key().clone();
                self.strings.push(arc_str);
                *entry.insert(next_id)
            }
        }
    }

    /// Create interner with pre-allocated capacity for better performance
    #[staticmethod]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: Vec::with_capacity(capacity),
            string_to_id: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Get the string for a given ID.
    fn get_string(&self, id: u32) -> PyResult<&str> {
        self.strings
            .get(id as usize)
            .map(|s| s.as_ref())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Invalid string ID"))
    }

    /// Get the number of interned strings.
    fn len(&self) -> usize {
        self.strings.len()
    }

    /// Check if the interner is empty.
    fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Core entity representation using roaring bitmaps for record ID sets.
/// Uses interned dataset IDs (u32) instead of strings for massive memory savings.
#[pyclass]
#[derive(Clone)]
pub struct Entity {
    datasets: HashMap<u32, RoaringBitmap>,
}

#[pymethods]
impl Entity {
    #[new]
    fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    /// Add a record ID to a dataset using interned dataset ID (internal method).
    fn add_record_by_id(&mut self, dataset_id: u32, record_id: u32) {
        self.datasets
            .entry(dataset_id)
            .or_default()
            .insert(record_id);
    }

    /// Add multiple record IDs to a dataset using interned dataset ID (internal method).
    fn add_records_by_id(&mut self, dataset_id: u32, record_ids: Vec<u32>) {
        let bitmap = self.datasets.entry(dataset_id).or_default();
        // Use RoaringBitmap's bulk insertion method for better performance
        bitmap.extend(&record_ids);
    }

    /// Add a record ID to a dataset (Python API - requires string lookup).
    fn add_record(&mut self, dataset: &str, record_id: u32) {
        // For Python API compatibility, we use a simple hash of the dataset name
        // In practice, this should be used with proper interner context
        let dataset_id = self.hash_dataset_name(dataset);
        self.add_record_by_id(dataset_id, record_id);
    }

    /// Add multiple record IDs to a dataset (Python API - requires string lookup).
    fn add_records(&mut self, dataset: &str, record_ids: Vec<u32>) {
        let dataset_id = self.hash_dataset_name(dataset);
        self.add_records_by_id(dataset_id, record_ids);
    }

    /// Simple hash function for dataset names (used in Python API).
    fn hash_dataset_name(&self, dataset: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        dataset.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Get record IDs for a dataset as a list (Python API).
    fn get_records(&self, dataset: &str) -> PyResult<Vec<u32>> {
        let dataset_id = self.hash_dataset_name(dataset);
        match self.datasets.get(&dataset_id) {
            Some(bitmap) => Ok(bitmap.iter().collect()),
            None => Ok(Vec::new()),
        }
    }

    /// Get record IDs for a dataset by ID (internal method).
    fn get_records_by_id(&self, dataset_id: u32) -> Vec<u32> {
        match self.datasets.get(&dataset_id) {
            Some(bitmap) => bitmap.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Get all dataset IDs (internal method for performance).
    fn get_dataset_ids(&self) -> Vec<u32> {
        self.datasets.keys().copied().collect()
    }

    /// Get all dataset names (Python API - limited without interner context).
    fn get_datasets(&self) -> Vec<String> {
        // For Python API compatibility, return placeholder names
        // In practice, this should be called through EntityCollection which has interner access
        self.datasets
            .keys()
            .map(|id| format!("dataset_{}", id))
            .collect()
    }

    /// Check if the entity contains records in a given dataset (Python API).
    fn has_dataset(&self, dataset: &str) -> bool {
        let dataset_id = self.hash_dataset_name(dataset);
        self.datasets.contains_key(&dataset_id)
    }

    /// Check if the entity contains records for a dataset ID (internal method).
    fn has_dataset_id(&self, dataset_id: u32) -> bool {
        self.datasets.contains_key(&dataset_id)
    }

    /// Get the total number of records across all datasets.
    fn total_records(&self) -> u64 {
        self.datasets.values().map(|bitmap| bitmap.len()).sum()
    }

    /// Compute Jaccard similarity with another entity.
    fn jaccard_similarity(&self, other: &Entity) -> f64 {
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

/// EntityCollection: A collection of entities from a single process (like pandas Series)
#[pyclass]
#[derive(Clone)]
pub struct EntityCollection {
    entities: Vec<Entity>,
    process_name: String,
}

#[pymethods]
impl EntityCollection {
    #[new]
    fn new(process_name: &str) -> Self {
        Self {
            entities: Vec::new(),
            process_name: process_name.to_string(),
        }
    }

    /// Create EntityCollection with pre-allocated capacity for better performance.
    #[staticmethod]
    fn with_capacity(process_name: String, capacity: usize) -> Self {
        Self {
            entities: Vec::with_capacity(capacity),
            process_name,
        }
    }

    /// Legacy method for backward compatibility (will auto-declare datasets)
    fn add_entities(
        &mut self,
        entity_data: Vec<HashMap<String, Vec<String>>>,
        interner: &mut StringInterner,
    ) {
        // For backward compatibility, we'll use a simple hash-based approach
        // This is less efficient than the Frame-managed approach
        self.entities.reserve(entity_data.len());

        for entity_dict in entity_data {
            let mut entity = Entity::new();

            for (dataset_name, record_ids) in entity_dict {
                // Use hash-based dataset ID for compatibility
                let dataset_id = {
                    use std::collections::hash_map::DefaultHasher;
                    let mut hasher = DefaultHasher::new();
                    dataset_name.hash(&mut hasher);
                    hasher.finish() as u32
                };

                let mut interned_record_ids = Vec::with_capacity(record_ids.len());
                for record_id in record_ids {
                    interned_record_ids.push(interner.intern(&record_id));
                }

                entity.add_records_by_id(dataset_id, interned_record_ids);
            }

            self.entities.push(entity);
        }
    }

    /// Get all entities in this collection
    fn get_entities(&self) -> Vec<Entity> {
        self.entities.clone()
    }

    /// Get the process name for this collection
    #[getter]
    fn process_name(&self) -> &str {
        &self.process_name
    }

    /// Get the number of entities in this collection
    fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if the collection is empty
    fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get the total number of records across all entities
    fn total_records(&self) -> u64 {
        self.entities
            .iter()
            .map(|entity| entity.total_records())
            .sum()
    }

    /// Get an entity by index
    fn get_entity(&self, index: usize) -> PyResult<Entity> {
        self.entities.get(index).cloned().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })
    }

    /// Compare this collection with another collection entity-by-entity
    fn compare_with(&self, other: &EntityCollection) -> PyResult<Vec<HashMap<String, PyObject>>> {
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
                comparison.insert("entity_index".to_string(), i.into_py(py));
                comparison.insert(
                    "process1".to_string(),
                    self.process_name.clone().into_py(py),
                );
                comparison.insert(
                    "process2".to_string(),
                    other.process_name.clone().into_py(py),
                );
                comparison.insert("jaccard".to_string(), jaccard.into_py(py));

                comparisons.push(comparison);
            }

            Ok(comparisons)
        })
    }
}

impl EntityCollection {
    /// Add entities from this process's output, using shared interner and dataset mappings (internal method)
    fn add_entities_with_datasets(
        &mut self,
        entity_data: Vec<HashMap<String, Vec<String>>>,
        interner: &mut StringInterner,
        dataset_name_to_id: &HashMap<String, u32>,
    ) {
        // Pre-allocate space for entities
        self.entities.reserve(entity_data.len());

        // Build entities using pre-interned dataset IDs
        for entity_dict in entity_data {
            let mut entity = Entity::new();

            for (dataset_name, record_ids) in entity_dict {
                // Get the pre-interned dataset ID
                let dataset_id = *dataset_name_to_id.get(&dataset_name).unwrap_or_else(|| {
                    panic!(
                        "Dataset '{}' not declared in Frame. Use declare_dataset() first.",
                        dataset_name
                    )
                });

                // Pre-allocate the interned_ids vector
                let mut interned_record_ids = Vec::with_capacity(record_ids.len());

                // Intern the record ID strings and convert to u32 IDs
                for record_id in record_ids {
                    interned_record_ids.push(interner.intern(&record_id));
                }

                // Use the efficient internal method with dataset ID
                entity.add_records_by_id(dataset_id, interned_record_ids);
            }

            self.entities.push(entity);
        }
    }
}

/// EntityFrame: A collection of EntityCollections with shared interner (like pandas DataFrame)
#[pyclass]
pub struct EntityFrame {
    collections: HashMap<String, EntityCollection>,
    interner: StringInterner,
    // Dataset name interning for massive memory savings
    dataset_names: HashMap<u32, String>,      // ID -> name
    dataset_name_to_id: HashMap<String, u32>, // name -> ID
}

#[pymethods]
impl EntityFrame {
    #[new]
    fn new() -> Self {
        Self {
            collections: HashMap::new(),
            interner: StringInterner::new(),
            dataset_names: HashMap::new(),
            dataset_name_to_id: HashMap::new(),
        }
    }

    /// Create EntityFrame with pre-declared datasets for better performance.
    #[staticmethod]
    fn with_datasets(dataset_names: Vec<String>) -> Self {
        let mut frame = Self::new();
        for name in dataset_names {
            frame.declare_dataset(&name);
        }
        frame
    }

    /// Declare a dataset name upfront for efficient interning.
    fn declare_dataset(&mut self, dataset_name: &str) -> u32 {
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
    fn get_dataset_id(&self, dataset_name: &str) -> Option<u32> {
        self.dataset_name_to_id.get(dataset_name).copied()
    }

    /// Get dataset name from ID (internal method).
    fn get_dataset_name(&self, dataset_id: u32) -> Option<&String> {
        self.dataset_names.get(&dataset_id)
    }

    /// Get a mutable reference to the interner for adding entities
    #[getter]
    fn interner(&self) -> StringInterner {
        self.interner.clone()
    }

    /// Add a collection to the frame
    fn add_collection(&mut self, name: &str, collection: EntityCollection) {
        self.collections.insert(name.to_string(), collection);
    }

    /// Create and add a collection with entity data in one step
    fn add_method(&mut self, method_name: &str, entity_data: Vec<HashMap<String, Vec<String>>>) {
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
    fn get_collection_names(&self) -> Vec<String> {
        self.collections.keys().cloned().collect()
    }

    /// Get a collection by name
    fn get_collection(&self, name: &str) -> Option<EntityCollection> {
        self.collections.get(name).cloned()
    }

    /// Compare two collections and return similarities
    fn compare_collections(
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
    fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Get the total number of entities across all collections
    fn total_entities(&self) -> usize {
        self.collections
            .values()
            .map(|collection| collection.len())
            .sum()
    }

    /// Get the total number of unique strings in the interner
    fn interner_size(&self) -> usize {
        self.interner.len()
    }

    /// Get all declared dataset names
    fn get_dataset_names(&self) -> Vec<String> {
        self.dataset_names.values().cloned().collect()
    }

    /// Check if an entity in a collection has a dataset (with proper name resolution)
    fn entity_has_dataset(
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

    // Legacy methods for backward compatibility
    /// Get method names (alias for get_collection_names)
    fn get_method_names(&self) -> Vec<String> {
        self.get_collection_names()
    }

    /// Get entities for a method (alias for get_collection)
    fn get_entities(&self, method_name: &str) -> Option<Vec<Entity>> {
        self.collections
            .get(method_name)
            .map(|collection| collection.get_entities())
    }

    /// Compare methods (alias for compare_collections)
    fn compare_methods(
        &self,
        method1: &str,
        method2: &str,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        self.compare_collections(method1, method2)
    }

    /// Get method count (alias for collection_count)
    fn method_count(&self) -> usize {
        self.collection_count()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StringInterner>()?;
    m.add_class::<Entity>()?;
    m.add_class::<EntityCollection>()?;
    m.add_class::<EntityFrame>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner() {
        let mut interner = StringInterner::new();

        // Test interning
        let id1 = interner.intern("hello");
        let id2 = interner.intern("world");
        let id3 = interner.intern("hello"); // Should return same ID

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 0); // Same as id1

        // Test retrieval
        assert_eq!(interner.get_string(id1).unwrap(), "hello");
        assert_eq!(interner.get_string(id2).unwrap(), "world");

        // Test length
        assert_eq!(interner.len(), 2);
        assert!(!interner.is_empty());
    }

    #[test]
    fn test_entity() {
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

        // Test datasets
        let datasets = entity.get_datasets();
        assert!(datasets.contains(&"customers".to_string()));
        assert!(datasets.contains(&"transactions".to_string()));

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
    fn test_entity_collection_creation() {
        let collection = EntityCollection::new("splink");
        assert_eq!(collection.process_name(), "splink");
        assert_eq!(collection.len(), 0);
        assert!(collection.is_empty());
        assert_eq!(collection.total_records(), 0);
    }

    #[test]
    fn test_entity_collection_add_entities() {
        let mut collection = EntityCollection::new("splink");
        let mut interner = StringInterner::new();

        let entity_data = vec![
            {
                let mut entity = HashMap::new();
                entity.insert(
                    "customers".to_string(),
                    vec!["cust_001".to_string(), "cust_002".to_string()],
                );
                entity.insert("transactions".to_string(), vec!["txn_100".to_string()]);
                entity
            },
            {
                let mut entity = HashMap::new();
                entity.insert("customers".to_string(), vec!["cust_003".to_string()]);
                entity.insert(
                    "transactions".to_string(),
                    vec!["txn_101".to_string(), "txn_102".to_string()],
                );
                entity
            },
        ];

        collection.add_entities(entity_data, &mut interner);

        assert_eq!(collection.len(), 2);
        assert!(!collection.is_empty());
        assert_eq!(collection.total_records(), 6); // 3 + 3 records

        // Check that the interner was used
        assert_eq!(interner.len(), 5); // cust_001, cust_002, cust_003, txn_100, txn_101, txn_102

        // Test entity retrieval
        let entity1 = collection.get_entity(0).unwrap();
        assert_eq!(entity1.total_records(), 3);

        let entity2 = collection.get_entity(1).unwrap();
        assert_eq!(entity2.total_records(), 3);
    }

    #[test]
    fn test_entity_collection_compare() {
        let mut collection1 = EntityCollection::new("splink");
        let mut collection2 = EntityCollection::new("dedupe");
        let mut interner = StringInterner::new();

        // Add identical data to both collections
        let entity_data = vec![{
            let mut entity = HashMap::new();
            entity.insert(
                "customers".to_string(),
                vec!["cust_001".to_string(), "cust_002".to_string()],
            );
            entity
        }];

        collection1.add_entities(entity_data.clone(), &mut interner);
        collection2.add_entities(entity_data, &mut interner);

        // Since we can't easily test the PyObject return in unit tests,
        // we can verify the logic by comparing entities directly
        let entities1 = collection1.get_entities();
        let entities2 = collection2.get_entities();

        assert_eq!(entities1.len(), entities2.len());

        // Entities should be identical (Jaccard = 1.0)
        let similarity = entities1[0].jaccard_similarity(&entities2[0]);
        assert_eq!(similarity, 1.0);
    }

    #[test]
    fn test_entity_frame_creation() {
        let frame = EntityFrame::new();
        assert_eq!(frame.collection_count(), 0);
        assert_eq!(frame.total_entities(), 0);
        assert!(frame.get_collection_names().is_empty());
        assert_eq!(frame.interner_size(), 0);
    }

    #[test]
    fn test_entity_frame_add_collection() {
        let mut frame = EntityFrame::new();
        let mut collection = EntityCollection::new("splink");
        let mut interner = frame.interner();

        let entity_data = vec![{
            let mut entity = HashMap::new();
            entity.insert("customers".to_string(), vec!["cust_001".to_string()]);
            entity
        }];

        collection.add_entities(entity_data, &mut interner);
        frame.add_collection("splink", collection);

        assert_eq!(frame.collection_count(), 1);
        assert_eq!(frame.total_entities(), 1);
        assert!(frame.get_collection_names().contains(&"splink".to_string()));

        let retrieved_collection = frame.get_collection("splink").unwrap();
        assert_eq!(retrieved_collection.len(), 1);
        assert_eq!(retrieved_collection.process_name(), "splink");
    }

    #[test]
    fn test_entity_frame_add_method() {
        let mut frame = EntityFrame::new();

        let entity_data = vec![
            {
                let mut entity = HashMap::new();
                entity.insert(
                    "customers".to_string(),
                    vec!["cust_001".to_string(), "cust_002".to_string()],
                );
                entity.insert("transactions".to_string(), vec!["txn_100".to_string()]);
                entity
            },
            {
                let mut entity = HashMap::new();
                entity.insert("customers".to_string(), vec!["cust_003".to_string()]);
                entity.insert(
                    "transactions".to_string(),
                    vec!["txn_101".to_string(), "txn_102".to_string()],
                );
                entity
            },
        ];

        frame.add_method("splink", entity_data);

        assert_eq!(frame.collection_count(), 1);
        assert_eq!(frame.total_entities(), 2);
        assert_eq!(frame.interner_size(), 5); // 5 unique strings

        let collection = frame.get_collection("splink").unwrap();
        assert_eq!(collection.len(), 2);
        assert_eq!(collection.process_name(), "splink");
    }

    #[test]
    fn test_entity_frame_shared_interner() {
        let mut frame = EntityFrame::new();

        // Add two collections with overlapping record IDs
        let method1_data = vec![{
            let mut entity = HashMap::new();
            entity.insert(
                "customers".to_string(),
                vec!["cust_001".to_string(), "cust_002".to_string()],
            );
            entity
        }];

        let method2_data = vec![{
            let mut entity = HashMap::new();
            entity.insert(
                "customers".to_string(),
                vec!["cust_001".to_string(), "cust_003".to_string()],
            );
            entity
        }];

        frame.add_method("splink", method1_data);
        frame.add_method("dedupe", method2_data);

        // The interner should have deduplicated "cust_001"
        assert_eq!(frame.interner_size(), 3); // cust_001, cust_002, cust_003
        assert_eq!(frame.collection_count(), 2);
        assert_eq!(frame.total_entities(), 2);
    }

    #[test]
    fn test_entity_frame_legacy_methods() {
        let mut frame = EntityFrame::new();

        let entity_data = vec![{
            let mut entity = HashMap::new();
            entity.insert("customers".to_string(), vec!["cust_001".to_string()]);
            entity
        }];

        frame.add_method("splink", entity_data);

        // Test legacy method aliases
        assert_eq!(frame.get_method_names(), frame.get_collection_names());
        assert_eq!(frame.method_count(), frame.collection_count());
        assert!(frame.get_entities("splink").is_some());
    }
}
