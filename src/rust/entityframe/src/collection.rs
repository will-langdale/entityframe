use pyo3::prelude::*;
use std::collections::HashMap;

use crate::entity::Entity;
use crate::interner::StringInterner;

/// EntityCollection: A collection of entities from a single process (like pandas Series)
#[pyclass]
#[derive(Clone)]
pub struct EntityCollection {
    pub entities: Vec<Entity>,
    process_name: String,
    // Own interner for standalone mode
    interner: StringInterner,
    // Keep track of dataset name-to-ID mapping for proper entity API support
    dataset_name_to_id: HashMap<String, u32>,
}

#[pymethods]
impl EntityCollection {
    #[new]
    pub fn new(process_name: &str) -> Self {
        Self {
            entities: Vec::new(),
            process_name: process_name.to_string(),
            interner: StringInterner::new(),
            dataset_name_to_id: HashMap::new(),
        }
    }

    /// Create EntityCollection with pre-allocated capacity for better performance.
    #[staticmethod]
    pub fn with_capacity(process_name: String, capacity: usize) -> Self {
        Self {
            entities: Vec::with_capacity(capacity),
            process_name,
            interner: StringInterner::with_capacity(capacity * 10), // Estimate for strings
            dataset_name_to_id: HashMap::new(),
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

    /// Get access to the collection's interner (for standalone mode)
    #[getter]
    pub fn interner(&self) -> StringInterner {
        self.interner.clone()
    }

    /// Get the size of the collection's interner
    pub fn interner_size(&self) -> usize {
        self.interner.len()
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

    /// Check if an entity at given index has a dataset (using collection's name mapping)
    pub fn entity_has_dataset(&self, entity_index: usize, dataset_name: &str) -> PyResult<bool> {
        let entity = self.entities.get(entity_index).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyIndexError, _>("Entity index out of range")
        })?;

        if let Some(&dataset_id) = self.dataset_name_to_id.get(dataset_name) {
            Ok(entity.has_dataset_id(dataset_id))
        } else {
            Ok(false)
        }
    }

    /// Add entities to this collection using collection's own interner (standalone mode)
    pub fn add_entities_standalone(&mut self, entity_data: Vec<HashMap<String, Vec<String>>>) {
        // Use collection's own interner for standalone operation
        for entity_dict in &entity_data {
            for dataset_name in entity_dict.keys() {
                if !self.dataset_name_to_id.contains_key(dataset_name) {
                    let dataset_id = self.interner.intern(dataset_name);
                    self.dataset_name_to_id
                        .insert(dataset_name.clone(), dataset_id);
                }
            }
        }

        // Handle entities inline to avoid borrowing issues
        self.entities.reserve(entity_data.len());

        for entity_dict in entity_data {
            let mut entity = Entity::new();

            for (dataset_name, record_ids) in entity_dict {
                let dataset_id = *self.dataset_name_to_id.get(&dataset_name).unwrap();

                let mut interned_record_ids = Vec::with_capacity(record_ids.len());
                for record_id in record_ids {
                    interned_record_ids.push(self.interner.intern(&record_id));
                }

                entity.add_records_by_id(dataset_id, interned_record_ids);
            }

            self.entities.push(entity);
        }
    }

    /// Add entities to this collection using external shared interner (for cross-collection comparison)
    /// Returns the updated interner since PyO3 doesn't support mutable references
    pub fn add_entities(
        &mut self,
        entity_data: Vec<HashMap<String, Vec<String>>>,
        mut external_interner: StringInterner,
    ) -> StringInterner {
        // Use external interner for dataset IDs - this ensures collections using the same interner
        // will have consistent dataset IDs for comparison
        for entity_dict in &entity_data {
            for dataset_name in entity_dict.keys() {
                if !self.dataset_name_to_id.contains_key(dataset_name) {
                    let dataset_id = external_interner.intern(dataset_name);
                    self.dataset_name_to_id
                        .insert(dataset_name.clone(), dataset_id);
                }
            }
        }

        // Copy external interner's strings to our own interner for consistency
        for entity_dict in &entity_data {
            for record_ids in entity_dict.values() {
                for record_id in record_ids {
                    self.interner.intern(record_id);
                }
            }
        }

        let dataset_mapping = self.dataset_name_to_id.clone();
        self.add_entities_with_datasets(entity_data, &mut external_interner, &dataset_mapping);
        external_interner
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
                    i.into_pyobject(py)?.unbind().into(),
                );
                comparison.insert(
                    "process1".to_string(),
                    self.process_name.clone().into_pyobject(py)?.unbind().into(),
                );
                comparison.insert(
                    "process2".to_string(),
                    other
                        .process_name
                        .clone()
                        .into_pyobject(py)?
                        .unbind()
                        .into(),
                );
                comparison.insert(
                    "jaccard".to_string(),
                    jaccard.into_pyobject(py)?.unbind().into(),
                );

                comparisons.push(comparison);
            }

            Ok(comparisons)
        })
    }
}

impl EntityCollection {
    /// Get access to the dataset name-to-ID mapping (for frame migration)
    pub fn dataset_name_to_id(&self) -> &HashMap<String, u32> {
        &self.dataset_name_to_id
    }

    /// Add entities from this process's output, using shared interner and dataset mappings (internal method)
    pub fn add_entities_with_datasets(
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
