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
        // Pre-allocate space for entities
        self.entities.reserve(entity_data.len());

        for entity_dict in entity_data {
            let mut entity = Entity::new();

            for (dataset_name, record_ids) in entity_dict {
                // Get or create dataset ID using frame's system
                let dataset_id = if let Some(&existing_id) = dataset_name_to_id.get(&dataset_name) {
                    existing_id
                } else {
                    let new_id = interner.intern(&dataset_name);
                    dataset_name_to_id.insert(dataset_name.clone(), new_id);
                    new_id
                };

                // Intern all record IDs using shared interner
                let mut interned_record_ids = Vec::with_capacity(record_ids.len());
                for record_id in record_ids {
                    interned_record_ids.push(interner.intern(&record_id));
                }

                entity.add_records_by_id(dataset_id, interned_record_ids);
            }

            self.entities.push(entity);
        }
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
}
