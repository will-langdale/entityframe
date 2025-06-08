use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

/// String interning for efficient record ID storage and lookup.
#[pyclass]
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
        if let Some(&id) = self.string_to_id.get(s) {
            return id;
        }
        
        let arc_str: Arc<str> = Arc::from(s);
        let id = self.strings.len() as u32;
        self.strings.push(arc_str.clone());
        self.string_to_id.insert(arc_str, id);
        id
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
#[pyclass]
pub struct Entity {
    datasets: HashMap<String, RoaringBitmap>,
}

#[pymethods]
impl Entity {
    #[new]
    fn new() -> Self {
        Self {
            datasets: HashMap::new(),
        }
    }

    /// Add a record ID to a dataset.
    fn add_record(&mut self, dataset: &str, record_id: u32) {
        self.datasets
            .entry(dataset.to_string())
            .or_insert_with(RoaringBitmap::new)
            .insert(record_id);
    }

    /// Add multiple record IDs to a dataset.
    fn add_records(&mut self, dataset: &str, record_ids: Vec<u32>) {
        let bitmap = self.datasets
            .entry(dataset.to_string())
            .or_insert_with(RoaringBitmap::new);
        
        for id in record_ids {
            bitmap.insert(id);
        }
    }

    /// Get record IDs for a dataset as a list.
    fn get_records(&self, dataset: &str) -> PyResult<Vec<u32>> {
        match self.datasets.get(dataset) {
            Some(bitmap) => Ok(bitmap.iter().collect()),
            None => Ok(Vec::new()),
        }
    }

    /// Get all dataset names.
    fn get_datasets(&self) -> Vec<String> {
        self.datasets.keys().cloned().collect()
    }

    /// Check if the entity contains records in a given dataset.
    fn has_dataset(&self, dataset: &str) -> bool {
        self.datasets.contains_key(dataset)
    }

    /// Get the total number of records across all datasets.
    fn total_records(&self) -> u64 {
        self.datasets.values().map(|bitmap| bitmap.len()).sum()
    }

    /// Compute Jaccard similarity with another entity.
    fn jaccard_similarity(&self, other: &Entity) -> f64 {
        let mut union_size = 0u64;
        let mut intersection_size = 0u64;

        // Get all unique dataset names
        let mut all_datasets: std::collections::HashSet<&String> = std::collections::HashSet::new();
        all_datasets.extend(self.datasets.keys());
        all_datasets.extend(other.datasets.keys());

        for dataset in all_datasets {
            let self_bitmap = self.datasets.get(dataset);
            let other_bitmap = other.datasets.get(dataset);

            match (self_bitmap, other_bitmap) {
                (Some(a), Some(b)) => {
                    intersection_size += (a & b).len();
                    union_size += (a | b).len();
                }
                (Some(a), None) => {
                    union_size += a.len();
                }
                (None, Some(b)) => {
                    union_size += b.len();
                }
                (None, None) => {
                    // Both empty for this dataset, no contribution
                }
            }
        }

        if union_size == 0 {
            1.0 // Both entities are empty
        } else {
            intersection_size as f64 / union_size as f64
        }
    }
}

/// Return a hello message from Rust.
#[pyfunction]
fn hello_rust() -> PyResult<String> {
    Ok("Hello from Rust!".to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_rust, m)?)?;
    m.add_class::<StringInterner>()?;
    m.add_class::<Entity>()?;
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
        assert!((similarity - 3.0/7.0).abs() < 1e-10);
    }
}