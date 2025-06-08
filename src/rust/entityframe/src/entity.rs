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
