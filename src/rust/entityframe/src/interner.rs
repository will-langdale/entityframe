use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

/// String interning for efficient record ID storage and lookup.
#[pyclass]
#[derive(Clone)]
pub struct StringInterner {
    strings: Vec<Arc<str>>,
    string_to_id: FxHashMap<Arc<str>, u32>,
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl StringInterner {
    #[new]
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            string_to_id: FxHashMap::default(),
        }
    }

    /// Intern a string and return its ID.
    pub fn intern(&mut self, s: &str) -> u32 {
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
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            strings: Vec::with_capacity(capacity),
            string_to_id: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
        }
    }

    /// Get the string for a given ID.
    pub fn get_string(&self, id: u32) -> PyResult<&str> {
        self.strings
            .get(id as usize)
            .map(|s| s.as_ref())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Invalid string ID"))
    }

    /// Get the number of interned strings.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Check if the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}
