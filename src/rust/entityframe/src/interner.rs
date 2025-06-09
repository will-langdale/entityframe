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

impl StringInterner {
    /// Get the string for a given ID (internal method returning Option).
    pub fn get_string_internal(&self, id: u32) -> Option<&str> {
        self.strings.get(id as usize).map(|s| s.as_ref())
    }

    /// Merge another interner into this one, returning the ID remapping table
    pub fn merge(&mut self, other: &StringInterner) -> std::collections::HashMap<u32, u32> {
        use std::collections::HashMap;
        let mut remapping = HashMap::new();

        for (old_id, string) in other.strings.iter().enumerate() {
            let new_id = self.intern(string);
            if old_id as u32 != new_id {
                remapping.insert(old_id as u32, new_id);
            }
        }

        remapping
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner_basic() {
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
    fn test_string_interner_merge() {
        let mut interner1 = StringInterner::new();
        let mut interner2 = StringInterner::new();

        // Add strings to both interners
        let id1_a = interner1.intern("apple");
        let id1_b = interner1.intern("banana");
        let id1_c = interner1.intern("cherry");

        let id2_b = interner2.intern("banana"); // Same string, different ID
        let id2_d = interner2.intern("date");
        let id2_e = interner2.intern("elderberry");

        // Before merge
        assert_eq!(id1_a, 0);
        assert_eq!(id1_b, 1);
        assert_eq!(id1_c, 2);
        assert_eq!(id2_b, 0); // Different ID for "banana"
        assert_eq!(id2_d, 1);
        assert_eq!(id2_e, 2);

        // Merge interner2 into interner1
        let remapping = interner1.merge(&interner2);

        // Check remapping table
        assert_eq!(remapping.get(&0), Some(&1)); // "banana": 0 -> 1
        assert_eq!(remapping.get(&1), Some(&3)); // "date": 1 -> 3
        assert_eq!(remapping.get(&2), Some(&4)); // "elderberry": 2 -> 4

        // Check final interner
        assert_eq!(interner1.len(), 5);
        assert_eq!(interner1.get_string_internal(0), Some("apple"));
        assert_eq!(interner1.get_string_internal(1), Some("banana"));
        assert_eq!(interner1.get_string_internal(2), Some("cherry"));
        assert_eq!(interner1.get_string_internal(3), Some("date"));
        assert_eq!(interner1.get_string_internal(4), Some("elderberry"));
    }
}
