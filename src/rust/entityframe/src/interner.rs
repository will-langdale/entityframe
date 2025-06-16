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
    sorted_ids: Option<Vec<u32>>,
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
            sorted_ids: None,
        }
    }

    /// Intern a string and return its ID.
    pub fn intern(&mut self, s: &str) -> u32 {
        // Invalidate sorted cache when new string is added
        self.sorted_ids = None;

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
            sorted_ids: None,
        }
    }

    /// Get the string for a given ID.
    pub fn get_string(&self, id: u32) -> PyResult<&str> {
        self.strings
            .get(id as usize)
            .map(|s| s.as_ref())
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Invalid string ID"))
    }

    /// Look up a string and return its ID if it exists (does not add new strings).
    pub fn lookup(&self, s: &str) -> Option<u32> {
        self.string_to_id.get(s).copied()
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

    /// Fast unsafe string access for high-performance scenarios
    /// Only use when you're certain the ID is valid
    ///
    /// # Safety
    /// The caller must ensure that `id` is a valid index into the strings vector.
    /// Using an invalid ID will result in undefined behavior.
    pub unsafe fn get_string_unchecked(&self, id: u32) -> &str {
        self.strings.get_unchecked(id as usize).as_ref()
    }

    /// Get string with bounds checking but optimized for hot paths
    pub fn get_string_fast(&self, id: u32) -> Option<&str> {
        // Direct array access - should be fastest possible
        if (id as usize) < self.strings.len() {
            Some(unsafe { self.get_string_unchecked(id) })
        } else {
            None
        }
    }

    /// Bulk string lookup for batch processing optimisation
    /// Returns HashMap mapping IDs to their string values
    pub fn bulk_get_strings(&self, ids: &[u32]) -> std::collections::HashMap<u32, &str> {
        let mut result = std::collections::HashMap::with_capacity(ids.len());
        for &id in ids {
            if let Some(string) = self.get_string_internal(id) {
                result.insert(id, string);
            }
        }
        result
    }

    /// Get IDs in sorted string order (cached).
    /// This method is not exposed to Python and requires mutable access for caching.
    pub fn get_sorted_ids(&mut self) -> &[u32] {
        if self.sorted_ids.is_none() {
            let mut ids: Vec<u32> = (0..self.strings.len() as u32).collect();
            ids.sort_by(|&a, &b| self.strings[a as usize].cmp(&self.strings[b as usize]));
            self.sorted_ids = Some(ids);
        }
        self.sorted_ids.as_ref().unwrap()
    }

    /// Batch intern multiple strings and compute sorted order in single pass.
    /// Returns (string_ids, sorted_ids) for efficient entity creation.
    /// This achieves O(S log S) instead of O(S log S) per entity for batch processing.
    pub fn batch_intern_with_sort(&mut self, strings: &[String]) -> (Vec<u32>, Vec<u32>) {
        // First pass: intern all strings
        let mut string_ids = Vec::with_capacity(strings.len());

        for s in strings {
            let id = self.intern(s);
            string_ids.push(id);
        }

        // Second pass: get sorted order for these specific strings
        // This is the key optimisation - we only sort the local strings, not all strings
        let mut local_sorted_ids = string_ids.clone();
        local_sorted_ids.sort_by(|&a, &b| self.strings[a as usize].cmp(&self.strings[b as usize]));

        (string_ids, local_sorted_ids)
    }

    /// Batch intern strings from multiple datasets, computing sorted order for each dataset.
    /// This is optimised for entity creation where each dataset needs separate sorting.
    /// Returns HashMap mapping dataset_id -> (record_ids, sorted_record_ids).
    pub fn batch_intern_by_dataset(
        &mut self,
        dataset_records: &std::collections::HashMap<u32, Vec<String>>,
    ) -> std::collections::HashMap<u32, (Vec<u32>, Vec<u32>)> {
        use std::collections::HashMap;

        let mut results = HashMap::new();

        // Process each dataset
        for (&dataset_id, records) in dataset_records {
            let (record_ids, sorted_record_ids) = self.batch_intern_with_sort(records);
            results.insert(dataset_id, (record_ids, sorted_record_ids));
        }

        results
    }

    /// Merge another interner into this one, returning the ID remapping table
    pub fn merge(&mut self, other: &StringInterner) -> std::collections::HashMap<u32, u32> {
        use std::collections::HashMap;

        // Invalidate sorted cache since we're adding new strings
        self.sorted_ids = None;

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

        // Test lookup
        assert_eq!(interner.lookup("hello"), Some(0));
        assert_eq!(interner.lookup("world"), Some(1));
        assert_eq!(interner.lookup("nonexistent"), None);

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

    #[test]
    fn test_batch_intern_with_sort() {
        let mut interner = StringInterner::new();

        // Test batch processing with sorting
        let strings = vec![
            "zebra".to_string(),
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
        ];

        let (string_ids, sorted_ids) = interner.batch_intern_with_sort(&strings);

        // Check that strings were interned
        assert_eq!(string_ids.len(), 4);
        assert_eq!(sorted_ids.len(), 4);

        // Verify sorted order (apple, banana, cherry, zebra alphabetically)
        let sorted_strings: Vec<&str> = sorted_ids
            .iter()
            .map(|&id| interner.get_string_internal(id).unwrap())
            .collect();
        assert_eq!(sorted_strings, vec!["apple", "banana", "cherry", "zebra"]);

        // Test with duplicate strings
        let strings2 = vec![
            "banana".to_string(), // Already exists
            "date".to_string(),   // New
            "apple".to_string(),  // Already exists
        ];

        let (string_ids2, sorted_ids2) = interner.batch_intern_with_sort(&strings2);
        assert_eq!(string_ids2.len(), 3);
        assert_eq!(sorted_ids2.len(), 3);

        // Verify sorted order for this batch (apple, banana, date)
        let sorted_strings2: Vec<&str> = sorted_ids2
            .iter()
            .map(|&id| interner.get_string_internal(id).unwrap())
            .collect();
        assert_eq!(sorted_strings2, vec!["apple", "banana", "date"]);
    }

    #[test]
    fn test_batch_intern_by_dataset() {
        use std::collections::HashMap;

        let mut interner = StringInterner::new();

        // Prepare dataset records
        let mut dataset_records = HashMap::new();
        dataset_records.insert(
            0,
            vec![
                "cust_3".to_string(),
                "cust_1".to_string(),
                "cust_2".to_string(),
            ],
        );
        dataset_records.insert(1, vec!["order_b".to_string(), "order_a".to_string()]);

        let results = interner.batch_intern_by_dataset(&dataset_records);

        // Check that we got results for both datasets
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));

        // Check dataset 0 (customers)
        let (cust_ids, cust_sorted) = &results[&0];
        assert_eq!(cust_ids.len(), 3);
        assert_eq!(cust_sorted.len(), 3);

        // Verify sorted order for customers (cust_1, cust_2, cust_3)
        let cust_sorted_strings: Vec<&str> = cust_sorted
            .iter()
            .map(|&id| interner.get_string_internal(id).unwrap())
            .collect();
        assert_eq!(cust_sorted_strings, vec!["cust_1", "cust_2", "cust_3"]);

        // Check dataset 1 (orders)
        let (order_ids, order_sorted) = &results[&1];
        assert_eq!(order_ids.len(), 2);
        assert_eq!(order_sorted.len(), 2);

        // Verify sorted order for orders (order_a, order_b)
        let order_sorted_strings: Vec<&str> = order_sorted
            .iter()
            .map(|&id| interner.get_string_internal(id).unwrap())
            .collect();
        assert_eq!(order_sorted_strings, vec!["order_a", "order_b"]);
    }
}
