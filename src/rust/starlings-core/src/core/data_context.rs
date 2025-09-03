use crate::core::key::Key;
use crate::core::record::InternedRecord;
use dashmap::DashMap;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner, Symbol};

#[derive(Debug)]
pub struct DataContext {
    pub records: RwLock<Vec<InternedRecord>>,
    pub source_interner: RwLock<StringInterner<DefaultBackend>>,
    pub identity_map: DashMap<InternedRecord, u32>,
    pub source_index: DashMap<u32, RoaringBitmap>,
    next_record_id: AtomicU32,
}

impl DataContext {
    pub fn new() -> Self {
        DataContext {
            records: RwLock::new(Vec::new()),
            source_interner: RwLock::new(StringInterner::new()),
            identity_map: DashMap::new(),
            source_index: DashMap::new(),
            next_record_id: AtomicU32::new(0),
        }
    }

    /// Thread-safe record interning with lock-free fast path
    pub fn ensure_record(&self, source: &str, key: Key) -> u32 {
        // Get or intern source string (requires write lock, but infrequent)
        let source_id = {
            // Try read-only access first for existing sources
            if let Ok(interner) = self.source_interner.try_read() {
                if let Some(symbol) = interner.get(source) {
                    symbol.to_usize() as u32
                } else {
                    drop(interner);
                    // Need to intern new source
                    self.source_interner
                        .write()
                        .unwrap()
                        .get_or_intern(source)
                        .to_usize() as u32
                }
            } else {
                // Fallback to write lock
                self.source_interner
                    .write()
                    .unwrap()
                    .get_or_intern(source)
                    .to_usize() as u32
            }
        };

        let record = InternedRecord::new(source_id, key);

        // Lock-free fast path: check if record already exists
        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        // Need to create new record - use atomic counter for ID
        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        // Try to insert the mapping
        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                // Someone else inserted it first - return their ID
                *entry.get()
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                // We won the race - insert our record
                entry.insert(record_id);

                // Add to records vector (requires write lock)
                {
                    let mut records = self.records.write().unwrap();
                    // Ensure vector is large enough
                    while records.len() <= record_id as usize {
                        records.push(InternedRecord::new(0, Key::U32(0))); // placeholder
                    }
                    records[record_id as usize] = record;
                }

                // Update source index (lock-free)
                self.source_index
                    .entry(source_id)
                    .or_default()
                    .insert(record_id);

                record_id
            }
        }
    }

    /// Thread-safe record interning with attributes
    pub fn ensure_record_with_attributes(
        &self,
        source: &str,
        key: Key,
        attributes: HashMap<String, String>,
    ) -> u32 {
        // Get or intern source and attribute strings
        let (source_id, interned_attrs) = {
            let mut interner = self.source_interner.write().unwrap();
            let source_id = interner.get_or_intern(source).to_usize() as u32;

            let mut interned_attrs = HashMap::new();
            for (k, v) in attributes {
                let key_id = interner.get_or_intern(k).to_usize() as u32;
                let val_id = interner.get_or_intern(v).to_usize() as u32;
                interned_attrs.insert(key_id, val_id);
            }
            (source_id, interned_attrs)
        };

        let record = InternedRecord::with_attributes(source_id, key, interned_attrs);

        // Lock-free fast path: check if record already exists
        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        // Need to create new record
        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(record_id);

                // Add to records vector
                {
                    let mut records = self.records.write().unwrap();
                    while records.len() <= record_id as usize {
                        records.push(InternedRecord::new(0, Key::U32(0)));
                    }
                    records[record_id as usize] = record;
                }

                // Update source index
                self.source_index
                    .entry(source_id)
                    .or_default()
                    .insert(record_id);

                record_id
            }
        }
    }

    pub fn get_record(&self, id: u32) -> Option<InternedRecord> {
        let records = self.records.read().unwrap();
        records.get(id as usize).cloned()
    }

    pub fn len(&self) -> usize {
        self.next_record_id.load(Ordering::Relaxed) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_source_name(&self, source_id: u32) -> Option<String> {
        let interner = self.source_interner.read().unwrap();
        let symbol = DefaultSymbol::try_from_usize(source_id as usize)?;
        interner.resolve(symbol).map(|s| s.to_string())
    }

    pub fn get_records_by_source(&self, source_name: &str) -> Option<Vec<u32>> {
        let interner = self.source_interner.read().unwrap();
        let symbol = interner.get(source_name)?;
        let source_id = symbol.to_usize() as u32;
        self.source_index
            .get(&source_id)
            .map(|bitmap| bitmap.iter().collect())
    }

    /// Reserve space for the expected number of records (for performance)
    pub fn reserve(&self, additional: usize) {
        let mut records = self.records.write().unwrap();
        records.reserve(additional);
    }
}

impl Default for DataContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_deduplication() {
        let ctx = DataContext::new();

        let id1 = ctx.ensure_record("source1", Key::String("key1".to_string()));
        let id2 = ctx.ensure_record("source1", Key::String("key1".to_string()));
        let id3 = ctx.ensure_record("source1", Key::String("key2".to_string()));

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_different_sources_different_records() {
        let ctx = DataContext::new();

        let id1 = ctx.ensure_record("source1", Key::U32(42));
        let id2 = ctx.ensure_record("source2", Key::U32(42));

        assert_ne!(id1, id2);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_source_index() {
        let ctx = DataContext::new();

        ctx.ensure_record("source1", Key::String("a".to_string()));
        ctx.ensure_record("source1", Key::String("b".to_string()));
        ctx.ensure_record("source2", Key::String("c".to_string()));
        ctx.ensure_record("source1", Key::String("d".to_string()));

        let source1_records = ctx.get_records_by_source("source1").unwrap();
        let source2_records = ctx.get_records_by_source("source2").unwrap();

        assert_eq!(source1_records.len(), 3);
        assert_eq!(source2_records.len(), 1);
    }

    #[test]
    fn test_record_with_attributes() {
        let ctx = DataContext::new();

        let mut attrs1 = HashMap::new();
        attrs1.insert("name".to_string(), "Alice".to_string());
        attrs1.insert("age".to_string(), "30".to_string());

        let mut attrs2 = HashMap::new();
        attrs2.insert("name".to_string(), "Alice".to_string());
        attrs2.insert("age".to_string(), "30".to_string());

        let mut attrs3 = HashMap::new();
        attrs3.insert("name".to_string(), "Bob".to_string());

        let id1 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs1);
        let id2 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs2);
        let id3 = ctx.ensure_record_with_attributes("people", Key::U32(1), attrs3);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_index_stability() {
        let ctx = DataContext::new();

        let ids: Vec<u32> = (0..100)
            .map(|i| ctx.ensure_record("test", Key::U32(i)))
            .collect();

        for (i, &id) in ids.iter().enumerate() {
            assert_eq!(id, i as u32);
            let record = ctx.get_record(id).unwrap();
            assert_eq!(record.key, Key::U32(i as u32));
        }
    }

    #[test]
    fn test_get_source_name() {
        let ctx = DataContext::new();

        ctx.ensure_record("source_a", Key::U32(1));
        ctx.ensure_record("source_b", Key::U32(2));

        assert_eq!(ctx.get_source_name(0), Some("source_a".to_string()));
        assert_eq!(ctx.get_source_name(1), Some("source_b".to_string()));
        assert_eq!(ctx.get_source_name(999), None);
    }
}
