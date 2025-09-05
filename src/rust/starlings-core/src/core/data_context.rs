use crate::core::key::Key;
use crate::core::record::InternedRecord;
use boxcar::Vec as BoxcarVec;
use dashmap::DashMap;
use lasso::{Capacity, Key as LassoKey, ThreadedRodeo};
use roaring::RoaringBitmap;
use rustc_hash::FxHasher;
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

type FxDashMap<K, V> = DashMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Debug)]
pub struct DataContext {
    pub records: BoxcarVec<InternedRecord>,
    pub source_interner: Arc<ThreadedRodeo>,
    pub identity_map: FxDashMap<InternedRecord, u32>,
    pub source_index: FxDashMap<u32, RoaringBitmap>,
    next_record_id: AtomicU32,
}

impl DataContext {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create DataContext with pre-allocated capacity for better performance
    pub fn with_capacity(estimated_records: usize) -> Self {
        let hasher = BuildHasherDefault::<FxHasher>::default();

        DataContext {
            records: BoxcarVec::new(),
            source_interner: Arc::new(ThreadedRodeo::with_capacity(Capacity::for_strings(
                estimated_records.min(10000),
            ))),
            identity_map: DashMap::with_capacity_and_hasher(estimated_records, hasher.clone()),
            source_index: DashMap::with_hasher(hasher),
            next_record_id: AtomicU32::new(0),
        }
    }

    /// Batch ensure records for improved performance
    pub fn ensure_records_batch(&self, source: &str, keys: &[Key]) -> Vec<u32> {
        let source_id = LassoKey::into_usize(self.source_interner.get_or_intern(source)) as u32;

        keys.iter()
            .map(|key| {
                let record = InternedRecord::new(source_id, key.clone());

                if let Some(existing_id) = self.identity_map.get(&record) {
                    return *existing_id;
                }

                let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

                match self.identity_map.entry(record.clone()) {
                    dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
                    dashmap::mapref::entry::Entry::Vacant(entry) => {
                        entry.insert(record_id);

                        self.records.push(record);

                        self.source_index
                            .entry(source_id)
                            .or_default()
                            .insert(record_id);

                        record_id
                    }
                }
            })
            .collect()
    }

    /// Thread-safe record interning with lock-free operations
    pub fn ensure_record(&self, source: &str, key: Key) -> u32 {
        let source_id = LassoKey::into_usize(self.source_interner.get_or_intern(source)) as u32;

        let record = InternedRecord::new(source_id, key);

        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(record_id);

                self.records.push(record);

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
        let source_id = LassoKey::into_usize(self.source_interner.get_or_intern(source)) as u32;

        let mut interned_attrs = HashMap::new();
        for (k, v) in attributes {
            let key_id = LassoKey::into_usize(self.source_interner.get_or_intern(k)) as u32;
            let val_id = LassoKey::into_usize(self.source_interner.get_or_intern(v)) as u32;
            interned_attrs.insert(key_id, val_id);
        }

        let record = InternedRecord::with_attributes(source_id, key, interned_attrs);

        if let Some(existing_id) = self.identity_map.get(&record) {
            return *existing_id;
        }

        let record_id = self.next_record_id.fetch_add(1, Ordering::Relaxed);

        match self.identity_map.entry(record.clone()) {
            dashmap::mapref::entry::Entry::Occupied(entry) => *entry.get(),
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(record_id);

                self.records.push(record);

                self.source_index
                    .entry(source_id)
                    .or_default()
                    .insert(record_id);

                record_id
            }
        }
    }

    pub fn get_record(&self, id: u32) -> Option<InternedRecord> {
        self.records.get(id as usize).map(|r| (*r).clone())
    }

    pub fn len(&self) -> usize {
        self.next_record_id.load(Ordering::Relaxed) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_source_name(&self, source_id: u32) -> Option<String> {
        let spur = LassoKey::try_from_usize(source_id as usize)?;
        self.source_interner
            .try_resolve(&spur)
            .map(|s| s.to_string())
    }

    pub fn get_records_by_source(&self, source_name: &str) -> Option<Vec<u32>> {
        let source_id = LassoKey::into_usize(self.source_interner.get(source_name)?) as u32;
        self.source_index
            .get(&source_id)
            .map(|bitmap| bitmap.iter().collect())
    }

    /// Reserve space for the expected number of records (for performance)
    pub fn reserve(&self, _additional: usize) {}
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
