use crate::core::key::Key;
use crate::core::record::InternedRecord;
use fxhash::FxHashMap;
use roaring::RoaringBitmap;
use std::collections::HashMap;
use string_interner::{DefaultBackend, DefaultSymbol, StringInterner, Symbol};

#[derive(Debug)]
pub struct DataContext {
    pub records: Vec<InternedRecord>,
    pub source_interner: StringInterner<DefaultBackend>,
    pub identity_map: FxHashMap<InternedRecord, u32>,
    pub source_index: HashMap<u32, RoaringBitmap>,
}

impl DataContext {
    pub fn new() -> Self {
        DataContext {
            records: Vec::new(),
            source_interner: StringInterner::new(),
            identity_map: FxHashMap::default(),
            source_index: HashMap::new(),
        }
    }

    pub fn ensure_record(&mut self, source: &str, key: Key) -> u32 {
        let source_id = self.source_interner.get_or_intern(source).to_usize() as u32;

        let record = InternedRecord::new(source_id, key);

        if let Some(&existing_id) = self.identity_map.get(&record) {
            existing_id
        } else {
            let record_id = self.records.len() as u32;
            self.records.push(record.clone());
            self.identity_map.insert(record, record_id);

            self.source_index
                .entry(source_id)
                .or_default()
                .insert(record_id);

            record_id
        }
    }

    pub fn ensure_record_with_attributes(
        &mut self,
        source: &str,
        key: Key,
        attributes: HashMap<String, String>,
    ) -> u32 {
        let source_id = self.source_interner.get_or_intern(source).to_usize() as u32;

        let mut interned_attrs = HashMap::new();
        for (k, v) in attributes {
            let key_id = self.source_interner.get_or_intern(k).to_usize() as u32;
            let val_id = self.source_interner.get_or_intern(v).to_usize() as u32;
            interned_attrs.insert(key_id, val_id);
        }

        let record = InternedRecord::with_attributes(source_id, key, interned_attrs);

        if let Some(&existing_id) = self.identity_map.get(&record) {
            existing_id
        } else {
            let record_id = self.records.len() as u32;
            self.records.push(record.clone());
            self.identity_map.insert(record, record_id);

            self.source_index
                .entry(source_id)
                .or_default()
                .insert(record_id);

            record_id
        }
    }

    pub fn get_record(&self, id: u32) -> Option<&InternedRecord> {
        self.records.get(id as usize)
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn get_source_name(&self, source_id: u32) -> Option<&str> {
        let symbol = DefaultSymbol::try_from_usize(source_id as usize)?;
        self.source_interner.resolve(symbol)
    }

    pub fn get_records_by_source(&self, source_name: &str) -> Option<Vec<u32>> {
        let source_id = self.source_interner.get(source_name)?.to_usize() as u32;
        self.source_index
            .get(&source_id)
            .map(|bitmap| bitmap.iter().collect())
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
        let mut ctx = DataContext::new();

        let id1 = ctx.ensure_record("source1", Key::String("key1".to_string()));
        let id2 = ctx.ensure_record("source1", Key::String("key1".to_string()));
        let id3 = ctx.ensure_record("source1", Key::String("key2".to_string()));

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_different_sources_different_records() {
        let mut ctx = DataContext::new();

        let id1 = ctx.ensure_record("source1", Key::U32(42));
        let id2 = ctx.ensure_record("source2", Key::U32(42));

        assert_ne!(id1, id2);
        assert_eq!(ctx.len(), 2);
    }

    #[test]
    fn test_source_index() {
        let mut ctx = DataContext::new();

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
        let mut ctx = DataContext::new();

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
        let mut ctx = DataContext::new();

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
        let mut ctx = DataContext::new();

        ctx.ensure_record("source_a", Key::U32(1));
        ctx.ensure_record("source_b", Key::U32(2));

        assert_eq!(ctx.get_source_name(0), Some("source_a"));
        assert_eq!(ctx.get_source_name(1), Some("source_b"));
        assert_eq!(ctx.get_source_name(999), None);
    }
}
