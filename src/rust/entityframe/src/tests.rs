#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
        assert!((similarity - 3.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_entity_collection_creation() {
        let collection = EntityCollection::new("splink");
        assert_eq!(collection.process_name(), "splink");
        assert_eq!(collection.len(), 0);
        assert!(collection.is_empty());
        assert_eq!(collection.total_records(), 0);
    }

    #[test]
    fn test_entity_frame_creation() {
        let frame = EntityFrame::new();
        assert_eq!(frame.collection_count(), 0);
        assert_eq!(frame.total_entities(), 0);
        assert!(frame.get_collection_names().is_empty());
        assert_eq!(frame.interner_size(), 0);
    }

    #[test]
    fn test_entity_frame_add_collection() {
        let mut frame = EntityFrame::new();
        let collection = EntityCollection::new("splink");

        frame.add_collection("splink", collection);

        assert_eq!(frame.collection_count(), 1);
        assert_eq!(frame.total_entities(), 0); // Empty collection
        assert!(frame.get_collection_names().contains(&"splink".to_string()));

        let retrieved_collection = frame.get_collection("splink").unwrap();
        assert_eq!(retrieved_collection.len(), 0);
        assert_eq!(retrieved_collection.process_name(), "splink");
    }

    #[test]
    fn test_entity_frame_add_method() {
        let mut frame = EntityFrame::new();

        let entity_data = vec![
            {
                let mut entity = HashMap::new();
                entity.insert(
                    "customers".to_string(),
                    vec!["cust_001".to_string(), "cust_002".to_string()],
                );
                entity.insert("transactions".to_string(), vec!["txn_100".to_string()]);
                entity
            },
            {
                let mut entity = HashMap::new();
                entity.insert("customers".to_string(), vec!["cust_003".to_string()]);
                entity.insert(
                    "transactions".to_string(),
                    vec!["txn_101".to_string(), "txn_102".to_string()],
                );
                entity
            },
        ];

        frame.add_method("splink", entity_data);

        assert_eq!(frame.collection_count(), 1);
        assert_eq!(frame.total_entities(), 2);
        assert_eq!(frame.interner_size(), 5); // 5 unique strings

        let collection = frame.get_collection("splink").unwrap();
        assert_eq!(collection.len(), 2);
        assert_eq!(collection.process_name(), "splink");
    }

    #[test]
    fn test_entity_frame_shared_interner() {
        let mut frame = EntityFrame::new();

        // Add two collections with overlapping record IDs
        let method1_data = vec![{
            let mut entity = HashMap::new();
            entity.insert(
                "customers".to_string(),
                vec!["cust_001".to_string(), "cust_002".to_string()],
            );
            entity
        }];

        let method2_data = vec![{
            let mut entity = HashMap::new();
            entity.insert(
                "customers".to_string(),
                vec!["cust_001".to_string(), "cust_003".to_string()],
            );
            entity
        }];

        frame.add_method("splink", method1_data);
        frame.add_method("dedupe", method2_data);

        // The interner should have deduplicated "cust_001"
        assert_eq!(frame.interner_size(), 3); // cust_001, cust_002, cust_003
        assert_eq!(frame.collection_count(), 2);
        assert_eq!(frame.total_entities(), 2);
    }
}
