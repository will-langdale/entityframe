use lru::LruCache;
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::bitmap_pool::BitmapPool;
use super::merge_event::MergeEvent;
use super::partition::PartitionLevel;
use super::union_find::UnionFind;
use crate::core::DataContext;

/// Hierarchy of merge events that can generate partitions at any threshold
#[derive(Debug, Clone)]
pub struct PartitionHierarchy {
    context: Arc<DataContext>,
    merges: Vec<MergeEvent>,
    partition_cache: LruCache<u32, PartitionLevel>,
    #[allow(dead_code)]
    threshold_index: BTreeMap<u32, usize>,
    bitmap_pool: BitmapPool,
    #[allow(dead_code)]
    cache_size: usize,
}

impl PartitionHierarchy {
    /// Fixed-point precision factor - supports up to 6 decimal places
    pub const PRECISION_FACTOR: f64 = 1_000_000.0;
    /// Default LRU cache size
    pub const CACHE_SIZE: usize = 10;

    /// Convert f64 threshold to u32 key for exact comparison and caching
    fn threshold_to_key(threshold: f64) -> u32 {
        (threshold.clamp(0.0, 1.0) * Self::PRECISION_FACTOR).round() as u32
    }

    /// Convert u32 key back to f64 threshold
    #[allow(dead_code)]
    fn key_to_threshold(key: u32) -> f64 {
        key as f64 / Self::PRECISION_FACTOR
    }

    /// Build a hierarchy from edges using union-find algorithm
    pub fn from_edges(
        edges: Vec<(u32, u32, f64)>,
        context: Arc<DataContext>,
        quantise: u32,
    ) -> Self {
        // Validate quantise is between 1 and 6
        assert!(
            (1..=6).contains(&quantise),
            "quantise must be between 1 and 6, got {}",
            quantise
        );

        if edges.is_empty() {
            return Self {
                context,
                merges: Vec::new(),
                partition_cache: LruCache::new(Self::CACHE_SIZE.try_into().unwrap()),
                threshold_index: BTreeMap::new(),
                bitmap_pool: BitmapPool::new(),
                cache_size: Self::CACHE_SIZE,
            };
        }

        let num_records = context.len();

        // Store edge count for pool scaling before consuming edges
        let num_edges = edges.len();

        // Apply quantisation to weights
        let factor = 10_f64.powi(quantise as i32);
        let mut quantised_edges: Vec<_> = edges
            .into_iter()
            .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
            .collect();

        // Sort edges by threshold (highest first)
        quantised_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Create a temporary instance with scaled bitmap pool for construction
        let mut temp_hierarchy = Self {
            context: context.clone(),
            merges: Vec::new(),
            partition_cache: LruCache::new(Self::CACHE_SIZE.try_into().unwrap()),
            threshold_index: BTreeMap::new(),
            bitmap_pool: BitmapPool::new_for_scale(num_edges),
            cache_size: Self::CACHE_SIZE,
        };

        // Group edges by threshold and build merge events
        let threshold_groups = Self::group_edges_by_threshold(quantised_edges);
        let merges = temp_hierarchy.build_merge_events(threshold_groups, num_records);

        // Build threshold index for fast lookup
        let threshold_index = Self::build_threshold_index(&merges);

        // Reuse the bitmap pool from temporary instance for efficiency
        Self {
            context,
            merges,
            partition_cache: LruCache::new(Self::CACHE_SIZE.try_into().unwrap()),
            threshold_index,
            bitmap_pool: temp_hierarchy.bitmap_pool,
            cache_size: Self::CACHE_SIZE,
        }
    }

    /// Group consecutive edges with the same threshold
    fn group_edges_by_threshold(sorted_edges: Vec<(u32, u32, f64)>) -> Vec<(f64, Vec<(u32, u32)>)> {
        if sorted_edges.is_empty() {
            return Vec::new();
        }

        let mut threshold_groups = Vec::new();
        let mut current_threshold = sorted_edges[0].2;
        let mut current_group = Vec::new();

        for (src, dst, weight) in sorted_edges {
            if (weight - current_threshold).abs() < f64::EPSILON {
                current_group.push((src, dst));
            } else {
                threshold_groups.push((current_threshold, current_group));
                current_threshold = weight;
                current_group = vec![(src, dst)];
            }
        }

        if !current_group.is_empty() {
            threshold_groups.push((current_threshold, current_group));
        }

        threshold_groups
    }

    /// Build merge events from grouped edges using union-find
    fn build_merge_events(
        &mut self,
        threshold_groups: Vec<(f64, Vec<(u32, u32)>)>,
        num_records: usize,
    ) -> Vec<MergeEvent> {
        let mut merges = Vec::new();
        let mut uf = UnionFind::new(num_records);
        let mut active_components: HashMap<usize, RoaringBitmap> = HashMap::new();

        for (threshold, edges_at_threshold) in threshold_groups.iter() {
            let mut processed_pairs = HashSet::new();

            for &(src, dst) in edges_at_threshold {
                let root_src = uf.find(src as usize);
                let root_dst = uf.find(dst as usize);

                if root_src != root_dst {
                    let pair = if root_src < root_dst {
                        (root_src, root_dst)
                    } else {
                        (root_dst, root_src)
                    };

                    if !processed_pairs.contains(&pair) {
                        processed_pairs.insert(pair);

                        let mut merging_components = Vec::new();

                        if let Some(component_src) = active_components.remove(&root_src) {
                            merging_components.push(component_src);
                        } else {
                            let (mut bitmap, _) = self.bitmap_pool.get(1);
                            bitmap.insert(src);
                            merging_components.push(bitmap);
                        }

                        if let Some(component_dst) = active_components.remove(&root_dst) {
                            merging_components.push(component_dst);
                        } else {
                            let (mut bitmap, _) = self.bitmap_pool.get(1);
                            bitmap.insert(dst);
                            merging_components.push(bitmap);
                        }

                        uf.union(src as usize, dst as usize);
                        let new_root = uf.find(src as usize);

                        if merging_components.len() > 1 {
                            let (mut merged_component, _) =
                                self.bitmap_pool
                                    .get(merging_components.iter().map(|b| b.len()).sum::<u64>()
                                        as u32);
                            for old_component in &merging_components {
                                merged_component |= old_component;
                            }

                            merges.push(MergeEvent::new(*threshold, merging_components));
                            active_components.insert(new_root, merged_component);
                        }
                    }
                }
            }
        }

        for (_, bitmap) in active_components {
            self.bitmap_pool
                .put(bitmap, super::bitmap_pool::PoolSizeClass::Small);
        }

        merges
    }

    /// Build index for binary search on thresholds
    fn build_threshold_index(merges: &[MergeEvent]) -> BTreeMap<u32, usize> {
        merges
            .iter()
            .enumerate()
            .map(|(idx, merge)| (Self::threshold_to_key(merge.threshold), idx))
            .collect()
    }

    /// Get the number of records in the context
    pub fn num_records(&self) -> usize {
        self.context.len()
    }

    /// Get all merge events (for testing)
    pub fn merge_events(&self) -> &[MergeEvent] {
        &self.merges
    }

    /// Get a partition at a specific threshold
    pub fn at_threshold(&mut self, threshold: f64) -> &PartitionLevel {
        // Validate threshold
        assert!(
            (0.0..=1.0).contains(&threshold),
            "Threshold must be between 0.0 and 1.0, got {}",
            threshold
        );

        let key = Self::threshold_to_key(threshold);

        // Check if already cached
        if self.partition_cache.contains(&key) {
            return self.partition_cache.get(&key).unwrap();
        }

        // Reconstruct the partition
        let partition = self.reconstruct_at_threshold(threshold);

        // Store in cache and return reference
        self.partition_cache.put(key, partition);
        self.partition_cache.get(&key).unwrap()
    }

    /// Reconstruct a partition at a specific threshold
    fn reconstruct_at_threshold(&self, threshold: f64) -> PartitionLevel {
        let num_records = self.context.len();

        // Start with all records as singletons
        let mut uf = UnionFind::new(num_records);

        // Apply all merges with threshold >= requested threshold
        for merge in &self.merges {
            if merge.threshold >= threshold {
                // Collect all records from all merging groups
                let mut all_records = Vec::new();
                for group in &merge.merging_groups {
                    for record in group.iter() {
                        all_records.push(record);
                    }
                }

                // Union all records together (using first as representative)
                if let Some(&first) = all_records.first() {
                    for &record in all_records.iter().skip(1) {
                        uf.union(first as usize, record as usize);
                    }
                }
            } else {
                // Merges are sorted by descending threshold, so we can stop
                break;
            }
        }

        // Convert union-find to partition with entities
        let mut entities_map: HashMap<usize, RoaringBitmap> = HashMap::new();

        // Include ALL records from the context (handles isolates)
        for record_idx in 0..num_records {
            let root = uf.find(record_idx);
            entities_map.entry(root).or_insert_with(|| {
                // Estimate entity size for pool selection - use small size as default
                let estimated_size = (num_records / 100).max(10) as u32; // Reasonable default
                let (bitmap, _) = self.bitmap_pool.get(estimated_size);
                bitmap
            });
            entities_map
                .get_mut(&root)
                .unwrap()
                .insert(record_idx as u32);
        }

        // Convert HashMap to Vec of entities
        let entities: Vec<RoaringBitmap> = entities_map.into_values().collect();

        PartitionLevel::new(threshold, entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataContext, Key};

    fn create_test_context() -> Arc<DataContext> {
        let ctx = DataContext::new();
        ctx.ensure_record("test", Key::String("A".to_string()));
        ctx.ensure_record("test", Key::String("B".to_string()));
        ctx.ensure_record("test", Key::String("C".to_string()));
        Arc::new(ctx)
    }

    #[test]
    fn test_empty_edges() {
        let ctx = create_test_context();
        let hierarchy = PartitionHierarchy::from_edges(vec![], ctx.clone(), 2);

        assert_eq!(hierarchy.merge_events().len(), 0);
        assert_eq!(hierarchy.num_records(), 3);
    }

    #[test]
    fn test_three_node_graph() {
        let ctx = create_test_context();

        // Create a simple 3-node graph: A-B (0.8), B-C (0.6)
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);
        let merges = hierarchy.merge_events();

        // Should have two merge events: one at 0.8 and one at 0.6
        assert_eq!(merges.len(), 2);

        // First merge should be at threshold 0.8 (A-B)
        assert_eq!(merges[0].threshold, 0.8);
        assert_eq!(merges[0].merging_groups.len(), 2); // Two singletons merge

        // Second merge should be at threshold 0.6 ((A,B)-C)
        assert_eq!(merges[1].threshold, 0.6);
        assert_eq!(merges[1].merging_groups.len(), 2); // Group {A,B} merges with {C}
    }

    #[test]
    fn test_disconnected_components() {
        let ctx = DataContext::new();
        // Create 4 records: A, B, C, D
        ctx.ensure_record("test", Key::String("A".to_string()));
        ctx.ensure_record("test", Key::String("B".to_string()));
        ctx.ensure_record("test", Key::String("C".to_string()));
        ctx.ensure_record("test", Key::String("D".to_string()));
        let ctx = Arc::new(ctx);

        // Create two disconnected components: A-B (0.8), C-D (0.7)
        let edges = vec![
            (0, 1, 0.8), // A-B
            (2, 3, 0.7), // C-D
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);
        let merges = hierarchy.merge_events();

        // Should have two independent merge events
        assert_eq!(merges.len(), 2);

        // Each merge should involve exactly 2 singletons
        for merge in merges {
            assert_eq!(merge.merging_groups.len(), 2);
            for group in &merge.merging_groups {
                assert_eq!(group.len(), 1); // Each group is a singleton
            }
        }
    }

    #[test]
    fn test_same_threshold_edges_nway_merge() {
        let ctx = DataContext::new();
        // Create 4 records: A, B, C, D
        ctx.ensure_record("test", Key::String("A".to_string()));
        ctx.ensure_record("test", Key::String("B".to_string()));
        ctx.ensure_record("test", Key::String("C".to_string()));
        ctx.ensure_record("test", Key::String("D".to_string()));
        let ctx = Arc::new(ctx);

        // All edges have the same threshold - should create n-way merge
        let edges = vec![
            (0, 1, 0.5), // A-B
            (1, 2, 0.5), // B-C
            (2, 3, 0.5), // C-D
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);
        let merges = hierarchy.merge_events();

        // Should create merge events as the union-find processes the edges
        // The exact number depends on the order of processing, but all should be at 0.5
        assert!(!merges.is_empty());
        for merge in merges {
            assert_eq!(merge.threshold, 0.5);
        }
    }

    #[test]
    fn test_quantisation_enforcement() {
        let ctx = create_test_context();

        // Test with high precision input that should be quantised
        let edges = vec![
            (0, 1, 0.123456789), // Should be quantised to 0.12 with quantise=2
        ];

        let hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);
        let merges = hierarchy.merge_events();

        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0].threshold, 0.12); // Quantised to 2 decimal places
    }

    #[test]
    fn test_threshold_conversion() {
        // Test fixed-point conversion
        let threshold = 0.123456;
        let key = PartitionHierarchy::threshold_to_key(threshold);
        let back = PartitionHierarchy::key_to_threshold(key);

        // Should be close due to fixed-point precision
        assert!((back - threshold).abs() < 0.000001);
    }

    #[test]
    #[should_panic(expected = "quantise must be between 1 and 6")]
    fn test_invalid_quantise() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];

        // Should panic with quantise=0
        PartitionHierarchy::from_edges(edges, ctx, 0);
    }

    #[test]
    fn test_threshold_0_one_entity() {
        let ctx = create_test_context();

        // Create edges connecting all nodes
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // At threshold 0.0, all records should be in one entity
        let partition = hierarchy.at_threshold(0.0);

        assert_eq!(partition.threshold(), 0.0);
        assert_eq!(partition.num_entities(), 1);
        assert_eq!(partition.total_records(), 3);

        // Verify all records are in the same entity
        let entity = &partition.entities()[0];
        assert!(entity.contains(0));
        assert!(entity.contains(1));
        assert!(entity.contains(2));
    }

    #[test]
    fn test_threshold_1_all_singletons() {
        let ctx = create_test_context();

        // Create edges with weights less than 1.0
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.6), // B-C
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // At threshold 1.0, each record should be a singleton
        let partition = hierarchy.at_threshold(1.0);

        assert_eq!(partition.threshold(), 1.0);
        assert_eq!(partition.num_entities(), 3);
        assert_eq!(partition.total_records(), 3);

        // Each entity should have exactly one record
        for entity in partition.entities() {
            assert_eq!(entity.len(), 1);
        }
    }

    #[test]
    fn test_threshold_intermediate() {
        let ctx = create_test_context();

        // Create edges: A-B at 0.8, B-C at 0.4
        let edges = vec![
            (0, 1, 0.8), // A-B
            (1, 2, 0.4), // B-C
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // At threshold 0.5, A-B should be merged but C separate
        let partition = hierarchy.at_threshold(0.5);

        assert_eq!(partition.threshold(), 0.5);
        assert_eq!(partition.num_entities(), 2);

        // Find which entity contains A and B
        let entity_a = partition.find_entity_for_record(0).unwrap();
        let entity_b = partition.find_entity_for_record(1).unwrap();
        let entity_c = partition.find_entity_for_record(2).unwrap();

        // A and B should be in the same entity
        assert_eq!(entity_a, entity_b);
        // C should be in a different entity
        assert_ne!(entity_a, entity_c);
    }

    #[test]
    fn test_isolates_as_singletons() {
        let ctx = DataContext::new();

        // Create 5 records
        for i in 0..5 {
            ctx.ensure_record("test", Key::U32(i));
        }
        let ctx = Arc::new(ctx);

        // Only connect some records, leaving 3 and 4 as isolates
        let edges = vec![
            (0, 1, 0.8), // Connect 0-1
            (1, 2, 0.6), // Connect 1-2
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // At threshold 0.5, should have:
        // - One entity with {0, 1, 2}
        // - Two singleton entities for isolates {3} and {4}
        let partition = hierarchy.at_threshold(0.5);

        assert_eq!(partition.num_entities(), 3);
        assert_eq!(partition.total_records(), 5);

        // Verify isolates exist as singletons
        assert!(partition.contains_record(3));
        assert!(partition.contains_record(4));

        // Count singleton entities
        let singleton_count = partition.entities().iter().filter(|e| e.len() == 1).count();
        assert_eq!(singleton_count, 2); // Two isolates
    }

    #[test]
    fn test_cache_functionality() {
        let ctx = create_test_context();

        let edges = vec![(0, 1, 0.8), (1, 2, 0.6)];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // First access - should reconstruct
        let partition1 = hierarchy.at_threshold(0.7);
        assert_eq!(partition1.threshold(), 0.7);

        // Second access to same threshold - should hit cache
        let partition2 = hierarchy.at_threshold(0.7);
        assert_eq!(partition2.threshold(), 0.7);

        // Access different thresholds to test cache capacity
        for i in 0..15 {
            let threshold = (i as f64 * 0.05 * 100.0).round() / 100.0; // Round to avoid fp precision issues
            let partition = hierarchy.at_threshold(threshold);
            assert!((partition.threshold() - threshold).abs() < 1e-10);
        }

        // Original threshold might be evicted due to LRU cache size limit
        // But should still work correctly
        let partition3 = hierarchy.at_threshold(0.7);
        assert!((partition3.threshold() - 0.7).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Threshold must be between 0.0 and 1.0")]
    fn test_invalid_threshold_negative() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];
        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // Should panic with negative threshold
        hierarchy.at_threshold(-0.1);
    }

    #[test]
    #[should_panic(expected = "Threshold must be between 0.0 and 1.0")]
    fn test_invalid_threshold_too_large() {
        let ctx = create_test_context();
        let edges = vec![(0, 1, 0.5)];
        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // Should panic with threshold > 1.0
        hierarchy.at_threshold(1.1);
    }

    #[test]
    fn test_separate_components_same_threshold() {
        // This test specifically addresses the "single giant component" bug
        let ctx = DataContext::new();
        // Create 6 records: A, B, C, D, E, F
        for i in 0..6 {
            ctx.ensure_record("test", Key::String(format!("{}", (b'A' + i) as char)));
        }
        let ctx = Arc::new(ctx);

        // Create three separate pairs at the same threshold
        let edges = vec![
            (0, 1, 0.9), // A-B
            (2, 3, 0.9), // C-D
            (4, 5, 0.9), // E-F
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // At threshold 0.9, should have 3 components (3 pairs)
        let partition = hierarchy.at_threshold(0.9);
        let num_entities_09 = partition.entities().len();
        assert_eq!(
            num_entities_09, 3,
            "Should have exactly 3 components at threshold 0.9"
        );

        // Verify each component has exactly 2 members
        for entity in partition.entities() {
            assert_eq!(
                entity.len(),
                2,
                "Each component should have exactly 2 members"
            );
        }

        // At threshold 1.0, should have 6 singletons
        let partition_high = hierarchy.at_threshold(1.0);
        assert_eq!(
            partition_high.entities().len(),
            6,
            "Should have 6 singletons at threshold 1.0"
        );
    }

    #[test]
    fn test_hierarchical_component_formation() {
        let ctx = DataContext::new();
        // Create 8 records for complex hierarchy
        for i in 0..8 {
            ctx.ensure_record("test", Key::String(format!("record_{}", i)));
        }
        let ctx = Arc::new(ctx);

        // Hierarchical structure:
        // Threshold 0.9: (0,1), (2,3) -> 2 pairs + 4 singletons = 6 entities
        // Threshold 0.8: (4,5) -> 3 pairs + 2 singletons = 5 entities
        // Threshold 0.7: (6,7) -> 4 pairs = 4 entities
        let edges = vec![
            (0, 1, 0.9), // Pair 1
            (2, 3, 0.9), // Pair 2
            (4, 5, 0.8), // Pair 3 (lower threshold)
            (6, 7, 0.7), // Pair 4 (lowest threshold)
        ];

        let mut hierarchy = PartitionHierarchy::from_edges(edges, ctx, 2);

        // Test hierarchical behavior
        let partition_high = hierarchy.at_threshold(0.95);
        assert_eq!(
            partition_high.entities().len(),
            8,
            "Above all thresholds: 8 singletons"
        );

        let partition_09 = hierarchy.at_threshold(0.9);
        assert_eq!(
            partition_09.entities().len(),
            6,
            "At 0.9: 2 pairs + 4 singletons = 6 entities"
        );

        let partition_08 = hierarchy.at_threshold(0.8);
        assert_eq!(
            partition_08.entities().len(),
            5,
            "At 0.8: 3 pairs + 2 singletons = 5 entities"
        );

        let partition_07 = hierarchy.at_threshold(0.7);
        assert_eq!(
            partition_07.entities().len(),
            4,
            "At 0.7: 4 pairs = 4 entities"
        );

        let partition_low = hierarchy.at_threshold(0.5);
        assert_eq!(
            partition_low.entities().len(),
            4,
            "Below all thresholds: same as 0.7"
        );
    }

    #[test]
    fn test_quantisation_preserves_components() {
        let ctx = DataContext::new();
        // Create 4 records
        for i in 0..4 {
            ctx.ensure_record("test", Key::String(format!("record_{}", i)));
        }
        let ctx = Arc::new(ctx);

        // Test different quantisation levels preserve component structure
        let edges = vec![
            (0, 1, 0.85432), // Should quantise to 0.85
            (2, 3, 0.75678), // Should quantise to 0.76
        ];

        // Test with quantise=2 (2 decimal places)
        let mut hierarchy = PartitionHierarchy::from_edges(edges.clone(), ctx.clone(), 2);

        let partition_high = hierarchy.at_threshold(0.9);
        assert_eq!(
            partition_high.entities().len(),
            4,
            "Above quantised thresholds: 4 singletons"
        );

        let partition_mid = hierarchy.at_threshold(0.8);
        assert_eq!(
            partition_mid.entities().len(),
            3,
            "Between quantised thresholds: 1 pair + 2 singletons"
        );

        let partition_low = hierarchy.at_threshold(0.7);
        assert_eq!(
            partition_low.entities().len(),
            2,
            "Below quantised thresholds: 2 pairs"
        );
    }
}
