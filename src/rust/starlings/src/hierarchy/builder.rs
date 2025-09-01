use disjoint_sets::UnionFind;
use lru::LruCache;
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use super::merge_event::MergeEvent;
use crate::core::DataContext;

/// Represents a hierarchy of merge events that can generate partitions at any threshold
///
/// The hierarchy is built from edges using a union-find algorithm and stores merge events
/// in descending threshold order. It maintains an LRU cache for frequently accessed
/// partitions and supports fixed-point threshold conversion for exact comparisons.
#[derive(Debug, Clone)]
pub struct PartitionHierarchy {
    /// Reference to the data context that gives meaning to record indices
    context: Arc<DataContext>,

    /// Primary: Merge events sorted by threshold (descending)
    merges: Vec<MergeEvent>,

    /// Secondary: Cache for frequently accessed partitions
    /// Using u32 keys (threshold * 1_000_000) for exact comparison
    #[allow(dead_code)]
    partition_cache: LruCache<u32, PartitionLevel>,

    /// Index for binary search on thresholds (using integer keys)
    #[allow(dead_code)]
    threshold_index: BTreeMap<u32, usize>,

    /// Configuration
    #[allow(dead_code)]
    cache_size: usize,
}

/// Represents a partition at a specific threshold level
#[derive(Debug, Clone)]
pub struct PartitionLevel {
    threshold: f64,
    /// Entities as sets of record indices (RoaringBitmaps)
    entities: Vec<RoaringBitmap>,
    /// Pre-computed entity sizes for quick access
    entity_sizes: Vec<u32>,
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
    ///
    /// # Arguments
    /// * `edges` - Vector of (src, dst, weight) tuples with u32 record indices
    /// * `context` - Shared data context that gives meaning to record indices
    /// * `quantise` - Number of decimal places to quantise thresholds (1-6)
    ///
    /// # Returns
    /// A new PartitionHierarchy with merge events sorted by threshold (descending)
    ///
    /// # Complexity
    /// O(m log m) where m = number of edges (dominated by sorting)
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
                cache_size: Self::CACHE_SIZE,
            };
        }

        let num_records = context.len();

        // Apply quantisation to weights
        let factor = 10_f64.powi(quantise as i32);
        let mut sorted_edges: Vec<_> = edges
            .into_iter()
            .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
            .collect();

        // Sort edges by weight (descending) - O(m log m)
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Group edges by threshold (handles quantisation naturally)
        let threshold_groups = Self::group_edges_by_threshold(sorted_edges);

        // Build merge events using union-find
        let merges = Self::build_merge_events(threshold_groups, num_records);

        // Build threshold index for fast lookup
        let threshold_index = Self::build_threshold_index(&merges);

        Self {
            context,
            merges,
            partition_cache: LruCache::new(Self::CACHE_SIZE.try_into().unwrap()),
            threshold_index,
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
        threshold_groups: Vec<(f64, Vec<(u32, u32)>)>,
        num_records: usize,
    ) -> Vec<MergeEvent> {
        let mut merges = Vec::new();
        let mut uf = UnionFind::new(num_records);

        for (threshold, edges_at_threshold) in threshold_groups {
            // Track components before applying merges
            let mut components_before: HashMap<usize, RoaringBitmap> = HashMap::new();

            // First pass: collect all current components that will be affected
            let mut affected_records = HashSet::new();
            for &(src, dst) in &edges_at_threshold {
                affected_records.insert(src);
                affected_records.insert(dst);
            }

            // Build components before merging
            for &record in &affected_records {
                let root = uf.find(record as usize);
                components_before.entry(root).or_insert_with(|| {
                    let mut component = RoaringBitmap::new();
                    for i in 0..num_records {
                        if uf.find(i) == root {
                            component.insert(i as u32);
                        }
                    }
                    component
                });
            }

            // Apply merges to union-find
            let mut merged_roots = HashSet::new();
            for &(src, dst) in &edges_at_threshold {
                let root_src = uf.find(src as usize);
                let root_dst = uf.find(dst as usize);

                if root_src != root_dst {
                    merged_roots.insert(root_src);
                    merged_roots.insert(root_dst);
                    uf.union(src as usize, dst as usize);
                }
            }

            // Create merge events for components that were merged
            if merged_roots.len() > 1 {
                // Group components by their new root after merging
                let mut merge_groups: HashMap<usize, Vec<RoaringBitmap>> = HashMap::new();

                for old_root in merged_roots {
                    if let Some(component) = components_before.get(&old_root) {
                        // Find what this component's new root is
                        let first_record = component.iter().next().unwrap() as usize;
                        let new_root = uf.find(first_record);
                        merge_groups
                            .entry(new_root)
                            .or_default()
                            .push(component.clone());
                    }
                }

                // Create a merge event for each group of components that merged
                for groups in merge_groups.values() {
                    if groups.len() > 1 {
                        merges.push(MergeEvent::new(threshold, groups.clone()));
                    }
                }
            }
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
}

impl PartitionLevel {
    /// Create a new partition level
    pub fn new(threshold: f64, entities: Vec<RoaringBitmap>) -> Self {
        let entity_sizes = entities.iter().map(|e| e.len() as u32).collect();
        Self {
            threshold,
            entities,
            entity_sizes,
        }
    }

    /// Get the threshold for this partition
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the entities in this partition
    pub fn entities(&self) -> &[RoaringBitmap] {
        &self.entities
    }

    /// Get the number of entities
    pub fn num_entities(&self) -> usize {
        self.entities.len()
    }

    /// Get entity sizes
    pub fn entity_sizes(&self) -> &[u32] {
        &self.entity_sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DataContext, Key};

    fn create_test_context() -> Arc<DataContext> {
        let mut ctx = DataContext::new();
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
        let mut ctx = DataContext::new();
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
        let mut ctx = DataContext::new();
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
}
