# Starlings Design Document B: Technical Implementation

## Layer 2: Computer Science Techniques

### Core Data Structure Architecture

**Multi-collection EntityFrame with Contextual Ownership**

```rust
pub struct EntityFrame {
    // Shared data context (append-only arena)
    context: Arc<DataContext>,
    
    // Multiple hierarchies (collections) over shared record space
    collections: HashMap<CollectionId, PartitionHierarchy>,
    
    // Cross-collection comparison engine
    comparison_engine: CollectionComparer,
    
    // Track garbage for automatic compaction
    garbage_records: RoaringBitmap,
}

pub struct DataContext {
    // Append-only record storage
    records: Vec<InternedRecord>,
    
    // String interning for sources
    source_interner: StringInterner<FxHashMap>,
    
    // Canonical identity mapping
    identity_map: HashMap<(SourceId, Key), RecordIndex>,
    
    // Index for fast lookup
    source_index: HashMap<SourceId, RoaringBitmap>,
}

impl DataContext {
    /// Ensure a record exists in the context (deduplication)
    /// O(1) lookup and optional insertion
    pub fn ensure_record(&mut self, source: &str, key: Key) -> RecordIndex {
        let source_id = self.source_interner.get_or_intern(source);
        let identity = (source_id, key.clone());
        
        if let Some(&idx) = self.identity_map.get(&identity) {
            idx
        } else {
            let idx = self.records.len() as RecordIndex;
            self.records.push(InternedRecord {
                source_id,
                key: key.clone(),
                attributes: None,
            });
            self.identity_map.insert(identity, idx);
            self.source_index.entry(source_id).or_default().insert(idx);
            idx
        }
    }
    
    /// Deep copy the context for collection copying
    pub fn deep_copy(&self) -> Self {
        DataContext {
            records: self.records.clone(),
            source_interner: self.source_interner.clone(),
            identity_map: self.identity_map.clone(),
            source_index: self.source_index.clone(),
        }
    }
}

pub struct InternedRecord {
    source_id: u16,     // Interned source identifier
    key: Key,           // Key within that source (flexible type)
    attributes: Option<RecordData>,  // Optional attribute data
}

/// Flexible key types for record identification
pub enum Key {
    U32(u32),
    U64(u64),
    String(String),  // Keys don't need interning - they're unique by definition
    Bytes(Vec<u8>),
}

/// Record specification for standalone collection construction
pub enum RecordSpec {
    Keys(Vec<Key>),           // Additional records (isolates)
    Entities(Vec<EntityData>), // Pre-grouped entities for hierarchical resolution
}

/// Entity data for hierarchical resolution
pub struct EntityData {
    id: u32,  // Temporary ID for edge references
    members: Vec<(SourceId, Key)>,
}
```

### Contextual Ownership Architecture

**Separation of data and logic with immutable views**

```rust
pub struct Collection {
    // Reference to data (may be shared or owned)
    context: Arc<DataContext>,
    
    // Logical structure (hierarchy of merge events)
    hierarchy: PartitionHierarchy,
    
    // Track if this is a view from a frame
    is_view: bool,
}

impl Collection {
    /// Create standalone collection with owned context
    pub fn from_edges(edges: Vec<(Key, Key, f64)>, 
                     records: Option<RecordSpec>) -> Self {
        let mut context = DataContext::new();
        
        // Add all records mentioned in edges
        for (key1, key2, _) in &edges {
            context.ensure_record("default", key1.clone());
            context.ensure_record("default", key2.clone());
        }
        
        // Add any additional records (isolates or all records - we dedupe)
        if let Some(records) = records {
            match records {
                RecordSpec::Keys(keys) => {
                    for key in keys {
                        context.ensure_record("default", key);  // Deduplicates automatically
                    }
                },
                RecordSpec::Entities(entities) => {
                    // Add all entity members to context
                    for entity in &entities {
                        for (source_id, key) in &entity.members {
                            context.ensure_record_with_source(*source_id, key.clone());
                        }
                    }
                    // Entities are then converted to edges for hierarchy building
                }
            }
        }
        
        let context_arc = Arc::new(context);
        
        // Convert edges to u32 indices for hierarchy construction
        let indexed_edges = edges_to_indices(&edges, &context_arc);
        let hierarchy = PartitionHierarchy::from_edges(indexed_edges, context_arc.clone(), 6);
        
        Collection {
            context: context_arc,
            hierarchy,
            is_view: false,
        }
    }
    
    /// Create standalone collection from pre-resolved entities
    pub fn from_entities(entities: Vec<EntitySet>) -> Self {
        let mut context = DataContext::new();
        
        // Add all records from entities to context
        for entity_set in &entities {
            for (source, key) in entity_set {
                context.ensure_record(source, key.clone());
            }
        }
        
        let context_arc = Arc::new(context);
        
        // Convert entities to edges at threshold 1.0
        // All pairs within each entity get weight 1.0
        let edges = entities_to_edges(&entities, 1.0);
        let hierarchy = PartitionHierarchy::from_edges(edges, context_arc.clone(), 6);
        
        Collection {
            context: context_arc,
            hierarchy,
            is_view: false,
        }
    }
    
    /// Create an explicit owned copy of this collection
    pub fn copy(&self) -> Self {
        let new_context = Arc::new(self.context.deep_copy());
        let mut new_hierarchy = self.hierarchy.clone();
        new_hierarchy.context = new_context.clone();  // Update to new context
        
        Collection {
            context: new_context,
            hierarchy: new_hierarchy,
            is_view: false,
        }
    }
    
    /// Check if this is an immutable view from a frame
    pub fn is_view(&self) -> bool {
        self.is_view
    }
}
```

### Efficient Hierarchical Data Structures

**Merge-event based hierarchy with smart caching and fixed-point thresholds**

```rust
pub struct PartitionHierarchy {
    // Reference to the data context that gives meaning to record indices
    context: Arc<DataContext>,
    
    // Primary: Merge events sorted by threshold (descending)
    merges: Vec<MergeEvent>,
    
    // Secondary: Cache for frequently accessed partitions
    // Using u32 keys (threshold * 1_000_000) for exact comparison
    partition_cache: LruCache<u32, PartitionLevel>,
    
    // Tertiary: State for incremental metric computation
    metric_state: Option<IncrementalMetricState>,
    
    // Index for binary search on thresholds (using integer keys)
    threshold_index: BTreeMap<u32, usize>,
    
    // Configuration
    cache_size: usize,  // Non-configurable size 10 per hierarchy for v1
}

impl PartitionHierarchy {
    const PRECISION_FACTOR: f64 = 1_000_000.0;  // Supports quantize up to 6 decimal places
    
    fn threshold_to_key(threshold: f64) -> u32 {
        (threshold.clamp(0.0, 1.0) * Self::PRECISION_FACTOR).round() as u32
    }
    
    fn key_to_threshold(key: u32) -> f64 {
        key as f64 / Self::PRECISION_FACTOR
    }
}

pub struct MergeEvent {
    threshold: f64,
    
    // Components merging at this threshold (supports n-way)
    merging_components: Vec<ComponentId>,
    
    // The new component formed
    result_component: ComponentId,
    
    // Records involved for incremental updates
    affected_records: RoaringBitmap,
}

pub struct PartitionLevel {
    threshold: f64,
    
    // Entities as sets of record indices (RoaringBitmaps)
    entities: Vec<RoaringBitmap>,
    
    // Pre-computed cheap statistics
    entity_sizes: Vec<u32>,
    
    // Lazy expensive metrics
    lazy_metrics: LazyMetricsCache,
}

pub struct IncrementalMetricState {
    last_threshold: f64,
    last_partition: PartitionLevel,
    contingency_table: SparseContingencyTable,
    affected_entities: RoaringBitmap,
}
```

**Memory analysis**

For 1M records with 1M edges in the Contextual Ownership model:
- DataContext: ~10MB for records + interning
- Merge events: 1M × 50-100 bytes = 50-100MB (includes RoaringBitmaps)
- LRU cache (10 partitions): 10 × 500KB = 5MB
- Total per collection: ~60-115MB
- Shared context when in frame: No duplication
- Total for 5 collections in frame: ~300-500MB (not 5× due to sharing)

### Helper Functions for Hierarchy Construction

```rust
/// Convert entities to edges for hierarchy building
/// O(n²) within each entity, but entities are typically small
/// All pairs within an entity get weight 1.0
fn entities_to_edges(entities: &[EntitySet], weight: f64) -> Vec<(u32, u32, f64)> {
    let mut edges = Vec::new();
    for (entity_id, entity_set) in entities.iter().enumerate() {
        let members: Vec<_> = entity_set.iter().collect();
        // Create edges between all pairs in the entity
        for i in 0..members.len() {
            for j in i+1..members.len() {
                edges.push((members[i].to_index(), members[j].to_index(), weight));
            }
        }
    }
    edges
}

/// Convert Key-based edges to u32 indices using DataContext
/// O(m) where m = number of edges
fn edges_to_indices(edges: &[(Key, Key, f64)], context: &DataContext) -> Vec<(u32, u32, f64)> {
    edges.iter().map(|(k1, k2, w)| {
        let idx1 = context.get_index(k1).expect("Key not in context");
        let idx2 = context.get_index(k2).expect("Key not in context");
        (idx1 as u32, idx2 as u32, *w)
    }).collect()
}

/// Translate all record indices in hierarchy according to map
/// O(m) where m = number of merge events
impl PartitionHierarchy {
    pub fn translate_indices(&mut self, translation_map: &TranslationMap) {
        for merge in &mut self.merges {
            // Translate affected records
            let mut translated_records = RoaringBitmap::new();
            for old_idx in merge.affected_records.iter() {
                if let Some(&new_idx) = translation_map.get(old_idx) {
                    translated_records.insert(new_idx as u32);
                }
            }
            merge.affected_records = translated_records;
            
            // Component IDs may also need translation depending on implementation
        }
    }
}
```

**Adding collections to frames with O(k + m) complexity**

```rust
impl EntityFrame {
    pub fn add_collection(&mut self, name: &str, collection: Collection) {
        // Collections from frames are immutable views
        if collection.is_view {
            panic!("Cannot add a view collection to a frame. Use copy() first.");
        }
        
        if Arc::ptr_eq(&collection.context, &self.context) {
            // Same context, just add hierarchy
            self.collections.insert(name.into(), collection.hierarchy);
        } else {
            // Different context, need assimilation
            let translation = self.assimilate(collection);
            self.collections.insert(name.into(), translation.hierarchy);
        }
    }
    
    fn assimilate(&mut self, collection: Collection) -> AssimilatedCollection {
        // Complexity: O(k) where k = incoming collection's records
        //           + O(m) where m = merge events to rewrite
        let mut translation_map = TranslationMap::new();
        
        // Identity resolution: (source, key) determines identity
        // O(k) iterations with O(1) HashMap lookups
        for (idx, record) in collection.context.records.iter().enumerate() {
            let identity = (record.source_id, &record.key);
            
            if let Some(&existing_idx) = self.context.identity_map.get(&identity) {
                // Record exists, map to existing - O(1)
                translation_map.insert(idx, existing_idx);
            } else {
                // New record, append to context - O(1)
                let new_idx = self.context.records.len();
                self.context.records.push(record.clone());
                self.context.identity_map.insert(identity, new_idx);
                translation_map.insert(idx, new_idx);
            }
        }
        
        // Translate hierarchy to use new indices - O(m)
        let mut new_hierarchy = collection.hierarchy.clone();
        new_hierarchy.translate_indices(&translation_map);
        new_hierarchy.context = self.context.clone();  // Update context reference
        
        AssimilatedCollection {
            hierarchy: new_hierarchy,
            translation_map,
        }
    }
}
```

### Connected Components Algorithm for Hierarchy Construction

**Union-find based merge event extraction with mandatory quantization**

```rust
use disjoint_sets::UnionFind;

impl PartitionHierarchy {
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, 
                     context: Arc<DataContext>,
                     quantize: u32) -> Self {
        // Validate quantize is 1-6
        assert!(quantize >= 1 && quantize <= 6, "quantize must be between 1 and 6");
        
        // Apply quantization
        let factor = 10_f64.powi(quantize as i32);
        let mut sorted_edges: Vec<_> = edges.into_iter()
            .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
            .collect();
        
        // Sort edges by weight (descending) - O(m log m)
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Group edges by threshold (handles quantization naturally)
        let mut threshold_groups: Vec<(f64, Vec<(u32, u32)>)> = Vec::new();
        let mut current_threshold = sorted_edges[0].2;
        let mut current_group = Vec::new();
        
        for (src, dst, weight) in sorted_edges {
            if weight == current_threshold {
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
        
        // Build merge events with proper component tracking
        let mut merges = Vec::new();
        let mut uf = UnionFind::new(num_records);
        let mut component_map: HashMap<usize, ComponentId> = HashMap::new();
        let mut next_component_id = 0;
        
        // Initialize each record as its own component (handles isolates)
        for i in 0..num_records {
            component_map.insert(i, ComponentId(i as u32));
        }
        
        for (threshold, edges_at_threshold) in threshold_groups {
            // Track which components are merging
            let mut merging_groups: HashMap<usize, HashSet<usize>> = HashMap::new();
            
            for (src, dst) in &edges_at_threshold {
                let root_src = uf.find(*src as usize);
                let root_dst = uf.find(*dst as usize);
                
                if root_src != root_dst {
                    merging_groups.entry(root_src).or_default().insert(root_dst);
                    merging_groups.entry(root_dst).or_default().insert(root_src);
                }
            }
            
            // Apply merges
            for (src, dst) in &edges_at_threshold {
                uf.union(*src as usize, *dst as usize);
            }
            
            // Create merge events (handles n-way merges naturally)
            let mut processed = HashSet::new();
            for root in merging_groups.keys() {
                if processed.contains(root) {
                    continue;
                }
                
                // Find all components in this connected merge group
                let mut to_visit = vec![*root];
                let mut merge_group = Vec::new();
                let mut affected = RoaringBitmap::new();
                
                while let Some(current) = to_visit.pop() {
                    if !processed.insert(current) {
                        continue;
                    }
                    
                    merge_group.push(component_map[&current]);
                    
                    // Add all records in this component to affected set
                    for i in 0..num_records {
                        if uf.find(i) == current {
                            affected.insert(i as u32);
                        }
                    }
                    
                    if let Some(neighbors) = merging_groups.get(&current) {
                        to_visit.extend(neighbors.iter().cloned());
                    }
                }
                
                if merge_group.len() > 1 {
                    let result_id = ComponentId(next_component_id);
                    next_component_id += 1;
                    
                    // Update component map for merged roots
                    let new_root = uf.find(*root);
                    component_map.insert(new_root, result_id);
                    
                    merges.push(MergeEvent {
                        threshold,
                        merging_components: merge_group,
                        result_component: result_id,
                        affected_records: affected,
                    });
                }
            }
        }
        
        PartitionHierarchy {
            context,  // Store reference to context
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size: 10,
        }
    }
    
    fn build_threshold_index(merges: &[MergeEvent]) -> BTreeMap<u32, usize> {
        merges.iter()
            .enumerate()
            .map(|(idx, merge)| (Self::threshold_to_key(merge.threshold), idx))
            .collect()
    }
}
```

### Partition Reconstruction from Merge Events

**Efficient reconstruction algorithm with fixed-point thresholds**

```rust
impl PartitionHierarchy {
    pub fn at_threshold(&mut self, threshold: f64) -> Result<&PartitionLevel, HierarchyError> {
        // Validate threshold
        if threshold < 0.0 || threshold > 1.0 {
            return Err(HierarchyError::InvalidThreshold(threshold));
        }
        
        let key = Self::threshold_to_key(threshold);
        
        // Check cache first using integer key
        if self.partition_cache.contains(&key) {
            return Ok(self.partition_cache.get(&key).unwrap());
        }
        
        // Check if cache is full and warn
        if self.partition_cache.len() >= self.cache_size {
            log::debug!("Cache full, evicting LRU partition");
        }
        
        // Reconstruct from merge events
        let partition = self.reconstruct_at_threshold(threshold)?;
        self.partition_cache.put(key, partition);
        Ok(self.partition_cache.get(&key).unwrap())
    }
    
    fn reconstruct_at_threshold(&self, threshold: f64) -> PartitionLevel {
        // Get complete record space from context
        let num_records = self.context.records.len();
        
        // Start with all singletons (including isolates)
        let mut uf = UnionFind::new(num_records);
        
        // Apply all merges with threshold >= t
        for merge in &self.merges {
            if merge.threshold >= threshold {
                // Apply this merge
                for window in merge.merging_components.windows(2) {
                    if let [comp_a, comp_b] = window {
                        // In practice, we'd map components back to records
                        // This is simplified for clarity
                        uf.union(comp_a.0 as usize, comp_b.0 as usize);
                    }
                }
            } else {
                // Merges are sorted, so we can stop
                break;
            }
        }
        
        // Convert union-find to partition
        let mut entities: HashMap<usize, RoaringBitmap> = HashMap::new();
        for record in 0..num_records {
            let root = uf.find(record);
            entities.entry(root).or_default().insert(record as u32);
        }
        
        let entities: Vec<RoaringBitmap> = entities.into_values().collect();
        let entity_sizes = entities.iter().map(|e| e.cardinality()).collect();
        
        PartitionLevel {
            threshold,
            entities,
            entity_sizes,
            lazy_metrics: LazyMetricsCache::new(),
        }
    }
}
```

### Automatic Memory Management

**Compaction on collection removal**

```rust
impl EntityFrame {
    pub fn drop(&mut self, name: &str) {
        if let Some(hierarchy) = self.collections.remove(name) {
            // Track which records were used by this collection
            let collection_records = hierarchy.get_all_record_indices();
            
            // Mark as garbage
            self.garbage_records.or_inplace(&collection_records);
            
            // Check if compaction needed
            let garbage_ratio = self.garbage_records.cardinality() as f64 / 
                              self.context.records.len() as f64;
            
            if garbage_ratio > 0.5 {
                self.auto_compact();
            }
        }
    }
    
    fn auto_compact(&mut self) {
        // Find live records
        let mut live_records = RoaringBitmap::new();
        for hierarchy in self.collections.values() {
            live_records.or_inplace(&hierarchy.get_all_record_indices());
        }
        
        // Build new context with only live records
        let mut new_context = DataContext::new();
        let mut translation_map = TranslationMap::new();
        
        for &old_idx in live_records.iter() {
            let record = &self.context.records[old_idx as usize];
            let new_idx = new_context.records.len();
            new_context.records.push(record.clone());
            translation_map.insert(old_idx as usize, new_idx);
        }
        
        // Update all hierarchies
        for hierarchy in self.collections.values_mut() {
            hierarchy.translate_indices(&translation_map);
        }
        
        // Swap contexts
        self.context = Arc::new(new_context);
        self.garbage_records.clear();
    }
}
```

### Lazy Metric Computation Framework

**LazyMetric pattern implementation with complexity annotations**

```rust
pub struct LazyMetricsCache {
    // Computed on first access, then cached
    precision: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    recall: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    f1_score: HashMap<OrderedFloat<f64>, OnceCell<f64>>,
    
    // LRU cache for expensive pairwise computations
    jaccard_cache: LruCache<(EntityId, EntityId), f64>,
    
    // Computation tracking
    computation_stats: ComputationStats,
}

pub struct ComputationStats {
    total_computations: usize,
    incremental_computations: usize,
    cache_hits: usize,
    avg_incremental_speedup: f64,
}

impl LazyMetricsCache {
    pub fn get_metric(&mut self, 
                      metric: MetricType, 
                      threshold: f64,
                      ground_truth: &PartitionLevel) -> f64 {
        let key = OrderedFloat(threshold);
        
        match metric {
            MetricType::Precision => {
                self.precision.entry(key).or_insert_with(|| {
                    self.compute_or_update_metric(
                        MetricType::Precision, 
                        threshold, 
                        ground_truth
                    ).into()
                }).get().copied().unwrap()
            },
            MetricType::BCubed => {
                // B-cubed is semi-incremental: O(k × avg_entity_size)
                // where k = affected entities
                self.compute_bcubed(threshold, ground_truth)
            },
            // Similar for other metrics...
        }
    }
    
    fn compute_or_update_metric(&mut self,
                                metric: MetricType,
                                threshold: f64,
                                truth: &PartitionLevel) -> f64 {
        // Find nearest computed threshold
        if let Some((prev_t, prev_val)) = self.find_nearest_computed(metric, threshold) {
            // Incremental update - O(k) where k = affected entities
            let delta = self.compute_delta(metric, prev_t, threshold, truth);
            self.computation_stats.incremental_computations += 1;
            prev_val + delta
        } else {
            // First computation - O(n²) but only once
            self.computation_stats.total_computations += 1;
            self.compute_from_scratch(metric, threshold, truth)
        }
    }
}

// Metric computation complexity annotations
// Fully incremental O(k): Precision, Recall, F1, ARI, NMI
// Semi-incremental O(k × avg_entity_size): B-cubed metrics
// Note: All metrics benefit from caching between repeated queries
```

### Sparse Structure Exploitation

**Natural sparsity patterns**

```rust
// 1. Sparse contingency tables
pub struct SparseContingencyTable {
    // Most entity pairs have zero overlap
    nonzero_cells: HashMap<(EntityId, EntityId), u32>,
    row_marginals: Vec<u32>,
    col_marginals: Vec<u32>,
    total: u32,
}

impl SparseContingencyTable {
    pub fn update_incremental(&mut self, merge: &MergeEvent) {
        // Only update cells affected by merge - O(k)
        for entity_id in &merge.merging_components {
            // Update only non-zero cells involving this entity
            let affected_cells = self.nonzero_cells
                .keys()
                .filter(|(r, c)| *r == *entity_id || *c == *entity_id)
                .cloned()
                .collect::<Vec<_>>();
            
            for cell in affected_cells {
                self.update_cell(cell, merge);
            }
        }
    }
}

// 2. Block structure for disconnected components
pub struct BlockedHierarchy {
    blocks: Vec<PartitionHierarchy>,
    block_assignments: Vec<BlockId>,
    
    pub fn process_parallel<T>(&self, op: impl Fn(&PartitionHierarchy) -> T + Sync) 
        -> Vec<T> 
    where T: Send 
    {
        self.blocks.par_iter().map(op).collect()
    }
}

// 3. Sparse edge representation
pub struct SparseEdgeList {
    // Only store actual edges, not full matrix
    edges: Vec<(u32, u32, f64)>,
    
    // For efficient lookup
    adjacency: HashMap<u32, Vec<(u32, f64)>>,
}
```

### SIMD and Cache Optimisations

**Vectorised metric computation**

```rust
use std::simd::{f64x4, u32x8, SimdFloat, SimdUint};

impl PartitionLevel {
    pub fn compute_sizes_vectorised(&self) -> Vec<u32> {
        let mut sizes = Vec::with_capacity(self.entities.len());
        
        // Process 8 entities at once
        for chunk in self.entities.chunks(8) {
            let chunk_sizes: [u32; 8] = chunk
                .iter()
                .map(|e| e.cardinality())
                .chain(std::iter::repeat(0))
                .take(8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            
            let simd_sizes = u32x8::from_array(chunk_sizes);
            sizes.extend_from_slice(&simd_sizes.as_array()[..chunk.len()]);
        }
        
        sizes
    }
    
    pub fn compute_f1_scores_vectorised(&self, 
                                        precisions: &[f64], 
                                        recalls: &[f64]) -> Vec<f64> {
        assert_eq!(precisions.len(), recalls.len());
        let mut f1_scores = Vec::with_capacity(precisions.len());
        
        for (p_chunk, r_chunk) in precisions.chunks(4).zip(recalls.chunks(4)) {
            let p_simd = f64x4::from_slice(p_chunk);
            let r_simd = f64x4::from_slice(r_chunk);
            
            let two = f64x4::splat(2.0);
            let numerator = two * p_simd * r_simd;
            let denominator = p_simd + r_simd;
            let f1_simd = numerator / denominator;
            
            f1_scores.extend_from_slice(&f1_simd.to_array()[..p_chunk.len()]);
        }
        
        f1_scores
    }
}
```

**Cache-optimised memory layout**

```rust
// Align to cache lines for better performance
#[repr(C, align(64))]
pub struct CacheAlignedPartition {
    threshold: f64,
    num_entities: u32,
    _pad1: [u8; 4],  // Padding for alignment
    
    // Hot data together
    entity_sizes: *const u32,
    entity_data: *const RoaringBitmap,
    
    // Cold data separately
    metadata: *const PartitionMetadata,
    _pad2: [u8; 32],  // Prevent false sharing
}

// Structure of Arrays for better vectorisation
pub struct SoAMetrics {
    precisions: Vec<f64>,
    recalls: Vec<f64>,
    f1_scores: Vec<f64>,
    // Better cache locality than Array of Structs
}
```

### Parallel Processing Architecture

**Rayon-based parallel merge event processing**

```rust
use rayon::prelude::*;

impl PartitionHierarchy {
    pub fn build_parallel(edges: Vec<WeightedEdge>, context: Arc<DataContext>) -> Self {
        // Sort edges in parallel
        let mut edges = edges;
        edges.par_sort_unstable_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        
        // Group by threshold
        let threshold_groups = Self::group_by_threshold(edges);
        
        // Process each threshold's merges in parallel where possible
        let merges: Vec<MergeEvent> = threshold_groups
            .into_iter()
            .flat_map(|(threshold, edges)| {
                Self::find_merges_at_threshold(threshold, edges)
            })
            .collect();
        
        PartitionHierarchy {
            context,
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size: 10,
        }
    }
}
        
        // Group by threshold
        let threshold_groups = Self::group_by_threshold(edges);
        
        // Process each threshold's merges in parallel where possible
        let merges: Vec<MergeEvent> = threshold_groups
            .into_iter()
            .flat_map(|(threshold, edges)| {
                Self::find_merges_at_threshold(threshold, edges)
            })
            .collect();
        
        PartitionHierarchy {
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            num_records,
            cache_size: 10,
        }
    }
}
```

**Work-stealing for multi-collection comparison**

```rust
impl EntityFrame {
    pub fn compare_all_collections_parallel(&self, 
                                           truth_id: CollectionId,
                                           threshold_truth: f64,
                                           thresholds: HashMap<CollectionId, f64>) 
        -> HashMap<CollectionId, Metrics> {
        let truth = &self.collections[&truth_id];
        let truth_partition = truth.at_threshold(threshold_truth);
        
        thresholds
            .par_iter()
            .filter(|(id, _)| **id != truth_id)
            .map(|(id, threshold)| {
                let collection = &self.collections[id];
                let partition = collection.at_threshold(*threshold);
                let metrics = self.comparison_engine.compare_partitions(
                    partition,
                    truth_partition
                );
                (*id, metrics)
            })
            .collect()
    }
}
```

### Dual Processing for Entity Operations

**Built-in operations with parallel Rust execution**

```rust
pub enum EntityProcessor {
    // Fast built-in operations run in parallel
    BuiltIn(BuiltInOp),
    // Custom Python function - flexible but slower
    Python(PyObject),
}

pub enum BuiltInOp {
    Hash(HashAlgorithm),  // SHA256, MD5, Blake3, etc.
    Size,                 // Entity cardinality
    Density,              // Internal connectivity
    Fingerprint,          // MinHash or similar
}

impl PartitionLevel {
    pub fn map_over_entities<T>(&self, 
                               processor: EntityProcessor) -> HashMap<EntityId, T> 
    where T: Send + Sync 
    {
        match processor {
            EntityProcessor::BuiltIn(op) => {
                // Parallel execution with Rayon
                self.entities
                    .par_iter()
                    .enumerate()
                    .map(|(id, entity)| {
                        let result = match op {
                            BuiltInOp::Hash(algo) => compute_hash(entity, algo),
                            BuiltInOp::Size => entity.cardinality(),
                            BuiltInOp::Density => compute_density(entity),
                            BuiltInOp::Fingerprint => compute_fingerprint(entity),
                        };
                        (EntityId(id), result)
                    })
                    .collect()
            },
            EntityProcessor::Python(func) => {
                // Single-threaded Python execution
                self.entities
                    .iter()
                    .enumerate()
                    .map(|(id, entity)| {
                        let result = call_python_func(func, entity);
                        (EntityId(id), result)
                    })
                    .collect()
            }
        }
    }
}
```

**High-performance hashing for large-scale deduplication**

```rust
use blake3::Hasher;
use rayon::prelude::*;

fn compute_entity_hashes(partition: &PartitionLevel) -> Vec<[u8; 32]> {
    partition.entities
        .par_iter()
        .map(|entity| {
            let mut hasher = Hasher::new();
            // Sort for consistent hashing
            let mut members: Vec<_> = entity.iter().collect();
            members.sort_unstable();
            for member in members {
                hasher.update(&member.to_le_bytes());
            }
            *hasher.finalize().as_bytes()
        })
        .collect()
}
```

### Arrow Integration

**Hierarchical Arrow schema with optimal dictionary encoding**

```rust
use arrow::datatypes::{DataType, Field, Schema, UnionMode};

pub fn create_starlings_schema() -> Schema {
    Schema::new(vec![
        // Parallel arrays for maximum deduplication efficiency
        Field::new("record_sources", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8),
        ), false),
        
        Field::new("record_keys", DataType::Dictionary(
            Box::new(DataType::UInt32),
            Box::new(DataType::Union(
                vec![
                    Field::new("u32", DataType::UInt32, false),
                    Field::new("u64", DataType::UInt64, false),
                    Field::new("string", DataType::Utf8, false),
                    Field::new("bytes", DataType::Binary, false),
                ],
                vec![0, 1, 2, 3],  // Type ids
                UnionMode::Dense,
            )),
        ), false),
        
        Field::new("record_indices", DataType::UInt32, false),  // Simple array of indices
        
        // Collections with merge events
        Field::new("collections", DataType::List(Box::new(DataType::Struct(vec![
            Field::new("name", DataType::Dictionary(
                Box::new(DataType::UInt8),
                Box::new(DataType::Utf8),
            ), false),
            Field::new("merge_events", DataType::List(
                Box::new(DataType::Struct(vec![
                    Field::new("threshold", DataType::Float64, false),
                    Field::new("merging_components", DataType::List(
                        Box::new(DataType::UInt32)
                    ), false),
                    Field::new("result_component", DataType::UInt32, false),
                    Field::new("affected_records", DataType::List(
                        Box::new(DataType::UInt32)
                    ), false),
                ]))
            ), false),
        ]))), false),
        
        // Metadata
        Field::new("version", DataType::Utf8, false),
        Field::new("created_at", DataType::Timestamp(TimeUnit::Millisecond, None), false),
    ])
}
```

## Layer 3: Practical Engineering Design

### Core EntityFrame API

**Multi-collection management with contextual ownership**

```rust
pub struct EntityFrame {
    context: Arc<DataContext>,
    collections: HashMap<CollectionId, PartitionHierarchy>,
    comparison_engine: FrameAnalyzer,
    garbage_records: RoaringBitmap,
}

impl EntityFrame {
    pub fn from_records(source_name: &str, records: Vec<Record>) -> Self {
        let mut context = DataContext::new();
        let source_id = context.source_interner.get_or_intern(source_name);
        
        for record in records {
            let key = Key::from(record.id);
            context.records.push(InternedRecord {
                source_id,
                key,
                attributes: Some(record.data),
            });
        }
        
        EntityFrame {
            context: Arc::new(context),
            collections: HashMap::new(),
            comparison_engine: FrameAnalyzer::new(),
            garbage_records: RoaringBitmap::new(),
        }
    }
    
    pub fn add_collection_from_edges(&mut self, 
                                     name: &str, 
                                     edges: Vec<(u32, u32, f64)>) -> CollectionId {
        let id = CollectionId::new(name);
        // Pass the frame's context to the hierarchy
        let hierarchy = PartitionHierarchy::from_edges(edges, self.context.clone(), 6);
        self.collections.insert(id, hierarchy);
        id
    }
    
    pub fn add_collection_from_entities(&mut self,
                                        name: &str,
                                        entities: EntitySpec) -> CollectionId {
        let id = CollectionId::new(name);
        let edges = match entities {
            EntitySpec::Sets(sets) => sets_to_edges(sets, 1.0),
            EntitySpec::Entities(entities) => entities_to_edges(entities, 1.0),
        };
        // Pass the frame's context to the hierarchy
        let hierarchy = PartitionHierarchy::from_edges(edges, self.context.clone(), 6);
        self.collections.insert(id, hierarchy);
        id
    }
    
    pub fn add_collection(&mut self, 
                         name: &str, 
                         collection: Collection) -> CollectionId {
        // Only accept owned collections, not views
        if collection.is_view {
            panic!("Cannot add a view collection. Use copy() first.");
        }
        
        let id = CollectionId::new(name);
        
        // Assimilate if necessary
        if !Arc::ptr_eq(&collection.context, &self.context) {
            let assimilated = self.assimilate(collection);
            self.collections.insert(id, assimilated.hierarchy);
        } else {
            self.collections.insert(id, collection.hierarchy);
        }
        
        id
    }
    
    pub fn get_collection(&self, name: &str) -> Collection {
        // Return immutable view that shares context
        let hierarchy = self.collections[&CollectionId::from(name)].clone();
        
        // The hierarchy already has a reference to the frame's context
        // Create a view collection that shares this context
        Collection {
            context: self.context.clone(),  // Share the frame's context
            hierarchy,
            is_view: true,  // Mark as immutable view
        }
    }
    
    pub fn drop(&mut self, name: &str) {
        if let Some(hierarchy) = self.collections.remove(name) {
            let collection_records = hierarchy.get_all_record_indices();
            self.garbage_records.or_inplace(&collection_records);
            
            // Auto-compact if needed
            if self.garbage_records.cardinality() > self.context.records.len() / 2 {
                self.auto_compact();
            }
        }
    }
}
```

### Python Interface via PyO3

**Core Python bindings with starlings package**

The Python API follows the polars-inspired expression design, providing a clear separation between data containers and operations through composable expressions.

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct PyEntityFrame {
    inner: Arc<Mutex<EntityFrame>>,
}

#[pymethods]
impl PyEntityFrame {
    #[staticmethod]
    pub fn from_records(source: &str, data: &PyAny) -> PyResult<Self> {
        // Handle pandas DataFrame, Arrow table, or list of dicts
        let records = if data.is_instance_of::<PyDataFrame>()? {
            records_from_dataframe(data)?
        } else if data.is_instance_of::<PyArrowTable>()? {
            records_from_arrow(data)?
        } else {
            records_from_list(data)?
        };
        
        let frame = EntityFrame::from_records(source, records);
        Ok(PyEntityFrame {
            inner: Arc::new(Mutex::new(frame)),
        })
    }
    
    pub fn add_collection_from_edges(&mut self, 
                                     name: &str, 
                                     edges: Vec<(PyObject, PyObject, f64)>) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        
        // Convert Python Key objects directly to u32 indices for performance
        // This is the most efficient conversion point
        let internal_edges: Vec<(u32, u32, f64)> = edges.into_iter()
            .map(|(k1, k2, w)| {
                let idx1 = python_key_to_index(k1, &frame.context)?;
                let idx2 = python_key_to_index(k2, &frame.context)?;
                Ok((idx1, idx2, w))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        frame.add_collection_from_edges(name, internal_edges);
        Ok(())
    }
    
    pub fn add_collection_from_entities(&mut self,
                                        name: &str,
                                        entities: &PyAny) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        
        // Detect type and convert
        let entity_spec = if is_entity_list(entities)? {
            EntitySpec::Entities(extract_entities(entities)?)
        } else {
            EntitySpec::Sets(extract_sets(entities)?)
        };
        
        frame.add_collection_from_entities(name, entity_spec);
        Ok(())
    }
    
    pub fn add_collection(&mut self,
                         name: &str,
                         collection: PyCollection) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        frame.add_collection(name, collection.inner);
        Ok(())
    }
    
    pub fn __getitem__(&self, name: &str) -> PyResult<PyCollection> {
        let frame = self.inner.lock().unwrap();
        let collection = frame.get_collection(name);
        Ok(PyCollection { inner: collection })
    }
    
    pub fn analyse(&mut self, 
                  expressions: Vec<PyExpression>,
                  metrics: Option<Vec<PyMetric>>) -> PyResult<PyObject> {
        let mut frame = self.inner.lock().unwrap();
        
        // Parse expressions to determine operation type
        let operation = parse_expressions(expressions)?;
        
        // Python wrapper provides defaults if metrics not specified
        let metrics = metrics.unwrap_or_else(|| {
            match &operation {
                Operation::SingleCollection(_) => vec![
                    PyMetric::EntityCount,
                    PyMetric::Entropy,
                ],
                Operation::Comparison(_) => vec![
                    PyMetric::F1,
                    PyMetric::Precision,
                    PyMetric::Recall,
                    PyMetric::ARI,
                    PyMetric::NMI,
                ],
            }
        });
        
        // Call Rust implementation with required metrics
        let results = match operation {
            Operation::PointComparison(cuts) => {
                let metrics = frame.compare_cuts(cuts, metrics);
                vec![metrics]  // Single dict in list
            },
            Operation::Sweep(sweep_spec) => {
                frame.sweep(sweep_spec, metrics)  // Already returns List[Dict]
            },
        };
        
        Python::with_gil(|py| {
            // Convert to Python list of dicts
            let py_list = PyList::new(py, 
                results.into_iter().map(|dict| {
                    dict.to_pydict(py)
                })
            );
            Ok(py_list.into())
        })
    }
    
    // American spelling alias
    pub fn analyze(&mut self, 
                  expressions: Vec<PyExpression>,
                  metrics: Vec<PyMetric>) -> PyResult<PyObject> {
        self.analyse(expressions, metrics)
    }
    
    pub fn drop(&mut self, names: Vec<&str>) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        for name in names {
            frame.drop(name);
        }
        Ok(())
    }
}

#[pyclass]
pub struct PyCollection {
    inner: Collection,
}

#[pymethods]
impl PyCollection {
    #[staticmethod]
    pub fn from_edges(edges: Vec<(PyObject, PyObject, f64)>, 
                     records: Option<&PyAny>,
                     quantize: u32) -> PyResult<Self> {
        // Validate quantize
        if quantize < 1 || quantize > 6 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "quantize must be between 1 and 6"
            ));
        }
        
        // Convert Python Key objects to u32 for Rust processing
        let rust_edges: Vec<(u32, u32, f64)> = edges.into_iter()
            .map(|(k1, k2, w)| {
                let id1 = key_to_u32(k1)?;
                let id2 = key_to_u32(k2)?;
                Ok((id1, id2, w))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        // Handle optional records parameter (can be all records or just additional)
        let record_spec = if let Some(records) = records {
            Some(parse_record_spec(records)?)
        } else {
            None
        };
        
        // Build collection with Rust deduplicating between edges and records
        let collection = Collection::from_edges(
            rust_edges, 
            record_spec,
            quantize
        );
        Ok(PyCollection { inner: collection })
    }
    
    #[staticmethod]
    pub fn from_entities(entities: &PyAny) -> PyResult<Self> {
        let entity_spec = if is_entity_list(entities)? {
            extract_entities(entities)?
        } else {
            extract_sets(entities)?
        };
        
        let collection = Collection::from_entities(entity_spec);
        Ok(PyCollection { inner: collection })
    }
    
    pub fn at(&mut self, threshold: f64) -> PyResult<PyPartition> {
        let partition = self.inner.at_threshold(threshold)?;
        Ok(PyPartition { inner: partition })
    }
    
    pub fn copy(&self) -> PyResult<Self> {
        let new_collection = self.inner.copy();
        Ok(PyCollection { inner: new_collection })
    }
    
    #[getter]
    pub fn is_view(&self) -> PyResult<bool> {
        Ok(self.inner.is_view())
    }
}

// Expression API implementation
#[pyclass]
pub struct PyColExpression {
    collection_name: String,
}

#[pymethods]
impl PyColExpression {
    pub fn at(&self, threshold: f64) -> PyExpression {
        PyExpression::At(self.collection_name.clone(), threshold)
    }
    
    pub fn sweep(&self, start: f64, stop: f64, step: f64) -> PyExpression {
        PyExpression::Sweep(self.collection_name.clone(), start, stop, step)
    }
}

// Python Key type wrapper
#[pyclass]
pub struct PyKey {
    inner: Key,
}

#[pymethods]
impl PyKey {
    #[new]
    pub fn new(value: &PyAny) -> PyResult<Self> {
        // Automatically convert Python types to appropriate Key variant
        let inner = if let Ok(v) = value.extract::<u32>() {
            Key::U32(v)
        } else if let Ok(v) = value.extract::<u64>() {
            Key::U64(v)
        } else if let Ok(v) = value.extract::<String>() {
            Key::String(v)  // Keys don't need interning - they're unique
        } else if let Ok(v) = value.extract::<Vec<u8>>() {
            Key::Bytes(v)
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Key must be int, str, or bytes"
            ));
        };
        Ok(PyKey { inner })
    }
}

// Python Entity type (not mirrored in Rust)
#[pyclass]
pub struct PyEntity {
    members: HashSet<(String, PyKey)>,
}

#[pymethods]
impl PyEntity {
    #[new]
    pub fn new(members: HashSet<(String, PyKey)>) -> Self {
        PyEntity { members }
    }
}

// Injectable operations pattern implementation
// These are marker types that route to optimised Rust implementations
#[pyclass]
pub struct F1Metric;

#[pyclass]
pub struct PrecisionMetric;

#[pyclass]
pub struct RecallMetric;

#[pyclass]
pub struct ARIMetric;

#[pyclass]
pub struct NMIMetric;

#[pyclass]
pub struct VMeasureMetric;

#[pyclass]
pub struct BCubedPrecisionMetric;

#[pyclass]
pub struct BCubedRecallMetric;

#[pyclass]
pub struct EntropyMetric;

#[pyclass]
pub struct EntityCountMetric;

#[pyclass]
pub struct SHA256Op;

#[pyclass]
pub struct Blake3Op;

#[pyclass]
pub struct MD5Op;

#[pyclass]
pub struct SizeOp;

#[pyclass]
pub struct DensityOp;

#[pyclass]
pub struct FingerprintOp;

// Nested class structure for metrics and operations
#[pyclass]
pub struct MetricsEval;

#[pymethods]
impl MetricsEval {
    #[classattr]
    fn f1() -> F1Metric { F1Metric }
    
    #[classattr]
    fn precision() -> PrecisionMetric { PrecisionMetric }
    
    #[classattr]
    fn recall() -> RecallMetric { RecallMetric }
    
    #[classattr]
    fn ari() -> ARIMetric { ARIMetric }
    
    #[classattr]
    fn nmi() -> NMIMetric { NMIMetric }
    
    #[classattr]
    fn v_measure() -> VMeasureMetric { VMeasureMetric }
    
    #[classattr]
    fn bcubed_precision() -> BCubedPrecisionMetric { BCubedPrecisionMetric }
    
    #[classattr]
    fn bcubed_recall() -> BCubedRecallMetric { BCubedRecallMetric }
}

#[pyclass]
pub struct MetricsStats;

#[pymethods]
impl MetricsStats {
    #[classattr]
    fn entropy() -> EntropyMetric { EntropyMetric }
    
    #[classattr]
    fn entity_count() -> EntityCountMetric { EntityCountMetric }
}

#[pyclass]
pub struct Metrics;

#[pymethods]
impl Metrics {
    #[classattr]
    fn eval() -> MetricsEval { MetricsEval }
    
    #[classattr]
    fn stats() -> MetricsStats { MetricsStats }
}

#[pyclass]
pub struct OpsHash;

#[pymethods]
impl OpsHash {
    #[classattr]
    fn sha256() -> SHA256Op { SHA256Op }
    
    #[classattr]
    fn blake3() -> Blake3Op { Blake3Op }
    
    #[classattr]
    fn md5() -> MD5Op { MD5Op }
}

#[pyclass]
pub struct OpsCompute;

#[pymethods]
impl OpsCompute {
    #[classattr]
    fn size() -> SizeOp { SizeOp }
    
    #[classattr]
    fn density() -> DensityOp { DensityOp }
    
    #[classattr]
    fn fingerprint() -> FingerprintOp { FingerprintOp }
}

#[pyclass]
pub struct Ops;

#[pymethods]
impl Ops {
    #[classattr]
    fn hash() -> OpsHash { OpsHash }
    
    #[classattr]
    fn compute() -> OpsCompute { OpsCompute }
}

// Key conversion at Python/Rust boundary
// Performance note: Convert Python Keys directly to u32 indices for hierarchy operations
// Only use Rust Key enum when preserving actual key values is needed
fn python_key_to_index(key: PyObject, context: &DataContext) -> PyResult<u32> {
    Python::with_gil(|py| {
        // Extract the key value from Python
        let rust_key = if let Ok(v) = key.extract::<u32>(py) {
            Key::U32(v)
        } else if let Ok(v) = key.extract::<u64>(py) {
            Key::U64(v)
        } else if let Ok(v) = key.extract::<String>(py) {
            Key::String(v)
        } else if let Ok(v) = key.extract::<Vec<u8>>(py) {
            Key::Bytes(v)
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Key must be int, str, or bytes"
            ));
        };
        
        // Look up or create index in context
        Ok(context.get_or_create_index(&rust_key) as u32)
    })
}
#[pymodule]
fn starlings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEntityFrame>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyPartition>()?;
    m.add_class::<PyColExpression>()?;
    m.add_class::<PyKey>()?;
    m.add_class::<PyEntity>()?;  // Python-only type
    
    // Add nested metrics and ops classes
    m.add_class::<Metrics>()?;
    m.add_class::<MetricsEval>()?;
    m.add_class::<MetricsStats>()?;
    m.add_class::<Ops>()?;
    m.add_class::<OpsHash>()?;
    m.add_class::<OpsCompute>()?;
    
    // Add the individual metric and operation types
    m.add_class::<F1Metric>()?;
    m.add_class::<PrecisionMetric>()?;
    m.add_class::<RecallMetric>()?;
    m.add_class::<ARIMetric>()?;
    m.add_class::<NMIMetric>()?;
    m.add_class::<VMeasureMetric>()?;
    m.add_class::<BCubedPrecisionMetric>()?;
    m.add_class::<BCubedRecallMetric>()?;
    m.add_class::<EntropyMetric>()?;
    m.add_class::<EntityCountMetric>()?;
    m.add_class::<SHA256Op>()?;
    m.add_class::<Blake3Op>()?;
    m.add_class::<MD5Op>()?;
    m.add_class::<SizeOp>()?;
    m.add_class::<DensityOp>()?;
    m.add_class::<FingerprintOp>()?;
    
    // Add col function
    m.add_function(wrap_pyfunction!(col, m)?)?;
    m.add_function(wrap_pyfunction!(from_records, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn col(name: &str) -> PyColExpression {
    PyColExpression {
        collection_name: name.to_string(),
    }
}

#[pyfunction]
fn from_records(source: &str, data: &PyAny) -> PyResult<PyEntityFrame> {
    PyEntityFrame::from_records(source, data)
}
```

### Batch Processing Optimisations

**Optimised batch construction**

```rust
pub struct BatchProcessor {
    frame: EntityFrame,
    batch_size: usize,
}

impl BatchProcessor {
    pub fn process_large_dataset(&mut self, 
                                 edges: impl Iterator<Item = WeightedEdge>) {
        // Process in memory-efficient batches
        let chunks = edges.chunks(self.batch_size);
        
        for chunk in chunks {
            // Pass the frame's context to hierarchy
            let hierarchy = PartitionHierarchy::from_edges(
                chunk.collect(), 
                self.frame.context.clone(),
                6  // Default quantization
            );
            // Process hierarchy...
        }
    }
    
    pub fn parallel_batch_comparison(&self,
                                    collections: Vec<CollectionId>,
                                    thresholds: Vec<f64>) -> ComparisonMatrix {
        use rayon::prelude::*;
        
        // Parallel comparison across collections and thresholds
        let comparisons: Vec<_> = collections
            .par_iter()
            .flat_map(|col_a| {
                thresholds.par_iter().flat_map(move |t_a| {
                    collections.par_iter().flat_map(move |col_b| {
                        thresholds.par_iter().map(move |t_b| {
                            self.frame.compare_cuts(
                                (col_a, *t_a),
                                (col_b, *t_b)
                            )
                        })
                    })
                })
            })
            .collect();
        
        ComparisonMatrix::from_comparisons(comparisons)
    }
}
```

### Production Deployment

**Monitoring and observability**

```rust
#[derive(Serialize, Deserialize)]
pub struct FrameMetrics {
    // Data statistics
    num_collections: usize,
    num_records: usize,
    num_sources: usize,
    total_memory_mb: f64,
    
    // Performance metrics
    avg_reconstruction_time_ms: f64,
    cache_hit_rate: f64,
    incremental_computation_rate: f64,
    
    // Quality indicators
    avg_entity_stability: f64,
    threshold_sensitivity: f64,
    
    // Memory management
    garbage_ratio: f64,
    last_compaction: Option<DateTime<Utc>>,
}

impl EntityFrame {
    pub fn collect_metrics(&self) -> FrameMetrics {
        let total_merges: usize = self.collections
            .values()
            .map(|h| h.merges.len())
            .sum();
        
        let cache_stats = self.get_cache_statistics();
        
        FrameMetrics {
            num_collections: self.collections.len(),
            num_records: self.context.records.len(),
            num_sources: self.context.source_interner.len(),
            total_memory_mb: self.estimate_memory_usage() / 1_048_576.0,
            avg_reconstruction_time_ms: cache_stats.avg_miss_time_ms,
            cache_hit_rate: cache_stats.hit_rate,
            incremental_computation_rate: self.compute_incremental_rate(),
            avg_entity_stability: self.compute_avg_stability(),
            threshold_sensitivity: self.compute_sensitivity(),
            garbage_ratio: self.garbage_records.cardinality() as f64 / 
                          self.context.records.len() as f64,
            last_compaction: self.last_compaction_time,
        }
    }
    
    fn estimate_memory_usage(&self) -> f64 {
        let context_memory = self.context.records.len() * 
                           std::mem::size_of::<InternedRecord>();
        
        let merge_memory: usize = self.collections
            .values()
            .map(|h| h.merges.len() * std::mem::size_of::<MergeEvent>())
            .sum();
        
        let cache_memory: usize = self.collections
            .values()
            .map(|h| h.partition_cache.len() * 500_000) // ~500KB per cached partition
            .sum();
        
        (context_memory + merge_memory + cache_memory) as f64
    }
}
```

**Performance characteristics**

```rust
/// Expected performance for different scales
/// Based on contextual ownership with automatic compaction
pub struct PerformanceProfile {
    pub scale: DataScale,
    pub hierarchy_build_time: Duration,
    pub cached_query_time: Duration,
    pub uncached_query_time: Duration,
    pub sweep_time_per_threshold: Duration,
    pub memory_usage: ByteSize,
}

impl PerformanceProfile {
    pub fn estimate(num_records: usize, num_edges: usize) -> Self {
        match (num_records, num_edges) {
            (n, m) if n <= 10_000 => PerformanceProfile {
                scale: DataScale::Small,
                hierarchy_build_time: Duration::from_millis(100),
                cached_query_time: Duration::from_micros(100),
                uncached_query_time: Duration::from_millis(10),
                sweep_time_per_threshold: Duration::from_millis(1),
                memory_usage: ByteSize::mb(10),
            },
            (n, m) if n <= 1_000_000 => PerformanceProfile {
                scale: DataScale::Medium,
                hierarchy_build_time: Duration::from_secs(10),
                cached_query_time: Duration::from_millis(1),
                uncached_query_time: Duration::from_millis(200),
                sweep_time_per_threshold: Duration::from_millis(10),
                memory_usage: ByteSize::mb(100),
            },
            _ => PerformanceProfile {
                scale: DataScale::Large,
                hierarchy_build_time: Duration::from_secs(300),
                cached_query_time: Duration::from_millis(10),
                uncached_query_time: Duration::from_secs(2),
                sweep_time_per_threshold: Duration::from_millis(100),
                memory_usage: ByteSize::gb(1),
            },
        }
    }
}
```
