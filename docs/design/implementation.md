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

pub struct InternedRecord {
    source_id: u16,     // Interned source identifier
    key: Key,           // Key within that source (flexible type)
    attributes: Option<RecordData>,  // Optional attribute data
}

/// Flexible key types for record identification
pub enum Key {
    U32(u32),
    U64(u64),
    String(InternedString),
    Bytes(Vec<u8>),
}
```

### Contextual Ownership Architecture

**Separation of data and logic**

```rust
pub struct Collection {
    // Reference to data (may be shared or owned)
    context: Arc<DataContext>,
    
    // Logical structure (hierarchy of merge events)
    hierarchy: PartitionHierarchy,
    
    // Track if this is a view from a frame
    parent_frame: Option<Arc<EntityFrame>>,
}

impl Collection {
    /// Create standalone collection with owned context
    pub fn from_edges(edges: Vec<(Key, Key, f64)>) -> Self {
        let context = Arc::new(DataContext::new());
        // Populate context with records from edges
        let hierarchy = build_hierarchy(edges, &context);
        
        Collection {
            context,
            hierarchy,
            parent_frame: None,
        }
    }
    
    /// Copy-on-Write for mutations
    pub fn add_edges(&mut self, new_edges: Vec<(Key, Key, f64)>) {
        // Atomic check for unique ownership
        if let Some(context) = Arc::get_mut(&mut self.context) {
            // We own the context, mutate in place
            context.append_records(/* ... */);
        } else {
            // Shared context, trigger CoW to create owned collection
            let new_context = Arc::new(self.context.deep_copy());
            self.context = new_context;
            self.parent_frame = None;  // Now standalone with owned DataContext
            // Now safe to mutate
            Arc::get_mut(&mut self.context).unwrap().append_records(/* ... */);
        }
    }
}
```

### Efficient Hierarchical Data Structures

**Merge-event based hierarchy with smart caching**

```rust
pub struct PartitionHierarchy {
    // Primary: Merge events sorted by threshold (descending)
    merges: Vec<MergeEvent>,
    
    // Secondary: Cache for frequently accessed partitions
    partition_cache: LruCache<OrderedFloat<f64>, PartitionLevel>,
    
    // Tertiary: State for incremental metric computation
    metric_state: Option<IncrementalMetricState>,
    
    // Index for binary search on thresholds
    threshold_index: BTreeMap<OrderedFloat<f64>, usize>,
    
    // Configuration
    cache_size: usize,
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

### Assimilation Process

**Adding collections to frames**

```rust
impl EntityFrame {
    pub fn add_collection(&mut self, name: &str, collection: Collection) {
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
        let mut translation_map = TranslationMap::new();
        
        // Identity resolution: (source, key) determines identity
        for (idx, record) in collection.context.records.iter().enumerate() {
            let identity = (record.source_id, &record.key);
            
            if let Some(&existing_idx) = self.context.identity_map.get(&identity) {
                // Record exists, map to existing
                translation_map.insert(idx, existing_idx);
            } else {
                // New record, append to context
                let new_idx = self.context.records.len();
                self.context.records.push(record.clone());
                self.context.identity_map.insert(identity, new_idx);
                translation_map.insert(idx, new_idx);
            }
        }
        
        // Translate hierarchy to use new indices
        let mut new_hierarchy = collection.hierarchy.clone();
        new_hierarchy.translate_indices(&translation_map);
        
        AssimilatedCollection {
            hierarchy: new_hierarchy,
            translation_map,
        }
    }
}
```

### Connected Components Algorithm for Hierarchy Construction

**Union-find based merge event extraction with quantization support**

```rust
use disjoint_sets::UnionFind;

impl PartitionHierarchy {
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, 
                     num_records: u32, 
                     quantize: Option<u32>) -> Self {
        // Optional quantization
        let mut sorted_edges = if let Some(decimals) = quantize {
            let factor = 10_f64.powi(decimals as i32);
            edges.into_iter()
                .map(|(i, j, w)| (i, j, (w * factor).round() / factor))
                .collect()
        } else {
            edges
        };
        
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
        let mut uf = UnionFind::new(num_records as usize);
        let mut component_map: HashMap<usize, ComponentId> = HashMap::new();
        let mut next_component_id = 0;
        
        // Initialize each record as its own component
        for i in 0..num_records {
            component_map.insert(i as usize, ComponentId(i));
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
                        if uf.find(i as usize) == current {
                            affected.insert(i);
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
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size: 10,
        }
    }
}
```

### Partition Reconstruction from Merge Events

**Efficient reconstruction algorithm**

```rust
impl PartitionHierarchy {
    pub fn at_threshold(&mut self, threshold: f64) -> Result<&PartitionLevel, HierarchyError> {
        // Validate threshold
        if threshold < 0.0 || threshold > 1.0 {
            return Err(HierarchyError::InvalidThreshold(threshold));
        }
        
        let key = OrderedFloat(threshold);
        
        // Check cache first
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
        // Start with all singletons
        let mut uf = UnionFind::new(self.num_records());
        
        // Apply all merges with threshold >= t
        for merge in &self.merges {
            if merge.threshold >= threshold {
                // Apply this merge
                for window in merge.merging_components.windows(2) {
                    if let [comp_a, comp_b] = window {
                        // In practice, we'd map components back to records
                        // This is simplified for clarity
                        uf.union(comp_a.0, comp_b.0);
                    }
                }
            } else {
                // Merges are sorted, so we can stop
                break;
            }
        }
        
        // Convert union-find to partition
        let mut entities: HashMap<usize, RoaringBitmap> = HashMap::new();
        for record in 0..self.num_records() {
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

**LazyMetric pattern implementation**

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
    pub fn build_parallel(edges: Vec<WeightedEdge>) -> Self {
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
            merges,
            partition_cache: LruCache::new(10),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
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

**Hierarchical Arrow schema with dictionary encoding**

```rust
use arrow::datatypes::{DataType, Field, Schema};

pub fn create_starlings_schema() -> Schema {
    Schema::new(vec![
        // Collection metadata
        Field::new("collection_id", DataType::Utf8, false),
        
        // Interned references with dictionary encoding
        Field::new("source_dictionary", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8),
        ), false),
        
        // Records
        Field::new("records", DataType::List(Box::new(DataType::Struct(vec![
            Field::new("source", DataType::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(DataType::Utf8),
            ), false),
            Field::new("key", DataType::Utf8, false),
            Field::new("record_index", DataType::UInt32, false),
        ]))), false),
        
        // Merge events (expanded, not binary)
        Field::new("merge_events", DataType::List(Box::new(Field::new(
            "merge", 
            DataType::Struct(vec![
                Field::new("threshold", DataType::Float64, false),
                Field::new("merging_components", DataType::List(
                    Box::new(DataType::UInt32)
                ), false),
                Field::new("result_component", DataType::UInt32, false),
                Field::new("affected_records", DataType::List(
                    Box::new(DataType::UInt32)
                ), false),
            ]), 
            false
        ))), false),
        
        // Statistics
        Field::new("statistics", DataType::Struct(vec![
            Field::new("num_merges", DataType::UInt32, false),
            Field::new("threshold_range", DataType::Struct(vec![
                Field::new("min", DataType::Float64, false),
                Field::new("max", DataType::Float64, false),
            ]), false),
        ]), true),
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
    
    pub fn add_collection(&mut self, 
                         name: &str, 
                         collection: Collection) -> CollectionId {
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
        // Return view that shares context
        Collection {
            context: self.context.clone(),
            hierarchy: self.collections[&CollectionId::from(name)].clone(),
            parent_frame: Some(Arc::new(self.clone())),
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
    
    pub fn add_collection(&mut self, 
                         name: &str, 
                         edges: Option<Vec<(PyObject, PyObject, f64)>>,
                         collection: Option<PyCollection>) -> PyResult<()> {
        let mut frame = self.inner.lock().unwrap();
        
        if let Some(edges) = edges {
            // Build collection from edges
            let collection = Collection::from_edges(edges);
            frame.add_collection(name, collection);
        } else if let Some(collection) = collection {
            // Add existing collection
            frame.add_collection(name, collection.inner);
        }
        
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
        
        // Use default metrics if none provided
        // sl.report defaults: [f1, precision, recall, ari, nmi] for comparisons
        //                     [entity_count, entropy] for single collections
        let metrics = metrics.unwrap_or_else(|| {
            match &operation {
                Operation::SingleCollection(_) => vec![PyMetric::EntityCount, PyMetric::Entropy],
                Operation::Comparison(_) => vec![
                    PyMetric::F1, 
                    PyMetric::Precision, 
                    PyMetric::Recall,
                    PyMetric::ARI,
                    PyMetric::NMI
                ],
                Operation::Sweep(_) => vec![
                    PyMetric::F1, 
                    PyMetric::Precision, 
                    PyMetric::Recall,
                    PyMetric::ARI,
                    PyMetric::NMI
                ],
            }
        });
        
        match operation {
            Operation::PointComparison(cuts) => {
                let result = frame.compare_cuts(cuts, metrics);
                Python::with_gil(|py| Ok(result.to_pydict(py)))
            },
            Operation::Sweep(sweep_spec) => {
                let result = frame.sweep(sweep_spec, metrics);
                Python::with_gil(|py| Ok(result.to_dataframe(py)))
            },
        }
    }
    
    // American spelling alias
    pub fn analyze(&mut self, 
                  expressions: Vec<PyExpression>,
                  metrics: Option<Vec<PyMetric>>) -> PyResult<PyObject> {
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
    pub fn from_edges(edges: Vec<(PyObject, PyObject, f64)>) -> PyResult<Self> {
        let collection = Collection::from_edges(convert_edges(edges)?);
        Ok(PyCollection { inner: collection })
    }
    
    #[staticmethod]
    pub fn from_entities(entities: Vec<HashSet<PyObject>>, 
                        threshold: f64) -> PyResult<Self> {
        let collection = Collection::from_entities(convert_entities(entities)?, threshold);
        Ok(PyCollection { inner: collection })
    }
    
    #[staticmethod]
    pub fn from_merge_events(merges: Vec<PyDict>, 
                            num_records: usize) -> PyResult<Self> {
        let collection = Collection::from_merge_events(
            convert_merges(merges)?, 
            num_records
        );
        Ok(PyCollection { inner: collection })
    }
    
    pub fn at(&mut self, threshold: f64) -> PyResult<PyPartition> {
        let partition = self.inner.at_threshold(threshold)?;
        Ok(PyPartition { inner: partition })
    }
    
    pub fn copy(&self) -> PyResult<Self> {
        let mut new_collection = self.inner.clone();
        // Trigger CoW to detach and create owned collection with its own DataContext
        new_collection.make_standalone();
        Ok(PyCollection { inner: new_collection })
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

// Module definition
#[pymodule]
fn starlings(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyEntityFrame>()?;
    m.add_class::<PyCollection>()?;
    m.add_class::<PyPartition>()?;
    m.add_class::<PyColExpression>()?;
    
    // Add metrics as module constants
    m.add("f1", PyMetric::F1)?;
    m.add("precision", PyMetric::Precision)?;
    m.add("recall", PyMetric::Recall)?;
    m.add("ari", PyMetric::ARI)?;
    m.add("nmi", PyMetric::NMI)?;
    
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
            let hierarchy = PartitionHierarchy::from_edges(
                chunk.collect(), 
                self.num_records,
                None  // No quantization by default
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
