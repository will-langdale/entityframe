# EntityFrame Design Document B: Technical Implementation

## Layer 2: Computer Science Techniques

### Core Data Structure Architecture

**Multi-collection EntityFrame structure**

```rust
pub struct EntityFrame {
    // Multiple hierarchies (collections) sharing same record space
    // Collections ARE hierarchies mathematically
    collections: HashMap<CollectionId, PartitionHierarchy>,
    
    // Shared interned record storage
    records: InternedRecordStorage,
    
    // Cross-collection comparison engine
    comparison_engine: CollectionComparer,
    
    // Global string interning for sources
    source_interner: StringInterner,
}

pub struct InternedRecordStorage {
    // Map source names to compact IDs
    source_interner: StringInterner<FxHashMap>,
    
    // Actual record data
    records: Vec<InternedRecord>,
    
    // Index for fast lookup
    source_index: HashMap<SourceId, RoaringBitmap>,
}

pub struct InternedRecord {
    source_id: u16,     // Interned source identifier
    local_id: LocalId,  // ID within that source (flexible type)
    attributes: Option<RecordData>,  // Optional attribute data
}

/// Flexible key types for record identification
pub enum LocalId {
    U32(u32),
    U64(u64),
    String(InternedString),
    Bytes(Vec<u8>),
}
```

### Efficient Hierarchical Data Structures

**Merge-event based hierarchy with smart caching**

```rust
pub struct PartitionHierarchy {
    // Primary: Merge events sorted by threshold (descending)
    merges: Vec<MergeEvent>,
    
    // Secondary: Cache for frequently accessed partitions (per collection)
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

For 1M records with 1M edges:
- Merge events: 1M × 50-100 bytes = 50-100MB (includes RoaringBitmaps)
- LRU cache (10 partitions): 10 × 500KB = 5MB
- Total: ~50-250MB vs ~10GB for full materialization
- Tradeoff: O(m) reconstruction cost for uncached queries

### String Interning Architecture

**Multi-level interning strategy**

```rust
use string_interner::{StringInterner, DefaultSymbol};

pub struct InternSystem {
    // Source dataset names (e.g., "CRM", "MailingList")
    source_interner: StringInterner<DefaultSymbol>,
    
    // Attribute names across all sources
    attribute_interner: StringInterner<DefaultSymbol>,
    
    // Frequently occurring values (optional)
    value_interner: Option<StringInterner<DefaultSymbol>>,
}

impl InternSystem {
    pub fn intern_reference(&mut self, source: &str, local_id: u32) -> InternedRef {
        let source_sym = self.source_interner.get_or_intern(source);
        InternedRef {
            source_id: source_sym.to_usize() as u16,
            local_id,
        }
    }
    
    pub fn resolve_reference(&self, interned: InternedRef) -> (String, u32) {
        let source = self.source_interner
            .resolve(DefaultSymbol::from(interned.source_id as usize))
            .unwrap()
            .to_string();
        (source, interned.local_id)
    }
}
```

**Compression results**
- Source names: 20-30 bytes → 2 bytes (93% reduction)
- References: (String, u32) ~24 bytes → 6 bytes (75% reduction)
- Total memory: 80-90% reduction for typical workloads

### Connected Components Algorithm for Hierarchy Construction

**Union-find based merge event extraction**

```rust
use disjoint_sets::UnionFind;

impl PartitionHierarchy {
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, num_records: u32, cache_size: usize) -> Self {
        // Sort edges by weight (descending) - O(m log m)
        let mut sorted_edges = edges;
        sorted_edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        // Group edges by threshold
        let mut threshold_groups: Vec<(f64, Vec<(u32, u32)>)> = Vec::new();
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
            
            // Create merge events
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
            partition_cache: LruCache::new(cache_size),
            metric_state: None,
            threshold_index: Self::build_threshold_index(&merges),
            cache_size,
        }
    }
    
    fn build_threshold_index(merges: &[MergeEvent]) -> BTreeMap<OrderedFloat<f64>, usize> {
        merges.iter()
            .enumerate()
            .map(|(idx, merge)| (OrderedFloat(merge.threshold), idx))
            .collect()
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

**Incremental computation chain**

```rust
impl PartitionHierarchy {
    pub fn compute_metric_range(&mut self,
                                start: f64,
                                end: f64,
                                step: f64,
                                truth: &Self) -> Vec<(f64, Metrics)> {
        let mut results = Vec::new();
        let mut current_metrics = None;
        let mut current_partition = None;
        
        for threshold in (0..)
            .map(|i| start + i as f64 * step)
            .take_while(|&t| t <= end) {
            
            // Get partition (from cache or reconstruct)
            let partition = self.at_threshold(threshold);
            
            let metrics = if let Some(prev) = current_metrics {
                // Update from previous - O(k) where k = affected entities
                self.update_metrics_incremental(
                    prev, 
                    &current_partition.unwrap(),
                    partition,
                    truth
                )
            } else {
                // Only compute from scratch once
                self.compute_metrics_full(partition, truth)
            };
            
            results.push((threshold, metrics.clone()));
            current_metrics = Some(metrics);
            current_partition = Some(partition.clone());
        }
        
        results
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

**Hierarchical Arrow schema**

```rust
use arrow::datatypes::{DataType, Field, Schema};

pub fn create_entityframe_schema() -> Schema {
    Schema::new(vec![
        // Collection metadata
        Field::new("collection_id", DataType::Utf8, false),
        
        // Interned references
        Field::new("source_dictionary", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8),
        ), false),
        
        // Merge events (not full partitions)
        Field::new("merge_events", DataType::List(Box::new(Field::new(
            "merge", 
            DataType::Struct(vec![
                Field::new("threshold", DataType::Float64, false),
                Field::new("merging_components", DataType::List(
                    Box::new(DataType::UInt32)
                ), false),
                Field::new("result_component", DataType::UInt32, false),
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

**Multi-collection management**

```rust
pub struct EntityFrame {
    collections: HashMap<CollectionId, PartitionHierarchy>,
    records: InternedRecordStorage,
    interner: InternSystem,
    analyzer: FrameAnalyzer,
}

impl EntityFrame {
    pub fn new() -> Self {
        EntityFrame {
            collections: HashMap::new(),
            records: InternedRecordStorage::new(),
            interner: InternSystem::new(),
            analyzer: FrameAnalyzer::new(),
        }
    }
    
    pub fn add_collection(&mut self, 
                         name: &str, 
                         hierarchy: PartitionHierarchy) -> CollectionId {
        let id = CollectionId::new(name);
        self.collections.insert(id, hierarchy);
        id
    }
    
    pub fn add_records_from_source(&mut self, 
                                   source_name: &str,
                                   records: Vec<Record>) -> Vec<RecordId> {
        let source_id = self.interner.intern_source(source_name);
        self.records.add_from_source(source_id, records)
    }
    
    pub fn compare_cuts(&mut self,
                       cut_a: (&str, f64),
                       cut_b: (&str, f64)) -> ComparisonResult {
        let collection_a = &mut self.collections.get_mut(&CollectionId::from(cut_a.0)).unwrap();
        let collection_b = &mut self.collections.get_mut(&CollectionId::from(cut_b.0)).unwrap();
        
        // Partitions may be cached or reconstructed
        let partition_a = collection_a.at_threshold(cut_a.1);
        let partition_b = collection_b.at_threshold(cut_b.1);
        
        self.analyzer.compare_partitions(partition_a, partition_b)
    }
}
```

### Python Interface via PyO3

**Core Python bindings**

```rust
use pyo3::prelude::*;

#[pyclass]
pub struct PyEntityFrame {
    inner: Arc<Mutex<EntityFrame>>,
}

#[pymethods]
impl PyEntityFrame {
    #[new]
    pub fn new() -> Self {
        PyEntityFrame {
            inner: Arc::new(Mutex::new(EntityFrame::new())),
        }
    }
    
    pub fn add_collection(&mut self, 
                         name: &str, 
                         py_hierarchy: PyObject) -> PyResult<String> {
        let hierarchy = PartitionHierarchy::from_pyobject(py_hierarchy)?;
        let mut frame = self.inner.lock().unwrap();
        let id = frame.add_collection(name, hierarchy);
        Ok(id.to_string())
    }
    
    pub fn load_records(&mut self, 
                       source: &str, 
                       data: &PyAny) -> PyResult<usize> {
        // Handle pandas DataFrame, Arrow table, or list of dicts
        let records = if data.is_instance_of::<PyDataFrame>()? {
            records_from_dataframe(data)?
        } else if data.is_instance_of::<PyArrowTable>()? {
            records_from_arrow(data)?
        } else {
            records_from_list(data)?
        };
        
        let mut frame = self.inner.lock().unwrap();
        let ids = frame.add_records_from_source(source, records);
        Ok(ids.len())
    }
    
    #[pyo3(text_signature = "($self, cut_a, cut_b)")]
    pub fn compare(&mut self, 
                   cut_a: PyCollectionCut,
                   cut_b: PyCollectionCut) -> PyResult<PyDict> {
        let mut frame = self.inner.lock().unwrap();
        
        // All cuts are now (name, threshold) tuples
        let rust_cut_a = cut_a.to_rust()?;
        let rust_cut_b = cut_b.to_rust()?;
        
        let result = frame.compare_cuts(rust_cut_a, rust_cut_b);
        
        // Convert to Python dict
        let py_dict = PyDict::new();
        py_dict.set_item("precision", result.precision)?;
        py_dict.set_item("recall", result.recall)?;
        py_dict.set_item("f1", result.f1)?;
        py_dict.set_item("ari", result.ari)?;
        py_dict.set_item("nmi", result.nmi)?;
        Ok(py_dict)
    }
}

// All collections now use (name, threshold) format
struct PyCollectionCut {
    name: String,
    threshold: f64,
}

impl PyCollectionCut {
    fn from_python(obj: &PyAny) -> PyResult<Self> {
        let (name, threshold) = obj.extract::<(String, f64)>()?;
        Ok(PyCollectionCut { name, threshold })
    }
    
    fn to_rust(&self) -> PyResult<(&str, f64)> {
        Ok((&self.name, self.threshold))
    }
}
```

**Python usage examples**

```python
import entityframe as ef
import pandas as pd

# Create frame and load data from multiple sources
frame = ef.EntityFrame()

# Load records from different sources
frame.load_records("CRM", pd.read_csv("crm_customers.csv"))
frame.load_records("MailingList", pd.read_csv("mailing_list.csv"))
frame.load_records("OrderSystem", pd.read_parquet("orders.parquet"))

# Add different resolution attempts as collections (all hierarchies)
frame.add_collection("splink_output", splink_hierarchy)
frame.add_collection("dedupe_output", dedupe_hierarchy)
frame.add_collection("ground_truth", truth_hierarchy)  # Even "fixed" data is at threshold 1.0

# Compare collections at specific thresholds
# Note: ALL comparisons now use (collection, threshold) format
comparison = frame.compare(
    ("splink_output", 0.85),
    ("dedupe_output", 0.76)
)
print(f"Agreement: F1={comparison['f1']:.3f}")

# Compare against "fixed" ground truth (at threshold 1.0)
comparison = frame.compare(
    ("splink_output", 0.85),
    ("ground_truth", 1.0)  # Explicit threshold for all collections
)
print(f"Splink at 0.85: F1={comparison['f1']:.3f}")

# Sweep thresholds to find optimal
results = []
for threshold in np.arange(0.5, 1.0, 0.01):
    metrics = frame.compare(
        ("splink_output", threshold),
        ("ground_truth", 1.0)
    )
    results.append((threshold, metrics['f1']))

optimal = max(results, key=lambda x: x[1])
print(f"Optimal threshold: {optimal[0]:.2f} (F1={optimal[1]:.3f})")

# Analyse specific entity
entity = frame.get_entity("splink_output", 0.85, entity_id=42)
for source, record_id in entity.members:
    record = frame.get_record(source, record_id)
    print(f"  {source}: {record}")

# Export for further analysis
frame.to_arrow("entity_analysis.arrow")
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
            let hierarchy = PartitionHierarchy::from_edges(chunk.collect(), self.num_records);
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
            num_records: self.records.len(),
            num_sources: self.interner.num_sources(),
            total_memory_mb: self.estimate_memory_usage() / 1_048_576.0,
            avg_reconstruction_time_ms: cache_stats.avg_miss_time_ms,
            cache_hit_rate: cache_stats.hit_rate,
            incremental_computation_rate: self.compute_incremental_rate(),
            avg_entity_stability: self.compute_avg_stability(),
            threshold_sensitivity: self.compute_sensitivity(),
        }
    }
    
    fn estimate_memory_usage(&self) -> f64 {
        let merge_memory: usize = self.collections
            .values()
            .map(|h| h.merges.len() * std::mem::size_of::<MergeEvent>())
            .sum();
        
        let cache_memory: usize = self.collections
            .values()
            .map(|h| h.partition_cache.len() * 500_000) // ~500KB per cached partition
            .sum();
        
        (merge_memory + cache_memory) as f64
    }
}
```

**Performance characteristics**

```rust
/// Expected performance for different scales
/// Based on merge event representation with LRU caching
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
