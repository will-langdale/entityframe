# EntityFrame Design Document B: Technical implementation

## Layer 2: Computer science techniques

### Core data structure architecture

**Multi-collection EntityFrame structure**

```rust
pub struct EntityFrame {
    // Multiple hierarchies (collections) sharing same record space
    // Note: We use HashMap<CollectionId, EntityCollection> for implementation,
    // but mathematically this represents {H₁, H₂, ..., Hₙ}
    collections: HashMap<CollectionId, EntityCollection>,
    
    // Shared interned record storage
    records: InternedRecordStorage,
    
    // Cross-collection comparison engine
    comparison_engine: CollectionComparer,
    
    // Global string interning for sources
    source_interner: StringInterner,
}

pub struct EntityCollection {
    // The hierarchical partition structure
    // Note: Mathematically, the collection IS the hierarchy.
    // This struct wraps it with metadata for implementation purposes.
    hierarchy: PartitionHierarchy,
    
    // Collection metadata
    metadata: CollectionMetadata,
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

### Efficient hierarchical data structures

**Partition hierarchy with RoaringBitmaps**

```rust
pub struct PartitionHierarchy {
    // Partition at each threshold where merges occur
    levels: BTreeMap<OrderedFloat<f64>, PartitionLevel>,
    
    // Transition information between levels
    transitions: Vec<ThresholdTransition>,
    
    // Cache for frequently accessed computations
    computation_cache: ComputationCache,
}

pub struct PartitionLevel {
    threshold: f64,
    
    // Entities as sets of record indices (RoaringBitmaps)
    entities: Vec<RoaringBitmap>,
    
    // Pre-computed cheap statistics
    entity_sizes: Vec<u32>,
    internal_edges: Vec<u32>,
    sum_weights: Vec<f64>,
    
    // Lazy expensive metrics
    lazy_metrics: LazyMetricsCache,
}

pub struct ThresholdTransition {
    from_threshold: f64,
    to_threshold: f64,
    
    // N-way merge support
    merging_groups: Vec<Vec<EntityId>>,
    
    // For incremental computation
    added_edges: u32,
    removed_edges: u32,
    affected_records: RoaringBitmap,
}
```

**Memory analysis**

For 1M records, 100 threshold levels, 100K entities average:
- RoaringBitmap per entity: ~5 bytes (sparse) to ~125KB (dense)
- Per level: 100K × 5 bytes = 500KB typical
- Total hierarchy: 100 × 500KB = 50MB
- Acceptable tradeoff for computational efficiency

**Quantisation and memory warnings**

```rust
impl PartitionHierarchy {
    pub fn check_memory_usage(&self, num_records: usize) -> MemoryStatus {
        let num_levels = self.levels.len();
        let estimated_entries = num_records * num_levels;
        
        if estimated_entries > 100_000_000 {
            MemoryStatus::Warning(format!(
                "Large hierarchy: {}M entries. Consider quantisation to reduce precision",
                estimated_entries / 1_000_000
            ))
        } else {
            MemoryStatus::Ok
        }
    }
    
    pub fn with_quantisation(mut self, decimal_places: usize) -> Self {
        // Optionally quantise thresholds to reduce unique levels
        let quantisation_factor = 10_f64.powi(decimal_places as i32);
        // Implementation merges levels within quantisation bounds
        self
    }
}
```

### String interning architecture

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

### Connected components algorithm for hierarchy construction

**Union-find based hierarchy builder**

```rust
use disjoint_sets::UnionFind;

impl PartitionHierarchy {
    pub fn from_edges(edges: Vec<(u32, u32, f64)>, num_records: u32) -> Self {
        // Group edges by similarity value
        let mut threshold_groups: BTreeMap<OrderedFloat<f64>, Vec<(u32, u32)>> = 
            BTreeMap::new();
        
        for (src, dst, weight) in edges {
            threshold_groups
                .entry(OrderedFloat(weight))
                .or_default()
                .push((src, dst));
        }
        
        let mut levels = BTreeMap::new();
        let mut current_partition = (0..num_records)
            .map(|i| {
                let mut bitmap = RoaringBitmap::new();
                bitmap.insert(i);
                bitmap
            })
            .collect::<Vec<_>>();
        
        // Process from highest to lowest threshold
        for (threshold, edges_at_threshold) in threshold_groups.into_iter().rev() {
            // Union-find handles n-way merges naturally
            let mut uf = UnionFind::new(num_records as usize);
            
            // Apply all edges at this threshold simultaneously
            for (src, dst) in edges_at_threshold {
                uf.union(src as usize, dst as usize);
            }
            
            // Extract connected components
            let partition = uf.to_partition();
            let entities = partition_to_bitmaps(partition);
            
            levels.insert(threshold, PartitionLevel {
                threshold: threshold.0,
                entities,
                entity_sizes: compute_sizes(&entities),
                internal_edges: vec![],
                sum_weights: vec![],
                lazy_metrics: LazyMetricsCache::new(),
            });
            
            current_partition = entities;
        }
        
        PartitionHierarchy {
            levels,
            transitions: Vec::new(),  // Built lazily when needed
            computation_cache: Default::default(),
        }
    }
}
```

This approach naturally handles n-way merges: when edges (A,B,0.9), (B,C,0.9), and (C,D,0.9) all have the same weight, the union-find merges {A,B,C,D} simultaneously at threshold 0.9.

### Lazy metric computation framework

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
                // Must recalculate for all records in affected entities
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
            // Incremental update - O(changes) instead of O(n²)
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
        
        for threshold in (0..)
            .map(|i| start + i as f64 * step)
            .take_while(|&t| t <= end) {
            
            let metrics = if let Some(prev) = current_metrics {
                // Update from previous - extremely fast
                self.update_metrics_incremental(prev, threshold, truth)
            } else {
                // Only compute from scratch once
                self.compute_metrics_full(threshold, truth)
            };
            
            results.push((threshold, metrics.clone()));
            current_metrics = Some(metrics);
        }
        
        results
    }
}
```

### Sparse structure exploitation

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
        // Only update cells affected by merge
        for entity_id in &merge.merging_entities {
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

// 3. Sparse similarity matrix
pub struct SparseSimilarityMatrix {
    // Only store non-zero similarities
    edges: Vec<(u32, u32, f64)>,
    
    // For efficient lookup
    adjacency: HashMap<u32, Vec<(u32, f64)>>,
}
```

**Exploiting sparsity**

```rust
impl SparseSimilarityMatrix {
    pub fn find_merges_at_threshold(&self, threshold: f64) -> Vec<MergeEvent> {
        // Only examine edges above threshold - typically sparse
        let active_edges = self.edges
            .iter()
            .filter(|(_, _, w)| *w >= threshold);
        
        // Union-find on sparse graph
        let mut uf = UnionFind::new(self.num_nodes());
        for (u, v, _) in active_edges {
            uf.union(*u, *v);
        }
        
        uf.extract_components()
    }
}
```

### SIMD and cache optimisations

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

### Parallel processing architecture

**Rayon-based parallel hierarchy construction**

```rust
use rayon::prelude::*;

impl PartitionHierarchy {
    pub fn build_parallel(edges: Vec<WeightedEdge>) -> Self {
        // Group edges by threshold
        let mut threshold_groups: HashMap<OrderedFloat<f64>, Vec<(u32, u32)>> = 
            HashMap::new();
        
        for edge in edges {
            threshold_groups
                .entry(OrderedFloat(edge.weight))
                .or_default()
                .push((edge.src, edge.dst));
        }
        
        // Process each threshold level in parallel
        let levels: Vec<(OrderedFloat<f64>, PartitionLevel)> = threshold_groups
            .into_par_iter()
            .map(|(threshold, edges)| {
                let partition = Self::build_partition_level(edges);
                (threshold, partition)
            })
            .collect();
        
        PartitionHierarchy {
            levels: levels.into_iter().collect(),
            transitions: Vec::new(),  // Built after
            computation_cache: Default::default(),
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
        let truth_cut = (&self.collections[&truth_id], threshold_truth);
        
        thresholds
            .par_iter()
            .filter(|(id, _)| **id != truth_id)
            .map(|(id, threshold)| {
                let collection = &self.collections[id];
                let metrics = self.comparison_engine.compare_cuts(
                    (collection, *threshold),
                    truth_cut
                );
                (*id, metrics)
            })
            .collect()
    }
}
```

**Note on distributed processing**: The block structure of disconnected components naturally enables local parallelisation. Full distributed processing across machines would require careful handling of cross-partition merges and is deferred to future work.

### Dual processing for entity operations

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

### Arrow integration

**Hierarchical Arrow schema**

```rust
use arrow::datatypes::{DataType, Field, Schema};

pub fn create_entityframe_schema() -> Schema {
    Schema::new(vec![
        // Collection metadata
        Field::new("collection_id", DataType::Utf8, false),
        Field::new("collection_type", DataType::Utf8, false),
        
        // Interned references
        Field::new("source_dictionary", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8),
        ), false),
        
        // Partition levels
        Field::new("partitions", DataType::List(Box::new(Field::new(
            "partition", 
            DataType::Struct(vec![
                Field::new("threshold", DataType::Float64, false),
                Field::new("num_entities", DataType::UInt32, false),
                Field::new("entities", DataType::List(
                    Box::new(Field::new("entity", DataType::List(
                        Box::new(Field::new("record_id", DataType::UInt32, false))
                    ), false))
                ), false),
            ]), 
            false
        ))), false),
        
        // Statistics
        Field::new("statistics", DataType::Struct(vec![
            Field::new("entity_sizes", DataType::List(
                Box::new(DataType::UInt32)
            ), false),
            Field::new("internal_edges", DataType::List(
                Box::new(DataType::UInt32)
            ), false),
        ]), true),
    ])
}
```

## Layer 3: Practical engineering design

### Core EntityFrame API

**Multi-collection management**

```rust
pub struct EntityFrame {
    collections: HashMap<CollectionId, EntityCollection>,
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
        // Wrap hierarchy with metadata for implementation purposes
        // (mathematically, the collection IS the hierarchy)
        let collection = EntityCollection {
            hierarchy,
            metadata: CollectionMetadata::new(name),
        };
        self.collections.insert(id, collection);
        id
    }
    
    pub fn add_records_from_source(&mut self, 
                                   source_name: &str,
                                   records: Vec<Record>) -> Vec<RecordId> {
        let source_id = self.interner.intern_source(source_name);
        self.records.add_from_source(source_id, records)
    }
    
    pub fn compare_cuts(&self,
                       cut_a: (&str, f64),
                       cut_b: (&str, f64)) -> ComparisonResult {
        let collection_a = &self.collections[&CollectionId::from(cut_a.0)];
        let collection_b = &self.collections[&CollectionId::from(cut_b.0)];
        self.analyzer.compare_partitions(
            collection_a.hierarchy.at_threshold(cut_a.1),
            collection_b.hierarchy.at_threshold(cut_b.1)
        )
    }
}
```

### Python interface via PyO3

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
        let frame = self.inner.lock().unwrap();
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
    pub fn compare(&self, 
                   cut_a: PyCollectionCut,
                   cut_b: PyCollectionCut) -> PyResult<PyDict> {
        let frame = self.inner.lock().unwrap();
        
        // Convert Python tuples/strings to cuts
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

// Helper for Python collection cuts
enum PyCollectionCut {
    Hierarchical(String, f64),  // ("splink_v1", 0.85)
    Deterministic(String),       // "ground_truth"
}

impl PyCollectionCut {
    fn from_python(obj: &PyAny) -> PyResult<Self> {
        if let Ok(name) = obj.extract::<String>() {
            Ok(PyCollectionCut::Deterministic(name))
        } else if let Ok((name, threshold)) = obj.extract::<(String, f64)>() {
            Ok(PyCollectionCut::Hierarchical(name, threshold))
        } else {
            Err(PyErr::new::<PyTypeError, _>(
                "Expected string or (string, float) tuple for collection cut"
            ))
        }
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

# Add different resolution attempts as collections
frame.add_collection("splink_output", splink_hierarchy)
frame.add_collection("dedupe_output", dedupe_hierarchy)
frame.add_collection("ground_truth", truth_hierarchy)

# Compare collections at specific thresholds
# Note: thresholds are not comparable across methods
comparison = frame.compare(
    ("splink_output", 0.85),
    ("dedupe_output", 0.76)
)
print(f"Agreement: F1={comparison['f1']:.3f}")

# Compare against deterministic ground truth
comparison = frame.compare(
    ("splink_output", 0.85),
    "ground_truth"  # No threshold needed for deterministic
)
print(f"Splink at 0.85: F1={comparison['f1']:.3f}")

# Sweep thresholds to find optimal
results = []
for threshold in np.arange(0.5, 1.0, 0.01):
    metrics = frame.compare(
        ("splink_output", threshold),
        "ground_truth"
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

### Incremental processing

**Streaming updates**

```rust
pub struct StreamingEntityFrame {
    base_frame: EntityFrame,
    update_buffer: RingBuffer<RecordUpdate>,
    batch_size: usize,
}

impl StreamingEntityFrame {
    pub async fn process_stream<S>(&mut self, mut stream: S) 
    where S: Stream<Item = RecordUpdate> + Unpin
    {
        while let Some(update) = stream.next().await {
            self.update_buffer.push(update);
            
            if self.update_buffer.len() >= self.batch_size {
                self.flush_updates().await;
            }
        }
        
        // Final flush
        if !self.update_buffer.is_empty() {
            self.flush_updates().await;
        }
    }
    
    async fn flush_updates(&mut self) {
        let updates: Vec<_> = self.update_buffer.drain().collect();
        
        // Add new records
        let new_records = updates.iter()
            .filter_map(|u| u.as_new_record())
            .collect::<Vec<_>>();
        self.base_frame.records.add_batch(new_records);
        
        // Update hierarchies incrementally
        for collection in self.base_frame.collections.values_mut() {
            collection.hierarchy.update_incremental(&updates);
        }
    }
}
```

### Production deployment

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
    avg_comparison_time_ms: f64,
    cache_hit_rate: f64,
    incremental_computation_rate: f64,
    
    // Quality indicators
    avg_entity_stability: f64,
    threshold_sensitivity: f64,
}

impl EntityFrame {
    pub fn collect_metrics(&self) -> FrameMetrics {
        FrameMetrics {
            num_collections: self.collections.len(),
            num_records: self.records.len(),
            num_sources: self.interner.num_sources(),
            total_memory_mb: self.estimate_memory_usage() / 1_048_576.0,
            avg_comparison_time_ms: self.analyzer.avg_comparison_time(),
            cache_hit_rate: self.compute_cache_hit_rate(),
            incremental_computation_rate: self.compute_incremental_rate(),
            avg_entity_stability: self.compute_avg_stability(),
            threshold_sensitivity: self.compute_sensitivity(),
        }
    }
}
```