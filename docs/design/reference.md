# EntityFrame Design Document C: Reference architecture

## Complete API reference

### Python API

#### Core EntityFrame operations

```python
class EntityFrame:
    """
    Main container for multiple entity resolution collections over shared records.
    
    Mathematically: F = (R, {H₁, H₂, ..., Hₙ}, I) where hierarchies ARE the collections.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise EntityFrame.
        
        Args:
            config: Optional configuration dict with keys:
                - 'parallel_workers': Number of parallel threads (default: CPU count)
                - 'cache_size_mb': LRU cache size for metrics (default: 100)
                - 'arrow_batch_size': Batch size for Arrow operations (default: 10000)
                - 'string_intern_capacity': Initial interner capacity (default: 10000)
                - 'quantisation_decimals': Round thresholds to N decimal places (default: None)
        """
    
    def load_records(self, 
                    source_name: str, 
                    data: Union[pd.DataFrame, pa.Table, List[Dict]],
                    id_column: Optional[str] = None) -> int:
        """
        Load records from a data source.
        
        Args:
            source_name: Name of the source (e.g., "CRM", "MailingList")
            data: Records as DataFrame, Arrow Table, or list of dicts
            id_column: Column to use as local ID (default: auto-generate)
            
        Returns:
            Number of records loaded
            
        Example:
            frame.load_records("CRM", df_customers, id_column="customer_id")
        """
    
    def add_collection(self,
                      name: str,
                      hierarchy: Optional[Hierarchy] = None,
                      clusters: Optional[List[Set]] = None,
                      similarities: Optional[np.ndarray] = None) -> str:
        """
        Add an entity resolution collection to the frame.
        
        Note: Collections ARE hierarchies mathematically. This method creates
        a hierarchy from the provided data format.
        
        Args:
            name: Unique name for this collection
            hierarchy: Pre-built hierarchy object
            clusters: Fixed clustering (creates degenerate hierarchy)
            similarities: Build hierarchy from similarity matrix
            
        Returns:
            Collection ID
            
        Example:
            frame.add_collection("splink_v1", hierarchy=splink_hierarchy)
            frame.add_collection("ground_truth", clusters=truth_clusters)
        """
    
    def compare(self,
               cut_a: Union[str, Tuple[str, float]],
               cut_b: Union[str, Tuple[str, float]],
               metrics: List[str] = None) -> Dict[str, float]:
        """
        Compare two collections at specific thresholds.
        
        Args:
            cut_a: Collection name (if deterministic) or (name, threshold) tuple
            cut_b: Collection name (if deterministic) or (name, threshold) tuple
            metrics: List of metrics to compute (default: all)
                     Options: 'precision', 'recall', 'f1', 'ari', 'nmi', 
                              'v_measure', 'bcubed_precision', 'bcubed_recall'
        
        Returns:
            Dictionary of metric values
            
        Note:
            Most metrics update incrementally O(k) where k = changed entities.
            B-cubed metrics are semi-incremental O(k × avg_entity_size).
            
        Example:
            # Compare two hierarchical collections at their thresholds
            results = frame.compare(("splink_v1", 0.85), ("dedupe_v2", 0.76))
            
            # Compare hierarchical against deterministic ground truth
            results = frame.compare(("splink_v1", 0.85), "ground_truth")
        """
    
    def compare_sweep(self,
                     collection: str,
                     truth: str,
                     start: float = 0.0,
                     end: float = 1.0,
                     step: float = 0.01) -> pd.DataFrame:
        """
        Sweep thresholds and compare against truth.
        
        Args:
            collection: Collection to sweep
            truth: Ground truth collection (deterministic or hierarchical)
            start: Starting threshold
            end: Ending threshold
            step: Threshold increment
            
        Returns:
            DataFrame with columns: threshold, precision, recall, f1, ari, nmi, etc.
            
        Example:
            results = frame.compare_sweep("splink_v1", "ground_truth")
            results.plot(x='threshold', y=['precision', 'recall'])
        """
    
    def find_optimal_threshold(self,
                              collection: str,
                              truth: str,
                              metric: str = 'f1',
                              search_range: Tuple[float, float] = (0.0, 1.0)) -> OptimalResult:
        """
        Find threshold that optimises a metric.
        
        Args:
            collection: Collection to optimise
            truth: Ground truth collection
            metric: Metric to optimise
            search_range: Range of thresholds to search
            
        Returns:
            OptimalResult with threshold, score, and full metrics
            
        Example:
            optimal = frame.find_optimal_threshold("splink_v1", "ground_truth")
            print(f"Best threshold: {optimal.threshold} (F1={optimal.score})")
        """
    
    def get_entity(self,
                  collection: str,
                  threshold: float,
                  entity_id: int) -> Entity:
        """
        Get a specific entity from a collection at a threshold.
        
        Returns:
            Entity object with members as (source, local_id) pairs
            
        Example:
            entity = frame.get_entity("splink_v1", 0.85, entity_id=42)
            for source, record_id in entity.members:
                print(f"{source}: {record_id}")
        """
    
    def get_partition(self,
                     collection: str,
                     threshold: Optional[float] = None,
                     map_func: Optional[Callable[[Entity], Any]] = None) -> Union[Partition, Dict[EntityId, Any]]:
        """
        Get partition at threshold, optionally mapping a function over entities.
        
        Args:
            collection: Collection name
            threshold: Threshold (not needed for deterministic collections)
            map_func: Optional function to apply to each entity
                     Can be:
                     - Built-in operation string: 'hash:sha256', 'size', 'density'
                     - Custom Python callable
        
        Returns:
            Partition if no map_func, otherwise Dict[EntityId, result]
        
        Example:
            # Just get the partition
            partition = frame.get_partition("splink_v1", 0.85)
            
            # Compute SHA256 hashes (runs in parallel Rust)
            hashes = frame.get_partition("splink_v1", 0.85, map_func='hash:sha256')
            
            # Custom Python function
            custom = frame.get_partition(
                "splink_v1", 0.85,
                map_func=lambda e: len(e.members) * 2
            )
        """
    
    def load_resolved_collection(self,
                                name: str,
                                entities: List[Set[Tuple[str, Any]]],
                                source_frame: Optional['EntityFrame'] = None) -> str:
        """
        Load pre-resolved entities as a collection.
        
        Args:
            name: Name for this collection
            entities: List of entity sets, each containing (source, id) pairs
            source_frame: Optional source EntityFrame for provenance
            
        Returns:
            Collection ID
            
        Example:
            # Load entities resolved elsewhere
            frame.load_resolved_collection(
                "hospital_patients",
                patient_entities,  # List of sets of (source, id) tuples
                source_frame=hospital_frame
            )
        """
```

#### Hierarchy operations

```python
class Hierarchy:
    """
    Hierarchical partition structure that generates entities at any threshold.
    """
    
    @classmethod
    def from_similarities(cls,
                         similarity_matrix: np.ndarray,
                         method: str = 'connected_components',
                         record_ids: Optional[List] = None) -> 'Hierarchy':
        """
        Build hierarchy from similarity matrix.
        
        Args:
            similarity_matrix: Square matrix of pairwise similarities
            method: Method for hierarchy construction ('connected_components')
            record_ids: Optional record identifiers
            
        Returns:
            New Hierarchy object
            
        Note:
            Uses threshold-based connected components (equivalent to single-linkage)
            which naturally handles n-way merges at identical similarity values.
        """
    
    @classmethod
    def from_edges(cls,
                  edges: List[Tuple[int, int, float]],
                  num_records: int) -> 'Hierarchy':
        """
        Build hierarchy from weighted edges.
        
        Args:
            edges: List of (record_i, record_j, similarity) tuples
            num_records: Total number of records
            
        Returns:
            New Hierarchy object
            
        Example:
            # Edges with same weight merge simultaneously
            edges = [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9)]
            hierarchy = Hierarchy.from_edges(edges, 4)
            # At threshold 0.9: creates 4-way merge {0,1,2,3}
        """
    
    def at_threshold(self, threshold: float) -> Partition:
        """
        Get partition at specific threshold.
        
        Returns:
            Partition object with entities at that threshold
            
        Example:
            partition = hierarchy.at_threshold(0.85)
            print(f"Number of entities: {len(partition.entities)}")
        """
    
    def analyse_threshold_range(self,
                              start: float,
                              end: float,
                              step: float = 0.01) -> ThresholdAnalysis:
        """
        Analyse behaviour across threshold range.
        
        Returns:
            ThresholdAnalysis with stability regions, critical points, etc.
            
        Example:
            analysis = hierarchy.analyse_threshold_range(0.5, 0.95)
            for start, end in analysis.stable_regions:
                print(f"Stable from {start:.2f} to {end:.2f}")
        """
    
    def track_entity_lifetime(self, record_id: int) -> List[EntityLifetime]:
        """
        Track how a record's entity membership changes.
        
        Returns:
            List of (threshold_start, threshold_end, entity_size) tuples
            
        Example:
            lifetime = hierarchy.track_entity_lifetime(record_id=123)
            for t_start, t_end, size in lifetime:
                print(f"In size-{size} entity from {t_start} to {t_end}")
        """
    
    def diff_thresholds(self,
                       threshold_a: float,
                       threshold_b: float) -> ThresholdDiff:
        """
        Find differences between two thresholds.
        
        Returns:
            ThresholdDiff with merges, splits, affected records
            
        Example:
            diff = hierarchy.diff_thresholds(0.8, 0.85)
            print(f"Entities merging: {diff.merging_entities}")
        """
    
    def check_memory_usage(self, num_records: int) -> MemoryStatus:
        """
        Check if hierarchy size may cause memory issues.
        
        Returns:
            MemoryStatus with warnings if n×m exceeds thresholds
            
        Example:
            status = hierarchy.check_memory_usage(1_000_000)
            if status.has_warning:
                print(f"Warning: {status.message}")
        """
```

#### Analysis operations

```python
class ThresholdSweep:
    """
    Efficient threshold sweep analysis.
    """
    
    def __init__(self,
                frame: EntityFrame,
                collection: str,
                truth: str):
        """Initialise sweep analysis."""
    
    def compute_all_metrics(self,
                           start: float = 0.0,
                           end: float = 1.0,
                           step: float = 0.01) -> pd.DataFrame:
        """
        Compute all metrics across threshold range.
        
        Returns:
            DataFrame with columns: threshold, precision, recall, f1, ari, nmi, etc.
            
        Example:
            sweep = ThresholdSweep(frame, "splink_v1", "ground_truth")
            results = sweep.compute_all_metrics(0.5, 0.95, 0.01)
            results.plot(x='threshold', y=['precision', 'recall'])
        """
    
    def find_pareto_frontier(self,
                           metric_x: str = 'precision',
                           metric_y: str = 'recall') -> List[Tuple[float, float, float]]:
        """
        Find Pareto-optimal thresholds.
        
        Returns:
            List of (threshold, metric_x_value, metric_y_value) tuples
        """
    
    def stability_analysis(self) -> StabilityReport:
        """
        Analyse stability of resolution across thresholds.
        
        Returns:
            StabilityReport with stable regions, volatility scores, etc.
        """
```

### Data shape specifications

#### Input formats for load_records()

```python
# DataFrame input
df = pd.DataFrame({
    'customer_id': [1, 2, 3],      # Will be used as local_id if specified
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@ex.com', 'bob@ex.com', 'charlie@ex.com']
})
frame.load_records("CRM", df, id_column='customer_id')

# List of dicts input
records = [
    {'id': 1, 'name': 'Alice', 'email': 'alice@ex.com'},
    {'id': 2, 'name': 'Bob', 'email': 'bob@ex.com'}
]
frame.load_records("MailingList", records, id_column='id')

# Arrow Table input
table = pa.Table.from_pandas(df)
frame.load_records("OrderSystem", table, id_column='customer_id')
```

#### Input formats for add_collection()

```python
# From similarity matrix (square, symmetric)
similarities = np.array([
    [1.0, 0.8, 0.2],  # Record 0's similarities
    [0.8, 1.0, 0.3],  # Record 1's similarities  
    [0.2, 0.3, 1.0]   # Record 2's similarities
])
frame.add_collection("similarity_based", similarities=similarities)

# From edges (record_i, record_j, weight)
edges = [
    (0, 1, 0.95),  # Record 0 and 1 with 0.95 similarity
    (1, 2, 0.85),  # Record 1 and 2 with 0.85 similarity
    (0, 3, 0.75)   # Record 0 and 3 with 0.75 similarity
]
hierarchy = Hierarchy.from_edges(edges, num_records=4)
frame.add_collection("edge_based", hierarchy=hierarchy)

# From fixed clusters (deterministic)
clusters = [
    {0, 1, 4},      # First entity contains records 0, 1, 4
    {2, 3},         # Second entity contains records 2, 3
    {5, 6, 7, 8}    # Third entity contains records 5, 6, 7, 8
]
frame.add_collection("ground_truth", clusters=clusters)

# From pre-resolved entities with source attribution
entities = [
    {("CRM", 1), ("CRM", 9), ("MailingList", 17)},  # Cross-source entity
    {("OrderSystem", 42), ("OrderSystem", 43)}       # Single-source entity
]
frame.load_resolved_collection("pre_resolved", entities=entities)
```

### Built-in operations for entity processing

```python
# Available built-in operations (run in parallel Rust)
BUILT_IN_OPS = {
    'hash:sha256': 'SHA-256 hash of sorted entity members',
    'hash:md5': 'MD5 hash of sorted entity members', 
    'hash:blake3': 'BLAKE3 hash (faster than SHA)',
    'size': 'Number of records in entity',
    'density': 'Internal connectivity measure',
    'fingerprint': 'MinHash or similar sketch'
}

# Example usage
hashes = frame.get_partition("splink_v1", 0.85, map_func='hash:sha256')
sizes = frame.get_partition("splink_v1", 0.85, map_func='size')

# Custom Python function (slower, single-threaded)
def custom_metric(entity):
    return sum(1 for (source, _) in entity.members if source == "CRM")

crm_counts = frame.get_partition("splink_v1", 0.85, map_func=custom_metric)
```

## Rust API

### Core structures

```rust
/// Main EntityFrame structure
pub struct EntityFrame {
    collections: HashMap<CollectionId, EntityCollection>,
    records: InternedRecordStorage,
    interner: InternSystem,
    analyzer: FrameAnalyzer,
    config: FrameConfig,
}

/// Individual entity collection with hierarchy
/// Note: Mathematically, the collection IS the hierarchy.
/// This struct wraps it with metadata for implementation purposes.
pub struct EntityCollection {
    hierarchy: PartitionHierarchy,
    metadata: CollectionMetadata,
}

/// Hierarchical partition structure
pub struct PartitionHierarchy {
    levels: BTreeMap<OrderedFloat<f64>, PartitionLevel>,
    transitions: Vec<ThresholdTransition>,
    computation_cache: ComputationCache,
}

/// Single partition at a threshold
pub struct PartitionLevel {
    threshold: f64,
    entities: Vec<RoaringBitmap>,
    statistics: PartitionStatistics,
    lazy_metrics: LazyMetricsCache,
}

/// Interned record reference
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct InternedRef {
    source_id: u16,    // Interned source name
    local_id: LocalId, // ID within that source (flexible type)
}

/// Flexible local ID types
pub enum LocalId {
    U32(u32),
    U64(u64),
    String(InternedString),
    Bytes(Vec<u8>),
}

/// Entity as set of interned references
pub struct Entity {
    members: RoaringBitmap,  // Indices into record storage
    cached_size: OnceCell<u32>,
}

/// Collection cut (name + optional threshold)
pub enum CollectionCut {
    Hierarchical(String, f64),  // ("splink_v1", 0.85)
    Deterministic(String),       // "ground_truth"
}
```

### Key traits

```rust
/// Trait for hierarchical structures
pub trait Hierarchical {
    fn at_threshold(&self, threshold: f64) -> &PartitionLevel;
    fn threshold_range(&self) -> (f64, f64);
    fn num_levels(&self) -> usize;
    fn merge_events(&self) -> &[MergeEvent];
}

/// Trait for comparable partitions
pub trait Comparable {
    fn compare_with(&self, other: &Self) -> ComparisonMetrics;
    fn contingency_table(&self, other: &Self) -> ContingencyTable;
}

/// Trait for incremental updates
pub trait Incremental {
    fn update_with_records(&mut self, records: &[Record]) -> UpdateResult;
    fn update_with_similarities(&mut self, similarities: &[(u32, u32, f64)]) -> UpdateResult;
}
```

## Data structure specifications

### Memory layout

```rust
/// Memory-efficient entity storage using RoaringBitmaps
/// 
/// For 1M records, 100K entities:
/// - Traditional HashSet<u32>: ~400MB
/// - RoaringBitmap: ~50MB (87% reduction)
struct MemoryLayout {
    // Level 1: Source interning
    source_names: StringInterner,      // ~1KB for 100 sources
    
    // Level 2: Record storage
    record_refs: Vec<InternedRef>,     // 6 bytes per record
    record_attrs: Option<Vec<Attrs>>,  // Optional attribute data
    
    // Level 3: Entity storage
    entities: Vec<RoaringBitmap>,      // ~5 bytes per entity (sparse)
    
    // Level 4: Hierarchy
    partitions: BTreeMap<Threshold, PartitionLevel>,  // ~500KB per level
}
```

### Serialisation formats

#### Arrow schema

```rust
/// Arrow schema for EntityFrame serialisation
Schema {
    fields: vec![
        // Frame metadata
        Field::new("version", DataType::Utf8, false),
        Field::new("created_at", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        
        // Source dictionary
        Field::new("sources", DataType::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(DataType::Utf8)
        ), false),
        
        // Records with interned references
        Field::new("records", DataType::Struct(vec![
            Field::new("source_id", DataType::UInt16, false),
            Field::new("local_id", DataType::UInt32, false),
        ]), false),
        
        // Collections
        Field::new("collections", DataType::List(Box::new(DataType::Struct(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("hierarchy", DataType::Binary, false),  // Serialised hierarchy
        ]))), false),
    ]
}
```

#### JSON export format

```json
{
  "version": "1.0",
  "metadata": {
    "created_at": "2025-01-15T10:30:00Z",
    "num_records": 1000000,
    "num_sources": 5,
    "num_collections": 3
  },
  "sources": {
    "0": "CRM",
    "1": "MailingList",
    "2": "OrderSystem"
  },
  "collections": [
    {
      "name": "splink_v1",
      "type": "hierarchical",
      "num_partitions": 87,
      "threshold_range": [0.0, 1.0],
      "statistics": {
        "avg_entity_size": 12.5,
        "stability_score": 0.73
      }
    }
  ]
}
```

## Integration examples

### Integration with Splink

```python
import splink
import entityframe as ef

# Run Splink
linker = splink.Linker(df, settings)
linker.predict()
clusters = linker.cluster_pairwise_at_threshold(0.85)

# Import into EntityFrame
frame = ef.EntityFrame()
frame.load_records("source", df)
frame.add_collection("splink_output", clusters=clusters)

# Compare with different threshold
clusters_90 = linker.cluster_pairwise_at_threshold(0.90)
frame.add_collection("splink_0.90", clusters=clusters_90)

comparison = frame.compare(("splink_output", 1.0), ("splink_0.90", 1.0))
print(f"Agreement: {comparison['agreement']:.2%}")
```

### Integration with er-evaluation

```python
import er_evaluation as er_eval
import entityframe as ef

# Load data into EntityFrame
frame = ef.EntityFrame()
frame.load_records("dataset", records)
frame.add_collection("predicted", clusters=predicted_clusters)
frame.add_collection("truth", clusters=true_clusters)

# Use er-evaluation metrics
metrics = er_eval.evaluate(
    frame.get_partition("predicted", 0.85),
    frame.get_partition("truth")
)

# Or use EntityFrame's built-in metrics
metrics = frame.compare(("predicted", 0.85), "truth")
```

### Integration with Matchbox

```python
import entityframe as ef
from matchbox import MatchboxClient

# Get Matchbox results
client = MatchboxClient()
matches = client.get_matches(dataset_id)

# Convert to EntityFrame
frame = ef.EntityFrame()
frame.add_collection("matchbox", edges=matches, num_records=n)

# Export back to Matchbox format with hashes
hashes = frame.get_partition("matchbox", 0.9, map_func='hash:sha256')
matchbox_data = {
    'entities': frame.get_partition("matchbox", 0.9).to_list(),
    'hashes': hashes
}
client.upload_results(matchbox_data)
```

### Apache Arrow integration

```python
import pyarrow as pa
import pyarrow.parquet as pq
import entityframe as ef

# Save to Arrow/Parquet
frame = ef.EntityFrame()
# ... load data and collections ...

arrow_table = frame.to_arrow()
pq.write_table(arrow_table, "entityframe.parquet")

# Load from Arrow/Parquet
table = pq.read_table("entityframe.parquet")
frame = ef.EntityFrame.from_arrow(table)

# Stream processing with Arrow
reader = pa.ipc.RecordBatchStreamReader("entity_stream.arrow")
for batch in reader:
    frame.add_records_batch(batch)
```

## Implementation roadmap

### Phase 1: Foundation (Months 1-3)
**Goal**: Core functionality with single collection

- Week 1-2: Rust core data structures
  - InternedRecordStorage implementation with flexible LocalId types
  - Basic PartitionHierarchy with RoaringBitmaps
  - Simple threshold queries
  
- Week 3-4: Python bindings
  - PyO3 setup and basic API
  - Load records from pandas/Arrow
  - Create hierarchy from similarities
  
- Week 5-8: Essential algorithms
  - Connected components clustering
  - Basic metrics (precision, recall, F1)
  - Threshold extraction
  
- Week 9-12: Testing and optimisation
  - Comprehensive test suite
  - Performance benchmarks
  - Memory profiling

**Deliverable**: Working single-collection EntityFrame with Python API

### Phase 2: Multi-collection & analytics (Months 4-6)
**Goal**: Multiple collections and comprehensive metrics

- Week 1-3: Multi-collection support
  - Collection management in EntityFrame
  - Cross-collection comparison with (collection, threshold) cuts
  - Shared record storage
  
- Week 4-6: Complete metrics suite
  - ARI, NMI, V-measure implementation
  - B-cubed metrics (semi-incremental)
  - Lazy computation framework
  
- Week 7-9: Incremental computation
  - Metric caching system
  - Incremental updates between thresholds
  - Computation statistics
  
- Week 10-12: Analysis tools
  - Threshold sweep optimisation
  - Stability analysis
  - Entity lifetime tracking

**Deliverable**: Full multi-collection comparison capabilities

### Phase 3: Scale & performance (Months 7-9)
**Goal**: Production-ready performance

- Week 1-3: Parallelisation
  - Rayon integration
  - Parallel hierarchy construction
  - Concurrent collection comparison
  - Dual processing for entity operations (built-in vs Python)
  
- Week 4-6: SIMD optimisations
  - Vectorised metric computation
  - SIMD-accelerated set operations
  - Cache-optimised layouts
  - Built-in hash operations (SHA256, BLAKE3)
  
- Week 7-9: Sparse optimisations
  - Sparse contingency tables
  - Block-based processing
  - Memory-mapped storage
  - Quantisation support
  
- Week 10-12: Streaming support
  - Incremental record addition
  - Online hierarchy updates
  - Bounded memory streaming
  - Pre-resolved entity loading

**Deliverable**: 10-100x performance improvement over baseline

### Phase 4: Integration & polish (Months 10-12)
**Goal**: Ecosystem integration and production features

- Week 1-3: Tool integrations
  - Splink adapter
  - er-evaluation compatibility
  - Matchbox export/import with hashing
  
- Week 4-6: Advanced features
  - Custom metric plugins
  - Entity metadata attachment
  - GPU acceleration experiments
  
- Week 7-9: Production tooling
  - Monitoring and metrics
  - Error handling and recovery
  - Configuration management
  
- Week 10-12: Documentation & release
  - Complete API documentation
  - Tutorial notebooks
  - Performance guides
  - v1.0 release preparation

**Deliverable**: Production-ready EntityFrame 1.0

## Performance benchmarks

### Target performance metrics

```yaml
# Single machine performance targets
small_scale:  # 10K records
  hierarchy_build: < 100ms
  threshold_query: < 1ms
  full_comparison: < 10ms
  memory_usage: < 10MB

medium_scale:  # 1M records
  hierarchy_build: < 10s
  threshold_query: < 100ms
  full_comparison: < 1s
  memory_usage: < 1GB

large_scale:  # 100M records
  hierarchy_build: < 30min
  threshold_query: < 1s
  full_comparison: < 30s
  memory_usage: < 100GB

# Scaling characteristics
complexity:
  hierarchy_construction: O(n² log n) worst, O(n log n) typical
  threshold_query: O(n)
  metric_computation: O(n²) worst, O(k) incremental
  memory: O(n × m) where m = unique thresholds
```

### Benchmark suite

```python
# Performance testing harness
import entityframe as ef
import time
import memory_profiler

class EntityFrameBenchmark:
    def __init__(self, num_records, num_sources=5):
        self.num_records = num_records
        self.num_sources = num_sources
        self.frame = ef.EntityFrame()
        
    def benchmark_hierarchy_build(self):
        similarities = self.generate_similarities()
        start = time.time()
        hierarchy = ef.Hierarchy.from_similarities(similarities)
        return time.time() - start
    
    def benchmark_threshold_sweep(self, hierarchy, num_thresholds=100):
        start = time.time()
        for t in np.linspace(0, 1, num_thresholds):
            partition = hierarchy.at_threshold(t)
        return time.time() - start
    
    def benchmark_comparison(self, col_a, col_b, threshold_a, threshold_b):
        start = time.time()
        metrics = self.frame.compare((col_a, threshold_a), (col_b, threshold_b))
        return time.time() - start
    
    def benchmark_entity_hashing(self, collection, threshold):
        start = time.time()
        hashes = self.frame.get_partition(collection, threshold, map_func='hash:blake3')
        return time.time() - start
    
    def run_full_benchmark(self):
        results = {
            'hierarchy_build': self.benchmark_hierarchy_build(),
            'threshold_sweep': self.benchmark_threshold_sweep(),
            'comparison': self.benchmark_comparison(),
            'entity_hashing': self.benchmark_entity_hashing(),
            'memory_mb': self.measure_memory_usage()
        }
        return results
```

## Migration guide

### From pandas DataFrame resolution

```python
# Before: Manual threshold management
def resolve_entities(df, threshold):
    clusters = []
    for t in [0.7, 0.8, 0.9]:
        clusters.append(cluster_at_threshold(df, t))
    return clusters[threshold]

# After: EntityFrame with full hierarchy
frame = ef.EntityFrame()
frame.load_records("data", df)
hierarchy = ef.Hierarchy.from_edges(compute_edges(df), len(df))
frame.add_collection("resolved", hierarchy=hierarchy)

# Now can query any threshold instantly
partition_70 = frame.get_partition("resolved", 0.7)
partition_85 = frame.get_partition("resolved", 0.85)
partition_92 = frame.get_partition("resolved", 0.92)

# And compute hashes efficiently
entity_hashes = frame.get_partition("resolved", 0.85, map_func='hash:sha256')
```

### From multiple resolution attempts

```python
# Before: Separate scripts for each method
splink_results = run_splink(data, params1)
dedupe_results = run_dedupe(data, params2)
comparison = compare_results(splink_results, dedupe_results, truth)

# After: Unified EntityFrame
frame = ef.EntityFrame()
frame.load_records("data", data)
frame.add_collection("splink", splink_results)
frame.add_collection("dedupe", dedupe_results)
frame.add_collection("truth", truth)

# Compare everything systematically with proper thresholds
for threshold in np.arange(0.5, 1.0, 0.01):
    splink_metrics = frame.compare(("splink", threshold), "truth")
    dedupe_metrics = frame.compare(("dedupe", threshold * 0.9), "truth")  # Different calibration
    # ... analysis ...
```