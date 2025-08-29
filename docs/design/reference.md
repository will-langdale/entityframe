# EntityFrame Design Document C: Reference Architecture

## Complete API Reference

### Python API

#### Core EntityFrame Operations

```python
class EntityFrame:
    """
    Main container for multiple entity resolution collections over shared records.
    
    Mathematically: F = (R, {H₁, H₂, ..., Hₙ}, I) where collections ARE hierarchies.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise EntityFrame.
        
        Args:
            config: Optional configuration dict with keys:
                - 'parallel_workers': Number of parallel threads (default: CPU count)
                - 'cache_partitions_per_collection': Number of partitions to cache per collection (default: 10)
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
                      edges: Optional[List[Tuple[int, int, float]]] = None,
                      similarities: Optional[np.ndarray] = None) -> str:
        """
        Add an entity resolution collection to the frame.
        
        Collections are hierarchies built from edges or similarities.
        Fixed clusterings should be provided as edges with weight 1.0.
        
        Args:
            name: Unique name for this collection
            edges: List of (record_i, record_j, similarity) tuples
            similarities: Square similarity matrix (converts to edges internally)
            
        Returns:
            Collection ID
            
        Note:
            Internally creates a Hierarchy object from the provided data.
            Similarities are converted to edges (non-zero entries only).
            
        Example:
            # From probabilistic model
            frame.add_collection("splink_v1", edges=splink_edges)
            
            # From similarity matrix
            frame.add_collection("matrix_based", similarities=sim_matrix)
            
            # From fixed clustering (use weight 1.0)
            fixed_edges = [(i, j, 1.0) for cluster in clusters for i, j in pairs(cluster)]
            frame.add_collection("ground_truth", edges=fixed_edges)
        """
    
    def compare(self,
               cut_a: Tuple[str, float],
               cut_b: Tuple[str, float],
               metrics: List[str] = None) -> Dict[str, float]:
        """
        Compare two collections at specific thresholds.
        
        Args:
            cut_a: (collection_name, threshold) tuple
            cut_b: (collection_name, threshold) tuple
            metrics: List of metrics to compute (default: all)
                     Options: 'precision', 'recall', 'f1', 'ari', 'nmi', 
                              'v_measure', 'bcubed_precision', 'bcubed_recall'
        
        Returns:
            Dictionary of metric values
            
        Note:
            Most metrics update incrementally O(k) where k = changed entities between thresholds.
            Specifically:
            - k for contingency table = number of entity pairs whose relationship changes
            - k for pairwise metrics = number of record pairs affected by merges
            - B-cubed metrics are semi-incremental O(k × avg_entity_size)
            First comparison may reconstruct partitions O(m) if not cached.
            
        Example:
            # Compare two hierarchical collections at their thresholds
            results = frame.compare(("splink_v1", 0.85), ("dedupe_v2", 0.76))
            
            # Compare against fixed ground truth (at threshold 1.0)
            results = frame.compare(("splink_v1", 0.85), ("ground_truth", 1.0))
        """
    
    def compare_sweep(self,
                     collection: str,
                     truth: str,
                     truth_threshold: float = 1.0,
                     start: float = 0.0,
                     end: float = 1.0,
                     step: float = 0.01) -> pd.DataFrame:
        """
        Sweep thresholds and compare against truth.
        
        Args:
            collection: Collection to sweep
            truth: Ground truth collection
            truth_threshold: Threshold for truth collection (default: 1.0)
            start: Starting threshold
            end: Ending threshold
            step: Threshold increment
            
        Returns:
            DataFrame with columns: threshold, precision, recall, f1, ari, nmi, etc.
            
        Note:
            Uses incremental O(k) updates between adjacent thresholds.
            Initial partition reconstruction is O(m) if not cached.
            
        Example:
            results = frame.compare_sweep("splink_v1", "ground_truth")
            results.plot(x='threshold', y=['precision', 'recall'])
        """
    
    def find_optimal_threshold(self,
                              collection: str,
                              truth: str,
                              truth_threshold: float = 1.0,
                              metric: str = 'f1',
                              search_range: Tuple[float, float] = (0.0, 1.0)) -> OptimalResult:
        """
        Find threshold that optimises a metric.
        
        Args:
            collection: Collection to optimise
            truth: Ground truth collection
            truth_threshold: Threshold for truth collection
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
        
        Note: May trigger partition reconstruction O(m) if not cached.
        
        Returns:
            Entity object with members as (source, local_id) pairs
            
        Example:
            entity = frame.get_entity("splink_v1", 0.85, entity_id=42)
            for source, record_id in entity.members:
                print(f"{source}: {record_id}")
        """
    
    def get_partition(self,
                     collection: str,
                     threshold: float,
                     map_func: Optional[Callable[[Entity], Any]] = None) -> Union[Partition, Dict[EntityId, Any]]:
        """
        Get partition at threshold, optionally mapping a function over entities.
        
        Note: First access at a threshold reconstructs partition O(m).
              Subsequent accesses use cache O(1).
        
        Args:
            collection: Collection name
            threshold: Threshold value
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
                                entities: List[Set[Tuple[str, Any]]]) -> str:
        """
        Load pre-resolved entities as a collection.
        
        Converts fixed entities to a hierarchy at threshold 1.0.
        
        Args:
            name: Name for this collection
            entities: List of entity sets, each containing (source, id) pairs
            
        Returns:
            Collection ID
            
        Example:
            # Load entities resolved elsewhere
            frame.load_resolved_collection(
                "hospital_patients",
                patient_entities  # List of sets of (source, id) tuples
            )
            
            # Use like any other collection
            frame.compare(("hospital_patients", 1.0), ("other", 0.8))
        """
```

#### Hierarchy Operations

```python
class Hierarchy:
    """
    Hierarchical partition structure that generates entities at any threshold.
    Internally stored as merge events for space efficiency.
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
            Converts to edges internally, storing as merge events.
            Complexity: O(m log m) where m = number of non-zero similarities.
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
            New Hierarchy object storing merge events
            
        Note:
            Complexity: O(m log m) where m = len(edges).
            For sparse graphs from blocking, m << n².
            
        Example:
            # Edges with same weight merge simultaneously
            edges = [(0, 1, 0.9), (1, 2, 0.9), (2, 3, 0.9)]
            hierarchy = Hierarchy.from_edges(edges, 4)
            # At threshold 0.9: creates 4-way merge {0,1,2,3}
        """
    
    def at_threshold(self, threshold: float) -> Partition:
        """
        Get partition at specific threshold.
        
        Note:
            First call at threshold: O(m) reconstruction
            Subsequent calls: O(1) from cache
        
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
        
        Note:
            Uses incremental O(k) updates between thresholds.
        
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
    
    def memory_estimate(self) -> MemoryEstimate:
        """
        Estimate memory usage of the hierarchy.
        
        Returns:
            MemoryEstimate with merge event storage and cache usage
            
        Example:
            mem = hierarchy.memory_estimate()
            print(f"Merge events: {mem.merge_bytes / 1024 / 1024:.1f} MB")
            print(f"Cache: {mem.cache_bytes / 1024 / 1024:.1f} MB")
        """
```

#### Analysis Operations

```python
class ThresholdSweep:
    """
    Efficient threshold sweep analysis using incremental updates.
    """
    
    def __init__(self,
                frame: EntityFrame,
                collection: str,
                truth: str,
                truth_threshold: float = 1.0):
        """Initialise sweep analysis."""
    
    def compute_all_metrics(self,
                           start: float = 0.0,
                           end: float = 1.0,
                           step: float = 0.01) -> pd.DataFrame:
        """
        Compute all metrics across threshold range.
        
        Note:
            Uses O(k) incremental updates between adjacent thresholds.
            Initial reconstruction is O(m) if not cached.
        
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

### Data Shape Specifications

#### Input Formats for load_records()

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

#### Input Formats for add_collection()

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
frame.add_collection("edge_based", edges=edges)

# From fixed clusters (convert to edges with weight 1.0)
def clusters_to_edges(clusters):
    edges = []
    for cluster in clusters:
        members = list(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                edges.append((members[i], members[j], 1.0))
    return edges

clusters = [{0, 1, 4}, {2, 3}, {5, 6, 7, 8}]
edges = clusters_to_edges(clusters)
frame.add_collection("ground_truth", edges=edges)

# From pre-resolved entities with source attribution
entities = [
    {("CRM", 1), ("CRM", 9), ("MailingList", 17)},  # Cross-source entity
    {("OrderSystem", 42), ("OrderSystem", 43)}       # Single-source entity
]
frame.load_resolved_collection("pre_resolved", entities=entities)
```

### Built-in Operations for Entity Processing

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

### Core Structures

```rust
/// Main EntityFrame structure
pub struct EntityFrame {
    collections: HashMap<CollectionId, PartitionHierarchy>,
    records: InternedRecordStorage,
    interner: InternSystem,
    analyzer: FrameAnalyzer,
    config: FrameConfig,
}

/// Hierarchical partition structure using merge events
pub struct PartitionHierarchy {
    merges: Vec<MergeEvent>,
    partition_cache: LruCache<OrderedFloat<f64>, PartitionLevel>,
    metric_state: Option<IncrementalMetricState>,
    threshold_index: BTreeMap<OrderedFloat<f64>, usize>,
}

/// Single merge event in the hierarchy
pub struct MergeEvent {
    threshold: f64,
    merging_components: Vec<ComponentId>,
    result_component: ComponentId,
    affected_records: RoaringBitmap,
}

/// Cached partition at a threshold
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

/// Collection cut (always name + threshold)
pub struct CollectionCut {
    name: String,
    threshold: f64,
}
```

### Key Traits

```rust
/// Trait for hierarchical structures
pub trait Hierarchical {
    fn at_threshold(&mut self, threshold: f64) -> &PartitionLevel;
    fn threshold_range(&self) -> (f64, f64);
    fn num_merges(&self) -> usize;
    fn merge_events(&self) -> &[MergeEvent];
    fn reconstruct_at(&self, threshold: f64) -> PartitionLevel;
}

/// Trait for comparable partitions
pub trait Comparable {
    fn compare_with(&self, other: &Self) -> ComparisonMetrics;
    fn contingency_table(&self, other: &Self) -> ContingencyTable;
    fn incremental_compare(&self, other: &Self, prev_state: &MetricState) -> ComparisonMetrics;
}
```

## Data Structure Specifications

### Memory Layout

```rust
/// Memory-efficient entity storage using merge events
/// 
/// For 1M records, 1M edges:
/// - Full materialization: ~10GB (storing all partitions)
/// - Merge events: ~50-100MB (storing transitions with RoaringBitmaps)
/// - LRU cache (10 partitions): ~5MB
/// - Total: ~50-250MB (95-97% reduction vs full materialization)
struct MemoryLayout {
    // Level 1: Source interning
    source_names: StringInterner,      // ~1KB for 100 sources
    
    // Level 2: Record storage
    record_refs: Vec<InternedRef>,     // 6 bytes per record
    record_attrs: Option<Vec<Attrs>>,  // Optional attribute data
    
    // Level 3: Merge events
    merge_events: Vec<MergeEvent>,     // ~20 bytes per merge
    
    // Level 4: Cache
    partition_cache: LruCache<Threshold, PartitionLevel>,  // ~500KB per cached level
}
```

### Serialisation Formats

#### Arrow Schema

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
        
        // Collections (hierarchies)
        Field::new("collections", DataType::List(Box::new(DataType::Struct(vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("merge_events", DataType::Binary, false),  // Serialised merge events
        ]))), false),
    ]
}
```

#### JSON Export Format

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
      "num_merges": 87453,
      "threshold_range": [0.0, 1.0],
      "memory_mb": 1.7,
      "statistics": {
        "avg_merge_size": 2.3,
        "max_merge_size": 47,
        "stability_score": 0.73
      }
    }
  ]
}
```

## Integration Examples

### Integration with Splink

```python
import splink
import entityframe as ef

# Run Splink
linker = splink.Linker(df, settings)
linker.predict()

# Get edges from Splink (not clusters)
edges = linker.get_scored_comparisons()
edges = [(e['id_l'], e['id_r'], e['match_probability']) for e in edges]

# Import into EntityFrame
frame = ef.EntityFrame()
frame.load_records("source", df)
frame.add_collection("splink_output", edges=edges)

# Compare at different thresholds efficiently
for threshold in [0.80, 0.85, 0.90, 0.95]:
    metrics = frame.compare(
        ("splink_output", threshold),
        ("ground_truth", 1.0)
    )
    print(f"Threshold {threshold}: F1={metrics['f1']:.3f}")
```

### Integration with er-evaluation

```python
import er_evaluation as er_eval
import entityframe as ef

# Load data into EntityFrame
frame = ef.EntityFrame()
frame.load_records("dataset", records)

# Convert clusters to edges for EntityFrame
predicted_edges = clusters_to_edges(predicted_clusters)
true_edges = clusters_to_edges(true_clusters)

frame.add_collection("predicted", edges=predicted_edges)
frame.add_collection("truth", edges=true_edges)

# Use EntityFrame's efficient sweep
sweep_results = frame.compare_sweep("predicted", "truth")

# Or use er-evaluation metrics on extracted partitions
partition = frame.get_partition("predicted", 0.85)
metrics = er_eval.evaluate(partition, true_clusters)
```

### Integration with Matchbox

```python
import entityframe as ef
from matchbox import MatchboxClient

# Get Matchbox results
client = MatchboxClient()
matches = client.get_matches(dataset_id)

# Convert to edges for EntityFrame
edges = [(m['record_a'], m['record_b'], m['score']) for m in matches]

# Create EntityFrame
frame = ef.EntityFrame()
frame.add_collection("matchbox", edges=edges)

# Export with hashes at optimal threshold
optimal = frame.find_optimal_threshold("matchbox", "truth")
hashes = frame.get_partition("matchbox", optimal.threshold, map_func='hash:sha256')

matchbox_data = {
    'entities': frame.get_partition("matchbox", optimal.threshold).to_list(),
    'hashes': hashes,
    'threshold': optimal.threshold
}
client.upload_results(matchbox_data)
```

### Apache Arrow Integration

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

# Batch processing with Arrow
reader = pa.ipc.RecordBatchStreamReader("entity_stream.arrow")
for batch in reader:
    frame.add_records_batch(batch)
```

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Goal**: Core functionality with single collection

- Week 1-2: Rust core data structures
  - Merge event storage implementation
  - Basic PartitionHierarchy with reconstruction
  - LRU cache for partitions
  
- Week 3-4: Python bindings
  - PyO3 setup and basic API
  - Load records from pandas/Arrow
  - Create hierarchy from edges/similarities
  
- Week 5-8: Essential algorithms
  - Connected components with merge event extraction
  - Basic metrics (precision, recall, F1)
  - Partition reconstruction from merges
  
- Week 9-12: Testing and optimisation
  - Comprehensive test suite
  - Performance benchmarks
  - Memory profiling

**Deliverable**: Working single-collection EntityFrame with Python API

### Phase 2: Multi-collection & Analytics (Months 4-6)
**Goal**: Multiple collections and comprehensive metrics

- Week 1-3: Multi-collection support
  - Collection management in EntityFrame
  - Cross-collection comparison
  - Shared record storage
  
- Week 4-6: Complete metrics suite
  - ARI, NMI, V-measure implementation
  - B-cubed metrics (semi-incremental)
  - Lazy computation framework
  
- Week 7-9: Incremental computation
  - Metric state tracking between thresholds
  - O(k) updates for adjacent thresholds
  - Computation statistics
  
- Week 10-12: Analysis tools
  - Threshold sweep optimisation
  - Stability analysis
  - Entity lifetime tracking

**Deliverable**: Full multi-collection comparison capabilities

### Phase 3: Batch Performance Optimisation (Months 7-9)
**Goal**: Production-ready batch processing performance

- Week 1-3: Parallelisation
  - Rayon integration
  - Parallel merge event extraction
  - Concurrent collection comparison
  - Dual processing for entity operations
  
- Week 4-6: SIMD optimisations
  - Vectorised metric computation
  - SIMD-accelerated set operations
  - Cache-optimised layouts
  - Built-in hash operations (SHA256, BLAKE3)
  
- Week 7-9: Sparse optimisations
  - Sparse contingency tables
  - Block-based processing for disconnected components
  - Memory-mapped storage for large datasets
  - Quantisation support for memory reduction
  
- Week 10-12: Batch processing enhancements
  - Optimised cache strategies
  - Batch comparison operations
  - Performance profiling and tuning

**Deliverable**: 10-100x performance improvement for batch operations

### Phase 4: Integration & Polish (Months 10-12)
**Goal**: Ecosystem integration and production features

- Week 1-3: Tool integrations
  - Splink adapter
  - er-evaluation compatibility
  - Matchbox export/import with hashing
  
- Week 4-6: Advanced features
  - Custom metric plugins
  - Entity metadata attachment
  - Pre-resolved entity loading
  
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

## Performance Benchmarks

### Target Performance Metrics

```yaml
# Single machine performance targets
# Note: Assumes sparse graphs from blocking (m ~ n to n log n)

small_scale:  # 10K records, ~50K edges
  hierarchy_build: < 100ms
  cached_query: < 1ms
  uncached_query: < 10ms
  full_sweep_1000_thresholds: < 1s
  memory_usage: < 10MB

medium_scale:  # 1M records, ~5M edges
  hierarchy_build: < 10s
  cached_query: < 1ms
  uncached_query: < 200ms
  full_sweep_1000_thresholds: < 30s
  memory_usage: < 100MB

large_scale:  # 10M records, ~50M edges
  hierarchy_build: < 5min
  cached_query: < 10ms
  uncached_query: < 2s
  full_sweep_1000_thresholds: < 5min
  memory_usage: < 1GB

# Scaling characteristics
complexity:
  hierarchy_construction: O(m log m) where m = edges
  partition_reconstruction: O(m)
  cached_query: O(1)
  incremental_metric: O(k) where k = affected entities
  memory: O(m) for merge events + O(c × n) for c cached partitions
```

### Benchmark Suite

```python
# Performance testing harness
import entityframe as ef
import time
import memory_profiler

class EntityFrameBenchmark:
    def __init__(self, num_records, num_edges):
        self.num_records = num_records
        self.num_edges = num_edges
        self.frame = ef.EntityFrame()
        
    def benchmark_hierarchy_build(self):
        edges = self.generate_edges()
        start = time.time()
        hierarchy = ef.Hierarchy.from_edges(edges, self.num_records)
        return time.time() - start
    
    def benchmark_partition_reconstruction(self, hierarchy):
        # Clear cache to force reconstruction
        hierarchy.clear_cache()
        start = time.time()
        partition = hierarchy.at_threshold(0.85)
        return time.time() - start
    
    def benchmark_cached_access(self, hierarchy):
        # Warm cache
        _ = hierarchy.at_threshold(0.85)
        # Measure cached access
        start = time.time()
        partition = hierarchy.at_threshold(0.85)
        return time.time() - start
    
    def benchmark_threshold_sweep(self, hierarchy, num_thresholds=1000):
        start = time.time()
        for t in np.linspace(0, 1, num_thresholds):
            partition = hierarchy.at_threshold(t)
            # Incremental metrics would be computed here
        return time.time() - start
    
    def benchmark_entity_hashing(self, collection, threshold):
        start = time.time()
        hashes = self.frame.get_partition(collection, threshold, map_func='hash:blake3')
        return time.time() - start
    
    def measure_memory_usage(self):
        return self.frame.memory_estimate()
    
    def run_full_benchmark(self):
        results = {
            'hierarchy_build': self.benchmark_hierarchy_build(),
            'uncached_query': self.benchmark_partition_reconstruction(),
            'cached_query': self.benchmark_cached_access(),
            'threshold_sweep': self.benchmark_threshold_sweep(),
            'entity_hashing': self.benchmark_entity_hashing(),
            'memory_mb': self.measure_memory_usage() / 1024 / 1024
        }
        return results
```

## Migration Guide

### From Pandas DataFrame Resolution

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
edges = compute_edges(df)  # Your existing edge computation
frame.add_collection("resolved", edges=edges)

# Now can query any threshold instantly (after first reconstruction)
partition_70 = frame.get_partition("resolved", 0.7)  # O(m) first time, O(1) after
partition_85 = frame.get_partition("resolved", 0.85)  # O(1) if cached
partition_92 = frame.get_partition("resolved", 0.92)  # O(1) if cached

# Efficient sweep with incremental updates
results = frame.compare_sweep("resolved", "truth")  # O(k) between thresholds

# And compute hashes efficiently
entity_hashes = frame.get_partition("resolved", 0.85, map_func='hash:sha256')
```

### From Multiple Resolution Attempts

```python
# Before: Separate scripts for each method
splink_results = run_splink(data, params1)
dedupe_results = run_dedupe(data, params2)
comparison = compare_results(splink_results, dedupe_results, truth)

# After: Unified EntityFrame
frame = ef.EntityFrame()
frame.load_records("data", data)

# Add as edges, not clusters
frame.add_collection("splink", edges=splink_edges)
frame.add_collection("dedupe", edges=dedupe_edges)
frame.add_collection("truth", edges=truth_edges)

# Compare everything systematically
# Note: All collections now use (name, threshold) format
for threshold in np.arange(0.5, 1.0, 0.01):
    splink_metrics = frame.compare(("splink", threshold), ("truth", 1.0))
    dedupe_metrics = frame.compare(("dedupe", threshold * 0.9), ("truth", 1.0))
    # ... analysis ...

# Find optimal thresholds efficiently
splink_optimal = frame.find_optimal_threshold("splink", "truth")
dedupe_optimal = frame.find_optimal_threshold("dedupe", "truth")
```

### Important Notes on Performance

1. **First partition access**: Reconstructs from merge events O(m)
2. **Subsequent accesses**: Use cache O(1)
3. **Threshold sweeps**: Use incremental updates O(k) between adjacent thresholds
4. **Memory usage**: ~25MB for 1M edges vs ~10GB for full materialization
5. **Sparse graphs assumed**: EntityFrame expects m << n² from blocking/LSH
