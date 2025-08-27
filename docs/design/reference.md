# EntityFrame Design Document C: Reference Architecture

## Complete API Reference

### Python API

#### Core EntityFrame Operations

```python
class EntityFrame:
    """
    Main container for multiple entity resolution collections over shared records.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize EntityFrame.
        
        Args:
            config: Optional configuration dict with keys:
                - 'parallel_workers': Number of parallel threads (default: CPU count)
                - 'cache_size_mb': LRU cache size for metrics (default: 100)
                - 'arrow_batch_size': Batch size for Arrow operations (default: 10000)
                - 'string_intern_capacity': Initial interner capacity (default: 10000)
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
               collection_a: str,
               collection_b: str,
               threshold: float = None,
               metrics: List[str] = None) -> Dict[str, float]:
        """
        Compare two collections at a specific threshold.
        
        Args:
            collection_a: First collection name
            collection_b: Second collection name (often ground truth)
            threshold: Threshold for comparison (required if collection_a is hierarchical)
            metrics: List of metrics to compute (default: all)
                     Options: 'precision', 'recall', 'f1', 'ari', 'nmi', 
                              'v_measure', 'bcubed_precision', 'bcubed_recall'
        
        Returns:
            Dictionary of metric values
            
        Example:
            results = frame.compare("splink_v1", "ground_truth", threshold=0.85)
        """
    
    def find_optimal_threshold(self,
                              collection: str,
                              truth: str,
                              metric: str = 'f1',
                              search_range: Tuple[float, float] = (0.0, 1.0)) -> OptimalResult:
        """
        Find threshold that optimizes a metric.
        
        Args:
            collection: Collection to optimize
            truth: Ground truth collection
            metric: Metric to optimize
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
```

#### Hierarchy Operations

```python
class Hierarchy:
    """
    Hierarchical partition structure that generates entities at any threshold.
    """
    
    @classmethod
    def from_similarities(cls,
                         similarity_matrix: np.ndarray,
                         method: str = 'single',
                         record_ids: Optional[List] = None) -> 'Hierarchy':
        """
        Build hierarchy from similarity matrix.
        
        Args:
            similarity_matrix: Square matrix of pairwise similarities
            method: Linkage method ('single', 'average', 'complete')
            record_ids: Optional record identifiers
            
        Returns:
            New Hierarchy object
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
    
    def analyze_threshold_range(self,
                              start: float,
                              end: float,
                              step: float = 0.01) -> ThresholdAnalysis:
        """
        Analyze behavior across threshold range.
        
        Returns:
            ThresholdAnalysis with stability regions, critical points, etc.
            
        Example:
            analysis = hierarchy.analyze_threshold_range(0.5, 0.95)
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
```

#### Analysis Operations

```python
class ThresholdSweep:
    """
    Efficient threshold sweep analysis.
    """
    
    def __init__(self,
                frame: EntityFrame,
                collection: str,
                truth: str):
        """Initialize sweep analysis."""
    
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
        Analyze stability of resolution across thresholds.
        
        Returns:
            StabilityReport with stable regions, volatility scores, etc.
        """
```

### Rust API

#### Core Structures

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
pub struct EntityCollection {
    hierarchy: PartitionHierarchy,
    similarity_data: Option<SimilarityMatrix>,
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
    local_id: u32,     // ID within that source
}

/// Entity as set of interned references
pub struct Entity {
    members: RoaringBitmap,  // Indices into record storage
    cached_size: OnceCell<u32>,
}
```

#### Key Traits

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

## Data Structure Specifications

### Memory Layout

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

### Serialization Formats

#### Arrow Schema

```rust
/// Arrow schema for EntityFrame serialization
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
            Field::new("hierarchy", DataType::Binary, false),  // Serialized hierarchy
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
    ביט"num_sources": 5,
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

## Integration Examples

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

comparison = frame.compare("splink_output", "splink_0.90")
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
    frame.get_partition("truth", 1.0)
)

# Or use EntityFrame's built-in metrics
metrics = frame.compare("predicted", "truth", threshold=0.85)
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

# Export back to Matchbox format
hierarchy = frame.get_collection("matchbox")
matchbox_data = hierarchy.to_matchbox_format(threshold=0.9)
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

# Stream processing with Arrow
reader = pa.ipc.RecordBatchStreamReader("entity_stream.arrow")
for batch in reader:
    frame.add_records_batch(batch)
```

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Goal**: Core functionality with single collection

- Week 1-2: Rust core data structures
  - InternedRecordStorage implementation
  - Basic PartitionHierarchy with RoaringBitmaps
  - Simple threshold queries
  
- Week 3-4: Python bindings
  - PyO3 setup and basic API
  - Load records from pandas/Arrow
  - Create hierarchy from similarities
  
- Week 5-8: Essential algorithms
  - Single-linkage clustering
  - Basic metrics (precision, recall, F1)
  - Threshold extraction
  
- Week 9-12: Testing and optimization
  - Comprehensive test suite
  - Performance benchmarks
  - Memory profiling

**Deliverable**: Working single-collection EntityFrame with Python API

### Phase 2: Multi-Collection & Analytics (Months 4-6)
**Goal**: Multiple collections and comprehensive metrics

- Week 1-3: Multi-collection support
  - Collection management in EntityFrame
  - Cross-collection comparison
  - Shared record storage
  
- Week 4-6: Complete metrics suite
  - ARI, NMI, V-measure implementation
  - B-cubed metrics
  - Lazy computation framework
  
- Week 7-9: Incremental computation
  - Metric caching system
  - Incremental updates between thresholds
  - Computation statistics
  
- Week 10-12: Analysis tools
  - Threshold sweep optimization
  - Stability analysis
  - Entity lifetime tracking

**Deliverable**: Full multi-collection comparison capabilities

### Phase 3: Scale & Performance (Months 7-9)
**Goal**: Production-ready performance

- Week 1-3: Parallelization
  - Rayon integration
  - Parallel hierarchy construction
  - Concurrent collection comparison
  
- Week 4-6: SIMD optimizations
  - Vectorized metric computation
  - SIMD-accelerated set operations
  - Cache-optimized layouts
  
- Week 7-9: Sparse optimizations
  - Sparse contingency tables
  - Block-based processing
  - Memory-mapped storage
  
- Week 10-12: Streaming support
  - Incremental record addition
  - Online hierarchy updates
  - Bounded memory streaming

**Deliverable**: 10-100x performance improvement over baseline

### Phase 4: Integration & Polish (Months 10-12)
**Goal**: Ecosystem integration and production features

- Week 1-3: Tool integrations
  - Splink adapter
  - er-evaluation compatibility
  - Matchbox export/import
  
- Week 4-6: Advanced features
  - Distributed processing support
  - GPU acceleration experiments
  - Custom metric plugins
  
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

### Benchmark Suite

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
    
    def benchmark_comparison(self, col_a, col_b):
        start = time.time()
        metrics = self.frame.compare(col_a, col_b, threshold=0.85)
        return time.time() - start
    
    def run_full_benchmark(self):
        results = {
            'hierarchy_build': self.benchmark_hierarchy_build(),
            'threshold_sweep': self.benchmark_threshold_sweep(),
            'comparison': self.benchmark_comparison(),
            'memory_mb': self.measure_memory_usage()
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
hierarchy = ef.Hierarchy.from_similarities(compute_similarities(df))
frame.add_collection("resolved", hierarchy=hierarchy)

# Now can query any threshold instantly
partition_70 = frame.get_partition("resolved", 0.7)
partition_85 = frame.get_partition("resolved", 0.85)
partition_92 = frame.get_partition("resolved", 0.92)
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
frame.add_collection("splink", splink_results)
frame.add_collection("dedupe", dedupe_results)
frame.add_collection("truth", truth)

# Compare everything systematically
for threshold in np.arange(0.5, 1.0, 0.01):
    splink_metrics = frame.compare("splink", "truth", threshold)
    dedupe_metrics = frame.compare("dedupe", "truth", threshold)
    # ... analysis ...
```