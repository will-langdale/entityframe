# Starlings Design Document C: Reference Architecture

## Complete API Reference

### Python API

The Starlings API design is inspired by [Polars](https://pola.rs/), adopting its expression-based approach for composable, efficient data operations. Like Polars, we provide a clear separation between data containers (EntityFrame/Collection) and operations (expressions), enabling users to build complex analyses from simple, chainable components. This design philosophy emphasizes being opinionated about the "right way" to do things while maintaining flexibility for advanced use cases.

```python
import starlings as sl  # Standard import convention used throughout
```

#### Core Types

```python
class Key:
    """
    A flexible key type that can hold integers, strings, or bytes.
    
    Automatically converts based on input type:
    - int → stored as u32 or u64 based on size
    - str → stored as string
    - bytes → stored as bytes
    
    Example:
        Key(123)         # u32
        Key("cust_abc")  # string
        Key(b"\\x00\\x01")  # bytes
    """

class Entity:
    """
    A resolved entity containing records from one or more sources.
    
    Note: Entity objects are created by Starlings when you access partition.entities.
    You don't construct them directly - they're an export format for viewing resolved entities.
    
    Properties:
        members: Set[Tuple[str, Key]] - Set of (source, key) pairs
        size: int - Number of records in this entity
        sources: Set[str] - Unique sources contributing to this entity
    
    Example:
        entity.members = {
            ("CRM", Key(123)),
            ("CRM", Key(456)),
            ("MailingList", Key("user_abc"))
        }
        entity.size = 3
        entity.sources = {"CRM", "MailingList"}
    """
```

#### Core EntityFrame Operations

```python
import starlings as sl

class EntityFrame:
    """
    Main container for multiple entity resolution collections over shared records.
    
    Mathematically: F = (R, {H₁, H₂, ..., Hₙ}, I) where collections ARE hierarchies.
    """
    
    @classmethod
    def from_records(cls, source_name: str, 
                    data: Union[pd.DataFrame, pa.Table, List[Dict]],
                    key_column: Optional[str] = None) -> 'EntityFrame':
        """
        Create EntityFrame from records.
        
        Args:
            source_name: Name of the source (e.g., "CRM", "MailingList")
            data: Records as DataFrame, Arrow Table, or list of dicts
            key_column: Column to use as key (default: auto-generate)
            
        Returns:
            New EntityFrame containing the records
            
        Example:
            ef = sl.from_records("CRM", df_customers, key_column="customer_id")
        """
    
    def add_collection(self,
                      name: str,
                      edges: Optional[List[Tuple[int, int, float]]] = None,
                      collection: Optional['Collection'] = None) -> None:
        """
        Add an entity resolution collection to the frame.
        
        Collections are hierarchies built from edges or can be added directly.
        Fixed clusterings should be provided as edges with weight 1.0.
        
        Args:
            name: Unique name for this collection
            edges: List of (record_i, record_j, similarity) tuples
            collection: Existing Collection object
            
        Note:
            Internally creates a hierarchy from the provided data.
            
        Example:
            # From probabilistic model
            ef.add_collection("splink_v1", edges=splink_edges)
            
            # From existing collection
            ef.add_collection("dedupe_v2", collection=dedupe_collection)
            
            # From fixed clustering (use weight 1.0)
            fixed_edges = [(i, j, 1.0) for cluster in clusters for i, j in pairs(cluster)]
            ef.add_collection("ground_truth", edges=fixed_edges)
        """
    
    def __getitem__(self, name: str) -> 'Collection':
        """
        Get collection by name for direct operations.
        
        Returns a view of the collection that shares the frame's DataContext.
        
        Example:
            partition = ef["splink"].at(0.85)
            sweep_df = ef["splink"].sweep(0.5, 0.95)
        """
    
    def analyse(self, *expressions, metrics: Optional[List] = None) -> Union[Dict, pd.DataFrame]:
        """
        Universal analysis method using expressions.
        
        This is the primary method for cross-collection comparisons and analysis.
        Inspired by polars' expression API for composability.
        
        Args:
            *expressions: One or more sl.col() expressions
            metrics: List of metrics to compute. Defaults to eval metrics for comparisons
                     (f1, precision, recall, ari, nmi) or stats metrics for single
                     collections (entity_count, entropy).
                     Available metrics: sl.Metrics.eval.* for comparisons,
                                      sl.Metrics.stats.* for single collections
        
        Returns:
            Dict for point comparisons, DataFrame for sweeps
            
        Example:
            # Compare two collections at specific thresholds
            results = ef.analyse(
                sl.col("splink").at(0.85),
                sl.col("ground_truth").at(1.0),
                metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
            )
            
            # Sweep analysis
            sweep_df = ef.analyse(
                sl.col("splink").sweep(0.5, 0.95, 0.01),
                sl.col("ground_truth").at(1.0)
            )
            
            # Single collection analysis (uses sl.report by default)
            stats = ef.analyse(
                sl.col("splink").at(0.85)
            )  # Returns entity_count, entropy, and other single-collection metrics
        """
    
    # American spelling alias
    analyze = analyse
    
    def drop(self, *names: str) -> None:
        """
        Remove collections from the frame.
        
        Follows pandas/polars convention. Triggers automatic compaction
        when garbage exceeds 50% of total records.
        
        Args:
            *names: Names of collections to drop
            
        Example:
            ef.drop("splink_v1", "splink_v2")
        """
    
    def to_arrow(self) -> pa.Table:
        """
        Export EntityFrame to Arrow format.
        
        Uses dictionary encoding for efficient storage of repeated strings.
        
        Returns:
            Arrow Table with complete frame data
        """
    
    @classmethod
    def from_arrow(cls, table: pa.Table) -> 'EntityFrame':
        """
        Load EntityFrame from Arrow format.
        
        Args:
            table: Arrow Table created by to_arrow()
            
        Returns:
            Reconstructed EntityFrame
        """
```

#### Collection Operations

```python
class Collection:
    """
    Hierarchical partition structure that generates entities at any threshold.
    
    Collections exist in two states:
    - Standalone: Owns its DataContext exclusively (created via from_* methods)
    - View: Shares DataContext with parent EntityFrame (created via ef["name"])
    
    When a view is modified or copied, it automatically becomes standalone
    through Copy-on-Write, creating its own DataContext.
    """
    
    @property
    def is_view(self) -> bool:
        """Whether this collection is a view sharing data with a frame."""
    
    @classmethod
    def from_edges(cls,
                  edges: List[Tuple[Any, Any, float]],
                  quantize: Optional[int] = None) -> 'Collection':
        """
        Build collection from weighted edges.
        
        Args:
            edges: List of (record_i, record_j, similarity) tuples
                  Records can be any hashable type (int, str, bytes)
                  Raw values are automatically wrapped in Key objects
            quantize: Optional decimal places to round thresholds to
                     (e.g., quantize=4 rounds to 0.0001 precision)
            
        Returns:
            New Collection with hierarchy of merge events
            
        Note:
            Complexity: O(m log m) where m = len(edges).
            
        Example:
            edges = [
                ("cust_123", "cust_456", 0.95),
                (123, 456, 0.85),
                (b"hash1", b"hash2", 0.75)
            ]
            collection = sl.Collection.from_edges(edges, quantize=4)
        """
    
    @classmethod
    def from_entities(cls,
                     entities: List[Set[Any]],
                     threshold: float = 1.0) -> 'Collection':
        """
        Build collection from pre-resolved entities.
        
        Converts fixed entities to a hierarchy at specified threshold.
        
        Args:
            entities: List of entity sets
            threshold: Threshold at which these entities exist
            
        Returns:
            New Collection
            
        Example:
            entities = [
                {"cust_123", "cust_456"},
                {"user_xyz", "user_abc"}
            ]
            collection = sl.Collection.from_entities(entities)
        """
    
    @classmethod
    def from_merge_events(cls,
                         merges: List[Dict],
                         num_records: int) -> 'Collection':
        """
        Build collection from merge events directly.
        
        For resuming work or loading from database.
        
        Args:
            merges: List of merge event dictionaries
            num_records: Total number of records
            
        Returns:
            New Collection
            
        Example:
            merges = [
                {"threshold": 0.9, "merging": [0, 1], "result": 2},
                {"threshold": 0.8, "merging": [2, 3], "result": 4}
            ]
            collection = sl.Collection.from_merge_events(merges, 100)
        """
    
    def at(self, threshold: float) -> 'Partition':
        """
        Get partition at specific threshold.
        
        Note:
            First call at threshold: O(m) reconstruction
            Subsequent calls: O(1) from cache
        
        Returns:
            Partition object with entities at that threshold
            
        Example:
            partition = collection.at(0.85)
            print(f"Number of entities: {len(partition.entities)}")
        """
    
    def sweep(self, start: float = 0.0, 
             stop: float = 1.0, 
             step: float = 0.01) -> List[Dict[str, Any]]:
        """
        Analyse collection across threshold range.
        
        Note:
            Uses incremental O(k) updates between thresholds.
        
        Returns:
            List of dictionaries, one per threshold
            
        Example:
            analysis = collection.sweep(0.5, 0.95)
            # Returns [{"threshold": 0.5, "entity_count": 100}, ...]
        """
    
    def copy(self) -> 'Collection':
        """
        Create an owned copy of this collection.
        
        If collection is a view from an EntityFrame, triggers Copy-on-Write
        to create a deep copy with its own DataContext. This enables safe
        mutation without affecting the parent frame, at the cost of duplicating
        the underlying data.
        
        Returns:
            New Collection with independent data
            
        Example:
            view = ef["splink"]  # Lightweight view
            owned = view.copy()  # Deep copy with own data
        """
```

#### Expression API

```python
# Expression builders for analyse() method
class col:
    """
    Create a collection expression for analysis.
    
    Inspired by polars.col() for consistent API design.
    """
    
    def __init__(self, name: str):
        """
        Reference a collection by name.
        
        Example:
            sl.col("splink")
        """
    
    def at(self, threshold: float) -> 'Expression':
        """
        Specify threshold for this collection.
        
        Example:
            sl.col("splink").at(0.85)
        """
    
    def sweep(self, start: float, stop: float, step: float = 0.01) -> 'Expression':
        """
        Specify threshold range for sweeping.
        
        Example:
            sl.col("splink").sweep(0.5, 0.95, 0.01)
        """

# Pre-defined metrics as module constants (injectable functions)
# Accessed via sl.Metrics.eval.* and sl.Metrics.stats.*
class Metrics:
    class eval:
        f1 = F1Metric()
        precision = PrecisionMetric()
        recall = RecallMetric()
        ari = ARIMetric()
        nmi = NMIMetric()
        v_measure = VMeasureMetric()
        bcubed_precision = BCubedPrecisionMetric()
        bcubed_recall = BCubedRecallMetric()
    
    class stats:
        entropy = EntropyMetric()
        entity_count = EntityCountMetric()
```

#### Partition Operations

```python
class Partition:
    """
    A partition of records into entities at a specific threshold.
    """
    
    @property
    def entities(self) -> Set[Entity]:
        """Set of entities in this partition (unordered)."""
    
    @property
    def num_entities(self) -> int:
        """Number of entities in this partition."""
    
    def map(self, func: Union[Callable, 'Operation']) -> Dict[int, Any]:
        """
        Apply function to each entity.
        
        Args:
            func: Either a built-in operation or callable
                  Built-ins: sl.Ops.hash.sha256, sl.Ops.compute.size, etc.
                  
        Returns:
            Dictionary mapping entity ID to result
            
        Example:
            # Built-in hash function (runs in parallel Rust)
            hashes = partition.map(sl.Ops.hash.sha256)
            
            # Custom Python function
            custom = partition.map(lambda e: len(e.members) * 2)
        """
    
    def to_list(self) -> List[Set[Tuple[str, Key]]]:
        """
        Export entities as list of sets.
        
        Returns:
            List where each element is a set of (source, key) tuples
            
        Example:
            entities = partition.to_list()
            # [
            #   {("CRM", Key(123)), ("CRM", Key(456))},
            #   {("MailingList", Key(789))}
            # ]
        """
```

### Built-in Operations

```python
# Operations for partition.map() (parallel execution in Rust)
# Accessed via sl.Ops.hash.* and sl.Ops.compute.*
class Ops:
    class hash:
        sha256 = SHA256Op()
        sha512 = SHA512Op()
        md5 = MD5Op()
        blake3 = Blake3Op()
    
    class compute:
        size = SizeOp()      # Number of records in entity
        density = DensityOp() # Internal connectivity
        fingerprint = FingerprintOp() # MinHash or similar
```

### Error Handling

```python
# Exception hierarchy
class StarlingsError(Exception):
    """Base exception for all Starlings errors"""

class InvalidThresholdError(StarlingsError):
    """Threshold outside [0, 1] range"""

# ... other specific error types
```

### Data Shape Specifications

#### Input Formats for from_records()

```python
# DataFrame input
df = pd.DataFrame({
    'customer_id': [1, 2, 3],      # Will be used as key if specified
    'name': ['Alice', 'Bob', 'Charlie'],
    'email': ['alice@ex.com', 'bob@ex.com', 'charlie@ex.com']
})
ef = sl.from_records("CRM", df, key_column='customer_id')

# List of dicts input
records = [
    {'id': 1, 'name': 'Alice', 'email': 'alice@ex.com'},
    {'id': 2, 'name': 'Bob', 'email': 'bob@ex.com'}
]
ef = sl.from_records("MailingList", records, key_column='id')

# Arrow Table input
table = pa.Table.from_pandas(df)
ef = sl.from_records("OrderSystem", table, key_column='customer_id')
```

#### Input Formats for Collections

```python
# From edges (record pairs with similarities)
edges = [
    (0, 1, 0.95),  # Record 0 and 1 with 0.95 similarity
    (1, 2, 0.85),  # Record 1 and 2 with 0.85 similarity
    (0, 3, 0.75)   # Record 0 and 3 with 0.75 similarity
]
collection = sl.Collection.from_edges(edges)

# From fixed entities
entities = [
    {0, 1, 4},  # First entity contains records 0, 1, 4
    {2, 3},     # Second entity contains records 2, 3
    {5, 6, 7, 8}  # Third entity
]
collection = sl.Collection.from_entities(entities)

# From merge events (for persistence/resuming)
merges = [
    {"threshold": 0.9, "merging": [0, 1], "result": 2},
    {"threshold": 0.8, "merging": [2, 3], "result": 4}
]
collection = sl.Collection.from_merge_events(merges, num_records=10)
```

## Arrow Schema

### Serialisation Format

```python
# EntityFrame Arrow schema with optimal dictionary encoding
schema = pa.schema([
    # Parallel arrays for maximum deduplication efficiency
    ("record_sources", pa.dictionary(pa.int16(), pa.string())),  # Global source dictionary
    ("record_keys", pa.dictionary(pa.int32(), pa.dense_union([   # Dictionary of unions!
        pa.field("u32", pa.uint32()),
        pa.field("u64", pa.uint64()),
        pa.field("string", pa.string()),  # No nested dictionary needed
        pa.field("bytes", pa.binary())
    ]))),
    ("record_indices", pa.uint32()),  # Simple array of indices
    
    # Collections with merge events
    ("collections", pa.list_(pa.struct([
        ("name", pa.dictionary(pa.int8(), pa.string())),
        ("merge_events", pa.list_(pa.struct([
            ("threshold", pa.float64()),
            ("merging_components", pa.list_(pa.uint32())),
            ("result_component", pa.uint32()),
            ("affected_records", pa.list_(pa.uint32()))  # Expanded from RoaringBitmap
        ])))
    ]))),
    
    # Metadata
    ("version", pa.string()),
    ("created_at", pa.timestamp('ms'))
])
```

### Database Decomposition

The Arrow format can be decomposed into relational tables:

```sql
-- Records table
CREATE TABLE records (
    record_index INTEGER PRIMARY KEY,
    source VARCHAR(255),
    key VARCHAR(255),
    INDEX idx_source_key (source, key)
);

-- Collections table  
CREATE TABLE collections (
    collection_id INTEGER PRIMARY KEY,
    name VARCHAR(255) UNIQUE
);

-- Merge events table
CREATE TABLE merge_events (
    merge_id INTEGER PRIMARY KEY,
    collection_id INTEGER REFERENCES collections(collection_id),
    threshold DOUBLE,
    result_component INTEGER
);

-- Merge components junction table
CREATE TABLE merge_components (
    merge_id INTEGER REFERENCES merge_events(merge_id),
    component_id INTEGER,
    PRIMARY KEY (merge_id, component_id)
);

-- Affected records junction table
CREATE TABLE merge_affected_records (
    merge_id INTEGER REFERENCES merge_events(merge_id),
    record_index INTEGER REFERENCES records(record_index),
    PRIMARY KEY (merge_id, record_index)
);
```

## Integration Examples

### Integration with Splink

```python
import splink
import starlings as sl
import polars as pl

# Run Splink
linker = splink.Linker(df, settings)
linker.predict()

# Get edges from Splink (not clusters)
edges = linker.get_scored_comparisons()
edges = [(e['id_l'], e['id_r'], e['match_probability']) for e in edges]

# Import into Starlings
ef = sl.from_records("source", df)
ef.add_collection("splink_output", edges=edges)

# Find optimal threshold using polars-inspired API
sweep_results = ef.analyse(
    sl.col("splink_output").sweep(0.5, 0.95, 0.01),
    sl.col("ground_truth").at(1.0),
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns list of dicts: [{"threshold": 0.5, "f1": 0.72, ...}, ...]

# Convert to polars for analysis
df_results = pl.from_dicts(sweep_results)
optimal_row = df_results.filter(pl.col("f1") == pl.col("f1").max()).row(0, named=True)
print(f"Optimal threshold: {optimal_row['threshold']:.2f} (F1={optimal_row['f1']:.3f})")
```

### Integration with er-evaluation

```python
import er_evaluation as er_eval
import starlings as sl

# Load data into Starlings
ef = sl.from_records("dataset", records)

# Add collections
ef.add_collection("predicted", edges=predicted_edges)
ef.add_collection("truth", edges=true_edges)

# Use Starlings's efficient sweep
sweep_results = ef.analyse(
    sl.col("predicted").sweep(0.5, 0.95),
    sl.col("truth").at(1.0)
)

# Or extract partition for er-evaluation  
partition = ef["predicted"].at(0.85)
clusters = partition.to_list()
metrics = er_eval.evaluate(clusters, true_clusters)

# Also supports American spelling
comparison = ef.analyze(
    sl.col("predicted").at(0.85),
    sl.col("truth").at(1.0)
)
```

### Integration with Matchbox

```python
import starlings as sl
import polars as pl
from matchbox import MatchboxClient

# Get Matchbox results
client = MatchboxClient()
matches = client.get_matches(dataset_id)

# Convert to edges for Starlings
edges = [(m['record_a'], m['record_b'], m['score']) for m in matches]

# Create EntityFrame
ef = sl.from_records("dataset", records)
ef.add_collection("matchbox", edges=edges)

# Find optimal threshold
sweep_results = ef.analyse(
    sl.col("matchbox").sweep(0.7, 0.95),
    sl.col("truth").at(1.0),
    metrics=[sl.Metrics.eval.f1]
)
# Convert to polars and find optimal
df_results = pl.from_dicts(sweep_results)
optimal_threshold = df_results.filter(
    pl.col("f1") == pl.col("f1").max()
).select("threshold")[0, 0]

# Export with hashes at optimal threshold
partition = ef["matchbox"].at(optimal_threshold)
hashes = partition.map(sl.Ops.hash.sha256)

matchbox_data = {
    'entities': partition.to_list(),
    'hashes': hashes,
    'threshold': optimal_threshold
}
client.upload_results(matchbox_data)
```

### Apache Arrow Integration

```python
import pyarrow as pa
import pyarrow.parquet as pq
import starlings as sl

# Save to Arrow/Parquet
ef = sl.from_records("source", data)
# ... add collections ...

arrow_table = ef.to_arrow()
pq.write_table(arrow_table, "starlings_frame.parquet")

# Load from Arrow/Parquet
table = pq.read_table("starlings_frame.parquet")
ef = sl.EntityFrame.from_arrow(table)

# The Arrow format uses optimal dictionary encoding for efficiency
# Sources and keys are automatically deduplicated at the global level
```

## Implementation Roadmap

### Phase 1: Foundation (Months 1-3)
**Goal**: Core functionality with single collection

- Week 1-2: Rust core data structures
  - Merge event storage implementation
  - Basic hierarchy with reconstruction
  - DataContext with append-only guarantee
  
- Week 3-4: Python bindings
  - PyO3 setup with starlings package
  - Basic from_records and Collection.from_edges
  - Collection.at() for partition access
  
- Week 5-8: Essential algorithms
  - Connected components with merge event extraction
  - Basic metrics (precision, recall, F1)
  - Partition reconstruction from merges
  
- Week 9-12: Testing and optimisation
  - Comprehensive test suite
  - Performance benchmarks
  - Memory profiling

**Deliverable**: Working single-collection Starlings with basic Python API

### Phase 2: Multi-collection & Analytics (Months 4-6)
**Goal**: Multiple collections and comprehensive metrics

- Week 1-3: Multi-collection support
  - EntityFrame with multiple collections
  - DataContext sharing via Arc
  - ef.analyse() with expressions
  
- Week 4-6: Complete metrics suite
  - ARI, NMI, V-measure implementation
  - B-cubed metrics (semi-incremental)
  - Injectable metric functions
  
- Week 7-9: Incremental computation
  - Metric state tracking between thresholds
  - O(k) updates for adjacent thresholds
  - Computation statistics
  
- Week 10-12: Analysis tools
  - Threshold sweep optimisation
  - Collection.sweep() implementation
  - Entity lifetime tracking

**Deliverable**: Full multi-collection comparison capabilities

### Phase 3: Batch Performance Optimisation (Months 7-9)
**Goal**: Production-ready batch processing performance

- Week 1-3: Parallelisation
  - Rayon integration
  - Parallel merge event extraction
  - Concurrent collection comparison
  - partition.map() with parallel built-ins
  
- Week 4-6: SIMD optimisations
  - Vectorised metric computation
  - SIMD-accelerated set operations
  - Cache-optimised layouts
  - Built-in hash operations (SHA256, BLAKE3)
  
- Week 7-9: Memory optimisation
  - Automatic compaction on ef.drop()
  - Collection.copy() for view detachment
  - Memory-mapped storage for large datasets
  - Optional quantisation support
  
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
  - Collection.from_entities() and from_merge_events()
  - Advanced Arrow serialisation
  
- Week 7-9: Production tooling
  - Monitoring and metrics
  - Error handling and recovery
  - Configuration management
  
- Week 10-12: Documentation & release
  - Complete API documentation  
  - Tutorial notebooks
  - Performance guides
  - v1.0 release preparation

**Deliverable**: Production-ready Starlings 1.0

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
import starlings as sl
import time
import memory_profiler

class StarlingsBenchmark:
    def __init__(self, num_records, num_edges):
        self.num_records = num_records
        self.num_edges = num_edges
        
    def benchmark_hierarchy_build(self):
        edges = self.generate_edges()
        start = time.time()
        collection = sl.Collection.from_edges(edges, self.num_records)
        return time.time() - start
    
    def benchmark_partition_reconstruction(self, collection):
        # Clear cache to force reconstruction
        collection.clear_cache()
        start = time.time()
        partition = collection.at(0.85)
        return time.time() - start
    
    def benchmark_cached_access(self, collection):
        # Warm cache
        _ = collection.at(0.85)
        # Measure cached access
        start = time.time()
        partition = collection.at(0.85)
        return time.time() - start
    
    def benchmark_threshold_sweep(self, ef):
        start = time.time()
        results = ef.analyse(
            sl.col("test").sweep(0, 1, 0.001),
            sl.col("truth").at(1.0)
        )
        return time.time() - start
    
    def benchmark_entity_hashing(self, collection):
        partition = collection.at(0.85)
        start = time.time()
        hashes = partition.map(sl.Hash.blake3)
        return time.time() - start
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

# After: Starlings with full hierarchy
import starlings as sl

ef = sl.from_records("data", df)
edges = compute_edges(df)  # Your existing edge computation
ef.add_collection("resolved", edges=edges)

# Now can query any threshold instantly (after first reconstruction)
partition = ef["resolved"].at(0.7)  # O(m) first time, O(1) after
partition = ef["resolved"].at(0.85)  # O(1) if cached

# Efficient sweep with incremental updates
sweep_results = ef.analyse(
    sl.col("resolved").sweep(0.5, 0.95),
    sl.col("truth").at(1.0)
)  # Returns List[Dict], O(k) between thresholds

# Can convert to polars if needed
import polars as pl
df = pl.from_dicts(sweep_results)
```

### From Multiple Resolution Attempts

```python
# Before: Separate scripts for each method
splink_results = run_splink(data, params1)
dedupe_results = run_dedupe(data, params2)
comparison = compare_results(splink_results, dedupe_results, truth)

# After: Unified Starlings
import starlings as sl

ef = sl.from_records("data", data)

# Add as edges, not clusters
ef.add_collection("splink", edges=splink_edges)
ef.add_collection("dedupe", edges=dedupe_edges)
ef.add_collection("truth", edges=truth_edges)

# Compare systematically using expressions
comparison = ef.analyse(
    sl.col("splink").at(0.85),
    sl.col("dedupe").at(0.76),
    sl.col("truth").at(1.0),
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)

# Find optimal thresholds efficiently
splink_sweep = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95),
    sl.col("truth").at(1.0)
)
optimal = splink_sweep.loc[splink_sweep['f1'].idxmax()]
```

### Important Notes on Performance

1. **First partition access**: Reconstructs from merge events O(m)
2. **Subsequent accesses**: Use cache O(1)
3. **Threshold sweeps**: Use incremental updates O(k) between adjacent thresholds
4. **Memory usage**: ~25MB for 1M edges with DataContext architecture
5. **Sparse graphs assumed**: Starlings expects m << n² from blocking/LSH
