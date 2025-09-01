# Starlings Engineering Build Plan

**Audience**: Engineers and coding agents  
**Instruction**: Each task specifies exact files to create and what to implement. Refer to design docs for specifications.

## Milestone 1: Minimal Viable Collection

### Task 1.1: Core Types & Data Structures

**Create files**: ✅ COMPLETED
- ✅ `src/rust/starlings/src/core/key.rs`
- ✅ `src/rust/starlings/src/core/record.rs`
- ✅ `src/rust/starlings/src/core/data_context.rs`
- ✅ `src/rust/starlings/src/core/mod.rs`

**Implement**: ✅ COMPLETED
- ✅ Key enum with U32, U64, String, Bytes variants
- ✅ InternedRecord struct with source_id, key, attributes
- ✅ DataContext with records Vec, source_interner, identity_map, source_index
- ✅ DataContext::ensure_record() method for deduplication
- ✅ Unit tests: key equality, record deduplication, index stability
- ✅ Benchmark: 10k unique record insertions using criterion

**Dependencies**: ✅ COMPLETED - roaring, fxhash, string-interner  
**Reference**: `algorithms.md` - Core data structure architecture

### Task 1.2: Hierarchy Construction

**Create files**: ✅ COMPLETED
- ✅ `src/rust/starlings/src/hierarchy/mod.rs`
- ✅ `src/rust/starlings/src/hierarchy/builder.rs`
- ✅ `src/rust/starlings/src/hierarchy/merge_event.rs`

**Implement**: ✅ COMPLETED
- ✅ MergeEvent struct with threshold and RoaringBitmap merging_groups
- ✅ PartitionHierarchy struct with merges Vec, partition_cache LruCache, context Arc<DataContext>
- ✅ PartitionHierarchy::from_edges() using union-find (disjoint_sets crate)
- ✅ Quantisation enforcement (1-6 decimal places)
- ✅ Fixed-point threshold_to_key() conversion (multiply by 1_000_000)
- ✅ Tests: 3-node graph, disconnected components, same-threshold edges (n-way merge)
- ✅ Benchmark: from_edges with 1k and 10k edges

**Dependencies**: ✅ COMPLETED - disjoint-sets, lru  
**Reference**: `algorithms.md` - Connected components algorithm

### Task 1.3: Partition Reconstruction

**Create files**:
- `rust/starlings-core/src/hierarchy/partition.rs`

**Implement**:
- [ ] PartitionLevel struct with entities as Vec<RoaringBitmap>
- [ ] PartitionHierarchy::at_threshold() with cache check
- [ ] PartitionHierarchy::reconstruct_at_threshold() internal method
- [ ] Include all records from DataContext (handles isolates automatically)
- [ ] Tests: threshold 0.0 (one entity), 1.0 (all singletons), 0.5 (intermediate)
- [ ] Test: verify isolates appear as singletons
- [ ] Benchmark: cached vs uncached at_threshold() calls

**Reference**: `algorithms.md` - Partition reconstruction from merge events

### Task 1.4: Python MVP

**Create files**:
- `rust/starlings-py/Cargo.toml`
- `rust/starlings-py/src/lib.rs`
- `python/starlings/__init__.py`
- `tests/python/test_collection_basic.py`

**Implement**:
- [ ] PyO3 project setup with maturin
- [ ] PyCollection struct wrapping PartitionHierarchy
- [ ] PyCollection::from_edges() classmethod
- [ ] PyCollection::at() method returning PyPartition
- [ ] Key conversion: Python int/str/bytes → Rust Key → u32 index
- [ ] Python test: edges → collection → partition → entities
- [ ] Benchmark: compare to networkx connected_components
- [ ] **Hook into existing CI/CD - ensure tests pass before PyPI release**

**Dependencies**: pyo3, maturin  
**Reference**: `engine.md` - Python interface via PyO3

## Milestone 2: Multi-Collection Frame & Analysis

### Task 2.1: EntityFrame

**Create files**:
- `rust/starlings-core/src/frame/mod.rs`
- `rust/starlings-core/src/frame/collection_map.rs`

**Implement**:
- [ ] EntityFrame struct with context Arc<DataContext>, collections HashMap
- [ ] EntityFrame::add_collection() with Arc::ptr_eq check for same context
- [ ] assimilate() method for different contexts with TranslationMap
- [ ] Collection view pattern with is_view flag
- [ ] Collection::copy() creating deep copy with new DataContext
- [ ] PyEntityFrame with __getitem__ for ef["name"] syntax
- [ ] Test: multiple collections share memory
- [ ] Test: view immutability

**Reference**: `algorithms.md` - Adding collections to frames section

### Task 2.2: Expression API

**Create files**:
- `python/starlings/expressions.py`
- `rust/starlings-py/src/expressions.rs`

**Implement**:
- [ ] Python col() function returning ColExpression
- [ ] ColExpression.at() and .sweep() methods
- [ ] Rust expression parsing distinguishing point vs sweep
- [ ] EntityFrame.analyse() taking variable expressions
- [ ] Return type always List[Dict[str, float]]
- [ ] Test: sl.col("a").at(0.8), sl.col("b").at(1.0) comparison
- [ ] Test: sl.col("a").sweep(0.5, 0.9, 0.1) output format

**Reference**: `interface.md` - Expression API section

### Task 2.3: Core Metrics

**Create files**:
- `rust/starlings-core/src/metrics/mod.rs`
- `rust/starlings-core/src/metrics/pairwise.rs`

**Implement**:
- [ ] compute_precision(), compute_recall(), compute_f1()
- [ ] Contingency table construction for two partitions
- [ ] Python sl.Metrics.eval.f1 etc. as marker classes
- [ ] Metric computation in analyse() method
- [ ] Test: known partitions with expected precision/recall
- [ ] Benchmark: metric computation for 1k entities

**Reference**: `principles.md` - Pairwise classification metrics

## Milestone 3: Performance & Production

### Task 3.1: Incremental Metrics

**Create files**:
- `rust/starlings-core/src/metrics/incremental.rs`
- `rust/starlings-core/src/metrics/state.rs`

**Implement**:
- [ ] IncrementalMetricState struct with last_threshold, contingency_table
- [ ] compute_delta() for metrics between adjacent thresholds
- [ ] O(k) update logic where k = affected entities
- [ ] Test: incremental result equals full recomputation
- [ ] Benchmark: 1000-threshold sweep time reduction

**Reference**: `principles.md` - Incremental metric computation

### Task 3.2: Parallelisation

**Modify files**: Throughout hierarchy and metrics modules

**Implement**:
- [ ] Replace sort_by with par_sort_unstable_by for edges
- [ ] Use rayon par_iter for independent entity operations
- [ ] Test: results identical with/without parallelisation
- [ ] Benchmark: measure speedup on 1, 2, 4, 8 threads

**Dependencies**: rayon  
**Reference**: `engine.md` - Parallel processing architecture

### Task 3.3: Arrow Serialisation

**Create files**:
- `rust/starlings-core/src/io/mod.rs`
- `rust/starlings-core/src/io/arrow.rs`

**Implement**:
- [ ] Arrow schema with dictionary encoding for sources/keys
- [ ] EntityFrame::to_arrow() method
- [ ] EntityFrame::from_arrow() method
- [ ] RoaringBitmap serialisation as nested lists
- [ ] Test: round-trip preservation of all data
- [ ] Benchmark: serialisation of 100k records

**Dependencies**: arrow  
**Reference**: `engine.md` - Arrow integration

### Task 3.4: Memory Management

**Create files**:
- `rust/starlings-core/src/frame/compaction.rs`

**Implement**:
- [ ] garbage_records RoaringBitmap in EntityFrame
- [ ] EntityFrame::drop() marking records as garbage
- [ ] auto_compact() when garbage_ratio > 0.5
- [ ] build_translation_map() and update all hierarchies
- [ ] Test: add 5 collections, drop 3, verify compaction
- [ ] Test: data integrity after compaction

**Reference**: `algorithms.md` - Automatic memory management

## Milestone 4: Complete Features

### Task 4.1: Advanced Metrics

**Create files**:
- `rust/starlings-core/src/metrics/cluster.rs`

**Implement**:
- [ ] ARI with adjusted_rand_index()
- [ ] NMI with normalized_mutual_information()
- [ ] V-measure with homogeneity and completeness
- [ ] B-cubed precision and recall
- [ ] Test against er-evaluation expected values
- [ ] Benchmark: metric computation scaling

**Reference**: `principles.md` - Complete mathematical operation space

### Task 4.2: Operations

**Create files**:
- `rust/starlings-core/src/operations/mod.rs`
- `rust/starlings-core/src/operations/hash.rs`
- `rust/starlings-core/src/operations/compute.rs`

**Implement**:
- [ ] SHA256, Blake3, MD5 hash operations
- [ ] Size, density compute operations
- [ ] Partition::map() method with EntityProcessor enum
- [ ] Python sl.Ops.hash.sha256 etc. marker classes
- [ ] Test: hash consistency
- [ ] Benchmark: parallel map execution

**Dependencies**: sha2, blake3, md-5  
**Reference**: `algorithms.md` - Dual processing for entity operations

### Task 4.3: Hierarchical Resolution

**Modify files**:
- `rust/starlings-core/src/hierarchy/builder.rs`

**Implement**:
- [ ] Accept Entity objects in records parameter of from_edges()
- [ ] entities_to_edges_internal() expanding entities to edges
- [ ] All pairs within entity get weight 1.0
- [ ] Collection::from_entities() constructor
- [ ] Test: two-stage resolution workflow

**Reference**: `interface.md` - Hierarchical resolution workflow

### Task 4.4: Polish

**Create files**:
- `python/starlings/_starlings.pyi`

**Implement**:
- [ ] Complete type stubs for all Python-visible classes
- [ ] Property tests with proptest (Rust) and hypothesis (Python)
- [ ] Memory leak detection with valgrind in CI
- [ ] Integration examples for Splink, er-evaluation
- [ ] Performance regression suite
- [ ] **Version bump and release notes for PyPI (existing CI/CD will handle publishing)**

**Reference**: `interface.md` - Integration sections

## Critical Constants

```rust
const CACHE_SIZE: usize = 10;  // LRU cache per hierarchy
const PRECISION_FACTOR: f64 = 1_000_000.0;  // Fixed-point conversion
const COMPACTION_THRESHOLD: f64 = 0.5;  // Garbage ratio trigger
const MAX_QUANTISE: u32 = 6;  // Maximum decimal places
```
