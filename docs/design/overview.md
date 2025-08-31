# Starlings Design Documents Overview

## What is Starlings?

Starlings is a Python library for systematically exploring and comparing entity resolution results across different thresholds and methods. Instead of forcing threshold decisions at processing time, Starlings preserves the complete resolution space as a hierarchy of merge events, enabling instant threshold exploration, efficient metric computation, and lossless data transport between pipeline stages.

## The Core Insight: Merge Events, Not Partitions

Traditional entity resolution tools force you to choose a threshold and compute clusters at that point. To explore different thresholds, you recompute from scratch each time. Starlings takes a different approach:

```python
import starlings as sl

# Traditional: Fixed clusters at each threshold
clusters_at_85 = compute_clusters(edges, 0.85)  # Full computation
clusters_at_90 = compute_clusters(edges, 0.90)  # Full computation again

# Starlings: Hierarchy generates any threshold
collection = sl.Collection.from_edges(edges)
partition_at_85 = collection.at(0.85)  # O(m) first time, O(1) cached
partition_at_90 = collection.at(0.90)  # O(1) from cache
partition_at_8739 = collection.at(0.8739)  # Any threshold, instantly
```

By storing the merge events—the moments when entities combine as thresholds change—we can generate any partition on demand. This isn't possible with standard dataframe libraries because they lack this hierarchical concept.

## Key Capabilities

### Analyze: Efficient Threshold Exploration
- Query any threshold without recomputation
- Update metrics incrementally (O(k) where k = affected entities)
- Compare multiple resolution methods systematically
- Sweep thousands of thresholds in seconds, not hours

### Represent: Complete Resolution Space
- Store infinite thresholds in ~60-115MB (vs ~10GB for explicit storage)
- Support n-way merges naturally (when 5 entities merge simultaneously)
- Preserve all information for downstream decisions
- Handle both probabilistic scores and fixed clusters uniformly

### Transport: Seamless Integration
- Arrow format with dictionary encoding for efficient serialization
- Clean decomposition to relational database tables
- Native adapters for Splink, er-evaluation, Matchbox
- Preserve complete resolution history across pipeline stages

## API Design

The API borrows from Polars' expression-based approach, separating data containers from operations:

```python
import starlings as sl
import polars as pl

# Load data and add resolution attempts
ef = sl.from_records("customers", df)
ef.add_collection("splink", edges=splink_edges)
ef.add_collection("dedupe", edges=dedupe_edges)
ef.add_collection("truth", edges=truth_edges)

# Analyze with composable expressions
results = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95, 0.01),
    sl.col("truth").at(1.0),
    metrics=[sl.Metrics.eval.f1, sl.Metrics.eval.precision, sl.Metrics.eval.recall]
)
# Returns list of dicts: [{"threshold": 0.5, "f1": 0.72, ...}, ...]

# Convert to polars for analysis
df = pl.from_dicts(results)
optimal = df.filter(pl.col("f1") == pl.col("f1").max()).row(0, named=True)

# Direct access for simple operations
partition = ef["splink"].at(optimal["threshold"])
entities = partition.to_list()  # List of sets of (source, key) pairs

# Apply operations to entities
hashes = partition.map(sl.Ops.hash.sha256)  # Fast parallel hashing
sizes = partition.map(sl.Ops.compute.size)   # Entity sizes
```

## Technical Implementation

### Memory Architecture: Contextual Ownership
Collections use contextual ownership—they own their data when standalone, but share efficiently when combined in frames. This provides the simplicity of independent objects with the efficiency of shared storage. When collections are views from a frame and need modification, Copy-on-Write ensures safe mutations without affecting parent frames. This trade-off prioritises memory efficiency in the common read-heavy case while maintaining safety.

### Performance Characteristics
- **Construction**: O(m log m) where m = number of edges
- **Threshold query**: O(m) first time, O(1) when cached
- **Metric updates**: O(k) incremental between adjacent thresholds
- **Memory usage**: ~60-115MB for 1M edges
- **Assumptions**: Sparse graphs from blocking/LSH (m ~ n to n log n, not n²)

### Language Stack
- **Rust core**: For performance-critical merge event processing with parallel execution via Rayon
- **Python interface**: For familiar data science workflows via PyO3 with automatic type conversion
- **Arrow integration**: For cross-language data transport with optimal dictionary encoding at the global level

## Injectable Operations Pattern

Starlings uses a consistent pattern for extensible operations that separates concerns cleanly:

```python
# Metrics for comparing partitions (used in analyse)
sl.Metrics.eval.f1       # Evaluation against ground truth
sl.Metrics.eval.ari      # Adjusted Rand Index
sl.Metrics.eval.nmi      # Normalised Mutual Information
sl.Metrics.stats.entropy # Single partition statistics
sl.Metrics.stats.entity_count

# Operations for entities (used in partition.map)
sl.Ops.hash.sha256      # Hash entities for metadata attachment
sl.Ops.hash.blake3      # Alternative fast hashing
sl.Ops.compute.size     # Compute entity properties
sl.Ops.compute.density  # Internal connectivity metrics
```

These are marker types that route to optimised Rust implementations, providing both flexibility and performance. The separation makes it clear: metrics operate on partitions, operations operate on individual entities.

## Document Structure

### [Document A: Core Design & Mathematics](foundations.md)
Establishes the mathematical foundations, user requirements, and theoretical guarantees. Defines the multi-collection model as F = (R, {H₁, H₂, ..., Hₙ}, I) where collections ARE hierarchies. Proves the efficiency of incremental computation through additive metric properties and introduces the contextual ownership architecture with Copy-on-Write for safe mutations.

### [Document B: Technical Implementation](implementation.md)
Details the Rust implementation, including the contextual ownership architecture with Arc<DataContext> for efficient sharing, parallel processing strategies using Rayon, and Python bindings via PyO3. Shows how marker types route to optimised implementations, how the Python/Rust boundary handles automatic Key type conversion, and the union-find algorithm for natural n-way merge support.

### [Document C: Reference Architecture](reference.md)
Complete API specification with the polars-inspired expression system (sl.col().at(), sl.col().sweep()), integration examples for Splink/er-evaluation/Matchbox, and performance benchmarks. Provides migration guides showing the 10-100x speedup for threshold analysis, practical usage patterns, and demonstrates working with List[Dict] outputs using polars for analysis.

## Why This Matters

Entity resolution practitioners currently work with incomplete information, forced to make threshold decisions before understanding their impact. Consider a typical workflow:

1. **Today's Reality**: Run Splink, get edges, pick threshold 0.85 because "it worked last time", generate clusters, evaluate against truth, realise 0.85 was suboptimal, re-run with 0.90, repeat.

2. **With Starlings**: Run Splink, get edges, load into Starlings, sweep 1000 thresholds in seconds, visualise F1/precision/recall curves, identify optimal threshold with data-driven confidence, export entities.

The difference isn't just speed—it's about making informed decisions. When you can see how metrics evolve across the entire threshold space, you understand not just what the optimal threshold is, but why it's optimal and how sensitive your results are to that choice.

Starlings also enables workflows that were previously impractical:
- **Ensemble resolution**: Compare multiple algorithms at their respective optimal thresholds
- **Stability analysis**: Find threshold ranges where small changes don't dramatically alter results  
- **Progressive refinement**: Start with rough blocking, analyse, refine, without losing previous work
- **Collaborative threshold selection**: Share complete resolution spaces, let downstream users choose thresholds

This isn't about revolutionizing the field—it's about providing a better tool for a specific problem. If you need to systematically explore thresholds, compare resolution methods, or preserve resolution information across pipeline stages, Starlings offers a purpose-built solution that traditional dataframe libraries cannot provide.
