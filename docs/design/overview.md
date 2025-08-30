# Starlings Design Documents Overview

## What is Starlings?

Starlings is a data structure for entity resolution that stores hierarchical merge events instead of fixed clusters. This design choice enables efficient threshold exploration, comprehensive metric computation, and lossless data transport between pipeline stages.

## The Core Insight: Merge Events, Not Partitions

Traditional entity resolution tools force you to choose a threshold and compute clusters at that point. To explore different thresholds, you recompute from scratch each time. Starlings takes a different approach:

```python
# Traditional: Fixed clusters at each threshold
clusters_at_85 = compute_clusters(edges, 0.85)  # Full computation
clusters_at_90 = compute_clusters(edges, 0.90)  # Full computation again

# Starlings: Hierarchy generates any threshold
hierarchy = sl.Collection.from_edges(edges)
partition_at_85 = hierarchy.at(0.85)  # O(m) first time, O(1) cached
partition_at_90 = hierarchy.at(0.90)  # O(1) from cache
partition_at_8739 = hierarchy.at(0.8739)  # Any threshold, instantly
```

By storing the merge events—the moments when entities combine as thresholds change—we can generate any partition on demand. This isn't possible with standard dataframe libraries because they lack this hierarchical concept.

## Key Capabilities

### Analyze: Efficient Threshold Exploration
- Query any threshold without recomputation
- Update metrics incrementally (O(k) where k = affected entities)
- Compare multiple resolution methods systematically
- Sweep thousands of thresholds in seconds, not hours

### Represent: Complete Resolution Space
- Store infinite thresholds in ~50-250MB (vs ~10GB for explicit storage)
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

# Load data and add resolution attempts
ef = sl.from_records("customers", df)
ef.add_collection("splink", edges=splink_edges)
ef.add_collection("dedupe", edges=dedupe_edges)
ef.add_collection("truth", edges=truth_edges)

# Analyze with composable expressions
results = ef.analyse(
    sl.col("splink").sweep(0.5, 0.95, 0.01),
    sl.col("truth").at(1.0),
    metrics=[sl.f1, sl.precision, sl.recall]
)

# Direct access for simple operations
partition = ef["splink"].at(0.85)
entities = partition.to_list()
```

## Technical Implementation

### Memory Architecture
Collections use contextual ownership—they own their data when standalone, but share efficiently when combined in frames. This provides the simplicity of independent objects with the efficiency of shared storage.

### Performance Characteristics
- **Construction**: O(m log m) where m = number of edges
- **Threshold query**: O(m) first time, O(1) when cached
- **Metric updates**: O(k) incremental between adjacent thresholds
- **Memory usage**: ~50-250MB for 1M edges

### Language Stack
- **Rust core**: For performance-critical merge event processing
- **Python interface**: For familiar data science workflows
- **Arrow integration**: For cross-language data transport

## Document Structure

### [Document A: Core Design & Mathematics](foundations.md)
Establishes the mathematical foundations, user requirements, and theoretical guarantees. Defines the multi-collection model and proves the efficiency of incremental computation.

### [Document B: Technical Implementation](implementation.md)
Details the Rust implementation, including the contextual ownership architecture, parallel processing strategies, and Python bindings via PyO3.

### [Document C: Reference Architecture](reference.md)
Complete API specification, integration examples, and performance benchmarks. Provides migration guides and practical usage patterns.

## Why This Matters

Entity resolution practitioners currently work with incomplete information, forced to make threshold decisions before understanding their impact. Starlings provides the missing abstraction: a data structure that preserves the complete resolution space efficiently.

This isn't about revolutionizing the field—it's about providing a better tool for a specific problem. If you need to systematically explore thresholds, compare resolution methods, or preserve resolution information across pipeline stages, Starlings offers a purpose-built solution that traditional dataframe libraries cannot provide.
