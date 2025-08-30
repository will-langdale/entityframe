# Starlings Design Document A: Core Design & Mathematics

## Executive Summary

Starlings represents a paradigm shift in entity resolution infrastructure. **Rather than forcing threshold decisions at processing time, Starlings captures the complete hierarchical structure of entity formation, enabling instant exploration of the entire resolution space.** This revolutionary approach achieves 10-100x performance improvements for threshold analysis while providing unprecedented insight into entity stability and formation patterns.

The framework's core innovation lies in representing entity resolution outputs as hierarchical structures that generate partitions at any threshold. This enables O(k) incremental metric computation across thresholds (where k = affected entities), O(m) storage for infinite threshold granularity (where m = number of edges), and natural support for entity data at any confidence level. By storing the complete resolution space, Starlings becomes the universal transport format for entity resolution - preserving all information needed for downstream analysis and decision-making.

Starlings supports multiple collections within a single frame, enabling direct comparison of different resolution approaches. Each collection is a hierarchy representing one attempt at resolving the same underlying records, whether from different algorithms, parameter settings, or confidence thresholds. This multi-collection architecture is fundamental to Starlings's value proposition: finding optimal resolution strategies through systematic comparison.

## Core Design Principles

### Incremental computation: no work thrown away
Every computation in Starlings builds upon previous work. When moving between thresholds, we update only what changes rather than recomputing from scratch. This principle drives our O(k) metric updates (where k = affected entities) and makes complete threshold analysis computationally feasible.

### Hierarchy generates partitions
The hierarchy is not a fixed clustering but a generative structure. At any threshold, it produces a complete partition of records into entities. This distinction is fundamental - we store relationships and transformations, not just states.

### Multiple collections, shared records
Starlings can hold multiple resolution attempts (collections) over the same record space. Collections share the underlying record storage but maintain independent hierarchies, enabling efficient comparison and analysis across different resolution strategies.

### Lazy evaluation with caching
Expensive operations are computed only when needed and cached for reuse. Simple statistics (sizes, counts) are pre-computed; complex metrics (Jaccard, NMI) are computed on demand. This balances memory usage with computational efficiency.

### Information preservation
The hierarchy preserves all information needed for threshold decisions. Users can explore the complete resolution space without information loss, making Starlings ideal for collaboration and decision deferral.

### Optimised simplicity
We choose simple, correct solutions and optimise them using Rust's low-level control. RoaringBitmaps for set operations, sparse matrices where naturally sparse, SIMD where beneficial - but no premature optimisation that adds complexity without clear benefit.

## User Requirements and Stories

### Core Analysis Requirements

**1. Optimal threshold discovery**
*"I want to compare my new collection with an existing one at every threshold to figure out where to resolve it to truth"*
- Efficiently sweep all thresholds
- Compute comprehensive metrics (precision, recall, F1, ARI, NMI)
- Identify optimal cut points for different objectives

**2. Fixed threshold data comparison**
*"I have entities resolved at a fixed threshold and need to compare with other data"*
- Handle entities resolved at any confidence level
- Compare collections uniformly regardless of their threshold distribution
- Support evaluation without requiring probabilistic scores

**3. Multi-level quality assessment**
*"Evaluate quality at varying confidence levels, then drill down to see specific differences"*
- Explore different confidence thresholds
- Trace entities back to source records
- Understand what changes between thresholds

**4. Threshold sensitivity analysis**
*"Understand exactly what happens to specific entities as thresholds change"*
- Track entity lifetime across thresholds
- Identify stable vs unstable regions
- Find critical merge points

### Scale and Performance Requirements

**5. Million-record scale**
*"Handle millions of records without memory explosion"*
- Efficient memory usage for large-scale data
- No quadratic blowup in storage
- Practical performance at production scale
- Note: Assumes sparse graphs from blocking/LSH preprocessing (m << n²)

**6. Batch processing excellence**
*"Process large datasets efficiently in batch mode"*
- Optimized batch hierarchy construction
- Parallel processing capabilities
- Memory-mapped storage for out-of-core processing

**7. Production deployment**
*"Actually work at scale in production, not just in theory"*
- Predictable performance characteristics
- Monitoring and observability
- Clear scaling boundaries and expectations

### Integration and Collaboration Requirements

**8. Data transport and serialisation**
*"Share entity resolution results while preserving all information"*
- Arrow format for interoperability
- Preserve complete resolution space
- Enable collaborative threshold selection

**9. Python ecosystem integration**
*"Work seamlessly with er-evaluation, matchbox, splink"*
- Compatible data formats
- Pythonic API via PyO3
- Integration with existing workflows
- Starlings enters the entity resolution pipeline after pairwise scoring but before threshold selection

**10. Multi-source resolution**
*"Resolve entities across CRM, mailing lists, and other heterogeneous sources"*
- Handle records from multiple systems with different schemas
- Track source attribution
- Support heterogeneous key types (u32, u64, String, bytes)

### Decision Support Requirements

**11. Deferred threshold decisions**
*"Don't force threshold choices until I have enough information"*
- Store complete resolution space
- Enable exploratory analysis
- Support data-driven decision making

**12. Entity formation debugging**
*"Understand why specific entities formed or didn't form"*
- Trace merge decisions
- Visualise entity evolution
- Debug unexpected clusterings

**13. Comprehensive metrics suite**
*"Support all standard entity resolution evaluation metrics"*
- Implement er-evaluation metrics
- Enable custom metric definitions
- Provide confidence intervals

### Strategic Requirements

**14. Foundation for multiple projects**
*"Underpin different entity resolution projects with one framework"*
- Flexible, extensible architecture
- Support diverse use cases
- Maintain backward compatibility

**15. Batch analysis capabilities**
*"Analyse many resolution approaches efficiently"*
- Compare different algorithms
- Parameter sweep optimisation
- A/B testing support

**16. Information-theoretic analysis**
*"Understand information preservation and loss"*
- Quantify resolution uncertainty
- Measure information content
- Support decision theory metrics

**17. Resolution stability analysis**
*"I need to identify stable operating points where small threshold changes won't dramatically alter my entities"*
- Use entropy changes to find plateaus
- Identify critical thresholds where structure changes significantly
- Quantify confidence in threshold selection
- Connect relative information loss to practical threshold decisions

**18. Extensible metadata attachment**
*"I need to run arbitrary functions on resolved entities and attach the results as metadata"*
- Apply hash functions to resolved entities for deduplication
- Compute custom metrics for entity quality assessment
- Generate derived attributes for downstream processing
- Support both built-in operations and custom functions

**19. Load pre-resolved entities**
*"I need to join entities that have already been resolved in different EntityFrames"*
- Load pre-resolved entity sets as collections
- Support entity-to-entity linkage across collections
- Preserve provenance from multiple resolution stages
- Enable hierarchical resolution workflows

## Layer 1: Mathematical Foundations

### The Multi-Collection Entity Model

**Core representation: multiple hierarchies, shared records**

Starlings fundamentally supports multiple entity collections over a shared record space. This enables direct comparison of different resolution attempts without duplicating data.

An EntityFrame `F` is formally defined as: `F = (R, {H₁, H₂, ..., Hₙ}, I)` where:
- `R` is the shared set of records across all collections
- `Hᵢ` are the hierarchical structures (collections ARE hierarchies)
- `I` is the interning system for efficient reference storage

Each hierarchy `Hᵢ` represents a complete entity resolution attempt over the record space `R`. Collections can be constructed from three input formats: weighted edges (the most common), pre-resolved entity sets, or previously computed merge events for resuming work.

### Entity Representation: Sets of Interned References

**The fundamental entity structure**

An entity is a set of references to records across potentially multiple data sources:

```
Entity E = {(source₁, id₁), (source₂, id₂), ..., (sourceₙ, idₙ)}
```

For example, a customer entity might be:
```
E_customer = {
    (CRM, row_1),
    (CRM, row_9),
    (MailingList, row_17),
    (OrderSystem, row_42)
}
```

**Interning for space efficiency**

To prevent explosive growth in string storage, Starlings interns all source identifiers:
- Source names are mapped to small integers: "CRM" → 0, "MailingList" → 1
- References become compact tuples: (0, 1), (0, 9), (1, 17), (2, 42)
- Typical compression: 80-90% reduction in memory usage

### Hierarchical Partitions

**Hierarchy generates partitions at any threshold**

Each collection's hierarchy `H` consists of partition levels: `H = {(t₁, P₁), (t₂, P₂), ..., (tₙ, Pₙ)}` where:
- Each `Pᵢ` is a complete partition of R at threshold `tᵢ`
- `P(t) = {E₁, E₂, ..., Eₖ}` where `⋃Eᵢ = R` and `Eᵢ ∩ Eⱼ = ∅`

**Critical properties**

1. **No new entities created**: The hierarchy only tracks how existing records group together, preventing identifier explosion
2. **Partition monotonicity**: If `t₁ < t₂`, then `P(t₁)` is a refinement of `P(t₂)`
3. **Completeness**: Every record appears in exactly one entity at any threshold

### Hierarchy Construction via Connected Components

**Threshold-based connected components approach**

Starlings uses threshold-based connected components to build hierarchies from pairwise similarities. This is algorithmically equivalent to single-linkage clustering but more natural for entity resolution. Starlings requires all pairwise relationships to be expressed as similarity scores in [0,1], where higher values indicate stronger matches. Distance metrics must be converted to similarities before ingestion.

1. Given edges with similarities: `{(record_i, record_j, similarity)}`
2. At threshold `t`, include all edges where `similarity ≥ t`
3. Find connected components to form entities
4. Store merge events that transition between partitions

**Complexity analysis**: O(m log m) where m is the number of edges. For practical entity resolution with blocking/LSH preprocessing, m is typically O(n) to O(n log n), not O(n²).

### N-way Merges and Real-World Patterns

**Natural n-way merge representation**

Real entity resolution produces complex merge patterns at each threshold. When multiple edges share the same similarity score, they merge simultaneously:

```
Example edges: (A,B,0.9), (B,C,0.9), (C,D,0.9)

Threshold 1.0: [{A}, {B}, {C}, {D}]
Threshold 0.9: [{A,B,C,D}]  // 4-way merge, not sequential binary merges
```

The union-find algorithm naturally handles these n-way merges without forcing artificial binary tree structures.

### Incremental Metric Computation

**The mathematical foundation for efficiency**

The key insight: most entity resolution metrics can be updated incrementally as we move between partition levels.

For any additive metric `M`: 
```
M(P(t + Δt)) = M(P(t)) + ΔM(t, t + Δt)
```

Where `ΔM` depends only on entities that changed (k entities), giving us O(k) updates.

**Contingency table evolution**

For comparing two partitions:
```
N_{ij}(t + Δt) = N_{ij}(t) + ΔN_{ij}
```

This enables O(k) updates for metrics like ARI and NMI as we move between thresholds, where k is the number of affected entities.

**Float precision and quantization**

When constructing hierarchies from edges, float quantization requires explicit opt-in. This allows users to control the trade-off between precision and the number of distinct merge events in the hierarchy.

### Complete Mathematical Operation Space

**Pairwise classification metrics**
- **Precision**: `P(t) = |TP(t)| / (|TP(t)| + |FP(t)|)`
- **Recall**: `R(t) = |TP(t)| / (|TP(t)| + |FN(t)|)`
- **F-measure**: `F_β(t) = (1 + β²) · P(t) · R(t) / (β² · P(t) + R(t))`

These update incrementally in O(k) as only changed pairs need recomputation.

**Cluster evaluation metrics**

For partition `P₁` at threshold `t` compared with ground truth `P₂`:

- **Adjusted Rand Index (ARI)**:
  ```
  ARI(t) = (RI(t) - E[RI]) / (max(RI) - E[RI])
  ```
  Incrementally updatable in O(k) via contingency table updates.
  
- **Normalised Mutual Information (NMI)**:
  ```
  NMI(t) = 2·I(P₁(t); P₂) / (H(P₁(t)) + H(P₂))
  ```
  Incrementally updatable in O(k) via mutual information terms.
  
- **V-measure**:
  ```
  V(t) = 2·h(t)·c(t) / (h(t) + c(t))
  ```
  Where h = homogeneity, c = completeness. Incrementally updatable in O(k).

- **B-cubed metrics**:
  ```
  B³-Precision = Σᵢ (1/|R|) · (1/|Cᵢ|) · Σⱼ∈Cᵢ |Cᵢ ∩ Lⱼ|²/|Lⱼ|
  B³-Recall = Σᵢ (1/|R|) · (1/|Lᵢ|) · Σⱼ∈Lᵢ |Lᵢ ∩ Cⱼ|²/|Cⱼ|
  ```
  Semi-incremental: O(k × avg_entity_size) where k is number of affected entities.

**Set-theoretic operations**

For entities `E₁, E₂` at threshold `t`:
- **Jaccard similarity**: `J(E₁, E₂) = |E₁ ∩ E₂| / |E₁ ∪ E₂|`
- **Dice coefficient**: `D(E₁, E₂) = 2|E₁ ∩ E₂| / (|E₁| + |E₂|)`
- **Overlap coefficient**: `O(E₁, E₂) = |E₁ ∩ E₂| / min(|E₁|, |E₂|)`

Computed on-demand using efficient set operations.

**Stability and sensitivity metrics**

Unique to hierarchical representation:
- **Entity lifetime**: `L(e) = {(t_start, t_end) : e exists in [t_start, t_end]}`
- **Merge criticality**: `C(merge) = |E_left| × |E_right|` (impact of merge)
- **Stability score**: `S(t₁, t₂) = |P(t₁) ∩ P(t₂)| / |P(t₁) ∪ P(t₂)|`
- **Resolution entropy**: `H(t) = -Σᵢ (|Eᵢ|/|R|) log(|Eᵢ|/|R|)`

### Multi-Collection Comparison

**Cross-collection analysis**

Starlings enables sophisticated comparison across collections. Importantly, thresholds are not comparable across different resolution methods - a 0.85 from one algorithm has no meaningful relationship to 0.85 from another due to different feature sets, modelling assumptions, and calibration.

**Collection cuts**:
A cut through a collection is denoted as `(collection_name, threshold)`, for example:
- `("splink_v1", 0.85)` - Splink output at threshold 0.85
- `("dedupe_v2", 0.76)` - Dedupe output at threshold 0.76
- `("ground_truth", 1.0)` - Fixed entities represented as hierarchy at threshold 1.0

**Agreement analysis**:
```
Agreement(C₁, t₁, C₂, t₂) = |{(i,j) : same_entity_C₁(i,j,t₁) ↔ same_entity_C₂(i,j,t₂)}| / |R|²
```

**Consensus clustering**:
```
Consensus(t₁, ..., tₙ) = argmin_P Σᵢ d(P, Cᵢ(tᵢ))
```
Where each collection Cᵢ uses its own calibrated threshold tᵢ.

### Information-Theoretic Framework

**Hierarchy as information preservation**

The merge hierarchy preserves maximum information about entity formation:

**Information content**: 
```
I(H) = -Σ_m log₂(P(m))
```
Where P(m) is the probability of merge m.

**Relative information loss**:
```
L(t) = 1 - I(P(t)) / I(H)
```
Quantifies information lost by choosing a specific threshold. Large jumps in this metric indicate unstable regions where small threshold changes cause significant structural changes - critical for stability analysis.

**Cross-collection information**:
```
MI(C₁, t₁, C₂, t₂) = Σᵢⱼ p(i,j) log(p(i,j)/(p₁(i)p₂(j)))
```
Mutual information between two collections at their respective thresholds.

### Memory Architecture: Contextual Ownership

**Solving the multi-collection memory challenge**

To resolve the fundamental conflict between standalone collection encapsulation and efficient frame composition, Starlings employs a Contextual Ownership architecture. This model separates physical data storage from logical structure, leveraging an append-only data context for index stability and performance.

**Core components**:
- **DataContext**: An append-only arena containing record storage and string interning. Once a record is added, its index is immutable for the context's lifetime.
- **Collections**: Lightweight logical views that hold an Arc<DataContext> reference and structural metadata (hierarchies) composed of indices valid within that context. Collections exist in two states: standalone (exclusively owning their DataContext) or view (sharing a DataContext with a parent EntityFrame).

**Memory efficiency through sharing**:
When collections are combined into an EntityFrame, they share the same DataContext through Arc reference counting. This eliminates duplication while maintaining the ability for collections to exist independently. The append-only nature guarantees index stability, enabling lock-free concurrent reads.

**Automatic memory management**:
When collections are removed from a frame, automatic compaction can reclaim unused space when garbage exceeds a threshold. This happens transparently without user intervention, maintaining the simplicity of the user-facing API while ensuring long-term memory efficiency. Float quantization can be explicitly enabled during hierarchy construction to reduce the number of distinct threshold levels, trading precision for memory efficiency.

### Theoretical Guarantees

**Computational complexity**

- **Hierarchy construction**: O(m log m) where m = number of edges
- **Threshold query**: O(m) to reconstruct partition (O(1) if cached)
- **Metric update**: O(k) where k is number of changed entities
- **Full threshold sweep**: O(m·k) where m is number of merge events
- **Storage**: O(m) where m is number of edges/merge events
- **Memory in practice**: For 1M edges, expect 50-250MB depending on merge complexity

**Mathematical properties**

1. **Ultrametric property**: The hierarchy induces an ultrametric on records
2. **Lattice structure**: Partitions form a lattice under refinement ordering
3. **Monotone metrics**: Metrics evolve monotonically between merge events
4. **Convergence**: All records eventually merge into single entity as t→0

**Practical scalability assumptions**

Starlings is designed for realistic entity resolution scenarios where:
- Edge lists are sparse due to blocking/LSH preprocessing
- m (number of edges) is O(n) to O(n log n), not O(n²)
- For 1M records, we expect 1-10M edges, not 1 trillion
- If your blocking produces dense graphs approaching O(n²), the problem lies in the blocking strategy, not Starlings
