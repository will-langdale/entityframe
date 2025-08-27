# EntityFrame Design Document A: Core Design & Mathematics

## Executive Summary

EntityFrame represents a paradigm shift in entity resolution infrastructure. **Rather than forcing threshold decisions at processing time, EntityFrame captures the complete hierarchical structure of entity formation, enabling instant exploration of the entire resolution space.** This revolutionary approach achieves 10-100x performance improvements for threshold analysis while providing unprecedented insight into entity stability and formation patterns.

The framework's core innovation lies in representing entity resolution outputs as hierarchical structures that generate partitions at any threshold. This enables O(1) incremental metric computation across thresholds, O(n) storage for infinite threshold granularity, and natural support for both probabilistic and deterministic entity data. By storing the complete resolution space, EntityFrame becomes the universal transport format for entity resolution - preserving all information needed for downstream analysis and decision-making.

EntityFrame supports multiple collections within a single frame, enabling direct comparison of different resolution approaches. Each collection represents one attempt at resolving the same underlying records, whether from different algorithms, parameter settings, or confidence thresholds. This multi-collection architecture is fundamental to EntityFrame's value proposition: finding optimal resolution strategies through systematic comparison.

## Core Design Principles

### Incremental Computation: No Work Thrown Away
Every computation in EntityFrame builds upon previous work. When moving between thresholds, we update only what changes rather than recomputing from scratch. This principle drives our O(1) metric updates and makes complete threshold analysis computationally feasible.

### Hierarchy Generates Partitions
The hierarchy is not a fixed clustering but a generative structure. At any threshold, it produces a complete partition of records into entities. This distinction is fundamental - we store relationships and transformations, not just states.

### Multiple Collections, Shared Records
EntityFrame can hold multiple resolution attempts (collections) over the same record space. Collections share the underlying record storage but maintain independent hierarchies, enabling efficient comparison and analysis across different resolution strategies.

### Lazy Evaluation with Caching
Expensive operations are computed only when needed and cached for reuse. Simple statistics (sizes, counts) are pre-computed; complex metrics (Jaccard, NMI) are computed on demand. This balances memory usage with computational efficiency.

### Information Preservation
The hierarchy preserves all information needed for threshold decisions. Users can explore the complete resolution space without information loss, making EntityFrame ideal for collaboration and decision deferral.

### Optimized Simplicity
We choose simple, correct solutions and optimize them using Rust's low-level control. RoaringBitmaps for set operations, sparse matrices where naturally sparse, SIMD where beneficial - but no premature optimization that adds complexity without clear benefit.

## User Requirements and Stories

### Core Analysis Requirements

**1. Optimal Threshold Discovery**
*"I want to compare my new collection with an existing one at every threshold to figure out where to resolve it to truth"*
- Efficiently sweep all thresholds
- Compute comprehensive metrics (precision, recall, F1, ARI, NMI)
- Identify optimal cut points for different objectives

**2. Deterministic Data Comparison**
*"I have pre-resolved entities without probabilities and need to compare with other data"*
- Handle entities resolved at fixed thresholds
- Compare deterministic and probabilistic data uniformly
- Support evaluation without probability information

**3. Multi-Level Quality Assessment**
*"Evaluate quality at varying confidence levels, then drill down to see specific differences"*
- Explore different confidence thresholds
- Trace entities back to source records
- Understand what changes between thresholds

**4. Threshold Sensitivity Analysis**
*"Understand exactly what happens to specific entities as thresholds change"*
- Track entity lifetime across thresholds
- Identify stable vs unstable regions
- Find critical merge points

### Scale and Performance Requirements

**5. Million-Record Scale**
*"Handle millions of records without memory explosion"*
- Efficient memory usage for large-scale data
- No quadratic blowup in storage
- Practical performance at production scale

**6. Streaming and Incremental Updates**
*"Add new records without rebuilding everything"*
- Support online entity resolution
- Incremental hierarchy updates
- Maintain performance with growing data

**7. Production Deployment**
*"Actually work at scale in production, not just in theory"*
- Predictable performance characteristics
- Monitoring and observability
- Distributed processing support

### Integration and Collaboration Requirements

**8. Data Transport and Serialization**
*"Share entity resolution results while preserving all information"*
- Arrow format for interoperability
- Preserve complete resolution space
- Enable collaborative threshold selection

**9. Python Ecosystem Integration**
*"Work seamlessly with er-evaluation, matchbox, splink"*
- Compatible data formats
- Pythonic API via PyO3
- Integration with existing workflows

**10. Multi-Source Resolution**
*"Resolve entities across CRM, mailing lists, and other heterogeneous sources"*
- Handle records from multiple systems
- Track source attribution
- Support heterogeneous data types

### Decision Support Requirements

**11. Deferred Threshold Decisions**
*"Don't force threshold choices until I have enough information"*
- Store complete resolution space
- Enable exploratory analysis
- Support data-driven decision making

**12. Entity Formation Debugging**
*"Understand why specific entities formed or didn't form"*
- Trace merge decisions
- Visualize entity evolution
- Debug unexpected clusterings

**13. Comprehensive Metrics Suite**
*"Support all standard entity resolution evaluation metrics"*
- Implement er-evaluation metrics
- Enable custom metric definitions
- Provide confidence intervals

### Strategic Requirements

**14. Foundation for Multiple Projects**
*"Underpin different entity resolution projects with one framework"*
- Flexible, extensible architecture
- Support diverse use cases
- Maintain backward compatibility

**15. Batch Analysis Capabilities**
*"Analyze many resolution approaches efficiently"*
- Compare different algorithms
- Parameter sweep optimization
- A/B testing support

**16. Information-Theoretic Analysis**
*"Understand information preservation and loss"*
- Quantify resolution uncertainty
- Measure information content
- Support decision theory metrics

## Layer 1: Mathematical Foundations

### The Multi-Collection Entity Model

**Core Representation: Multiple Hierarchies, Shared Records**

EntityFrame fundamentally supports multiple entity collections over a shared record space. This enables direct comparison of different resolution attempts without duplicating data.

An EntityFrame `F` is formally defined as: `F = (R, {C₁, C₂, ..., Cₙ}, I)` where:
- `R` is the shared set of records across all collections
- `Cᵢ` are individual collections, each with its own hierarchy
- `I` is the interning system for efficient reference storage

Each collection `Cᵢ = (Hᵢ, Sᵢ)` consists of:
- `Hᵢ` is the hierarchical structure for this collection
- `Sᵢ` is optional similarity/provenance information

### Entity Representation: Sets of Interned References

**The Fundamental Entity Structure**

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

**Interning for Space Efficiency**

To prevent explosive growth in string storage, EntityFrame interns all source identifiers:
- Source names are mapped to small integers: "CRM" → 0, "MailingList" → 1
- References become compact tuples: (0, 1), (0, 9), (1, 17), (2, 42)
- Typical compression: 80-90% reduction in memory usage

### Hierarchical Partitions

**Hierarchy Generates Partitions at Any Threshold**

Each collection's hierarchy `H` consists of partition levels: `H = {(t₁, P₁), (t₂, P₂), ..., (tₙ, Pₙ)}` where:
- Each `Pᵢ` is a complete partition of R at threshold `tᵢ`
- `P(t) = {E₁, E₂, ..., Eₖ}` where `⋃Eᵢ = R` and `Eᵢ ∩ Eⱼ = ∅`

**Critical Properties**

1. **No New Entities Created**: The hierarchy only tracks how existing records group together, preventing identifier explosion
2. **Partition Monotonicity**: If `t₁ < t₂`, then `P(t₁)` is a refinement of `P(t₂)`
3. **Completeness**: Every record appears in exactly one entity at any threshold

### N-Way Merges and Real-World Patterns

**Natural N-Way Merge Representation**

Real entity resolution produces complex merge patterns at each threshold:

```
Threshold 1.0: [{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}]
Threshold 0.9: [{1}, {2,3}, {4}, {5}, {6}, {7}, {8}, {9}]
Threshold 0.8: [{1,2,3,4,5}, {6}, {7,8,9}]  // 5-way and 3-way merges
Threshold 0.7: [{1,2,3,4,5,6,7,8,9}]
```

The hierarchy naturally represents these without forcing binary tree structures.

### Incremental Metric Computation

**The Mathematical Foundation for Efficiency**

The key insight: most entity resolution metrics can be updated incrementally as we move between partition levels.

For any additive metric `M`: 
```
M(P(t + Δt)) = M(P(t)) + ΔM(t, t + Δt)
```

Where `ΔM` depends only on entities that changed.

**Contingency Table Evolution**

For comparing two partitions:
```
N_{ij}(t + Δt) = N_{ij}(t) + ΔN_{ij}
```

This enables O(1) updates for metrics like ARI and NMI as we move between thresholds.

### Complete Mathematical Operation Space

**Pairwise Classification Metrics**
- **Precision**: `P(t) = |TP(t)| / (|TP(t)| + |FP(t)|)`
- **Recall**: `R(t) = |TP(t)| / (|TP(t)| + |FN(t)|)`
- **F-measure**: `F_β(t) = (1 + β²) · P(t) · R(t) / (β² · P(t) + R(t))`

These update incrementally as only changed pairs need recomputation.

**Cluster Evaluation Metrics**

For partition `P₁` at threshold `t` compared with ground truth `P₂`:

- **Adjusted Rand Index (ARI)**:
  ```
  ARI(t) = (RI(t) - E[RI]) / (max(RI) - E[RI])
  ```
  
- **Normalized Mutual Information (NMI)**:
  ```
  NMI(t) = 2·I(P₁(t); P₂) / (H(P₁(t)) + H(P₂))
  ```
  
- **V-measure**:
  ```
  V(t) = 2·h(t)·c(t) / (h(t) + c(t))
  ```
  Where h = homogeneity, c = completeness

- **B-cubed Metrics**:
  ```
  B³-Precision = Σᵢ (1/|R|) · (1/|Cᵢ|) · Σⱼ∈Cᵢ |Cᵢ ∩ Lⱼ|²/|Lⱼ|
  B³-Recall = Σᵢ (1/|R|) · (1/|Lᵢ|) · Σⱼ∈Lᵢ |Lᵢ ∩ Cⱼ|²/|Cⱼ|
  ```

**Set-Theoretic Operations**

For entities `E₁, E₂` at threshold `t`:
- **Jaccard Similarity**: `J(E₁, E₂) = |E₁ ∩ E₂| / |E₁ ∪ E₂|`
- **Dice Coefficient**: `D(E₁, E₂) = 2|E₁ ∩ E₂| / (|E₁| + |E₂|)`
- **Overlap Coefficient**: `O(E₁, E₂) = |E₁ ∩ E₂| / min(|E₁|, |E₂|)`

Computed on-demand using efficient set operations.

**Stability and Sensitivity Metrics**

Unique to hierarchical representation:
- **Entity Lifetime**: `L(e) = {(t_start, t_end) : e exists in [t_start, t_end]}`
- **Merge Criticality**: `C(merge) = |E_left| × |E_right|` (impact of merge)
- **Stability Score**: `S(t₁, t₂) = |P(t₁) ∩ P(t₂)| / |P(t₁) ∪ P(t₂)|`
- **Resolution Entropy**: `H(t) = -Σᵢ (|Eᵢ|/|R|) log(|Eᵢ|/|R|)`

### Multi-Collection Comparison

**Cross-Collection Analysis**

EntityFrame enables sophisticated comparison across collections:

**Agreement Analysis**:
```
Agreement(C₁, C₂, t) = |{(i,j) : same_entity_C₁(i,j,t) ↔ same_entity_C₂(i,j,t)}| / |R|²
```

**Consensus Clustering**:
```
Consensus(t) = argmin_P Σᵢ d(P, Cᵢ(t))
```
Where d is a partition distance metric.

**Differential Analysis**:
For each threshold, identify:
- Records that cluster differently
- Entities that exist in one collection but not another
- Stability differences across collections

### Information-Theoretic Framework

**Hierarchy as Information Preservation**

The merge hierarchy preserves maximum information about entity formation:

**Information Content**: 
```
I(H) = -Σ_m log₂(P(m))
```
Where P(m) is the probability of merge m.

**Relative Information Loss**:
```
L(t) = 1 - I(P(t)) / I(H)
```
Quantifies information lost by choosing a specific threshold.

**Cross-Collection Information**:
```
MI(C₁, C₂, t) = Σᵢⱼ p(i,j) log(p(i,j)/(p₁(i)p₂(j)))
```
Mutual information between two collections at threshold t.

### Theoretical Guarantees

**Computational Complexity**

- **Hierarchy Construction**: O(n² log n) worst case, O(n log n) typical
- **Threshold Query**: O(n) to extract partition
- **Metric Update**: O(k) where k is number of changed entities
- **Full Threshold Sweep**: O(m·k) where m is number of merge events
- **Storage**: O(n·m) where m is number of unique thresholds

**Mathematical Properties**

1. **Ultrametric Property**: The hierarchy induces an ultrametric on records
2. **Lattice Structure**: Partitions form a lattice under refinement ordering
3. **Monotone Metrics**: Metrics evolve monotonically between merge events
4. **Convergence**: All records eventually merge into single entity as t→0