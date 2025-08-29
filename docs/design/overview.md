# EntityFrame Design Documents Overview

## The Problem: Entity Resolution Needs Better Infrastructure

Entity resolution produces rich hierarchical relationships between records, but the field lacks a unified data structure to:
- **Analyze**: Compare different resolution methods systematically
- **Represent**: Store complete resolution information without premature decisions
- **Transport**: Share results between teams, tools, and stages of processing

Current approaches force threshold decisions too early, lose information between pipeline stages, and make it impossible to compare outputs from different methods. EntityFrame provides the foundational data structure to solve all three problems.

## Core Architecture: Merge Events as Universal Representation

EntityFrame stores entity resolution outputs as sequences of merge events - the fundamental transitions that occur as similarity thresholds change. This representation serves as:

- **Analysis substrate**: O(k) incremental metric computation between thresholds
- **Complete representation**: All possible partitions recoverable from ~50-250MB of merge events (vs ~10GB if stored explicitly)
- **Transport format**: Preserves full resolution information for downstream decisions

```
Pipeline Stage A → EntityFrame → Pipeline Stage B
(Splink output)    (merge events)   (threshold selection)
                         ↓
                   Pipeline Stage C
                   (quality analysis)
```

## Document Structure

### [Document A: Core Design & Mathematics](foundations.md)
**The Foundation**

Establishes EntityFrame as the universal container for entity resolution:
- Multi-collection model: F = (R, {H₁, H₂, ..., Hₙ}, I) 
- Collections ARE hierarchies (even fixed clusters at threshold 1.0)
- 19 user stories covering analysis, transport, and integration needs
- Mathematical framework for correctness and performance guarantees

### [Document B: Technical Implementation](implementation.md)
**The Engine**

Optimized Rust implementation for production scale:
- **Memory efficiency**: String interning, RoaringBitmaps, sparse structures
- **Performance**: Parallel processing, SIMD operations, smart caching
- **Flexibility**: Dual processing model (Rust built-ins + Python extensions)
- **Integration**: PyO3 bindings, Arrow serialization, monitoring hooks

### [Document C: Reference Architecture](reference.md)
**The Interface**

Complete specification for users and integrators:
- **Unified API**: Single data model for all resolution types
- **Tool adapters**: Splink, er-evaluation, Matchbox integrations
- **Transport protocols**: Arrow/Parquet for cross-system compatibility
- **Production guidance**: Benchmarks, deployment patterns, migration paths

## Why EntityFrame Is The Right Foundation

### For Analysis
- Compare any two resolution methods at any thresholds
- Sweep 1000 thresholds in seconds, not hours
- Track entity stability and formation patterns
- Compute standard metrics (ARI, NMI, B-cubed) incrementally

### For Representation
- Store complete resolution space in minimal memory
- Defer threshold decisions until sufficient information available
- Support heterogeneous data sources and ID types
- Handle both probabilistic scores and fixed clusters uniformly

### For Transport
- Share resolution outputs without information loss
- Enable pipeline stages to operate independently
- Preserve provenance and enable reproducibility
- Support batch processing at million-record scale

## Key Design Principles

1. **Information preservation**: Never force premature threshold decisions
2. **Incremental computation**: Reuse work between adjacent thresholds
3. **Universal representation**: One format for all entity resolution outputs
4. **Production reality**: Optimize for sparse graphs from real blocking strategies

## The Value Proposition

EntityFrame transforms entity resolution from isolated threshold decisions into a continuous exploration space. It's the difference between:

**Without EntityFrame**: Each tool/stage makes irreversible decisions
```
Blocking → Matching → Clustering → Analysis
   ↓          ↓           ↓            ↓
(loses)    (loses)     (loses)    (too late)
```

**With EntityFrame**: Preserve everything, decide later
```
Blocking → Matching → EntityFrame → Any threshold
                           ↓         Any comparison  
                      (preserves all) Any transport
```

By providing a single, optimized data structure for the complete entity resolution lifecycle, EntityFrame enables systematic analysis, lossless representation, and seamless transport of resolution results across tools and teams.
