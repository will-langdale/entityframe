# EntityFrame Design Documents Overview

## Project Summary
EntityFrame is a Rust-backed Python framework for entity resolution that captures the complete hierarchical structure of entity formation, enabling instant exploration of the entire resolution space without forcing threshold decisions at processing time.

## Document Structure

### [Document A: Core Design & Mathematics](foundations.md)
**Purpose**: Establishes the theoretical foundation and user requirements

- **Core Principles**: Incremental computation, hierarchy generates partitions, lazy evaluation
- **User Requirements**: 19 detailed user stories covering analysis, scale, integration, and decision support
- **Mathematical Foundations**: 
  - Multi-collection entity model: F = (R, {H₁, H₂, ..., Hₙ}, I) where hierarchies ARE collections
  - Entities as sets of interned references across heterogeneous sources
  - Hierarchical partitions via connected components (single-linkage)
  - N-way merge support without forced binary structures
  - Complete mathematical operation space (ARI, NMI, V-measure, semi-incremental B-cubed)
  - Information-theoretic framework for stability analysis

### [Document B: Technical Implementation](implementation.md)
**Purpose**: Details the computer science techniques and engineering design

- **Layer 2 - CS Techniques**:
  - Core data structures with RoaringBitmaps
  - String interning with flexible key types (u32, u64, String, bytes)
  - Connected components algorithm for hierarchy construction
  - Lazy metric computation framework
  - Sparse structure exploitation
  - SIMD and cache optimisations
  - Parallel processing with Rayon
  - Dual processing: built-in Rust operations vs Python callbacks
  
- **Layer 3 - Engineering**:
  - Multi-collection EntityFrame architecture
  - Python interface via PyO3
  - Incremental/streaming updates
  - Production deployment considerations
  - High-performance entity hashing (SHA256, BLAKE3)
  - Entity processing with map functions

### [Document C: Reference Architecture](reference.md)
**Purpose**: Provides practical API reference and implementation roadmap

- **Complete API Reference**: Python and Rust APIs with examples
- **Data Structure Specifications**: Memory layouts and serialisation formats
- **Data Shape Specifications**: Input formats for records and collections
- **Built-in Operations**: Parallel hashing and entity processing functions
- **Integration Examples**: Splink, er-evaluation, Matchbox, Arrow
- **Implementation Roadmap**: 12-month plan with quarterly phases
- **Performance Benchmarks**: Target metrics for different scales
- **Migration Guide**: Moving from existing approaches

## Key Innovation
Rather than storing entities at fixed thresholds, EntityFrame stores hierarchical structures that can generate partitions at any threshold instantly, enabling O(1) incremental metric updates and preserving complete information for collaborative decision-making.

## Use Cases
- Finding optimal thresholds for entity resolution
- Comparing multiple resolution approaches with proper calibration
- Analysing threshold sensitivity and entity stability
- Supporting both probabilistic and deterministic entity data
- High-performance entity hashing for deduplication
- Joining pre-resolved entities from multiple sources
- Running arbitrary functions on resolved entities
- Scaling to millions of records efficiently
