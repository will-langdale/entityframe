# EntityFrame Design Documents Overview

## Project Summary
EntityFrame is a Rust-backed Python framework for entity resolution that captures the complete hierarchical structure of entity formation, enabling instant exploration of the entire resolution space without forcing threshold decisions at processing time.

## Document Structure

### [Document A: Core Design & Mathematics](foundations.md)

**Purpose**: Establishes the theoretical foundation and user requirements

- **Core Principles**: Incremental computation, hierarchy generates partitions, lazy evaluation
- **User Requirements**: 16 detailed user stories covering analysis, scale, integration, and decision support
- **Mathematical Foundations**: 
 - Multi-collection entity model with shared records
 - Entities as sets of interned references 
 - Hierarchical partitions and n-way merges
 - Complete mathematical operation space (ARI, NMI, V-measure, B-cubed, etc.)
 - Information-theoretic framework

### [Document B: Technical Implementation](implementation.md)

**Purpose**: Details the computer science techniques and engineering design

- **Layer 2 - CS Techniques**:
 - Core data structures with RoaringBitmaps
 - String interning architecture for space efficiency
 - Lazy metric computation framework
 - Sparse structure exploitation
 - SIMD and cache optimizations
 - Parallel processing with Rayon
 
- **Layer 3 - Engineering**:
 - Multi-collection EntityFrame architecture
 - Python interface via PyO3
 - Incremental/streaming updates
 - Production deployment considerations

### [Document C: Reference Architecture](reference.md)

**Purpose**: Provides practical API reference and implementation roadmap

- **Complete API Reference**: Python and Rust APIs with examples
- **Data Structure Specifications**: Memory layouts and serialization formats
- **Integration Examples**: Splink, er-evaluation, Matchbox, Arrow
- **Implementation Roadmap**: 12-month plan with quarterly phases
- **Performance Benchmarks**: Target metrics for different scales
- **Migration Guide**: Moving from existing approaches

## Key Innovation
Rather than storing entities at fixed thresholds, EntityFrame stores hierarchical structures that can generate partitions at any threshold instantly, enabling O(1) incremental metric updates and preserving complete information for collaborative decision-making.

## Use Cases
- Finding optimal thresholds for entity resolution
- Comparing multiple resolution approaches
- Analyzing threshold sensitivity and entity stability
- Supporting both probabilistic and deterministic entity data
- Scaling to millions of records efficiently