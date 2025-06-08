🍏 EntityFrame rethinks how we evaluate entity resolution methods by treating entities as what they truly are: **collections of records across multiple datasets**, not just pairs of records.

## The Problem

When comparing tools like Splink vs RecordLinkage vs Dedupe, we're forced into a lossy abstraction where entities become mere pairs of records. But **entities aren't pairs - they're sets of sets**.

The question isn't "do these two records match?" but rather "how well do these two methods agree on what constitutes the entity 'Michael'?"

## The Solution

EntityFrame provides a **three-layer architecture** optimised for entity-centric evaluation at scale:

### Layer 1: String Interning

Every unique record ID gets mapped to a compact integer exactly once, creating a global string pool that enables massive memory savings and faster comparisons.

### Layer 2: Roaring Bitmaps

Record ID sets become compressed bitmaps optimised for the set operations that drive entity resolution evaluation. Perfect for the sparse/clustered patterns typical in entity data.

### Layer 3: Entity Hashing

Each entity gets a deterministic hash based on its complete set-of-sets structure, enabling fast deduplication, lookup, and caching.

## Core Concepts

An entity is a mathematical object: a mapping from dataset names to sets of record identifiers.

```python
Entity("Michael") = {
    "customers": {"cust_2023_001", "cust_2023_045"},
    "transactions": {"txn_555", "txn_777", "txn_999"}, 
    "addresses": {"addr_chicago_123"}
}
```

Entity resolution evaluation becomes **pure set theory**: comparing how different methods partition the same universe of records into entity-sets.

## Quick Start

```python
import entityframe as ef

# Create entities from your resolution results
entities = ef.EntityCollection()

# Add entities from different methods
entities.add_method("splink", splink_results)
entities.add_method("recordlinkage", recordlinkage_results)

# Compare methods at entity level
comparison = entities.compare_methods("splink", "recordlinkage")

# Find high-disagreement cases for debugging
problematic = comparison.filter(jaccard < 0.5)
```

## Installation

```bash
# Install from PyPI (when available)
pip install entityframe

# Or build from source
git clone https://github.com/yourusername/entityframe
cd entityframe
just install
just build
```

## Performance

EntityFrame is built for scale:
- **Memory efficiency**: 10-100x smaller than naive string sets
- **Speed**: Set operations use SIMD-optimised roaring bitmaps
- **Cache friendly**: Integer operations with excellent locality

Evaluate millions of entities across multiple methods in minutes, not hours.

## Architecture

EntityFrame is a hybrid Python/Rust package using PyO3 bindings:

```
src/
├── python/entityframe/    # Python API with DataFrame-like interface
├── rust/entityframe/      # High-performance core in Rust
└── tests/                # Comprehensive test suite
```

The Rust core handles the computationally intensive operations while Python provides a familiar, DataFrame-like API that data scientists already understand.

## Development

This project uses `just` as a command runner and `uv` for Python dependency management:

```bash
just install     # Install all dependencies
just build       # Build the Rust extension
just test-all    # Run Python and Rust tests
just check       # Format and lint all code
```

## Why EntityFrame?

- **Entity-native evaluation**: Compare methods on actual entity structure, not proxy metrics
- **Massive scale**: Handle millions of entities efficiently
- **Method agnostic**: Works with any entity resolution tool's output
- **Familiar API**: DataFrame-like interface for data scientists
- **Debugging focused**: Instantly drill down to specific disagreements

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.