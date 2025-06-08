🍏 EntityFrame rethinks how we evaluate entity resolution methods by treating entities as what they truly are: **collections of records across multiple datasets**, not just pairs of records.

## The Problem

When comparing entity resolution tools, we're forced into a lossy abstraction where entities become mere pairs of records. But **entities aren't pairs - they're sets of sets**.

The question isn't "do these two records match?" but rather "how well do these two methods agree on what constitutes the entity 'Michael'?"

## The Solution

EntityFrame provides a **three-layer architecture** optimised for entity-centric evaluation at scale:

### Layer 1: String Interning

Every unique record ID gets mapped to a compact integer exactly once, creating a global string pool that enables massive memory savings and faster comparisons.

### Layer 2: Roaring Bitmaps

Record ID sets become compressed bitmaps optimised for the set operations that drive entity resolution evaluation. Perfect for the sparse/clustered patterns typical in entity data.

### Layer 3: Entity Collections & Frames

**EntityCollection** (like pandas Series): Contains entities from a single process
**EntityFrame** (like pandas DataFrame): Contains multiple collections with shared string interning

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

### Option 1: Manual Collection Building (Composable)

```python
import entityframe as ef

# Create frame with shared string interning
frame = ef.EntityFrame()

# Build collections separately
model1_collection = ef.EntityCollection("model_1_output")
model1_collection.add_entities(model1_results, frame.interner)

model2_collection = ef.EntityCollection("model_2_output") 
model2_collection.add_entities(model2_results, frame.interner)

# Add to frame
frame.add_collection("model_1", model1_collection)
frame.add_collection("model_2", model2_collection)

# Compare entity resolution methods
comparison = frame.compare_collections("model_1", "model_2")
print(f"Average Jaccard similarity: {sum(c['jaccard'] for c in comparison) / len(comparison)}")
```

### Option 2: Convenience Methods

```python
import entityframe as ef

# Create frame and add methods directly
frame = ef.EntityFrame()
frame.add_method("model_1_output", model1_results)
frame.add_method("model_2_output", model2_results)

# Compare methods (legacy API, maps to compare_collections)
comparison = frame.compare_methods("model_1_output", "model_2_output")

# Find high-disagreement cases for debugging
low_agreement = [c for c in comparison if c['jaccard'] < 0.5]
print(f"Found {len(low_agreement)} entities with low agreement")
```

### Data Format

Your entity data should be structured as:

```python
model_results = [
    {
        "customers": ["cust_001", "cust_002"],      # Records in customers dataset
        "transactions": ["txn_100", "txn_101"],    # Records in transactions dataset
    },
    {
        "customers": ["cust_003"],
        "transactions": ["txn_102", "txn_103", "txn_104"],
        "addresses": ["addr_001"]                   # Can include different datasets
    }
    # ... more entities
]
```

## API Reference

### EntityFrame (like pandas DataFrame)

The main container that holds multiple entity collections with shared string interning.

```python
frame = ef.EntityFrame()

# Add collections
frame.add_method("model_a", entity_data)           # Convenience method
frame.add_collection("model_b", collection)       # Manual addition

# Access collections
collection = frame.get_collection("model_a")
names = frame.get_collection_names()

# Compare entity resolution methods
comparisons = frame.compare_collections("model_a", "model_b")

# Statistics
frame.collection_count()    # Number of methods/collections
frame.total_entities()      # Total entities across all collections  
frame.interner_size()       # Number of unique strings interned
```

### EntityCollection (like pandas Series)

Contains entities from a single entity resolution process.

```python
collection = ef.EntityCollection("model_output")

# Add entities using shared interner
collection.add_entities(entity_data, shared_interner)

# Access entities
entity = collection.get_entity(0)       # Get by index
entities = collection.get_entities()    # Get all entities
collection.len()                        # Number of entities
collection.total_records()              # Total records across all entities

# Compare with another collection
similarities = collection.compare_with(other_collection)
```

### Entity

Individual entity containing record IDs across multiple datasets.

```python
entity = ef.Entity()

# Add records  
entity.add_record("customers", record_id)
entity.add_records("transactions", [id1, id2, id3])

# Query entity
records = entity.get_records("customers")
datasets = entity.get_datasets()
entity.total_records()
entity.has_dataset("customers")

# Compare entities
similarity = entity1.jaccard_similarity(entity2)
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
- **Memory efficiency**: 10-100x smaller than naive string sets through string interning
- **Speed**: Set operations use SIMD-optimised roaring bitmaps
- **Cache friendly**: Integer operations with excellent locality
- **Shared interning**: Multiple collections share the same string pool

Evaluate millions of entities across multiple methods in minutes, not hours.

## Architecture

EntityFrame follows a **pandas-like design pattern**:

- **EntityCollection** ≈ pandas Series (single process results) 
- **EntityFrame** ≈ pandas DataFrame (multiple process comparison)

Built as a hybrid Python/Rust package using PyO3 bindings:

```
src/
├── python/entityframe/    # Python API with DataFrame-like interface
├── rust/entityframe/      # High-performance core in Rust
└── tests/                # Comprehensive test suite (36 tests)
```

The Rust core handles computationally intensive operations while Python provides a familiar, DataFrame-like API.

## Example: Comparing Entity Resolution Methods

```python
import entityframe as ef

# Sample output from two different entity resolution models
model_1_results = [
    {
        "customers": ["cust_001", "cust_002"],
        "transactions": ["txn_100"],
    },
    {
        "customers": ["cust_003"],
        "transactions": ["txn_101", "txn_102"],
    }
]

model_2_results = [
    {
        "customers": ["cust_001", "cust_002", "cust_003"],  # Different clustering
        "transactions": ["txn_100", "txn_101"],
    },
    {
        "transactions": ["txn_102"],
    }
]

# Compare the models
frame = ef.EntityFrame()
frame.add_method("model_1_output", model_1_results)
frame.add_method("model_2_output", model_2_results)

print(f"Model 1 has {len(frame.get_collection('model_1_output').get_entities())} entities")
print(f"Model 2 has {len(frame.get_collection('model_2_output').get_entities())} entities")
print(f"Shared interner contains {frame.interner_size()} unique strings")

# Compare entity resolution quality
comparisons = frame.compare_collections("model_1_output", "model_2_output")
avg_jaccard = sum(c['jaccard'] for c in comparisons) / len(comparisons)
print(f"Average Jaccard similarity: {avg_jaccard:.3f}")

# Find entities with low agreement
low_agreement = [c for c in comparisons if c['jaccard'] < 0.8]
print(f"Entities with <80% agreement: {len(low_agreement)}")
```

## Development

This project uses `just` as a command runner and `uv` for Python dependency management:

```bash
just install     # Install all dependencies
just build       # Build the Rust extension
just test-all    # Run Python and Rust tests
just format       # Format and lint all code
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