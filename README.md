# EntityFrame

**High-performance entity resolution evaluation for Python**

EntityFrame treats entities as what they truly are: collections of records across multiple datasets. Compare entity resolution methods efficiently using set operations rather than pairwise record comparisons.

## Quick Start

```python
import entityframe as ef

# Your entity resolution results from two different methods
splink_results = [
    {"customers": ["cust_001", "cust_002"], "transactions": ["txn_100"]},
    {"customers": ["cust_003"], "transactions": ["txn_101", "txn_102"]},
]

dedupe_results = [
    {"customers": ["cust_001"], "transactions": ["txn_100"]},
    {"customers": ["cust_002", "cust_003"], "transactions": ["txn_101", "txn_102"]},
]

# Create a frame with known datasets (optional - add_method auto-declares)
frame = ef.EntityFrame.with_datasets(["customers", "transactions"])

# Add both methods
frame.add_method("splink", splink_results)
frame.add_method("dedupe", dedupe_results)

# Compare how well the methods agree
comparisons = frame.compare_collections("splink", "dedupe")
avg_similarity = sum(c['jaccard'] for c in comparisons) / len(comparisons)
print(f"Average agreement: {avg_similarity:.2f}")
```

## What is EntityFrame?

Traditional entity resolution evaluation compares pairs of records. EntityFrame compares **entities as complete objects** - collections of records that span multiple datasets.

Instead of asking "do these two records match?", EntityFrame asks "how well do these two methods agree on what constitutes this entity?"

## Core Concepts

**Entity**: A collection of record IDs across multiple datasets
```python
# Entity representing "Michael Smith"
{
    "customers": ["cust_001", "cust_045"],
    "transactions": ["txn_555", "txn_777"], 
    "addresses": ["addr_123"]
}
```

**EntityCollection**: Entities from one method (like pandas Series)  
**EntityFrame**: Multiple collections for comparison (like pandas DataFrame)

## Installation

```bash
# From PyPI (when available)
pip install entityframe

# From source
git clone https://github.com/yourusername/entityframe
cd entityframe
just install && just build
```

## Simple Example

```python
import entityframe as ef

# Method 1: Conservative clustering
method1 = [
    {"customers": ["john_1"], "emails": ["john@email.com"]},
    {"customers": ["john_2"], "emails": ["j.smith@work.com"]},
]

# Method 2: Aggressive clustering  
method2 = [
    {"customers": ["john_1", "john_2"], "emails": ["john@email.com", "j.smith@work.com"]},
]

# Create frame and declare datasets upfront for efficiency
frame = ef.EntityFrame()
frame.declare_dataset("customers")
frame.declare_dataset("emails")

# Add the methods
frame.add_method("conservative", method1)
frame.add_method("aggressive", method2)

# See how they differ
results = frame.compare_collections("conservative", "aggressive")
for result in results:
    print(f"Entity {result['entity_index']}: {result['jaccard']:.2f} similarity")
```

## Working with Individual Entities

```python
# Create an entity manually
entity = ef.Entity()
entity.add_records("customers", [1, 2, 3])
entity.add_records("transactions", [100, 101])

# Query the entity
print(f"Total records: {entity.total_records()}")
print(f"Customer records: {entity.get_records('customers')}")
print(f"Has transactions: {entity.has_dataset('transactions')}")

# Compare two entities
other_entity = ef.Entity()
other_entity.add_records("customers", [2, 3, 4])
similarity = entity.jaccard_similarity(other_entity)
print(f"Jaccard similarity: {similarity:.2f}")
```

## API Overview

### EntityFrame
```python
frame = ef.EntityFrame()
frame = ef.EntityFrame.with_datasets(["ds1", "ds2"])  # Pre-declare datasets (recommended)
frame.declare_dataset(name)                      # Declare single dataset
frame.add_method(name, entity_data)              # Add method results (auto-declares datasets)
frame.compare_collections(name1, name2)          # Compare two methods
frame.get_collection(name)                       # Get specific collection
frame.get_collection_names()                     # List all methods
frame.total_entities()                           # Count total entities
```

### EntityCollection  
```python
collection = ef.EntityCollection("method_name")
collection.get_entity(index)                     # Get entity by index
collection.get_entities()                        # Get all entities
collection.len()                                 # Number of entities
collection.compare_with(other_collection)        # Compare collections
```

### Entity
```python
entity = ef.Entity()
entity.add_record(dataset, record_id)           # Add single record
entity.add_records(dataset, record_ids)         # Add multiple records
entity.get_records(dataset)                     # Get records for dataset
entity.jaccard_similarity(other_entity)         # Compare entities
entity.total_records()                          # Count all records
```

## Data Format

EntityFrame expects your entity resolution results as a list of dictionaries:

```python
entity_data = [
    {
        "dataset1": ["record_1", "record_2"],    # Records from dataset1
        "dataset2": ["record_10"],               # Records from dataset2
    },
    {
        "dataset1": ["record_3"],
        "dataset3": ["record_20", "record_21"],  # Different datasets per entity
    },
    # ... more entities
]
```

## Performance

EntityFrame is built for scale:
- **String interning**: 10-100x memory reduction
- **Roaring bitmaps**: SIMD-optimized set operations  
- **Rust core**: High-performance implementation
- **Shared memory**: Multiple methods share string pools

Handle millions of entities efficiently.

## Development

```bash
just install    # Install dependencies
just build      # Build Rust extension  
just test-all   # Run all tests
just format     # Format code
```

## License

MIT License